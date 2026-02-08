"""
qnbench.envs.env
=================

Gymnasium-compatible wrapper around :class:`QuantumNetworkEngine`.

Observation shape : ``(num_nodes, 18)``
Action space      : ``MultiDiscrete([7] * num_nodes)``
"""

from __future__ import annotations

import heapq
import logging
from typing import List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import EnvConfig, ACTION_NAMES
from .engine import QuantumNetworkEngine

logger = logging.getLogger("qnbench.env")


class QuantumNetworkEnv(gym.Env):
    """
    Quantum Network Entanglement Distribution Environment.

    Each step:
    1. Validate & correct actions (invalid → Wait + penalty).
    2. Execute valid actions in random order (for fairness).
    3. Advance the discrete-event engine to the next critical event.
    4. Check for end-to-end delivery above the fidelity threshold.

    Returns
    -------
    obs, reward, terminated, truncated, info
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        cfg: Optional[EnvConfig] = None,
        verbose: bool = False,
        # backward-compat kwargs
        node_distances=None,
        oracle_mode=None,
    ):
        super().__init__()

        # ── Build config ─────────────────────────────────────────
        if cfg is None:
            cfg = EnvConfig()
        if node_distances is not None:
            cfg.node_distances = node_distances
        if oracle_mode is not None:
            cfg.oracle_mode = oracle_mode
        self.cfg = cfg
        self.verbose = verbose

        # ── Logging ─────────────────────────────────────────────────
        # 仿真细节日志由 setup_logging(sim_level=...) 统一控制。
        # verbose 标志仅用于旧版兼容: 开启时强制仿真 logger 为 INFO。
        if verbose:
            from qnbench.utils.logging import ensure_logging
            ensure_logging()
            logging.getLogger("qnbench.env").setLevel(logging.INFO)
            logging.getLogger("qnbench.engine").setLevel(logging.INFO)

        # ── Env dimensions ───────────────────────────────────────
        self.num_nodes = cfg.num_nodes
        self.mem_per_node = cfg.mem_per_node
        self.obs_dim = 18   # features per node

        self.action_space = spaces.MultiDiscrete([7] * self.num_nodes)
        self.observation_space = spaces.Box(
            low=0.0, high=10.0,
            shape=(self.num_nodes, self.obs_dim),
            dtype=np.float32,
        )

        # ── Engine ───────────────────────────────────────────────
        self._rng = np.random.default_rng()
        self.engine = QuantumNetworkEngine(cfg, self._rng)
        self._step_count = 0

    # =============================================================
    # Gym API
    # =============================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = self.np_random
        self.engine.set_rng(self._rng)
        self.engine.reset()
        self._step_count = 0

        # Build the linear chain: Endpoint — Repeater(s) — Endpoint
        self.engine.add_node(self.mem_per_node, is_repeater=False)
        for _ in range(1, self.num_nodes - 1):
            self.engine.add_node(self.mem_per_node, is_repeater=True)
        self.engine.add_node(self.mem_per_node, is_repeater=False)

        return self._build_obs(), {}

    def step(self, actions):
        t0 = self.engine.current_time
        self._step_count += 1

        # ── 奖励组件追踪 ────────────────────────────────────────
        r_ops = 0.0         # 动作操作成本
        r_invalid = 0.0     # 无效动作惩罚
        r_failed = 0.0      # 并发冲突失败惩罚
        r_engine = 0.0      # 引擎内部奖励 (gen/swap/purify 成功)
        r_delivery = 0.0    # 交付奖励
        r_time = 0.0        # 时间惩罚

        # ── 1. Validate ─────────────────────────────────────────
        masks = self.get_action_mask()
        corrected = []
        for ni, act in enumerate(actions):
            if act < 0 or act >= 7 or not masks[ni, act]:
                corrected.append(0)
                r_invalid += self.cfg.invalid_action_penalty
            else:
                corrected.append(act)

        # ── 2. Execute (random order) ────────────────────────────
        self.engine.accumulated_step_reward = 0.0
        order = list(enumerate(corrected))
        self._rng.shuffle(order)

        for ni, act in order:
            cost = self.cfg.op_cost[act]
            # Re-check validity (concurrent actions may have changed state)
            if act != 0 and not self.engine.can_do_action(ni, act):
                r_failed += self.cfg.failed_action_penalty
                continue
            r_ops += cost
            if act == 1:
                self.engine.req_entangle(ni, 0)
            elif act == 2:
                self.engine.req_entangle(ni, 1)
            elif act == 3:
                self.engine.req_swap(ni)
            elif act == 4:
                self.engine.req_purify(ni, "left")
            elif act == 5:
                self.engine.req_purify(ni, "right")
            elif act == 6:
                self.engine.req_discard(ni)

        # ── 3. Advance simulation ────────────────────────────────
        if not self.engine.event_queue:
            self.engine.current_time += 1.0
        else:
            while True:
                evt = self.engine.pop_event()
                if evt is None:
                    break
                self.engine.current_time = evt.timestamp
                if evt.callback(evt.data):
                    break  # critical event → return control

        # ── 4. Time penalty ──────────────────────────────────────
        dt = self.engine.current_time - t0
        r_time = self.cfg.time_penalty_per_ms * max(0.1, dt)
        r_engine = self.engine.accumulated_step_reward

        # ── 5. Delivery check ────────────────────────────────────
        terminated = False
        for m in self.engine.nodes[0].memories:
            if (m.is_entangled_and_available()
                    and m.entangled_node == self.num_nodes - 1):
                link = self.engine.links.get(m.link_id)
                if link is None:
                    continue
                cf = link.current_fidelity(self.engine.current_time,
                                           self.cfg.t_coherence)
                if cf >= self.cfg.delivery_fidelity_threshold:
                    bonus = ((cf - self.cfg.delivery_fidelity_threshold)
                             * self.cfg.delivery_bonus_factor)
                    r = self.cfg.delivery_reward + bonus
                    r_delivery += r
                    self.engine.delivered_pairs += 1
                    self.engine._destroy_link(link.link_id)
                    logger.info(
                        "🎉 DELIVERY F=%.4f span=%.0fkm swaps=%d +%.1f",
                        cf, link.span_distance, link.swap_count, r,
                    )

        # ── 6. Total reward ──────────────────────────────────────
        reward = r_ops + r_invalid + r_failed + r_engine + r_delivery + r_time

        # ── 7. Truncation ────────────────────────────────────────
        truncated = self._step_count >= self.cfg.max_steps

        info = {
            "delivered": self.engine.delivered_pairs,
            "time_ms": self.engine.current_time,
            "step": self._step_count,
            # 奖励分项 (用于 debug 和分析)
            "reward_breakdown": {
                "op_cost": r_ops,
                "invalid": r_invalid,
                "failed": r_failed,
                "engine": r_engine,     # gen + swap + purify rewards
                "delivery": r_delivery,
                "time": r_time,
            },
        }
        return self._build_obs(), reward, terminated, truncated, info

    # =============================================================
    # Action Mask
    # =============================================================

    def get_action_mask(self) -> np.ndarray:
        """
        Return a ``(num_nodes, 7)`` boolean mask of valid actions.

        Agents should use this to avoid selecting invalid actions.
        """
        masks = np.ones((self.num_nodes, 7), dtype=bool)
        for i in range(self.num_nodes):
            for a in range(7):
                masks[i, a] = self.engine.can_do_action(i, a)
        return masks

    # =============================================================
    # Observation
    # =============================================================

    def _build_obs(self) -> np.ndarray:
        """
        Build the observation tensor  ``(num_nodes, 18)``.

        Per-node features (18 total):
            0   : is_endpoint (1/0)
            1   : is_repeater (1/0)
            2   : memory utilisation (fraction busy + entangled)
            3   : # available left links
            4-5 : best / 2nd-best left fidelity  (oracle) or age (exp)
            6   : # available right links
            7-8 : best / 2nd-best right fidelity  (oracle) or age (exp)
            9-10: mean left / right fidelity or age
            11  : normalised distance to left neighbour
            12  : normalised distance to right neighbour
            13-14: mean left / right link span distance (normalised)
            15  : normalised position along chain
            16-17: mean left / right swap count (normalised)
        """
        obs = np.zeros((self.num_nodes, self.obs_dim), dtype=np.float32)
        max_d = sum(self.cfg.node_distances) or 1.0
        tc = self.cfg.t_coherence

        for i, node in enumerate(self.engine.nodes):
            ll = self.engine.get_links_by_direction(i, "left")
            rl = self.engine.get_links_by_direction(i, "right")
            la = self.engine.get_links_by_direction(i, "left", True)
            ra = self.engine.get_links_by_direction(i, "right", True)

            busy = sum(m.is_locked() for m in node.memories)
            ent = sum(m.is_entangled_and_available() for m in node.memories)

            obs[i, 0] = float(not node.is_repeater)
            obs[i, 1] = float(node.is_repeater)
            obs[i, 2] = (busy + ent) / len(node.memories)
            obs[i, 3] = len(la)
            obs[i, 6] = len(ra)

            if self.cfg.oracle_mode:
                # Oracle mode: expose exact fidelities
                lf = sorted(
                    [l.current_fidelity(self.engine.current_time, tc) for l in ll],
                    reverse=True,
                )
                rf = sorted(
                    [l.current_fidelity(self.engine.current_time, tc) for l in rl],
                    reverse=True,
                )
                obs[i, 4] = lf[0] if lf else 0
                obs[i, 5] = lf[1] if len(lf) > 1 else 0
                obs[i, 7] = rf[0] if rf else 0
                obs[i, 8] = rf[1] if len(rf) > 1 else 0
                obs[i, 9] = float(np.mean(lf)) if lf else 0
                obs[i, 10] = float(np.mean(rf)) if rf else 0
            else:
                # Experimental mode: expose normalised link age
                lag = sorted(
                    [l.age(self.engine.current_time) / tc for l in ll]
                )
                rag = sorted(
                    [l.age(self.engine.current_time) / tc for l in rl]
                )
                obs[i, 4] = lag[0] if lag else 1
                obs[i, 5] = lag[1] if len(lag) > 1 else 1
                obs[i, 7] = rag[0] if rag else 1
                obs[i, 8] = rag[1] if len(rag) > 1 else 1
                obs[i, 9] = float(np.mean(lag)) if lag else 1
                obs[i, 10] = float(np.mean(rag)) if rag else 1

            dists = self.cfg.node_distances
            obs[i, 11] = dists[i - 1] / max_d if i > 0 else 0
            obs[i, 12] = dists[i] / max_d if i < self.num_nodes - 1 else 0
            obs[i, 13] = (float(np.mean([l.span_distance for l in ll])) / max_d
                          if ll else 0)
            obs[i, 14] = (float(np.mean([l.span_distance for l in rl])) / max_d
                          if rl else 0)
            obs[i, 15] = node.position / max_d

            ls = [l.swap_count for l in ll]
            rs = [l.swap_count for l in rl]
            obs[i, 16] = float(np.mean(ls)) / 3 if ls else 0
            obs[i, 17] = float(np.mean(rs)) / 3 if rs else 0

        return obs

    # =============================================================
    # Render
    # =============================================================

    def render(self):
        lines = [
            f"t={self.engine.current_time:.2f}ms  step={self._step_count}  "
            f"delivered={self.engine.delivered_pairs}  "
            f"links={len(self.engine.links)}  "
            f"ops={self.engine.ops.get_active_count()}",
        ]
        for l in sorted(self.engine.links.values(), key=lambda x: x.link_id):
            f = l.current_fidelity(self.engine.current_time, self.cfg.t_coherence)
            lines.append(
                f"  Link {l.link_id}: {l.node_a}↔{l.node_b}  "
                f"F={f:.4f}  age={l.age(self.engine.current_time):.1f}ms  "
                f"ttl={l.ttl(self.engine.current_time):.1f}ms"
            )
        text = "\n".join(lines)
        print(text)
        return text

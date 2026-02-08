"""
qnbench.evaluation.runner
==========================

Run one or more agents on the quantum network environment and collect
per-episode metrics.

Usage::

    from qnbench.evaluation.runner import evaluate_agent
    from qnbench.baselines import SwapASAPAgent
    from qnbench.envs import QuantumNetworkEnv, EnvConfig

    env = QuantumNetworkEnv(EnvConfig())
    agent = SwapASAPAgent(num_nodes=4)
    results = evaluate_agent(env, agent, n_episodes=100, seed=42)
    print(results)
"""

from __future__ import annotations

import sys
import time
import numpy as np
from typing import Dict, List

from qnbench.envs import QuantumNetworkEnv, EnvConfig
from qnbench.baselines.base import BaseAgent
from qnbench.evaluation.metrics import EpisodeMetrics, aggregate_metrics
from qnbench.utils.logging import ensure_logging


def evaluate_agent(
    env: QuantumNetworkEnv,
    agent: BaseAgent,
    n_episodes: int = 100,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Run *agent* on *env* for *n_episodes* and return aggregated metrics.

    Parameters
    ----------
    env : the Gymnasium environment
    agent : any ``BaseAgent`` subclass
    n_episodes : number of evaluation episodes
    seed : base seed (episode i uses seed + i)
    verbose : print per-episode progress

    Returns
    -------
    dict with keys:
        mean_reward, std_reward,
        mean_delivered, total_delivered,
        mean_episode_time_ms, mean_steps,
        wall_time_s
    """
    ensure_logging()
    all_metrics: List[EpisodeMetrics] = []
    wall_start = time.time()

    # 计算 verbose 打印间隔 (至少 10 次输出, 至少每个 episode 都能触发)
    print_every = max(1, n_episodes // 10)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        agent.reset()

        ep_reward = 0.0
        ep_deliveries = 0
        terminated = truncated = False

        while not (terminated or truncated):
            mask = env.get_action_mask()
            actions = agent.act(obs, mask)
            obs, reward, terminated, truncated, info = env.step(actions)
            ep_reward += reward
            ep_deliveries = info.get("delivered", 0)

        all_metrics.append(EpisodeMetrics(
            reward=ep_reward,
            delivered=ep_deliveries,
            time_ms=info.get("time_ms", 0.0),
            steps=info.get("step", 0),
        ))

        if verbose and (ep + 1) % print_every == 0:
            m = all_metrics[-1]
            pct = (ep + 1) / n_episodes * 100
            print(
                f"  [{pct:5.1f}%] Episode {ep + 1:4d}/{n_episodes}: "
                f"reward={m.reward:+8.2f}  delivered={m.delivered}  "
                f"steps={m.steps}",
                flush=True,
            )

    wall_time = time.time() - wall_start
    result = aggregate_metrics(all_metrics)
    result["wall_time_s"] = wall_time
    result["agent"] = agent.name
    return result


def compare_agents(
    cfg: EnvConfig,
    agents: Dict[str, BaseAgent],
    n_episodes: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> List[Dict[str, float]]:
    """
    Evaluate multiple agents on the same environment config and return
    a list of result dicts (one per agent).
    """
    ensure_logging()
    env = QuantumNetworkEnv(cfg=cfg, verbose=False)
    results = []

    for name, agent in agents.items():
        if verbose:
            print(f"\n{'─'*60}", flush=True)
            print(f"  Evaluating: {name}  ({n_episodes} episodes)", flush=True)
            print(f"{'─'*60}", flush=True)

        r = evaluate_agent(env, agent, n_episodes, seed, verbose=verbose)
        r["agent"] = name
        results.append(r)

        if verbose:
            print(
                f"  → reward={r['mean_reward']:+.2f} ± {r['std_reward']:.2f}  "
                f"delivered={r['total_delivered']}  "
                f"wall={r['wall_time_s']:.1f}s",
                flush=True,
            )

    return results

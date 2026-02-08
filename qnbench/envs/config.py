"""
qnbench.envs.config
====================

Centralised configuration for the quantum network environment.

All tuneable parameters live in the ``EnvConfig`` dataclass so that
experiments are fully reproducible from a single config object (or
YAML file, via ``EnvConfig.from_yaml``).
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── Physical constants ───────────────────────────────────────────
SPEED_OF_LIGHT_FIBER = 2e5          # km/s  (≈ 2/3 c in fiber)
FIDELITY_MIXED_STATE = 0.25         # F of maximally mixed state

# ── Human-readable action names ──────────────────────────────────
ACTION_NAMES = {
    0: "Wait",
    1: "Gen_L",
    2: "Gen_R",
    3: "Swap",
    4: "Purify_L",
    5: "Purify_R",
    6: "Discard",
}
NUM_ACTIONS = len(ACTION_NAMES)


@dataclass
class EnvConfig:
    """
    Complete configuration for ``QuantumNetworkEnv``.

    Reward Design Principles (V8)
    =============================

    旧版 (V7) 的 step_penalty = -0.2 导致 500 步累计 -100 固定惩罚,
    占总奖励 145%~760%, 淹没了所有学习信号。RL agent 无法区分
    "好 episode" 和 "坏 episode"，因为奖励方差被固定惩罚主导。

    V8 修复原则:
    1. 删除 step_penalty → 用 time_penalty 按仿真时间惩罚 (物理合理)
    2. 中间奖励 (gen/swap/purify) 总量 ≈ episode 惩罚的 50~80%
       → 给 agent 一个从 "完全随机" 到 "有效操作" 的上坡梯度
    3. delivery 奖励 >> 所有中间奖励之和 → 明确的最终目标信号
    4. 并发冲突 (failed_action) 极轻惩罚 → 不是 agent 的错

    典型 episode 奖励预算 (SwapASAP+Purify, 500 steps):
        time_penalty    ≈ -12 to -15  (仿真 ~1200ms, 按 -0.01/ms)
        op_cost         ≈ -8 to -10   (轻微资源消耗信号)
        engine rewards  ≈ +15 to +25  (gen/swap/purify 成功)
        delivery (×2)   ≈ +100 to +120
        ---
        net             ≈ +90 to +120

    Random agent 预算:
        time_penalty    ≈ -12
        op_cost         ≈ -5
        engine          ≈ +5
        delivery (×0.5) ≈ +25
        ---
        net             ≈ +13  (正值! 但远低于好策略)
    """

    # ── Topology ─────────────────────────────────────────────────
    node_distances: List[float] = field(
        default_factory=lambda: [50.0, 50.0, 50.0],
        metadata={"help": "Distance (km) between adjacent nodes."},
    )
    num_nodes: int = 4
    mem_per_node: int = 4

    # ── Physics ──────────────────────────────────────────────────
    prob_gen: float = 0.6           # per-attempt generation success prob
    prob_swap: float = 0.8          # BSM success probability
    prob_purify: float = 0.7        # purification success probability
    t_coherence: float = 2000.0     # quantum memory coherence time (ms)
    init_fidelity: float = 0.95     # fidelity of freshly generated pairs
    swap_resets_coherence: bool = False  # True → V6 behaviour (full reset)

    # ── Timing ───────────────────────────────────────────────────
    gate_time_swap: float = 0.001       # Bell measurement gate (ms)
    gate_time_purify: float = 0.001     # purification gate (ms)
    classical_processing: float = 0.1   # classical result processing (ms)
    gen_clock_rate_mhz: float = 10.0    # entanglement attempt clock (MHz)

    # ── Reward: 惩罚项 ──────────────────────────────────────────

    # 仿真时间惩罚: 每 ms 惩罚。物理上合理 (链路老化, 资源占用)
    # 典型 episode 约 1000~2500ms → 惩罚 -10 to -25
    time_penalty_per_ms: float = -0.01

    # 无效动作惩罚: 选了不可执行的动作 (如 endpoint 选 Swap)
    # 有 action mask, 理论上不应触发; 但仍需惩罚以训练未使用 mask 的 agent
    invalid_action_penalty: float = -1.0

    # 并发冲突惩罚: 合法动作因其他节点先执行而变得不可用
    # 这不是 agent 的决策错误, 只是 multi-agent 同步限制, 极轻惩罚
    failed_action_penalty: float = -0.02

    # 各动作的资源消耗成本 (很小的 shaping 信号)
    op_cost: Dict[int, float] = field(default_factory=lambda: {
        0: 0.0,     # Wait: 免费
        1: -0.02,   # Gen_L: 消耗光子资源
        2: -0.02,   # Gen_R: 消耗光子资源
        3: -0.05,   # Swap: 消耗 BSM 设备时间
        4: -0.05,   # Purify_L: 牺牲一条链路
        5: -0.05,   # Purify_R: 牺牲一条链路
        6: -0.01,   # Discard: 微量清理成本
    })

    # ── Reward: 中间奖励 (shaping) ──────────────────────────────

    # 生成成功: 基础构建块, 每次 gen 给一个小正信号
    gen_success_reward: float = 0.3

    # 交换成功: 延长纠缠跨度, 接近交付目标
    swap_success_reward: float = 1.0

    # 链路跨度进度: swap 后链路跨度越长, 奖励越大 (归一化 0~1)
    # 这给 RL agent 一个从 "1-hop 链路" → "2-hop" → "end-to-end" 的梯度
    span_progress_reward: float = 2.0

    # 纯化成功: 提升保真度 (防止 swap 后 F 低于交付阈值)
    purify_base_reward: float = 0.5
    purify_bonus_factor: float = 5.0    # × (F_new - F_old) 额外奖励

    # ── Reward: 交付奖励 (核心目标) ─────────────────────────────

    # 必须远大于中间奖励之和, 确保 agent 追求交付而非只做中间操作
    delivery_reward: float = 50.0
    delivery_bonus_factor: float = 50.0     # × (F - threshold) 超阈值 bonus
    delivery_fidelity_threshold: float = 0.8

    # ── Environment ──────────────────────────────────────────────
    max_steps: int = 500
    oracle_mode: bool = True        # True → expose fidelity; False → age

    # ── Derived ──────────────────────────────────────────────────
    @property
    def gen_attempt_period(self) -> float:
        """Time between successive generation attempts (ms)."""
        if self.gen_clock_rate_mhz > 0:
            return 1.0 / (self.gen_clock_rate_mhz * 1000)
        return 0.001

    # ── Serialisation ────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str) -> "EnvConfig":
        """Load config from a YAML file (reads the ``env:`` section)."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        env_cfg = raw.get("env", {})
        return cls(**{k: v for k, v in env_cfg.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for logging / checkpoints)."""
        from dataclasses import asdict
        return asdict(self)

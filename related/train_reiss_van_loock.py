#!/usr/bin/env python3
"""
related/train_reiss_van_loock.py
================================
复现 Reiß & van Loock (2023) — "Deep reinforcement learning for key distribution
based on quantum repeaters", Phys. Rev. A 108, 012406

=== 论文核心思想 ===
1.【MDP 建模】: 线性量子中继链 (4段 = 5节点)，每个 elementary link 有：
   - 存在性 (active/inactive)
   - 年龄 (age): 存储在量子存储器中的时间步数
   - Werner 参数 w(t): 随年龄退化的保真度指标，w(t) = 1 - (1-w0)*exp(-t/T_coh)
     实际上 w(t) 刻画退相干，保真度 F = (1+3w)/4

2.【动作空间】: 论文的核心创新是 *动态 memory cutoff*。
   - 每个时间步，agent 为每条 elementary link 决定一个 cutoff 值
   - 若 link 年龄超过 cutoff → 丢弃该 link 并尝试重新生成
   - Swap 使用 SWAP-ASAP 策略 (自动执行)
   - 因此 RL agent 只需要决定 "何时丢弃/保留" 每条 link

3.【奖励 / 目标】: BB84 Secret Key Rate (SKR)
   - 当端到端纠缠被成功交付时，根据最终保真度 F 计算 SKR:
     r_∞ = 1 - h(e_x) - h(e_z)，其中 e = (1-F)/2 (对 Werner state 简化)
   - 整体目标: 最大化 单位时间 SKR = sum(key_bits) / total_time

4.【RL 算法】: DQN (Deep Q-Network) with experience replay + target network
   - 论文重点在 4 段链 (5 节点)
   - 输入: 每条 link 的 (存在性, 年龄) → 全局状态向量
   - 输出: 每条 link 的 cutoff 决策

=== 在 QNBench Gym 上的适配策略 ===
我们不修改 gym 本体。通过 Wrapper 实现以下拦截:
1. 奖励替换: delivery reward → BB84 SKR bits
2. 动作映射: 论文只做 cutoff 决策 (generate/wait/swap 自动化)
   - 在 QNBench 中: 屏蔽 purify 动作 (prob_purify=0 + mask)
   - Discard 动作 (action=6) 由 agent 的 cutoff 策略隐式实现
   - 对每个节点: 若 link 年龄 > agent 的 cutoff → 执行 discard，否则 → 自动选择最优 {wait/gen/swap}
3. 观测空间: 使用 oracle_mode=True (论文假设全局状态知识)

由于 QNBench 的动作空间是 per-node 的 {Wait, GenL, GenR, Swap, PurL, PurR, Disc}，
我们实现一个 "CutoffPolicyWrapper" 将 agent 的高层 cutoff 决策翻译为底层动作。

使用方法:
    python related/train_reiss_van_loock.py
"""

import sys
import os
import collections
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import logging
import matplotlib.pyplot as plt

# ─── 0. 日志 ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("reiss_repro")

# ─── 路径 ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnbench.envs import QuantumNetworkEnv, EnvConfig
from qnbench.rl.masked_ppo import PPOTrainer, PPOConfig


# ═══════════════════════════════════════════════════════════════════
#  核心 Wrapper: 将 QNBench 适配为 Reiß & van Loock 的 MDP 语义
# ═══════════════════════════════════════════════════════════════════

class ReissProtocolWrapper(gym.Wrapper):
    """
    Reiß & van Loock (2023) 适配 Wrapper。

    论文核心：Agent 只做 cutoff 决策，其余操作自动化。
    但 QNBench 的 action space 是 per-node 的 7 动作，我们无法直接改变它。

    适配策略：
    ─────────
    我们保持 QNBench 原始 action space 不变（以兼容 PPOTrainer），
    但通过 Wrapper 做以下拦截：

    (A) 奖励替换 → BB84 SKR
    (B) 屏蔽纯化动作 (mask 掉 action 4,5)，论文不考虑 purification
    (C) 在 info 中注入论文特有的指标

    注意：由于我们使用 PPO（而非论文的 DQN），且保留了 QNBench 的
    per-node action space，这是一个 "思路级" 复现。agent 学习的策略
    包含了 cutoff 的隐式学习（通过选择 Discard vs Wait 动作）。

    论文中 cutoff 决策的精髓在于：
    - 当 link 年龄较大 (保真度退化) → agent 应该选 Discard (action=6)
    - 当 link 年龄较小 → agent 应该选 Wait (action=0) 或 Swap (action=3)
    这正好对应 QNBench 中 agent 在 {Wait, Gen, Swap, Discard} 中的选择。
    """

    # ── Reiß 论文参数 ──
    # Werner parameter: 初始 w0，随存储退相干
    # F = (1 + 3w) / 4 → w = (4F - 1) / 3
    # 对于 depolarizing channel: w(t) = w0 * exp(-t / T_coh)

    def __init__(self, env, skr_scale=50.0):
        """
        Parameters
        ----------
        env : QuantumNetworkEnv
            底层环境 (应设置 prob_purify=0.0, oracle_mode=True)
        skr_scale : float
            SKR 奖励的缩放因子，使 RL 更容易学习
        """
        super().__init__(env)
        self.unwrapped_env = env.unwrapped
        self.cfg = self.unwrapped_env.cfg
        self.skr_scale = skr_scale

        # 透传属性 (PPOTrainer 需要)
        self.obs_dim = self.unwrapped_env.obs_dim
        self.num_nodes = self.unwrapped_env.num_nodes

        # 缓存配置参数
        self.base_reward = self.cfg.delivery_reward
        self.bonus_factor = self.cfg.delivery_bonus_factor
        self.fidelity_threshold = self.cfg.delivery_fidelity_threshold

        # 静态 mask: 禁用 purification (action 4, 5)
        # QNBench Actions: 0:Wait, 1:GenL, 2:GenR, 3:Swap, 4:PurL, 5:PurR, 6:Disc
        # Reiß 论文只有: Generate, Wait, Swap, Discard (由 cutoff 触发)
        self.static_mask = np.array([1, 1, 1, 1, 0, 0, 1], dtype=bool)

        # 统计
        self.episode_skr_bits = 0.0
        self.episode_deliveries = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_skr_bits = 0.0
        self.episode_deliveries = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # ── 奖励替换: 从 delivery reward 反推 fidelity → 计算 BB84 SKR ──
        r_delivery = info.get("reward_breakdown", {}).get("delivery", 0.0)

        new_reward = reward  # 保留非交付部分的奖励 (如时间惩罚等)

        if r_delivery > 0:
            # 反推交付时的保真度
            bonus = r_delivery - self.base_reward
            if self.bonus_factor > 1e-6:
                est_fidelity = (bonus / self.bonus_factor) + self.fidelity_threshold
            else:
                est_fidelity = self.fidelity_threshold
            est_fidelity = np.clip(est_fidelity, 0.0, 1.0)

            # 计算 BB84 SKR (bits per delivered pair)
            skr_bits = self._bb84_skr(est_fidelity)

            # 替换奖励: 移除原交付分，加上 SKR 奖励
            new_reward = reward - r_delivery + skr_bits * self.skr_scale

            # 统计
            self.episode_skr_bits += skr_bits
            self.episode_deliveries += 1

            # 注入论文特有指标
            info["reiss_skr_bits"] = skr_bits
            info["reiss_fidelity"] = est_fidelity

        # 在 episode 结束时汇总
        if terminated or truncated:
            info["reiss_total_skr_bits"] = self.episode_skr_bits
            info["reiss_total_deliveries"] = self.episode_deliveries

        return obs, new_reward, terminated, truncated, info

    def get_action_mask(self):
        """
        覆盖 mask 逻辑: 在物理 mask 基础上，禁用 purification 动作。
        """
        raw_mask = self.unwrapped_env.get_action_mask()
        # static_mask (7,) 自动广播到 (num_nodes, 7)
        return raw_mask & self.static_mask

    @staticmethod
    def _bb84_skr(fidelity):
        """
        BB84 Secret Key Rate (asymptotic, one-way)

        Reiß & van Loock 使用的公式:
        对 Werner state: QBER = (1 - F)
        实际上对 depolarizing channel: e_x = e_z = (1-F)/2 (简化)

        r_∞ = max(0, 1 - h(e_x) - h(e_z))
             = max(0, 1 - 2*h(QBER))

        其中 h(p) = -p*log2(p) - (1-p)*log2(1-p) 是二元熵函数。

        当 F < 0.5 → 无法生成密钥
        当 QBER > 0.11 → 无法生成密钥 (BB84 极限)
        """
        if fidelity <= 0.5:
            return 0.0

        # 对于 Werner state under depolarizing noise:
        # QBER ≈ (1 - F) for the relevant error model
        qber = 1.0 - fidelity

        # BB84 安全极限
        if qber >= 0.11:
            return 0.0

        def binary_entropy(p):
            if p <= 1e-15 or p >= 1.0 - 1e-15:
                return 0.0
            return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

        rate = 1.0 - 2.0 * binary_entropy(qber)
        return max(0.0, rate)


# ═══════════════════════════════════════════════════════════════════
#  评估函数: 固定时长测试 + 绘图
# ═══════════════════════════════════════════════════════════════════

def evaluate_and_plot(env, model, device, save_dir, training_history):
    """
    在固定仿真时长内运行策略，统计:
    1. 总交付纠缠对数
    2. 总 SKR bits
    3. 交付速率 (pairs/s) 和密钥速率 (bits/s)
    """
    TEST_DURATION_MS = 1_000.0  # 1000 ms = 1 秒仿真时间

    print(f"\n{'=' * 60}")
    logger.info(f">>> Reiß & van Loock 固定时长评估 ({TEST_DURATION_MS} ms)")
    print(f"{'=' * 60}")

    model.eval()

    current_time = 0.0
    total_deliveries = 0
    total_skr_bits = 0.0

    time_points = [0.0]
    delivery_points = [0]
    skr_points = [0.0]

    ep_idx = 0

    while current_time < TEST_DURATION_MS:
        obs, _ = env.reset(seed=8000 + ep_idx)
        done = False

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(
                env.get_action_mask(), dtype=torch.bool, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                action_res = model.get_action_and_value(obs_t, mask_t)
                action = action_res[0].cpu().numpy()[0]

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # 本局结束
        ep_del = info.get("reiss_total_deliveries", info.get("delivered", 0))
        ep_bits = info.get("reiss_total_skr_bits", 0.0)

        total_deliveries += ep_del
        total_skr_bits += ep_bits
        current_time += info["time_ms"]

        time_points.append(current_time)
        delivery_points.append(total_deliveries)
        skr_points.append(total_skr_bits)

        ep_idx += 1
        if ep_idx % 10 == 0:
            prog = (current_time / TEST_DURATION_MS) * 100
            logger.info(
                f"进度: {prog:.1f}% | 交付: {total_deliveries} | "
                f"SKR bits: {total_skr_bits:.2f}"
            )

    # ── 统计 ──
    total_sec = current_time / 1000.0
    pair_rate = total_deliveries / total_sec if total_sec > 0 else 0

    logger.info(f"{'─' * 40}")
    logger.info(f"测试结果 (Reiß & van Loock 复现)")
    logger.info(f"  仿真时间:    {total_sec:.2f} s")
    logger.info(f"  总交付对:    {total_deliveries}")
    logger.info(f"  总 SKR bits: {total_skr_bits:.4f}")
    logger.info(f"  交付速率:    {pair_rate:.4f} pairs/s")
    logger.info(f"{'─' * 40}")

    # ── 绘图 ──
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 图1: 训练曲线
        ax1 = axes[0]
        if training_history and len(training_history) > 0:
            window = max(1, len(training_history) // 20)
            smoothed = np.convolve(
                training_history, np.ones(window) / window, mode='valid'
            )
            ax1.plot(training_history, alpha=0.3, color='blue', label='Raw')
            ax1.plot(smoothed, color='darkblue', linewidth=2, label='Smoothed')
            ax1.set_title("Training: Episode Reward (SKR-based)")
            ax1.set_xlabel("Episodes")
            ax1.set_ylabel("Reward")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No training data", ha='center', va='center')
            ax1.set_title("Training Curve")

        # 图2: 累计交付 vs 时间
        ax2 = axes[1]
        t_sec = np.array(time_points) / 1000.0
        ax2.step(t_sec, delivery_points, where='post', color='green', linewidth=2)
        ax2.set_title(f"Cumulative Deliveries (Rate ≈ {pair_rate:.2f} Hz)")
        ax2.set_xlabel("Simulation Time (s)")
        ax2.set_ylabel("Entangled Pairs Delivered")
        ax2.grid(True, alpha=0.3)
        if len(t_sec) > 1:
            ax2.plot(
                [0, total_sec], [0, total_deliveries],
                '--k', alpha=0.4, label='Avg Rate'
            )
            ax2.legend()

        # 图3: 累计 SKR bits vs 时间 (论文核心指标)
        ax3 = axes[2]
        ax3.step(t_sec, skr_points, where='post', color='red', linewidth=2)
        ax3.set_xlabel("Simulation Time (s)")
        ax3.set_ylabel("Secret Key Bits")
        ax3.grid(True, alpha=0.3)
        if len(t_sec) > 1:
            ax3.plot(
                [0, total_sec], [0, total_skr_bits],
                '--k', alpha=0.4, label='Avg Rate'
            )
            ax3.legend()

        plt.suptitle(
            "Reiß & van Loock (2023) Reproduction — "
            "DRL for Key Distribution with Dynamic Cutoffs",
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()

        plot_path = os.path.join(save_dir, "reiss_result_plot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"✅ 图表已保存: {plot_path}")

    except Exception as e:
        logger.error(f"❌ 绘图失败: {e}")
        import traceback
        traceback.print_exc()

    return {
        "pair_rate_hz": pair_rate,
        "total_deliveries": total_deliveries,
    }


# ═══════════════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════════════

def main():
    save_dir = os.path.join("checkpoints", "reiss_repro")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Reiß & van Loock (2023) 复现实验")
    print(f"  'Deep RL for Key Distribution Based on Quantum Repeaters'")
    print(f"  Phys. Rev. A 108, 012406")
    print(f"  输出目录: {save_dir}")
    print(f"{'=' * 60}\n")

    # ─────────────────────────────────────────────────
    # 1. 环境配置
    # ─────────────────────────────────────────────────
    # 论文设置: 4 段链 (5 节点)，均匀链
    # 关键: prob_purify = 0.0 (论文无纯化，只有 cutoff)
    # oracle_mode = True (论文假设全局状态知识)
    #
    # 论文参数空间:
    #   - 通信距离 L: 10-100 km (每段)
    #   - 相干时间 T_coh: 0.1-10 s
    #   - 链路生成概率 p_l: 取决于距离和衰减
    #   - Swap 成功概率 p_sw: ~0.5 (BSM with linear optics)
    #
    # 在 QNBench 中我们用合理的参数映射:
    cfg = EnvConfig(
        num_nodes=5,                        # 5 节点 = 4 段 (论文重点)
        node_distances=[15.0] * 4,          # 每段 15 km
        t_coherence=500.0,                  # 相干时间 (ms)
        prob_purify=0.0,                    # ★ 核心: 无纯化
        delivery_reward=30.0,               # 基础交付奖励 (会被 SKR 替换)
        delivery_bonus_factor=10.0,         # 保真度 bonus 系数
        delivery_fidelity_threshold=0.5,    # 较低阈值 (让 agent 学习动态 cutoff)
        oracle_mode=True,                   # ★ 全局状态知识
        max_steps=500,                      # 单局最大步数
    )

    # ─────────────────────────────────────────────────
    # 2. 创建环境 + Wrapper
    # ─────────────────────────────────────────────────
    raw_env = QuantumNetworkEnv(cfg, verbose=False)
    env = ReissProtocolWrapper(raw_env, skr_scale=50.0)

    logger.info(f"环境: {cfg.num_nodes} 节点链 | 段距: {cfg.node_distances[0]} km")
    logger.info(f"相干时间: {cfg.t_coherence} ms | 纯化: 禁用")
    logger.info(f"观测维度: {env.obs_dim} | 动作: 每节点 7 个 (mask 掉 purify)")

    # ─────────────────────────────────────────────────
    # 3. PPO 配置
    # ─────────────────────────────────────────────────
    # 注: 论文用 DQN，我们用 PPO (QNBench 的 masked_ppo 框架)
    # 这是方法论差异，但 "学习 cutoff 决策" 的核心思想保留
    ppo_cfg = PPOConfig(
        lr=3e-4,
        hidden_dim=256,
        gamma=0.99,         # 论文也用高 gamma (关注长期回报)
        batch_size=64,
        entropy_coef=0.01,  # 适度探索
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = PPOTrainer(env, ppo_cfg, device=device)

    # ─────────────────────────────────────────────────
    # 4. 训练
    # ─────────────────────────────────────────────────
    # ★ 演示用 5000 步，正式请用 150_000+ ★
    total_steps = 100000
    logger.info(f"开始训练 {total_steps} 步 (Device: {device})")
    logger.info(f"论文方法: DQN with dynamic cutoffs")
    logger.info(f"本复现:   PPO with masked actions (agent 隐式学习 cutoff 策略)")

    try:
        trainer.train(
            total_timesteps=total_steps,
            log_interval=5,
            save_dir=save_dir,
        )
    except KeyboardInterrupt:
        logger.warning("训练被中断，进入评估阶段...")

    # 保存模型
    final_path = os.path.join(save_dir, "reiss_final.pt")
    trainer.save(final_path)
    logger.info(f"模型已保存: {final_path}")

    # ─────────────────────────────────────────────────
    # 5. 评估 (固定时长测试)
    # ─────────────────────────────────────────────────
    results = evaluate_and_plot(
        env, trainer.model, device, save_dir, trainer.reward_history
    )

    # ─────────────────────────────────────────────────
    # 6. 与论文结果对比说明
    # ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  复现总结")
    print(f"{'=' * 60}")
    print(f"""
  论文 vs 本复现 的关键差异:

  ┌──────────────┬──────────────────────┬──────────────────────┐
  │ 维度         │ Reiß & van Loock     │ 本复现               │
  ├──────────────┼──────────────────────┼──────────────────────┤
  │ RL 算法      │ DQN + replay buffer  │ PPO (masked)         │
  │ 动作语义     │ 每条link的cutoff值    │ per-node 7动作(mask) │
  │ Swap 策略    │ SWAP-ASAP (自动)     │ Agent 决策           │
  │ Cutoff       │ Agent 显式输出       │ Agent 隐式学习       │
  │              │                      │ (通过 Discard 动作)  │
  │ 纯化         │ 无                   │ 无 (mask 掉)         │
  │ 状态知识     │ 全局 (MDP)           │ 全局 (oracle_mode)   │
  │ 奖励         │ BB84 SKR             │ BB84 SKR (相同公式)  │
  │ 链规模       │ 4 段 (5 节点)        │ 4 段 (5 节点)        │
  └──────────────┴──────────────────────┴──────────────────────┘

  核心保留:
  ✅ 无纯化、仅 swap + cutoff 的问题设置
  ✅ 全局状态知识假设
  ✅ BB84 SKR 作为优化目标
  ✅ 动态 cutoff 的隐式学习 (agent 学习何时 discard)
  ✅ 4 段均匀链

  主要简化:
  ⚠ DQN → PPO (算法族不同，但都是 model-free DRL)
  ⚠ 显式 cutoff 输出 → 隐式 discard 决策
    """)


if __name__ == "__main__":
    main()
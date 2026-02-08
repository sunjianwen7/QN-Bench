#!/usr/bin/env python3
"""
related/train_haldar_2024b_deterministic.py
==============================================
复现 Haldar et al. (2024b) — 确定性策略评估版
(移除 PPO 对比，专注于 FN/SN 及 Distill 策略的有效性验证)

修改说明:
1. 修复了观测解析错误导致的 0 交付问题，改为基于 Action Mask 的鲁棒决策。
2. 移除了 RL (PPO) 部分，仅评估论文提出的手工策略。
"""

import sys
import os
import numpy as np
import gymnasium as gym
import logging
import matplotlib.pyplot as plt

# ─── 日志 ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("haldar2024b")

# ─── 路径 ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnbench.envs import QuantumNetworkEnv, EnvConfig


# ═══════════════════════════════════════════════════════════════
#  核心策略类 (基于 Action Mask 修复版)
# ═══════════════════════════════════════════════════════════════

class HaldarPolicyBase:
    """
    策略基类
    """

    def __init__(self, num_nodes, enable_distill=False, distill_first=True):
        self.num_nodes = num_nodes
        self.enable_distill = enable_distill
        self.distill_first = distill_first

    def get_action(self, obs, action_mask):
        """
        根据策略返回动作。
        action definitions:
        0: Wait, 1: GenL, 2: GenR, 3: Swap, 4: PurL, 5: PurR, 6: Discard
        """
        raise NotImplementedError


class ParallelSwapASAP(HaldarPolicyBase):
    """
    Parallel SWAP-ASAP (Baseline)
    逻辑：只要能换就换，不能换就生成。不进行纯化。
    """

    def get_action(self, obs, action_mask):
        actions = np.zeros(self.num_nodes, dtype=np.int64)
        for i in range(self.num_nodes):
            mask = action_mask[i]

            # 优先级 1: 交换 (Swap)
            if mask[3]:
                actions[i] = 3
            # 优先级 2: 生成 (GenL / GenR)
            elif mask[1]:
                actions[i] = 1
            elif mask[2]:
                actions[i] = 2
            else:
                actions[i] = 0
        return actions


class FNSwapASAP(HaldarPolicyBase):
    """
    FN SWAP-ASAP (含 Distill 选项)

    逻辑:
    - Distill-First: 优先尝试 Purify，然后 Swap，然后 Gen
    - Swap-Distill: 优先 Swap，无法 Swap 时尝试 Purify (针对端点或长链路)，最后 Gen
    """

    def get_action(self, obs, action_mask):
        actions = np.zeros(self.num_nodes, dtype=np.int64)

        for i in range(self.num_nodes):
            mask = action_mask[i]

            # ── Distill-First 模式 ──
            if self.enable_distill and self.distill_first:
                # 只要 mask 允许纯化，就优先纯化 (假设 mask 内部已经 check 了基本条件)
                if mask[4]:
                    actions[i] = 4;
                    continue
                if mask[5]:
                    actions[i] = 5;
                    continue

            # ── 核心 Swap ──
            if mask[3]:
                actions[i] = 3;
                continue

            # ── Swap-Distill 模式 (Fallback) ──
            # 如果不能 Swap (例如只有一侧连接，或者已经 Swap 完了变成长连接)
            # 此时检查是否需要纯化
            if self.enable_distill and not self.distill_first:
                if mask[4]:
                    actions[i] = 4;
                    continue
                if mask[5]:
                    actions[i] = 5;
                    continue

            # ── 基础生成 ──
            if mask[1]:
                actions[i] = 1
            elif mask[2]:
                actions[i] = 2
            else:
                actions[i] = 0

        return actions


class SNSwapASAP(HaldarPolicyBase):
    """
    SN SWAP-ASAP (Strongest Neighbor)
    近似实现：逻辑同 FN，但可以通过调整内部阈值来模拟 (此处为了保证运行，逻辑与 FN 保持一致，
    但概念上它是倾向于保真度的)。
    """

    def get_action(self, obs, action_mask):
        # 复用 FN 逻辑以确保连通性
        return FNSwapASAP.get_action(self, obs, action_mask)


# ═══════════════════════════════════════════════════════════════
#  Wrapper (统计用)
# ═══════════════════════════════════════════════════════════════

class HaldarStatsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.unwrapped_env = env.unwrapped
        self.cfg = self.unwrapped_env.cfg
        self.episode_deliveries = 0
        self.episode_total_fidelity = 0.0
        self.episode_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_deliveries = 0
        self.episode_total_fidelity = 0.0
        self.episode_steps = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1

        r_delivery = info.get("reward_breakdown", {}).get("delivery", 0.0)
        if r_delivery > 0:
            self.episode_deliveries += 1
            # 反推保真度
            base = self.cfg.delivery_reward
            bonus = r_delivery - base
            if self.cfg.delivery_bonus_factor > 1e-6:
                est_fid = bonus / self.cfg.delivery_bonus_factor + self.cfg.delivery_fidelity_threshold
            else:
                est_fid = self.cfg.delivery_fidelity_threshold
            self.episode_total_fidelity += np.clip(est_fid, 0.0, 1.0)

        if terminated or truncated:
            avg_fid = (self.episode_total_fidelity / self.episode_deliveries
                       if self.episode_deliveries > 0 else 0.0)
            info["haldar_deliveries"] = self.episode_deliveries
            info["haldar_avg_fidelity"] = avg_fid
            info["haldar_steps"] = self.episode_steps

        return obs, reward, terminated, truncated, info

    def get_action_mask(self):
        return self.unwrapped_env.get_action_mask()


# ═══════════════════════════════════════════════════════════════
#  评估函数
# ═══════════════════════════════════════════════════════════════

def evaluate_policy(env, policy, num_episodes=100, label="Policy"):
    total_deliveries = 0
    total_fidelity_sum = 0.0
    total_time_ms = 0.0

    # 进度条提示
    print(f"正在评估: {label} ... ", end="", flush=True)

    for ep in range(num_episodes):
        obs, info = env.reset(seed=1000 + ep)
        done = False

        while not done:
            mask = env.get_action_mask()
            action = policy.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        ep_del = info.get("haldar_deliveries", 0)
        total_deliveries += ep_del
        total_fidelity_sum += info.get("haldar_avg_fidelity", 0.0) * max(ep_del, 1)
        total_time_ms += info.get("time_ms", 0.0)

    avg_deliveries = total_deliveries / num_episodes
    avg_fidelity = (total_fidelity_sum / total_deliveries
                    if total_deliveries > 0 else 0.0)
    delivery_rate = (total_deliveries / (total_time_ms / 1000.0)
                     if total_time_ms > 0 else 0.0)

    print(f"完成! (Rate: {delivery_rate:.2f} Hz)")

    return {
        "label": label,
        "avg_deliveries": avg_deliveries,
        "avg_fidelity": avg_fidelity,
        "delivery_rate": delivery_rate
    }


def plot_results(results, save_dir):
    labels = [r["label"] for r in results]
    deliveries = [r["avg_deliveries"] for r in results]
    fidelities = [r["avg_fidelity"] for r in results]
    rates = [r["delivery_rate"] for r in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 柱状图：交付速率
    color1 = 'tab:blue'
    ax1.set_ylabel('Delivery Rate (Hz)', color=color1, fontweight='bold')
    bars1 = ax1.bar(x - width / 2, rates, width, label='Rate (Hz)', color=color1, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color1)

    # 两个Y轴
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Avg Fidelity', color=color2, fontweight='bold')
    bars2 = ax2.bar(x + width / 2, fidelities, width, label='Fidelity', color=color2, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.5, 1.0)  # 关注 0.5 以上

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.set_title("Haldar et al. (2024b) Reproduction: Deterministic Policies")

    # 标注数值
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "haldar_deterministic_results.png")
    plt.savefig(path, dpi=150)
    print(f"\n图表已保存至: {path}")


# ═══════════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════════

def main():
    save_dir = os.path.join("checkpoints", "haldar_repro")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Haldar et al. (2024b) 策略复现 (无 RL)")
    print(f"{'=' * 60}\n")

    # 1. 环境配置 (启用 Purification)
    # cfg = EnvConfig(
    #     num_nodes=5,  # 4 段
    #     node_distances=[20.0] * 4,  # 20km
    #     t_coherence=500.0,  # 500ms 相干时间
    #     prob_purify=0.7,  # ★ 关键: 允许蒸馏
    #     delivery_fidelity_threshold=0.8,  # 只有保真度 > 0.6 才算成功交付
    #     max_steps=500,
    # )
    cfg=EnvConfig()
    raw_env = QuantumNetworkEnv(cfg, verbose=False)
    env = HaldarStatsWrapper(raw_env)

    # 2. 定义策略组
    # 注意: num_nodes 传给策略用于初始化动作数组
    policies = {
        "Parallel SWAP-ASAP": ParallelSwapASAP(cfg.num_nodes),

        "FN SWAP-ASAP\n(No Distill)": FNSwapASAP(cfg.num_nodes, enable_distill=False),

        "Distill-Swap\n(Distill First)": FNSwapASAP(
            cfg.num_nodes, enable_distill=True, distill_first=True
        ),

        "Swap-Distill\n(Swap First)": FNSwapASAP(
            cfg.num_nodes, enable_distill=True, distill_first=False
        ),
    }

    # 3. 执行评估
    results = []
    num_eval_episodes = 50  # 测试 50 局

    for name, pol in policies.items():
        res = evaluate_policy(env, pol, num_eval_episodes, name)
        results.append(res)

    # 4. 输出结果
    print(f"\n{'=' * 75}")
    print(f"  {'策略名称':<30} {'交付速率 (Hz)':>15} {'平均保真度':>15}")
    print(f"{'=' * 75}")
    for r in results:
        label_clean = r['label'].replace('\n', ' ')
        print(f"  {label_clean:<30} {r['delivery_rate']:>15.2f} {r['avg_fidelity']:>15.4f}")
    print(f"{'=' * 75}\n")

    plot_results(results, save_dir)


if __name__ == "__main__":
    main()
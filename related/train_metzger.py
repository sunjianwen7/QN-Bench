#!/usr/bin/env python3
"""
related/train_metzger.py
========================
复现 Metzger et al. (2025) - 训练、定长测试与绘图一体化脚本 (最终修复版)

功能:
1. 训练: 基于 SKR 奖励和 POMDP 观测训练 PPO。
2. 测试: 在 **固定仿真时间** (Fixed Duration) 内运行策略，统计总交付数。
3. 绘图:
   - 左图: 训练奖励曲线。
   - 右图: 测试阶段的累计交付数量 vs 时间 (直观展示产出速率)。

使用方法:
    python related/train_metzger.py
"""

import sys
import os
import numpy as np
import gymnasium as gym
import torch
import logging
import matplotlib.pyplot as plt

# ─── 0. 强制日志配置 ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("metzger_repro")

# ─── 路径设置 ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnbench.envs import QuantumNetworkEnv, EnvConfig
from qnbench.rl.masked_ppo import PPOTrainer, PPOConfig


# ─── 1. 核心 Wrapper (Metzger 逻辑) ────────────────────────────────
class MetzgerProtocolWrapper(gym.Wrapper):
    """
    Wrapper:
    1. 拦截奖励 -> 改为 Secret Key Rate (SKR)。
    2. 观测空间 -> 依赖 Env 的 oracle_mode=False (POMDP)。
    """

    def __init__(self, env):
        super().__init__(env)
        self.unwrapped_env = env.unwrapped
        self.cfg = self.unwrapped_env.cfg

        # 透传属性
        self.obs_dim = self.unwrapped_env.obs_dim
        self.num_nodes = self.unwrapped_env.num_nodes
        self.get_action_mask = self.unwrapped_env.get_action_mask

        # 参数缓存
        self.base_reward = self.cfg.delivery_reward
        self.bonus_factor = self.cfg.delivery_bonus_factor
        self.threshold = self.cfg.delivery_fidelity_threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 从 info 获取原始交付奖励 (env.py 需支持 reward_breakdown)
        r_delivery = info.get("reward_breakdown", {}).get("delivery", 0.0)

        if r_delivery > 0:
            # 反推保真度
            bonus = r_delivery - self.base_reward
            if self.bonus_factor > 1e-6:
                est_fidelity = (bonus / self.bonus_factor) + self.threshold
            else:
                est_fidelity = self.threshold
            est_fidelity = min(1.0, max(0.0, est_fidelity))

            # 计算密钥量 (Bits)
            skr_bits = self._calc_secret_key_bits(est_fidelity)

            # 替换奖励: 移除原交付分，加上 (密钥量 * 缩放)
            # 缩放 50 倍是为了让 RL 容易学 (0.1 bit -> +5.0 reward)
            skr_reward = skr_bits * 50.0
            new_reward = reward - r_delivery + skr_reward

            # 记录到 info
            info["metzger_bits"] = skr_bits
            info["final_fidelity"] = est_fidelity
        else:
            new_reward = reward

        return obs, new_reward, terminated, truncated, info

    def _calc_secret_key_bits(self, fidelity):
        """BB84 渐近密钥率公式: R = 1 - 2*h(QBER)"""
        if fidelity <= 0.5: return 0.0
        qber = 1.0 - fidelity
        if qber >= 0.11: return 0.0  # 误码率过高，无法生成密钥

        def binary_entropy(p):
            if p <= 0 or p >= 1: return 0.0
            return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

        rate = 1.0 - 2 * binary_entropy(qber)
        return max(0.0, rate)


# ─── 2. 评估与绘图函数 (固定时长测试) ──────────────────────────────
def evaluate_and_plot(env, model, device, save_dir, training_history):
    """
    在固定仿真时间内运行测试，统计交付数量和速率。
    """
    # ── 测试参数 ──
    TEST_DURATION_MS = 1_000.0  # 测试总时长: 1000秒 (1M ms)
    # ────────────────

    print(f"\n{'=' * 50}")
    logger.info(f">>> 开始固定时长评估 (Target: {TEST_DURATION_MS} ms)...")
    print(f"{'=' * 50}")

    model.eval()

    current_total_time = 0.0
    cumulative_deliveries = 0
    cumulative_bits = 0.0

    # 用于绘图的数据点: [(time, total_delivered), ...]
    time_points = [0.0]
    delivery_points = [0]

    ep_idx = 0

    while current_total_time < TEST_DURATION_MS:
        obs, _ = env.reset(seed=5000 + ep_idx)  # 不同的种子
        done = False

        # 记录本局开始前的累计值
        ep_start_time = current_total_time
        ep_start_del = cumulative_deliveries

        while not done:
            # 1. 准备数据
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(env.get_action_mask(), dtype=torch.bool, device=device).unsqueeze(0)

            # 2. 推理
            with torch.no_grad():
                # 直接调用 PPO 模型 API
                action_result = model.get_action_and_value(obs_t, mask_t)
                # action_result: (action, logprob, entropy, value)
                action = action_result[0].cpu().numpy()[0]

            # 3. 执行
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 4. 收集实时数据 (如果有交付)
            if "metzger_bits" in info:
                cumulative_bits += info["metzger_bits"]
                # 记录每一个交付时刻 (为了画出精细的阶梯图)
                # info['time_ms'] 是当前局内时间
                real_time = ep_start_time + info['time_ms']

                # env.py 的 delivered 是本局累计值，我们需要总累计值
                # 本次交付增量 = 当前局delivered - 上次记录的局内delivered
                # 简单做法: 直接 +1 (因为 step 每一步只可能交付 1 个)
                cumulative_deliveries += 1

                time_points.append(real_time)
                delivery_points.append(cumulative_deliveries)

        # 本局结束，更新总时间
        # info['time_ms'] 在 done 时即为本局总时长
        current_total_time += info['time_ms']
        ep_idx += 1

        if ep_idx % 10 == 0:
            progress = (current_total_time / TEST_DURATION_MS) * 100
            logger.info(f"进度: {progress:.1f}% | 累计交付: {cumulative_deliveries}")

    # ── 统计结果 ──
    total_seconds = current_total_time / 1000.0
    rate_hz = cumulative_deliveries / total_seconds if total_seconds > 0 else 0
    bits_per_sec = cumulative_bits / total_seconds if total_seconds > 0 else 0

    logger.info(f"测试结束! 总耗时: {total_seconds:.2f} s")
    logger.info(f"总交付纠缠对: {cumulative_deliveries}")
    logger.info(f"平均吞吐率 (Rate): {rate_hz:.4f} pairs/s")
    logger.info(f"平均密钥率 (SKR):  {bits_per_sec:.4f} bits/s")

    # ── 绘图 ──
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 图 1: 训练曲线
        if training_history:
            window = max(1, len(training_history) // 20)
            smoothed = np.convolve(training_history, np.ones(window) / window, mode='valid')
            ax1.plot(training_history, alpha=0.3, color='blue', label='Raw Reward')
            ax1.plot(smoothed, color='red', linewidth=2, label='Smoothed')
            ax1.set_title("Training: Episode Reward")
            ax1.set_xlabel("Training Episodes")
            ax1.set_ylabel("Reward")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No training data", ha='center')

        # 图 2: 累计交付 vs 时间 (核心结果)
        # 将 ms 转换为秒
        time_sec = np.array(time_points) / 1000.0
        ax2.step(time_sec, delivery_points, where='post', color='green', linewidth=2)
        ax2.set_title(f"Test: Cumulative Deliveries (Rate ≈ {rate_hz:.2f} Hz)")
        ax2.set_xlabel("Simulation Time (seconds)")
        ax2.set_ylabel("Total Entangled Pairs Delivered")
        ax2.grid(True, alpha=0.3)

        # 添加一条参考线 (平均速率)
        if len(time_sec) > 1:
            ax2.plot([0, total_seconds], [0, cumulative_deliveries],
                     color='black', linestyle='--', alpha=0.5, label='Avg Rate')
            ax2.legend()

        plot_path = os.path.join(save_dir, "metzger_result_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"✅ 结果图表已保存至: {plot_path}")

    except Exception as e:
        logger.error(f"绘图失败: {e}")
        import traceback
        traceback.print_exc()


# ─── 3. 主程序 ─────────────────────────────────────────────────────
def main():
    save_dir = os.path.join("checkpoints", "metzger_repro")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"   Metzger et al. (2025) 复现实验 (固定时长测试)")
    print(f"   输出目录: {save_dir}")
    print(f"{'=' * 60}\n")

    # 1. 创建配置
    cfg = EnvConfig(
        num_nodes=5,
        node_distances=[10.0] * 4,
        t_coherence=200.0,
        delivery_bonus_factor=10.0,
        delivery_fidelity_threshold=0.5,
        oracle_mode=False,  # <--- 核心: 开启盲态 (POMDP)
        max_steps=500  # 训练时单局最长步数
    )

    # 2. 初始化环境
    raw_env = QuantumNetworkEnv(cfg, verbose=False)
    env = MetzgerProtocolWrapper(raw_env)

    # 3. PPO 配置
    ppo_cfg = PPOConfig(
        lr=3e-4,
        hidden_dim=256,
        gamma=0.95,
        batch_size=64,
        entropy_coef=0.02
    )

    # 4. 训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = PPOTrainer(env, ppo_cfg, device=device)

    # ★ 演示用步数，正式跑请改大 (e.g. 200_000) ★
    total_steps = 5000
    logger.info(f"开始训练 {total_steps} 步 (Device: {device})...")

    try:
        trainer.train(
            total_timesteps=total_steps,
            log_interval=5,
            save_dir=save_dir
        )
    except KeyboardInterrupt:
        logger.warning("训练被人为中断！尝试直接进入评估阶段...")

    # 保存模型
    final_path = os.path.join(save_dir, "metzger_final.pt")
    trainer.save(final_path)
    logger.info(f"模型已保存。")

    # 5. 进入测试阶段 (固定时长)
    evaluate_and_plot(env, trainer.model, device, save_dir, trainer.reward_history)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
related/train_li_et_al.py
=========================
复现 Li et al. (2024) - 通信延迟约束下的纠缠分发优化

核心特性:
1. 【无纯化 (No Purify)】:
   - Config设置 prob_purify=0.0。
   - Action Masking 强制屏蔽动作 4, 5。
2. 【无主动丢弃 (No Discard)】:
   - Action Masking 强制屏蔽动作 6 (依赖物理自动超时)。
3. 【通信延迟 (CC Delay)】:
   - 使用 History Buffer 模拟观测滞后。Agent 看到的不是当前 obs，而是 N 步之前的 obs。
4. 【测试对齐】:
   - 使用 1,000,000 ms 固定时长测试，统计总交付数量。

使用方法:
    python related/train_li_et_al.py
"""

import sys
import os
import collections
import time
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
logger = logging.getLogger("li_repro")

# ─── 路径设置 ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnbench.envs import QuantumNetworkEnv, EnvConfig
from qnbench.rl.masked_ppo import PPOTrainer, PPOConfig


# ─── 1. 核心 Wrapper: 延迟与动作屏蔽 ──────────────────────────────
class LiProtocolWrapper(gym.Wrapper):
    """
    Wrapper 功能:
    1. Action Masking: 仅允许 {Wait, Gen_L, Gen_R, Swap}。
    2. Observation Delay: 模拟经典通信延迟。
    """

    def __init__(self, env, delay_steps=5):
        super().__init__(env)
        self.delay_steps = delay_steps
        # 使用 deque 存储历史观测，maxlen 自动维护窗口大小
        # 长度 = 延迟步数 + 1 (当前步)
        self.obs_history = collections.deque(maxlen=delay_steps + 1)

        # 透传属性
        self.obs_dim = env.unwrapped.obs_dim
        self.num_nodes = env.unwrapped.num_nodes

        # 【核心策略】：预先计算静态 Mask
        # QNBench Actions: 0:Wait, 1:GenL, 2:GenR, 3:Swap, 4:PurL, 5:PurR, 6:Disc
        # 我们只保留前4个: [1, 1, 1, 1, 0, 0, 0]
        self.static_mask = np.array([1, 1, 1, 1, 0, 0, 0], dtype=bool)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 清空历史
        self.obs_history.clear()
        # 填充初始状态之前的"空历史" (用全0填充)
        for _ in range(self.delay_steps):
            self.obs_history.append(np.zeros_like(obs))

        # 推入当前第一帧观测
        self.obs_history.append(obs)

        # 返回滞后的观测 (队列最左边)
        delayed_obs = self.obs_history[0]
        return delayed_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 将最新的真实观测推入队列 (队尾)
        self.obs_history.append(obs)

        # 取出滞后的观测给 Agent (队头，即 N 步之前的)
        delayed_obs = self.obs_history[0]

        return delayed_obs, reward, terminated, truncated, info

    def get_action_mask(self):
        """
        覆盖原始环境的 Mask 逻辑。
        实现 "No Purify" 和 "No Discard"。
        """
        # 1. 获取物理层面的 Mask (比如: 没邻居就不能Gen, 没链路就不能Swap)
        raw_mask = self.env.unwrapped.get_action_mask()

        # 2. 与我们的静态限制取交集 (Logical AND)
        # static_mask (7,) 会自动广播到 (Num_Nodes, 7)
        final_mask = raw_mask & self.static_mask

        return final_mask


# ─── 2. 评估与绘图函数 (固定时长测试) ──────────────────────────────
def evaluate_and_plot(env, model, device, save_dir, training_history):
    """
    在固定仿真时间内运行测试，统计交付数量和速率。
    """
    # ── 测试参数 (与 Metzger/Haldar 对齐) ──
    TEST_DURATION_MS = 1_000_000.0  # 测试总时长: 1000秒
    # ─────────────────────────────────────

    print(f"\n{'=' * 50}")
    logger.info(f">>> 开始固定时长评估 (Target: {TEST_DURATION_MS} ms)...")
    print(f"{'=' * 50}")

    model.eval()

    current_total_time = 0.0
    cumulative_deliveries = 0

    # 用于绘图的数据点: [(time, total_delivered), ...]
    time_points = [0.0]
    delivery_points = [0]

    ep_idx = 0

    while current_total_time < TEST_DURATION_MS:
        obs, _ = env.reset(seed=6000 + ep_idx)  # 不同的种子
        done = False

        while not done:
            # 1. 准备数据 (手动转换，稳健)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(env.get_action_mask(), dtype=torch.bool, device=device).unsqueeze(0)

            # 2. 推理
            with torch.no_grad():
                # 直接调用 PPO 模型 API
                # get_action_and_value 返回 (action, logprob, entropy, value)
                action_result = model.get_action_and_value(obs_t, mask_t)
                action = action_result[0].cpu().numpy()[0]

            # 3. 执行
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # 本局结束，更新数据
        # info['delivered'] 是本局交付总数
        ep_deliveries = info.get('delivered', 0)
        cumulative_deliveries += ep_deliveries

        # 更新时间
        current_total_time += info['time_ms']

        # 记录数据点
        time_points.append(current_total_time)
        delivery_points.append(cumulative_deliveries)

        ep_idx += 1

        if ep_idx % 10 == 0:
            progress = (current_total_time / TEST_DURATION_MS) * 100
            logger.info(f"进度: {progress:.1f}% | 累计交付: {cumulative_deliveries}")

    # ── 统计结果 ──
    total_seconds = current_total_time / 1000.0
    rate_hz = cumulative_deliveries / total_seconds if total_seconds > 0 else 0

    logger.info(f"测试结束! 总耗时: {total_seconds:.2f} s")
    logger.info(f"总交付纠缠对: {cumulative_deliveries}")
    logger.info(f"平均吞吐率 (Entanglement Rate): {rate_hz:.4f} Hz")

    # ── 绘图 ──
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 图 1: 训练曲线
        if training_history:
            window = max(1, len(training_history) // 20)
            smoothed = np.convolve(training_history, np.ones(window) / window, mode='valid')
            ax1.plot(training_history, alpha=0.3, color='blue', label='Raw Reward')
            ax1.plot(smoothed, color='purple', linewidth=2, label='Smoothed')
            ax1.set_title("Li et al. Training (No Purify, Delayed Obs)")
            ax1.set_xlabel("Episodes")
            ax1.set_ylabel("Reward")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No training data", ha='center')

        # 图 2: 累计交付 vs 时间
        # 将 ms 转换为秒
        time_sec = np.array(time_points) / 1000.0
        ax2.step(time_sec, delivery_points, where='post', color='purple', linewidth=2)
        ax2.set_title(f"Test: Cumulative Deliveries (Rate ≈ {rate_hz:.2f} Hz)")
        ax2.set_xlabel("Simulation Time (seconds)")
        ax2.set_ylabel("Total Entangled Pairs Delivered")
        ax2.grid(True, alpha=0.3)

        # 添加参考线
        if len(time_sec) > 1:
            ax2.plot([0, total_seconds], [0, cumulative_deliveries],
                     color='black', linestyle='--', alpha=0.5, label='Avg Rate')
            ax2.legend()

        plot_path = os.path.join(save_dir, "li_result_plot.png")
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
    save_dir = os.path.join("checkpoints", "li_repro")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"   Li et al. (2024) 复现实验 (固定时长测试)")
    print(f"   输出目录: {save_dir}")
    print(f"{'=' * 60}\n")

    # 1. 配置
    # 关键点: prob_purify = 0.0 (物理层禁用)
    cfg = EnvConfig(
        num_nodes=5,
        node_distances=[20.0] * 4,  # 距离稍微拉远
        t_coherence=500.0,
        prob_purify=0.0,  # <--- 核心修改: 物理禁用纯化
        delivery_fidelity_threshold=0.6,  # 降低阈值，因为没有纯化只能靠初始保真度
        delivery_reward=20.0,
        oracle_mode=True,  # 底层开启Oracle，由Wrapper负责制造延迟
        max_steps=500
    )

    # 2. 环境初始化
    raw_env = QuantumNetworkEnv(cfg, verbose=False)

    # 3. 套上 Wrapper (模拟通信延迟 + 动作屏蔽)
    # delay_steps=3 意味着 Agent 看到的总是 3 步之前的世界
    env = LiProtocolWrapper(raw_env, delay_steps=3)

    # 4. PPO 配置
    # 显式指定 hidden_dim=256 以保持一致性
    ppo_cfg = PPOConfig(
        lr=3e-4,
        hidden_dim=256,
        gamma=0.99,
        batch_size=64
    )

    # 5. 训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = PPOTrainer(env, ppo_cfg, device=device)

    # ★ 演示用步数 (5000)，正式跑建议 150_000+ ★
    total_steps = 5000
    logger.info(f"开始训练 {total_steps} 步 (Device: {device})...")

    try:
        trainer.train(total_timesteps=total_steps, log_interval=5, save_dir=save_dir)
    except KeyboardInterrupt:
        logger.warning("训练被人为中断！尝试直接进入评估阶段...")

    final_path = os.path.join(save_dir, "li_final.pt")
    trainer.save(final_path)
    logger.info(f"模型已保存。")

    # 6. 评估
    evaluate_and_plot(env, trainer.model, device, save_dir, trainer.reward_history)


if __name__ == "__main__":
    main()
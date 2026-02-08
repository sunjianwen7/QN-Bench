#!/usr/bin/env python3
"""
related/train_haldar.py
=======================
复现 Haldar et al. (2024) - 课程学习 (Curriculum Learning) 与 固定时长测试 (修复版)

核心思想:
通过 "分治法" 解决长链训练难的问题。
1. Stage 1 (3节点): 学习基础物理 (生成、纯化、交换)。
2. Stage 2 (5节点): 加载 Stage 1 的特征提取层，学习协调左右子段。
3. Stage 3 (9节点): 加载 Stage 2 的权重，解决长链端到端交付。

测试阶段:
与对比算法对齐，使用固定仿真时长 (1000秒) 统计总交付数和速率。

输出:
- checkpoints/haldar_repro/ 下的模型文件
- haldar_delivery_plot.png: 交付性能评估图

使用方法:
    python related/train_haldar.py
"""

import sys
import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# ─── 0. 强制日志配置 ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("haldar_repro")

# ─── 路径设置 ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnbench.envs import QuantumNetworkEnv, EnvConfig
from qnbench.rl.masked_ppo import PPOTrainer, PPOConfig


# ─── 1. 核心工具: 智能权重迁移 ─────────────────────────────────────
def smart_weight_transfer(target_model, source_state_dict):
    """
    Haldar 核心魔法：将“短链”模型的知识迁移到“长链”模型。
    只迁移形状匹配的层（通常是 Hidden Layers），跳过输入/输出层。
    """
    target_state = target_model.state_dict()
    new_state = target_state.copy()

    transferred = 0
    skipped = 0

    for key, target_param in target_state.items():
        if key in source_state_dict:
            source_param = source_state_dict[key]

            # 只有形状完全匹配的层才迁移
            if source_param.shape == target_param.shape:
                new_state[key] = source_param
                transferred += 1
            else:
                skipped += 1
        else:
            skipped += 1

    target_model.load_state_dict(new_state)
    return transferred, skipped


# ─── 2. 单阶段训练函数 ─────────────────────────────────────────────
def run_stage(stage_name, num_nodes, steps, prev_model_path, save_dir):
    """
    运行一个课程阶段。
    """
    print(f"\n{'=' * 60}")
    logger.info(f"启动 {stage_name}: {num_nodes} 节点链 | 目标步数: {steps}")
    print(f"{'=' * 60}")

    # 1. 配置环境
    # Haldar 论文通常使用 Oracle 模式 (假设节点知道链路状态)
    # 开启纯化 (prob_purify > 0)
    cfg = EnvConfig(
        num_nodes=num_nodes,
        node_distances=[25.0] * (num_nodes - 1),
        t_coherence=2000.0,
        prob_purify=0.7,
        delivery_reward=50.0,
        delivery_fidelity_threshold=0.85,  # 高保真度要求
        oracle_mode=True,  # 开启全知模式
        max_steps=500
    )
    env = QuantumNetworkEnv(cfg, verbose=False)

    # 2. 配置 PPO
    # 注意: hidden_dim 必须统一为 256，方便后续迁移和加载
    lr = 3e-4 if prev_model_path is None else 2e-4
    ppo_cfg = PPOConfig(
        lr=lr,
        hidden_dim=256,
        gamma=0.99,
        batch_size=64
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = PPOTrainer(env, ppo_cfg, device=device)

    # 3. 权重迁移 (核心步骤)
    if prev_model_path and os.path.exists(prev_model_path):
        logger.info(f"正在从上一阶段 ({prev_model_path}) 迁移知识...")
        checkpoint = torch.load(prev_model_path, map_location=device)
        n_trans, n_skip = smart_weight_transfer(trainer.model, checkpoint["model_state"])
        logger.info(f"✅ 权重迁移成功: 复用了 {n_trans} 层通用特征, 重置了 {n_skip} 层特定维度权重")
    else:
        logger.info("⚡ 无前序模型，从零开始训练 (Scratch Training)...")

    # 4. 训练
    trainer.train(total_timesteps=steps, log_interval=10, save_dir=save_dir)

    # 5. 保存
    final_path = os.path.join(save_dir, f"{stage_name}.pt")
    trainer.save(final_path)
    logger.info(f"{stage_name} 完成。模型已保存至: {final_path}")

    return final_path


# ─── 3. 评估与绘图 (固定时长测试) ───────────────────────────────────
def evaluate_and_plot(final_model_path, save_dir):
    """
    对最终的长链模型进行固定时长的性能评估。
    """
    # ── 测试参数 (与 Metzger/Li 对齐) ──
    TEST_DURATION_MS = 1_000.0  # 测试总时长: 1000秒
    # ──────────────────────────────────

    print(f"\n{'=' * 50}")
    logger.info(f">>> 开始固定时长评估 (Target: {TEST_DURATION_MS} ms)...")
    print(f"{'=' * 50}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 重新构建 9 节点环境用于测试
    cfg_final = EnvConfig(
        num_nodes=9,
        node_distances=[25.0] * 8,
        t_coherence=2000.0,
        prob_purify=0.7,
        delivery_fidelity_threshold=0.85,
        oracle_mode=True,
        max_steps=500
    )
    env = QuantumNetworkEnv(cfg_final, verbose=False)

    # 2. 加载模型
    # 【关键】: 必须使用与训练时相同的 hidden_dim=256，否则报错 size mismatch
    eval_ppo_cfg = PPOConfig(hidden_dim=256)

    temp_trainer = PPOTrainer(env, eval_ppo_cfg, device=device)
    temp_trainer.load(final_model_path)
    model = temp_trainer.model
    model.eval()

    # 3. 执行测试循环
    current_total_time = 0.0
    cumulative_deliveries = 0

    # 数据记录: [time_sec, total_delivered]
    time_points = [0.0]
    delivery_points = [0]

    ep_idx = 0

    while current_total_time < TEST_DURATION_MS:
        obs, _ = env.reset(seed=7000 + ep_idx)
        done = False

        while not done:
            # ── 手动 Tensor 转换 (稳健) ──
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(env.get_action_mask(), dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                action_res = model.get_action_and_value(obs_t, mask_t)
                action = action_res[0].cpu().numpy()[0]

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # 本局结束
        # info["delivered"] 是本局的累计交付数
        ep_deliveries = info.get("delivered", 0)
        cumulative_deliveries += ep_deliveries

        # 更新时间
        current_total_time += info["time_ms"]

        # 记录数据点
        time_points.append(current_total_time)
        delivery_points.append(cumulative_deliveries)

        ep_idx += 1
        if ep_idx % 10 == 0:
            prog = (current_total_time / TEST_DURATION_MS) * 100
            logger.info(f"进度: {prog:.1f}% | 累计交付: {cumulative_deliveries}")

    # 4. 统计结果
    total_seconds = current_total_time / 1000.0
    rate_hz = cumulative_deliveries / total_seconds if total_seconds > 0 else 0

    logger.info(f"测试结束! 总耗时: {total_seconds:.2f} s")
    logger.info(f"总交付纠缠对: {cumulative_deliveries}")
    logger.info(f"平均吞吐率 (Rate): {rate_hz:.4f} Hz")

    # 5. 绘图
    try:
        plt.figure(figsize=(8, 6))

        time_sec = np.array(time_points) / 1000.0
        plt.step(time_sec, delivery_points, where='post', color='green', linewidth=2)

        plt.title(f"Haldar (9-Node) Cumulative Deliveries\nRate ≈ {rate_hz:.2f} Hz")
        plt.xlabel("Simulation Time (seconds)")
        plt.ylabel("Total Entangled Pairs Delivered")
        plt.grid(True, alpha=0.3)

        # 参考线
        if len(time_sec) > 1:
            plt.plot([0, total_seconds], [0, cumulative_deliveries],
                     color='black', linestyle='--', alpha=0.5, label='Avg Rate')
            plt.legend()

        plot_path = os.path.join(save_dir, "haldar_delivery_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"✅ 交付性能图已保存至: {plot_path}")

    except Exception as e:
        logger.error(f"❌ 绘图失败: {e}")


# ─── 4. 主程序 ─────────────────────────────────────────────────────
def main():
    save_dir = os.path.join("checkpoints", "haldar_repro")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"   Haldar et al. (2024) 复现实验启动")
    print(f"   输出目录: {save_dir}")
    print(f"{'=' * 60}\n")

    # ── Stage 1: 3 节点 (Base Skills) ──
    # 演示用 5000 步
    # 【修复】去掉下划线，直接接收返回值
    path1 = run_stage(
        stage_name="Stage1_3Nodes",
        num_nodes=3,
        steps=5000,
        prev_model_path=None,
        save_dir=save_dir
    )

    # ── Stage 2: 5 节点 (Scaling Up) ──
    # 【修复】去掉下划线
    path2 = run_stage(
        stage_name="Stage2_5Nodes",
        num_nodes=5,
        steps=5000,
        prev_model_path=path1,
        save_dir=save_dir
    )

    # ── Stage 3: 9 节点 (Long Chain) ──
    # 【修复】去掉下划线
    path3 = run_stage(
        stage_name="Stage3_9Nodes",
        num_nodes=9,
        steps=5000,
        prev_model_path=path2,
        save_dir=save_dir
    )

    # ── 最终测试 (固定时长) ──
    evaluate_and_plot(path3, save_dir)


if __name__ == "__main__":
    main()
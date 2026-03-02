#!/usr/bin/env python3
import sys
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 确保能找到 qnbench 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnbench.envs import QuantumNetworkEnv, EnvConfig
from qnbench.baselines import RandomAgent, GreedyAgent, SwapASAPAgent
from qnbench.evaluation.runner import compare_agents
from qnbench.rl.networks import ActorCritic, ActorCriticAgent


def get_ppo_agent(n, env, checkpoint_path, device="cpu"):
    """
    尝试加载 PPO 模型。
    如果 Checkpoint 的网络结构与当前节点数 n 不匹配，返回 None 并打印警告。
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return None

    try:
        # weights_only=False 是为了兼容旧版 PyTorch 保存格式，但要注意安全性
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = ActorCritic(obs_dim=env.obs_dim, num_actions=7, num_nodes=n)

        # 尝试加载权重
        model.load_state_dict(ckpt["model_state"])
        agent = ActorCriticAgent(model, device=device)
        return agent
    except RuntimeError as e:
        # 捕获维度不匹配错误 (例如加载 5节点的权重到 9节点模型)
        if "size mismatch" in str(e):
            print(f"  [Skip] PPO checkpoint dimensions do not match {n} nodes.")
            return None
        else:
            raise e
    except Exception as e:
        print(f"  [Skip] Error loading PPO: {e}")
        return None


def run_experiments(checkpoint_path):
    device = "cpu"
    results = []

    # === 实验设计 ===
    node_sizes = [3, 4, 5, 6]
    decoherence_rates = [0.01, 0.05, 0.1]

    print(f"Starting Benchmark...")
    print(f"Checkpoint: {checkpoint_path}")

    for n in node_sizes:
        for dr in decoherence_rates:
            print(f"\n>>> Testing: Nodes={n}, Decoherence={dr}")

            # --- 关键修复 1: 动态生成距离列表 ---
            # 线性链有 n 个节点，则有 n-1 段距离。这里设每段 50km
            dists = [50.0] * (n - 1)

            # --- 关键修复 2: 正确初始化 EnvConfig ---
            # 必须同时传递 num_nodes 和 node_distances 确保一致性
            cfg = EnvConfig(num_nodes=n, node_distances=dists)

            # 设置最大步数
            cfg.max_steps = 200

            # 设置物理参数
            if hasattr(cfg, 'physics_params') and isinstance(cfg.physics_params, dict):
                cfg.physics_params['decoherence_rate'] = dr
            else:
                cfg.decoherence_rate = dr  # 针对简单的 Config 实现

            # 初始化环境
            env = QuantumNetworkEnv(cfg=cfg)

            # --- 定义对比算法 ---
            agents = {
                "Swap-ASAP": SwapASAPAgent(n),
                "Greedy": GreedyAgent(n),
            }

            # --- 安全加载 PPO ---
            ppo_agent = get_ppo_agent(n, env, checkpoint_path, device)
            if ppo_agent:
                agents["PPO"] = ppo_agent

            # --- 运行评估 ---
            try:
                # n_episodes 设为 50 以获得稳定数据
                metrics_list = compare_agents(cfg, agents, n_episodes=50, verbose=False)
            except TypeError as e:
                # 如果旧版代码还在用 num_episodes，这里做个兼容尝试
                try:
                    metrics_list = compare_agents(cfg, agents, num_episodes=50, verbose=False)
                except:
                    print(f"Critical Error in runner: {e}")
                    return pd.DataFrame()

            # 收集数据
            for m in metrics_list:
                results.append({
                    "Nodes": n,
                    "Decoherence": dr,
                    "Agent": m["agent"],
                    # 归一化指标：每分钟(或每百步)产生的纠缠对
                    "Throughput": m["total_delivered"] / 50.0,
                    "Reward": m["mean_reward"],
                    "AvgSteps": m.get("mean_steps", 0),
                    # 如果你的 metrics 有保真度字段
                    "Fidelity": m.get("mean_fidelity", 0)
                })

    return pd.DataFrame(results)


def plot_results(df):
    if df.empty:
        print("No data to plot.")
        return

    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    # 创建 1行3列 的画布
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- 图 1: 扩展性 (Scalability) ---
    # 筛选标准退相干率 0.05
    df_scale = df[df['Decoherence'] == 0.05]
    if not df_scale.empty:
        sns.lineplot(data=df_scale, x="Nodes", y="Throughput", hue="Agent", style="Agent", markers=True, ax=axes[0],
                     linewidth=2.5)
        axes[0].set_title("Scalability (Decoherence=0.05)")
        axes[0].set_ylabel("Avg. Delivered Pairs / Episode")
        axes[0].set_xlabel("Chain Length (Nodes)")
        axes[0].set_xticks(df['Nodes'].unique())

    # --- 图 2: 鲁棒性 (Robustness) ---
    # 筛选 5 节点数据
    df_robust = df[df['Nodes'] == 5]
    if not df_robust.empty:
        # 使用柱状图更能体现对比
        sns.barplot(data=df_robust, x="Decoherence", y="Throughput", hue="Agent", ax=axes[1], alpha=0.8)
        axes[1].set_title("Robustness (Nodes=5)")
        axes[1].set_xlabel("Decoherence Rate (Noise)")
        axes[1].set_ylabel("Avg. Delivered Pairs")

    # --- 图 3: 效率权衡 (Efficiency) ---
    # 所有的点，展示 Reward vs Throughput
    sns.scatterplot(data=df, x="Throughput", y="Reward", hue="Agent", size="Nodes", sizes=(20, 200), ax=axes[2],
                    alpha=0.7)
    axes[2].set_title("Reward vs Throughput (All Experiments)")
    axes[2].set_xlabel("Throughput (Output)")
    axes[2].set_ylabel("RL Mean Reward")

    plt.tight_layout()
    output_file = "benchmark_result.png"
    plt.savefig(output_file, dpi=300)
    print(f"\n[Success] Plot saved to: {output_file}")


if __name__ == "__main__":
    # 路径根据你的实际情况
    CKPT = "/Users/jevonsun/PycharmProjects/qnbenchmark/scripts/checkpoints/ppo_best.pt"

    df = run_experiments(CKPT)

    if not df.empty:
        print("\n=== Summary Table ===")
        # 打印文本摘要
        print(df.groupby(["Nodes", "Decoherence", "Agent"])[["Throughput"]].mean().unstack())
        plot_results(df)
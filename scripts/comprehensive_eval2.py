import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 确保能找到项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnbench.envs import QuantumNetworkEnv, EnvConfig
from qnbench.baselines import SwapASAPAgent, GreedyAgent
from qnbench.rl.networks import ActorCritic, ActorCriticAgent


# =========================================================================
# 1. 强制采样环境 (无死锁版)
# =========================================================================
class ForcedSamplingEnv(QuantumNetworkEnv):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.fidelity_records = []
        self.current_agent_name = "Unknown"

    def set_agent_name(self, name):
        self.current_agent_name = name
        self.fidelity_records = []

    def step(self, actions):
        # 1. 劫持阈值，设为无穷大，阻止环境自动处理
        real_threshold = self.cfg.delivery_fidelity_threshold
        self.cfg.delivery_fidelity_threshold = 999.0

        # 2. 执行物理仿真
        obs, reward, terminated, truncated, info = super().step(actions)

        # 3. 恢复阈值
        self.cfg.delivery_fidelity_threshold = real_threshold

        # 4. 强制扫描并清理
        delivered_valid = 0
        delivery_reward = 0.0

        # 必须使用 list() 复制，因为我们会在遍历中删除元素
        memories = list(self.engine.nodes[0].memories)

        for m in memories:
            # 检查是否连接到了最后一个节点
            if (m.is_entangled_and_available() and m.entangled_node == self.num_nodes - 1):
                link = self.engine.links.get(m.link_id)
                if link is None: continue

                cf = link.current_fidelity(self.engine.current_time, self.cfg.t_coherence)

                # --- 核心修改：记录所有产出 ---
                self.fidelity_records.append({
                    "Agent": self.current_agent_name,
                    "Fidelity": cf,
                    "Steps": self._step_count,
                    "Is_Success": cf >= real_threshold
                })

                # --- 奖励计算 ---
                if cf >= real_threshold:
                    bonus = ((cf - real_threshold) * self.cfg.delivery_bonus_factor)
                    r = self.cfg.delivery_reward + bonus
                    delivery_reward += r
                    delivered_valid += 1

                # --- 核心修改：强制销毁 (防止死锁) ---
                # 只要链路贯通了，无论质量如何，立刻销毁，腾出内存做下一次实验
                self.engine._destroy_link(link.link_id)

        reward += delivery_reward
        info['delivered'] = delivered_valid

        return obs, reward, terminated, truncated, info


# =========================================================================
# 2. 实验运行
# =========================================================================

def run_fidelity_distribution(checkpoint_path):
    # 参数设置
    N_EPISODES = 50  # 50回合足够了，因为每回合现在能产出几十个样本
    MAX_STEPS = 500  # 单回合步数
    N_NODES = 5

    dists = [50.0] * (N_NODES - 1)
    cfg = EnvConfig(
        num_nodes=N_NODES,
        node_distances=dists,
        max_steps=MAX_STEPS,
        delivery_fidelity_threshold=0.8,
        oracle_mode=True
    )

    env = ForcedSamplingEnv(cfg)
    agents = {}

    # 加载 PPO
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            model = ActorCritic(obs_dim=env.obs_dim, num_actions=7, num_nodes=N_NODES)
            model.load_state_dict(ckpt["model_state"])
            agents["PPO"] = ActorCriticAgent(model, device="cpu")
        except Exception as e:
            print(f"PPO Load Error: {e}")

    agents["Swap-ASAP"] = SwapASAPAgent(N_NODES)
    agents["Greedy"] = GreedyAgent(N_NODES)

    all_data = []

    print(f"开始采样 (Nodes={N_NODES}, Ep={N_EPISODES})...")

    for name, agent in agents.items():
        print(f"Testing {name} ... ", end="", flush=True)
        env.set_agent_name(name)

        for ep in range(N_EPISODES):
            obs, _ = env.reset(seed=100 + ep)
            agent.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                mask = env.get_action_mask()
                action = agent.act(obs, mask)
                obs, r, terminated, truncated, info = env.step(action)

        count = len(env.fidelity_records)
        print(f"Collected {count} samples.")
        all_data.extend(env.fidelity_records)

    return pd.DataFrame(all_data)


# =========================================================================
# 3. 绘图 (密度图 + 统计表)
# =========================================================================

def plot_analysis(df):
    if df.empty: return

    # 设置绘图风格
    sns.set_theme(style="ticks", font_scale=1.1)
    fig = plt.figure(figsize=(14, 8), dpi=150)
    gs = fig.add_gridspec(2, 2)

    # --- 图 1: 保真度密度分布 (KDE) ---
    ax1 = fig.add_subplot(gs[0, :])  # 占满第一行
    sns.kdeplot(
        data=df, x="Fidelity", hue="Agent",
        fill=True, common_norm=False, palette="viridis",
        alpha=0.3, linewidth=2, clip=(0.4, 1.0), ax=ax1
    )
    ax1.axvline(0.8, color='red', linestyle='--', label="Threshold (0.8)")
    ax1.set_title("Fidelity Density Distribution (All Generated Links)")
    ax1.set_xlim(0.4, 1.0)
    ax1.legend()

    # --- 图 2: 累计分布 (CDF) ---
    ax2 = fig.add_subplot(gs[1, 0])
    sns.ecdfplot(data=df, x="Fidelity", hue="Agent", palette="viridis", linewidth=2, ax=ax2)
    ax2.axvline(0.8, color='red', linestyle='--')
    ax2.set_title("CDF: Quality Yield Curve")
    ax2.set_ylabel("Proportion <= x")

    # --- 图 3: 统计表格 ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    summary = df.groupby("Agent")["Fidelity"].agg(
        Count='count',
        Mean='mean',
        Std='std',
        Yield_Rate_08=lambda x: (x >= 0.8).mean()
    ).reset_index()

    print("\n=== Final Statistics ===")
    print(summary)

    # 绘制表格
    cell_text = []
    for row in summary.itertuples():
        cell_text.append([
            row.Agent,
            f"{row.Count}",
            f"{row.Mean:.3f}",
            f"{row.Yield_Rate_08:.1%}"  # 达标率
        ])

    table = ax3.table(
        cellText=cell_text,
        colLabels=["Agent", "Count", "Mean F", "Yield (>0.8)"],
        loc='center', cellLoc='center'
    )
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    ax3.set_title("Statistical Summary")

    plt.tight_layout()
    plt.savefig("fidelity_distribution_fixed.png")
    print("\n图表已保存至: scripts/fidelity_distribution_fixed.png")


if __name__ == "__main__":
    CKPT = "/Users/jevonsun/PycharmProjects/qnbenchmark/scripts/checkpoints/ppo_best.pt"
    if not os.path.exists(CKPT): CKPT = "checkpoints/ppo_best.pt"

    df = run_fidelity_distribution(CKPT)
    plot_analysis(df)
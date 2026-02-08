#!/usr/bin/env python3
"""
scripts/benchmark_epoch_stats.py
================================
多 Epoch 综合基准测试脚本：展示平均性能与波动范围。

功能升级:
1. 每个模型运行 10 个 Epoch (不同的随机种子)。
2. 自动计算平均 Entanglement Rate 和标准差。
3. 绘图：绘制平均曲线 + 标准差阴影 (Confidence Interval)，直观展示稳定性。
4. 保持了相对路径修复。

使用:
    python scripts/benchmark_epoch_stats.py
"""

import sys
import os
import collections
import numpy as np
import torch
import gymnasium as gym
import matplotlib

# 设置无头模式后端，防止服务器/IDE报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d  # 用于对齐不同Epoch的时间轴

# ─── 路径设置 ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from qnbench.envs import QuantumNetworkEnv, EnvConfig
from qnbench.rl.masked_ppo import PPOTrainer, PPOConfig

# ===================================================================
#  用户配置区域
# ===================================================================

# 1. 物理场景配置
BENCHMARK_SCENARIO = EnvConfig(
    num_nodes=4,
    node_distances=[50.0] * 3,
    t_coherence=1000.0,
    prob_gen=0.6,
    prob_swap=0.8,
    prob_purify=0.7,
    delivery_fidelity_threshold=0.8,
    max_steps=1000
)

# 2. 测试配置
# 你提到"现在1s是要测试代码"，这里设为 1000.0 ms (1秒)
# 正式跑分建议设为 1_000_000.0 (1000秒) 以获得更平滑的曲线
TEST_DURATION_MS = 2_000.0
NUM_EPOCHS = 10  # 测试 10 次取平均

# 3. 绘图采样点数 (将所有Epoch的数据对齐到这 100 个时间点上)
PLOT_POINTS = 100

MODELS_CONFIG = {
    "Ours (QNBench)": {
        "path": "scripts/checkpoints/ppo_best.pt",
        "type": "ours",
        "color": "#d62728",  # 红
        "hidden_dim": 128
    },
    "Metzger (POMDP)": {
        "path": "related/checkpoints/metzger_repro/metzger_final.pt",
        "type": "metzger",
        "color": "#1f77b4",  # 蓝
        "hidden_dim": 256
    },
    # 你可以注释掉不想测的模型以节省时间
    "Haldar (Curriculum)": {
        "path": "related/checkpoints/haldar_repro/Stage3_9Nodes.pt",
        "type": "haldar",
        "color": "#2ca02c",  # 绿
        "hidden_dim": 256
    },
    "Li et al. (Delay)": {
        "path": "related/checkpoints/li_repro/li_final.pt",
        "type": "li",
        "color": "#9467bd",  # 紫
        "hidden_dim": 256
    }
}


# ===================================================================
#  辅助 Wrapper & 工具函数
# ===================================================================

class LiWrapper(gym.Wrapper):
    """Li et al. 专用: 观测延迟 + 动作屏蔽"""

    def __init__(self, env, delay_steps=3):
        super().__init__(env)
        self.obs_history = collections.deque(maxlen=delay_steps + 1)
        # 固定屏蔽某些动作
        self.static_mask = np.array([1, 1, 1, 1, 0, 0, 0], dtype=bool)

        # =========== 修复核心 ============
        # 显式暴露底层环境的关键属性，防止 AttributeError
        if hasattr(env, "obs_dim"):
            self.obs_dim = env.obs_dim
        elif hasattr(env.unwrapped, "obs_dim"):
            self.obs_dim = env.unwrapped.obs_dim
        else:
            # 如果实在找不到，尝试通过 observation_space 推断
            self.obs_dim = env.observation_space.shape[0]

        # 如果你的 Trainer 还需要 num_nodes，也加上：
        if hasattr(env.unwrapped, "num_nodes"):
            self.num_nodes = env.unwrapped.num_nodes
        # =================================

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs_history.clear()
        # 填充历史 buffer
        for _ in range(self.obs_history.maxlen - 1):
            self.obs_history.append(np.zeros_like(obs))
        self.obs_history.append(obs)
        return self.obs_history[0], info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.obs_history.append(obs)
        return self.obs_history[0], r, term, trunc, info

    def get_action_mask(self):
        # 确保调用底层 mask 并叠加 static_mask
        if hasattr(self.env, "get_action_mask"):
            base = self.env.get_action_mask()
        else:
            base = self.env.unwrapped.get_action_mask()
        return base & self.static_mask

def get_model_path(rel_path):
    return os.path.join(PROJECT_ROOT, rel_path)


def make_env_for_model(algo_type: str, base_cfg: EnvConfig):
    cfg = EnvConfig(**base_cfg.to_dict())
    if algo_type == "ours":
        cfg.oracle_mode = True
    elif algo_type == "metzger":
        cfg.oracle_mode = False
    elif algo_type == "haldar":
        cfg.oracle_mode = True
    elif algo_type == "li":
        cfg.oracle_mode = True
        cfg.prob_purify = 0.0
    else:
        raise ValueError(f"Unknown type: {algo_type}")

    env = QuantumNetworkEnv(cfg, verbose=False)
    if algo_type == "li":
        env = LiWrapper(env, delay_steps=3)
    return env


def get_safe_action_mask(env):
    if hasattr(env, "get_action_mask"):
        return env.get_action_mask()
    elif hasattr(env.unwrapped, "get_action_mask"):
        return env.unwrapped.get_action_mask()
    return None


# ===================================================================
#  核心逻辑
# ===================================================================

def run_benchmark():
    print(f"\n{'=' * 70}")
    print(f"🚀 QNBench 稳定性测试 (Multi-Epoch)")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Duration per Epoch: {TEST_DURATION_MS / 1000:.1f} s")
    print(f"{'=' * 70}\n")

    # 存储最终绘图数据
    plot_data = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name, config in MODELS_CONFIG.items():
        model_path = get_model_path(config["path"])
        if not os.path.exists(model_path):
            print(f"⚠️  跳过 [{name}]: 文件不存在")
            continue

        print(f"Testing [{name}] ", end="")

        # 1. 加载环境和模型
        try:
            env = make_env_for_model(config["type"], BENCHMARK_SCENARIO)
        except Exception as e:
            print(f"-> 环境错误: {e}")
            continue

        ppo_cfg = PPOConfig(hidden_dim=config.get("hidden_dim", 256))
        trainer = PPOTrainer(env, ppo_cfg, device=device)
        try:
            trainer.load(model_path)
        except RuntimeError:
            print("-> 权重加载失败")
            continue

        model = trainer.model
        model.eval()

        # 2. 运行多个 Epoch
        epoch_rates = []
        # 用于存储所有epoch的插值后数据: shape [NUM_EPOCHS, PLOT_POINTS]
        all_interpolated_y = np.zeros((NUM_EPOCHS, PLOT_POINTS))

        # 公共时间轴 (用于对齐所有曲线)
        common_time_axis = np.linspace(0, TEST_DURATION_MS / 1000.0, PLOT_POINTS)

        for seed in range(NUM_EPOCHS):
            # 每个 epoch 使用不同的种子
            current_seed = 42 + seed

            curr_time = 0.0
            cum_deliveries = 0

            # 记录轨迹：起始点 (0,0)
            t_history = [0.0]
            y_history = [0.0]

            obs, _ = env.reset(seed=current_seed)

            while curr_time < TEST_DURATION_MS:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mask = get_safe_action_mask(env)
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(
                    0) if mask is not None else None

                with torch.no_grad():
                    action_res = model.get_action_and_value(obs_t, mask_t)
                    action = action_res[0].cpu().numpy()[0]

                obs, _, term, trunc, info = env.step(action)

                if term or trunc:
                    dt = info.get("time_ms", 0)
                    delivered = info.get("delivered", 0)

                    curr_time += dt
                    cum_deliveries += delivered

                    # 记录数据点 (转换为秒)
                    t_history.append(curr_time / 1000.0)
                    y_history.append(cum_deliveries)

                    obs, _ = env.reset()

            # 计算该 Epoch 的速率
            duration_sec = curr_time / 1000.0
            rate = cum_deliveries / duration_sec if duration_sec > 0 else 0
            epoch_rates.append(rate)
            print(".", end="", flush=True)

            # 3. 数据对齐 (Interpolation)
            # 因为每个 Epoch 的事件时间点不同，我们需要将其映射到 common_time_axis 上
            # 使用 'previous' 插值，因为交付数是阶梯状增加的
            f = interp1d(t_history, y_history, kind='previous', bounds_error=False, fill_value="extrapolate")
            all_interpolated_y[seed, :] = f(common_time_axis)

        # 4. 统计分析
        avg_rate = np.mean(epoch_rates)
        std_rate = np.std(epoch_rates)

        # 计算曲线的均值和标准差
        mean_curve = np.mean(all_interpolated_y, axis=0)
        std_curve = np.std(all_interpolated_y, axis=0)

        plot_data[name] = {
            "x": common_time_axis,
            "mean": mean_curve,
            "std": std_curve,
            "avg_rate": avg_rate,
            "std_rate": std_rate,
            "color": config["color"]
        }

        print(f" Done. Rate: {avg_rate:.2f} ± {std_rate:.2f} Hz")

    if not plot_data:
        print("\n❌ 无数据生成。")
        return

    # ─── 绘图 (带波动阴影) ───
    plt.figure(figsize=(10, 6))

    for name, data in plot_data.items():
        x = data["x"]
        y_mean = data["mean"]
        y_std = data["std"]
        color = data["color"]

        # 绘制平均线
        plt.plot(x, y_mean, label=f"{name} ({data['avg_rate']:.2f} Hz)", color=color, linewidth=2)

        # 绘制波动区域 (Mean ± Std)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.15)

    plt.title(
        f"Benchmark: Stability Analysis ({NUM_EPOCHS} Epochs)\nDuration: {TEST_DURATION_MS / 1000}s, Nodes: {BENCHMARK_SCENARIO.num_nodes}",
        fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Cumulative Entangled Pairs", fontsize=12)

    # 图例
    plt.legend(fontsize=11, loc="upper left")
    plt.grid(True, alpha=0.3)

    # 保存
    save_path = os.path.join(SCRIPT_DIR, "benchmark_epoch_stats.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ 波动分析图已保存至: {save_path}")

    # 打印最终统计表
    print("\n" + "=" * 70)
    print(f"{'Model':<20} | {'Mean Rate':<12} | {'Std Dev':<12} | {'Stability'}")
    print("-" * 70)
    for name, data in sorted(plot_data.items(), key=lambda x: x[1]['avg_rate'], reverse=True):
        # 变异系数 (CV) 用于衡量不稳定性
        cv = (data['std_rate'] / data['avg_rate']) * 100 if data['avg_rate'] > 0 else 0
        print(f"{name:<20} | {data['avg_rate']:<12.2f} | {data['std_rate']:<12.2f} | CV: {cv:.1f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_benchmark()
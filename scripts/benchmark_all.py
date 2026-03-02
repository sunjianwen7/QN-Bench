#!/usr/bin/env python3
"""
scripts/benchmark_epoch_stats.py
================================
多 Epoch 综合基准测试脚本：展示平均性能与波动范围。

新增功能:
  --mode run    : 只跑数据，保存到 CSV
  --mode plot   : 只读 CSV 画图（不跑模型）
  --mode both   : 跑完数据后自动画图（默认）

使用示例:
    python scripts/benchmark_epoch_stats.py --mode run          # 只跑数据
    python scripts/benchmark_epoch_stats.py --mode plot         # 只画图
    python scripts/benchmark_epoch_stats.py --mode both         # 跑 + 画
    python scripts/benchmark_epoch_stats.py --mode plot --log   # 画图用对数坐标
"""

import sys
import os
import argparse
import collections
import numpy as np
import torch
import gymnasium as gym
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ─── 路径设置 ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from qnbench.envs import QuantumNetworkEnv, EnvConfig
from qnbench.rl.masked_ppo import PPOTrainer, PPOConfig

# ===================================================================
#  用户配置区域
# ===================================================================

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

TEST_DURATION_MS = 500_000.0  # 500 秒
NUM_EPOCHS = 10
PLOT_POINTS = 200

# CSV 保存路径
CSV_DIR = os.path.join(SCRIPT_DIR, "benchmark_data")

MODELS_CONFIG = {
    "Ours (QNBench)": {
        "path": "scripts/checkpoints/ppo_best.pt",
        "type": "ours",
        "color": "#d62728",
        "hidden_dim": 128
    },
    "Metzger (POMDP)": {
        "path": "related/checkpoints/metzger_repro/metzger_final.pt",
        "type": "metzger",
        "color": "#1f77b4",
        "hidden_dim": 256
    },
    "Haldar (Curriculum)": {
        "path": "related/checkpoints/haldar_repro/Stage3_9Nodes.pt",
        "type": "haldar",
        "color": "#2ca02c",
        "hidden_dim": 256
    },
    "Li et al. (Delay)": {
        "path": "related/checkpoints/li_repro/li_final.pt",
        "type": "li",
        "color": "#9467bd",
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
        self.static_mask = np.array([1, 1, 1, 1, 0, 0, 0], dtype=bool)

        if hasattr(env, "obs_dim"):
            self.obs_dim = env.obs_dim
        elif hasattr(env.unwrapped, "obs_dim"):
            self.obs_dim = env.unwrapped.obs_dim
        else:
            self.obs_dim = env.observation_space.shape[0]

        if hasattr(env.unwrapped, "num_nodes"):
            self.num_nodes = env.unwrapped.num_nodes

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs_history.clear()
        for _ in range(self.obs_history.maxlen - 1):
            self.obs_history.append(np.zeros_like(obs))
        self.obs_history.append(obs)
        return self.obs_history[0], info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.obs_history.append(obs)
        return self.obs_history[0], r, term, trunc, info

    def get_action_mask(self):
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
#  CSV 保存 / 加载
# ===================================================================

def save_results_to_csv(plot_data: dict):
    """
    保存格式:
      benchmark_data/
        {model_name}_curves.csv   — 每行: time, epoch_0, epoch_1, ...
        {model_name}_stats.csv    — 单行: avg_rate, std_rate, color
    """
    os.makedirs(CSV_DIR, exist_ok=True)

    for name, data in plot_data.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")

        # --- 保存曲线数据 (time + 每个epoch的插值数据) ---
        curves_path = os.path.join(CSV_DIR, f"{safe_name}_curves.csv")
        x = data["x"]  # shape: (PLOT_POINTS,)
        all_y = data["all_interpolated_y"]  # shape: (NUM_EPOCHS, PLOT_POINTS)

        # 构建矩阵: 第一列是 time，后续列是各 epoch
        n_epochs = all_y.shape[0]
        header = "time," + ",".join([f"epoch_{i}" for i in range(n_epochs)])
        matrix = np.column_stack([x, all_y.T])  # shape: (PLOT_POINTS, 1 + n_epochs)
        np.savetxt(curves_path, matrix, delimiter=",", header=header, comments="")

        # --- 保存统计数据 ---
        stats_path = os.path.join(CSV_DIR, f"{safe_name}_stats.csv")
        with open(stats_path, "w") as f:
            f.write("avg_rate,std_rate,color\n")
            f.write(f"{data['avg_rate']},{data['std_rate']},{data['color']}\n")

        print(f"   💾 [{name}] -> {curves_path}")

    print(f"\n✅ 数据已保存至: {CSV_DIR}/")


def load_results_from_csv() -> dict:
    """从 CSV 目录加载所有模型数据"""
    if not os.path.isdir(CSV_DIR):
        print(f"❌ 找不到数据目录: {CSV_DIR}")
        print(f"   请先运行 --mode run 生成数据。")
        sys.exit(1)

    plot_data = {}

    # 扫描所有 *_curves.csv
    curve_files = sorted([f for f in os.listdir(CSV_DIR) if f.endswith("_curves.csv")])
    if not curve_files:
        print(f"❌ {CSV_DIR} 中没有找到 *_curves.csv 文件。")
        sys.exit(1)

    for cf in curve_files:
        safe_name = cf.replace("_curves.csv", "")
        stats_file = os.path.join(CSV_DIR, f"{safe_name}_stats.csv")

        if not os.path.exists(stats_file):
            print(f"⚠️  跳过 {safe_name}: 缺少 _stats.csv")
            continue

        # 加载曲线
        curves = np.genfromtxt(os.path.join(CSV_DIR, cf), delimiter=",", skip_header=1)
        x = curves[:, 0]
        all_y = curves[:, 1:].T  # shape: (n_epochs, PLOT_POINTS)

        # 加载统计
        with open(stats_file, "r") as f:
            f.readline()  # skip header
            parts = f.readline().strip().split(",")
            avg_rate = float(parts[0])
            std_rate = float(parts[1])
            color = parts[2]

        # 反向查找显示名 (尝试匹配 MODELS_CONFIG)
        display_name = safe_name  # 默认用文件名
        for model_name, cfg in MODELS_CONFIG.items():
            candidate = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
            if candidate == safe_name:
                display_name = model_name
                color = cfg["color"]  # 优先用配置里的颜色
                break

        plot_data[display_name] = {
            "x": x,
            "mean": np.mean(all_y, axis=0),
            "std": np.std(all_y, axis=0),
            "avg_rate": avg_rate,
            "std_rate": std_rate,
            "color": color,
        }
        print(f"   📂 已加载 [{display_name}] ({all_y.shape[0]} epochs, {len(x)} points)")

    return plot_data


# ===================================================================
#  核心: 跑数据
# ===================================================================

def run_benchmark() -> dict:
    print(f"\n{'=' * 70}")
    print(f"🚀 QNBench 稳定性测试 (Multi-Epoch)")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Duration per Epoch: {TEST_DURATION_MS / 1000:.1f} s")
    print(f"{'=' * 70}\n")

    plot_data = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name, config in MODELS_CONFIG.items():
        model_path = get_model_path(config["path"])
        if not os.path.exists(model_path):
            print(f"⚠️  跳过 [{name}]: 文件不存在")
            continue

        print(f"Testing [{name}] ", end="")

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

        epoch_rates = []
        all_interpolated_y = np.zeros((NUM_EPOCHS, PLOT_POINTS))
        common_time_axis = np.linspace(0, TEST_DURATION_MS / 1000.0, PLOT_POINTS)

        for seed in range(NUM_EPOCHS):
            current_seed = 42 + seed
            curr_time = 0.0
            cum_deliveries = 0

            t_history = [0.0]
            y_history = [0.0]

            obs, _ = env.reset(seed=current_seed)

            while curr_time < TEST_DURATION_MS:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mask = get_safe_action_mask(env)
                mask_t = (
                    torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                    if mask is not None else None
                )

                with torch.no_grad():
                    action_res = model.get_action_and_value(obs_t, mask_t)
                    action = action_res[0].cpu().numpy()[0]

                obs, _, term, trunc, info = env.step(action)

                if term or trunc:
                    dt = info.get("time_ms", 0)
                    delivered = info.get("delivered", 0)

                    curr_time += dt
                    cum_deliveries += delivered

                    t_history.append(curr_time / 1000.0)
                    y_history.append(cum_deliveries)

                    obs, _ = env.reset()

            duration_sec = curr_time / 1000.0
            rate = cum_deliveries / duration_sec if duration_sec > 0 else 0
            epoch_rates.append(rate)
            print(".", end="", flush=True)

            f = interp1d(t_history, y_history, kind='linear',
                         bounds_error=False, fill_value="extrapolate")
            all_interpolated_y[seed, :] = f(common_time_axis)

        avg_rate = np.mean(epoch_rates)
        std_rate = np.std(epoch_rates)

        plot_data[name] = {
            "x": common_time_axis,
            "all_interpolated_y": all_interpolated_y,  # 保留原始数据用于存 CSV
            "mean": np.mean(all_interpolated_y, axis=0),
            "std": np.std(all_interpolated_y, axis=0),
            "avg_rate": avg_rate,
            "std_rate": std_rate,
            "color": config["color"]
        }

        print(f" Done. Rate: {avg_rate:.2f} ± {std_rate:.2f} Hz")

    return plot_data


# ===================================================================
#  核心: 画图
# ===================================================================

def plot_results(plot_data: dict, use_log: bool = False):
    if not plot_data:
        print("\n❌ 无数据可绘图。")
        return

    plt.figure(figsize=(10, 6))

    for name, data in plot_data.items():
        x = data["x"]
        y_mean = data["mean"]
        y_std = data["std"]
        color = data["color"]

        plt.plot(x, y_mean,
                 label=f"{name} ({data['avg_rate']:.2f} Hz)",
                 color=color, linewidth=2)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std,
                         color=color, alpha=0.15)

    plt.title("Benchmark: Stability Analysis", fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Cumulative Entangled Pairs", fontsize=12)

    if use_log:
        plt.yscale("log")

    plt.legend(fontsize=11, loc="upper left")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(SCRIPT_DIR, "benchmark_epoch_stats.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ 图表已保存至: {save_path}")

    # 打印统计表
    print("\n" + "=" * 70)
    print(f"{'Model':<25} | {'Mean Rate':<12} | {'Std Dev':<12} | {'Stability'}")
    print("-" * 70)
    for name, data in sorted(plot_data.items(), key=lambda x: x[1]['avg_rate'], reverse=True):
        cv = (data['std_rate'] / data['avg_rate']) * 100 if data['avg_rate'] > 0 else 0
        print(f"{name:<25} | {data['avg_rate']:<12.2f} | {data['std_rate']:<12.2f} | CV: {cv:.1f}%")
    print("=" * 70 + "\n")


# ===================================================================
#  主入口
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="QNBench Multi-Epoch Benchmark")
    parser.add_argument("--mode", choices=["run", "plot", "both"],
                        default="both",
                        help="run=只跑数据存CSV, plot=只读CSV画图, both=跑+画 (默认)")
    parser.add_argument(
        "--log",
        action="store_false",
        help="关闭对数坐标（默认开启）"
    )
    args = parser.parse_args()

    if args.mode in ("run", "both"):
        plot_data = run_benchmark()
        if plot_data:
            save_results_to_csv(plot_data)
        if args.mode == "run":
            print("📊 数据已保存。用 --mode plot 画图。")
            return
    else:
        plot_data = None

    if args.mode in ("plot", "both"):
        if plot_data is None:
            plot_data = load_results_from_csv()
        plot_results(plot_data, use_log=args.log)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
paper_eval.py
=============

Generate two paper-ready result figures:

Figure 1 (QNBench, your task):
    PPO (your trained checkpoint) vs. traditional heuristics on your Gym.

Figure 2 (Related-work comparison, two selected papers):
    Your PPO vs. two reproduced related-work PPO policies:
      - Reiß & van Loock (2023)  [oracle, no-purify, SKR reward]
      - Metzger et al. (2025)    [POMDP, SKR reward]

This script is designed to be *self-contained*:
- It does NOT modify your env or networks.
- It infers PPO network architecture (hidden_dim / n_layers) from checkpoints,
  so it can load checkpoints trained with different widths.

Outputs (by default into ./scripts/runs/paper_figs/):
    fig1_ppo_vs_heuristics.(png|pdf)
    fig2_ppo_vs_related.(png|pdf)
    table_fig1_summary.csv
    table_fig2_rates.csv
    (optional) per-episode raw logs as CSVs

Run example (from project root):
    python paper_eval.py \
        --ckpt scripts/checkpoints/ppo_best.pt \
        --cfg  configs/default.yaml \
        --out  scripts/runs/paper_figs \
        --episodes 200 \
        --device cpu

Notes
-----
- Requires: gymnasium, torch, numpy, pandas, matplotlib
- Uses only matplotlib (no seaborn).
"""

from __future__ import annotations

import os
import sys
import argparse
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure qnbench is importable when running from scripts/ or project root.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR))  # assume run from project root
if os.path.isdir(os.path.join(PROJ_ROOT, "qnbench")):
    sys.path.insert(0, PROJ_ROOT)
else:
    # If user copied this file into scripts/, go one level up.
    sys.path.insert(0, os.path.dirname(PROJ_ROOT))

import torch

from qnbench.envs import QuantumNetworkEnv, EnvConfig
from qnbench.baselines import RandomAgent, GreedyAgent, SwapASAPAgent
from qnbench.rl.networks import ActorCritic, ActorCriticAgent


# ============================================================
#  Helpers: checkpoint loading with architecture inference
# ============================================================

def _infer_actorcritic_arch(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """
    Infer (obs_dim, hidden_dim, n_layers) from an ActorCritic state_dict.

    Assumptions based on qnbench.rl.networks.ActorCritic:
        encoder = [Linear, ReLU] * n_layers
        actor_head = Linear(hidden_dim -> num_actions)
        critic_head = MLP(hidden_dim -> 1)
    """
    # Find encoder Linear layers
    enc_keys = []
    for k, v in state_dict.items():
        if k.startswith("encoder.") and k.endswith(".weight") and v.ndim == 2:
            # encoder.<idx>.weight
            try:
                idx = int(k.split(".")[1])
            except Exception:
                continue
            enc_keys.append((idx, k, v.shape))
    if not enc_keys:
        raise ValueError("Cannot infer architecture: no encoder.*.weight in checkpoint.")

    enc_keys.sort(key=lambda x: x[0])
    # Count number of Linear layers (each Linear has a '.weight')
    n_layers = len(enc_keys)

    # First Linear decides obs_dim and hidden_dim
    first_shape = enc_keys[0][2]  # (hidden_dim, obs_dim)
    hidden_dim = int(first_shape[0])
    obs_dim = int(first_shape[1])
    return obs_dim, hidden_dim, n_layers


def load_ppo_agent(
    ckpt_path: str,
    obs_dim: int,
    num_nodes: int,
    device: str = "cpu",
    name: str = "PPO",
) -> ActorCriticAgent:
    """
    Load an ActorCriticAgent from a checkpoint.

    - Infers hidden_dim and n_layers from ckpt (so checkpoints trained with
      different widths can be loaded).
    - Uses the *current* env obs_dim (must match ckpt's obs_dim).
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model_state': {ckpt_path}")

    state = ckpt["model_state"]
    ckpt_obs_dim, hidden_dim, n_layers = _infer_actorcritic_arch(state)

    if ckpt_obs_dim != obs_dim:
        raise ValueError(
            f"obs_dim mismatch: ckpt has {ckpt_obs_dim}, env has {obs_dim}. "
            f"Use the same observation definition as during training."
        )

    model = ActorCritic(
        obs_dim=obs_dim,
        num_actions=7,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        num_nodes=num_nodes,
    )
    model.load_state_dict(state)

    agent = ActorCriticAgent(model, device=device)

    # Patch a nicer name for plotting
    agent._custom_name = name  # type: ignore[attr-defined]
    return agent


def agent_name(agent) -> str:
    # Baselines have .name property; ActorCriticAgent provides "PPO" property.
    return getattr(agent, "_custom_name", getattr(agent, "name", agent.__class__.__name__))


def _infer_fidelity_from_delivery_reward(cfg: EnvConfig, avg_delivery_reward: float) -> float:
    """
    In env.py: delivery reward per delivered pair is:
        r = delivery_reward + (F - threshold) * delivery_bonus_factor
    => F = threshold + (r - delivery_reward) / delivery_bonus_factor
    """
    if cfg.delivery_bonus_factor <= 1e-9:
        return float(cfg.delivery_fidelity_threshold)
    f = cfg.delivery_fidelity_threshold + (avg_delivery_reward - cfg.delivery_reward) / cfg.delivery_bonus_factor
    return float(np.clip(f, 0.0, 1.0))


class MaskedAgent:
    """Wrap an agent and apply an additional *static* action mask.

    This is useful to evaluate policies trained under a restricted action
    set (e.g., no-purification in a related-work reproduction) fairly on
    your benchmark without changing the environment.
    """

    def __init__(self, inner, static_mask_7: np.ndarray, name: str):
        self.inner = inner
        self.static_mask_7 = static_mask_7.astype(bool)
        self._custom_name = name

    @property
    def name(self) -> str:  # for agent_name()
        return self._custom_name

    def reset(self):
        return getattr(self.inner, "reset", lambda: None)()

    def act(self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = True) -> np.ndarray:
        mask2 = mask & self.static_mask_7  # broadcast (7,) to (N,7)
        return self.inner.act(obs, mask2, deterministic=deterministic)


# ============================================================
#  Episode-based evaluation (Figure 1)
# ============================================================

def evaluate_agent_episodes(
    env: QuantumNetworkEnv,
    agent,
    n_episodes: int,
    seed: int,
) -> pd.DataFrame:
    """
    Evaluate an agent for n episodes.
    Collects multi-dimensional metrics per episode:
      - total reward
      - total delivered pairs
      - simulated time (ms)
      - pair rate (pairs/s)
      - inferred mean delivered fidelity
      - reward breakdown totals (op_cost, invalid, failed, engine, delivery, time)
    """
    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        agent.reset()

        ep_reward = 0.0
        breakdown = {"op_cost": 0.0, "invalid": 0.0, "failed": 0.0, "engine": 0.0, "delivery": 0.0, "time": 0.0}

        delivered_prev = 0
        delivered_fidelities: List[float] = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            mask = env.get_action_mask()
            action = agent.act(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += float(reward)

            rb = info.get("reward_breakdown", {})
            for k in breakdown.keys():
                breakdown[k] += float(rb.get(k, 0.0))

            # Track per-delivery inferred fidelities (handles multiple deliveries per step)
            delivered_now = int(info.get("delivered", 0))
            delta = delivered_now - delivered_prev
            delivered_prev = delivered_now

            r_del = float(rb.get("delivery", 0.0))
            if delta > 0 and r_del > 0.0:
                avg_r = r_del / float(delta)
                f = _infer_fidelity_from_delivery_reward(env.cfg, avg_r)
                delivered_fidelities.extend([f] * delta)

        t_ms = float(info.get("time_ms", 0.0))
        delivered = int(info.get("delivered", 0))
        steps = int(info.get("step", 0))
        pair_rate = delivered / (t_ms / 1000.0) if t_ms > 0 else 0.0
        mean_f = float(np.mean(delivered_fidelities)) if delivered_fidelities else np.nan

        row = {
            "agent": agent_name(agent),
            "episode": ep,
            "reward": ep_reward,
            "delivered": delivered,
            "time_ms": t_ms,
            "steps": steps,
            "pair_rate": pair_rate,
            "mean_delivered_fidelity": mean_f,
        }
        for k, v in breakdown.items():
            row[f"r_{k}"] = v
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std summary per agent."""
    agg = df.groupby("agent").agg(
        mean_reward=("reward", "mean"),
        std_reward=("reward", "std"),
        mean_delivered=("delivered", "mean"),
        std_delivered=("delivered", "std"),
        mean_time_ms=("time_ms", "mean"),
        std_time_ms=("time_ms", "std"),
        mean_pair_rate=("pair_rate", "mean"),
        std_pair_rate=("pair_rate", "std"),
        mean_fidelity=("mean_delivered_fidelity", "mean"),
        std_fidelity=("mean_delivered_fidelity", "std"),
        mean_r_op_cost=("r_op_cost", "mean"),
        mean_r_invalid=("r_invalid", "mean"),
        mean_r_failed=("r_failed", "mean"),
        mean_r_engine=("r_engine", "mean"),
        mean_r_delivery=("r_delivery", "mean"),
        mean_r_time=("r_time", "mean"),
    ).reset_index()
    return agg


def plot_fig1(summary: pd.DataFrame, out_path: str, title: str):
    """
    Figure 1: PPO vs heuristics on your benchmark.
    Two panels:
      (A) mean pair rate (pairs/s) with std error bars
      (B) stacked mean reward breakdown (per episode)
    """
    agents = summary["agent"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    # (A) Pair rate bar
    x = np.arange(len(agents))
    axes[0].bar(x, summary["mean_pair_rate"].values)
    axes[0].errorbar(x, summary["mean_pair_rate"].values, yerr=summary["std_pair_rate"].values, fmt="none", capsize=4)
    axes[0].set_xticks(x, agents, rotation=20, ha="right")
    axes[0].set_ylabel("Delivered pairs / second")
    axes[0].set_title("Throughput (higher is better)")

    # (B) Reward breakdown stacked bar
    parts = ["mean_r_op_cost", "mean_r_invalid", "mean_r_failed", "mean_r_engine", "mean_r_delivery", "mean_r_time"]
    labels = ["op_cost", "invalid", "failed", "engine", "delivery", "time"]
    bottom = np.zeros(len(agents))
    for p, lab in zip(parts, labels):
        vals = summary[p].values
        axes[1].bar(x, vals, bottom=bottom, label=lab)
        bottom += vals
    axes[1].set_xticks(x, agents, rotation=20, ha="right")
    axes[1].set_ylabel("Mean episode return (decomposed)")
    axes[1].set_title("Reward composition")
    axes[1].legend(fontsize=9, frameon=False, loc="best")

    fig.suptitle(title)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_fig2(summary: pd.DataFrame, out_path: str, title: str):
    """Figure 2 uses the same visual grammar as Figure 1 (throughput + reward composition)."""
    plot_fig1(summary=summary, out_path=out_path, title=title)


# ============================================================
#  Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="scripts/checkpoints/ppo_best.pt",
                    help="Your PPO checkpoint path.")
    ap.add_argument("--cfg", type=str, default="configs/default.yaml",
                    help="Your env YAML (env: section).")
    ap.add_argument("--out", type=str, default="scripts/runs/paper_figs",
                    help="Output directory.")
    ap.add_argument("--episodes", type=int, default=200,
                    help="Episodes for Figure 1 evaluation.")
    ap.add_argument("--seed", type=int, default=1000,
                    help="Base seed for Figure 1.")
    ap.add_argument("--device", type=str, default="cpu",
                    help="cpu or cuda")
    ap.add_argument("--save_raw", action="store_true",
                    help="Save per-episode raw CSVs.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --------------------------
    # Figure 1: your task
    # --------------------------
    cfg = EnvConfig.from_yaml(args.cfg)
    env = QuantumNetworkEnv(cfg=cfg, verbose=False)

    ppo_agent = load_ppo_agent(
        ckpt_path=args.ckpt,
        obs_dim=env.obs_dim,
        num_nodes=cfg.num_nodes,
        device=args.device,
        name="PPO (MDV)",
    )

    agents_fig1 = {
        "PPO (MDV)": ppo_agent,
        "Swap-ASAP": SwapASAPAgent(num_nodes=cfg.num_nodes, enable_purify=False),
        "Swap-ASAP+Purify": SwapASAPAgent(num_nodes=cfg.num_nodes, enable_purify=True),
        "Greedy": GreedyAgent(num_nodes=cfg.num_nodes),
        "Random": RandomAgent(num_nodes=cfg.num_nodes),
    }

    # Evaluate
    dfs = []
    for name, ag in agents_fig1.items():
        df = evaluate_agent_episodes(env, ag, n_episodes=args.episodes, seed=args.seed)
        df["agent"] = name  # enforce stable label
        dfs.append(df)
    df_raw = pd.concat(dfs, ignore_index=True)
    summary = summarize_episodes(df_raw)

    summary.to_csv(os.path.join(args.out, "table_fig1_summary.csv"), index=False)
    if args.save_raw:
        df_raw.to_csv(os.path.join(args.out, "raw_fig1_per_episode.csv"), index=False)

    plot_fig1(
        summary=summary,
        out_path=os.path.join(args.out, "fig1_ppo_vs_heuristics.png"),
        title="Figure 1: PPO (MDV) vs Heuristics on QNBench",
    )
    plot_fig1(
        summary=summary,
        out_path=os.path.join(args.out, "fig1_ppo_vs_heuristics.pdf"),
        title="Figure 1: PPO (MDV) vs Heuristics on QNBench",
    )

    # --------------------------
    # Figure 2: related work
    # (same visual grammar as Fig.1; no secret-key-rate in outputs)
    # --------------------------
    ckpt_reiss = os.path.join("related", "checkpoints", "reiss_repro", "reiss_final.pt")
    ckpt_metz  = os.path.join("related", "checkpoints", "metzger_repro", "metzger_final.pt")

    reiss_raw = load_ppo_agent(
        ckpt_path=ckpt_reiss,
        obs_dim=env.obs_dim,
        num_nodes=cfg.num_nodes,
        device=args.device,
        name="PPO (Reiß+vL repro)",
    )
    # Reiß reproduction disables purification; apply the same restriction for fair evaluation.
    reiss_static_mask = np.array([1, 1, 1, 1, 0, 0, 1], dtype=bool)
    reiss_agent = MaskedAgent(reiss_raw, reiss_static_mask, name="PPO (Reiß+vL repro)")

    metz_agent = load_ppo_agent(
        ckpt_path=ckpt_metz,
        obs_dim=env.obs_dim,
        num_nodes=cfg.num_nodes,
        device=args.device,
        name="PPO (Metzger repro)",
    )

    agents_fig2 = {
        "PPO (MDV)": ppo_agent,
        "PPO (Reiß+vL repro)": reiss_agent,
        "PPO (Metzger repro)": metz_agent,
    }

    dfs2 = []
    for name, ag in agents_fig2.items():
        df = evaluate_agent_episodes(env, ag, n_episodes=args.episodes, seed=args.seed + 50000)
        df["agent"] = name
        dfs2.append(df)
    df_raw2 = pd.concat(dfs2, ignore_index=True)
    summary2 = summarize_episodes(df_raw2)

    summary2.to_csv(os.path.join(args.out, "table_fig2_summary.csv"), index=False)
    if args.save_raw:
        df_raw2.to_csv(os.path.join(args.out, "raw_fig2_per_episode.csv"), index=False)

    plot_fig2(
        summary=summary2,
        out_path=os.path.join(args.out, "fig2_ppo_vs_related.png"),
        title="Figure 2: PPO (MDV) vs Related-Work PPO Policies on QNBench",
    )
    plot_fig2(
        summary=summary2,
        out_path=os.path.join(args.out, "fig2_ppo_vs_related.pdf"),
        title="Figure 2: PPO (MDV) vs Related-Work PPO Policies on QNBench",
    )

    print("\nDone.")
    print(f"Outputs saved to: {args.out}")


if __name__ == "__main__":
    main()

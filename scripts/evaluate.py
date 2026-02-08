#!/usr/bin/env python3
"""
scripts/evaluate.py
===================

Load a trained PPO checkpoint and evaluate it against baselines.

Usage::

    python scripts/evaluate.py --checkpoint checkpoints/ppo_best.pt
    python scripts/evaluate.py --checkpoint checkpoints/ppo_best.pt --episodes 200
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnbench.envs import QuantumNetworkEnv, EnvConfig
from qnbench.baselines import RandomAgent, GreedyAgent, SwapASAPAgent
from qnbench.evaluation import compare_agents, format_results_table
from qnbench.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent + baselines")
    parser.add_argument("--checkpoint", type=str,default="/Users/jevonsun/PycharmProjects/qnbenchmark/scripts/checkpoints/ppo_best.pt",
                        help="Path to PPO checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    setup_logging(level="INFO", sim_level="WARNING")

    # ── Lazy import ──────────────────────────────────────────────
    try:
        import torch
        from qnbench.rl.networks import ActorCritic, ActorCriticAgent
    except ImportError:
        print("ERROR: PyTorch required. pip install torch", flush=True)
        sys.exit(1)

    # ── Config & Env ─────────────────────────────────────────────
    cfg = EnvConfig.from_yaml(args.config) if args.config else EnvConfig()
    env = QuantumNetworkEnv(cfg=cfg, verbose=False)
    n = cfg.num_nodes

    # ── Load model ───────────────────────────────────────────────
    print(f"\n  Loading checkpoint: {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    model = ActorCritic(
        obs_dim=env.obs_dim,
        num_actions=7,
        num_nodes=n,
    )
    model.load_state_dict(ckpt["model_state"])
    ppo_agent = ActorCriticAgent(model, device=args.device)

    training_steps = ckpt.get("total_steps", "?")
    training_best = ckpt.get("best_mean_reward", "?")
    print(f"  Training steps: {training_steps}, Best mean reward: {training_best}",
          flush=True)

    # ── Build comparison agents ──────────────────────────────────
    agents = {
        "PPO (trained)":     ppo_agent,
        "Random":            RandomAgent(n, seed=args.seed),
        "Greedy":            GreedyAgent(n),
        "Swap-ASAP":         SwapASAPAgent(n),
        "Swap-ASAP+Purify":  SwapASAPAgent(n, enable_purify=True),
    }

    # ── Evaluate ─────────────────────────────────────────────────
    results = compare_agents(cfg, agents, args.episodes, args.seed, verbose=True)

    print(f"\n{'='*70}", flush=True)
    print("  FINAL COMPARISON", flush=True)
    print(f"{'='*70}", flush=True)
    print(format_results_table(results), flush=True)


if __name__ == "__main__":
    main()

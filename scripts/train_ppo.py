#!/usr/bin/env python3
"""
scripts/train_ppo.py
====================

Train a PPO agent on the quantum network environment.

Usage::

    python scripts/train_ppo.py --steps 200000
    python scripts/train_ppo.py --config configs/default.yaml --device cuda
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnbench.envs import QuantumNetworkEnv, EnvConfig
from qnbench.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train PPO on QNBench")
    parser.add_argument("--steps", type=int, default=2000_000,
                        help="Total training timesteps")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config path")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=5,
                        help="Print progress every N updates")
    args = parser.parse_args()

    # 初始化日志: 训练日志 INFO, 仿真引擎日志 WARNING (不刷屏)
    setup_logging(level="INFO", sim_level="WARNING")

    # ── Lazy import (requires torch) ─────────────────────────────
    try:
        from qnbench.rl.masked_ppo import PPOTrainer, PPOConfig
    except ImportError:
        print("ERROR: PyTorch is required for training. Install: pip install torch",
              flush=True)
        sys.exit(1)

    # ── Config ───────────────────────────────────────────────────
    if args.config:
        env_cfg = EnvConfig.from_yaml(args.config)
    else:
        env_cfg = EnvConfig()

    ppo_cfg = PPOConfig(
        lr=args.lr,
        hidden_dim=args.hidden,
        seed=args.seed,
    )

    # ── Environment ──────────────────────────────────────────────
    env = QuantumNetworkEnv(cfg=env_cfg, verbose=False)

    print(f"\n  Environment: {env_cfg.num_nodes} nodes, "
          f"distances={env_cfg.node_distances} km", flush=True)
    print(f"  Physics: p_gen={env_cfg.prob_gen}, p_swap={env_cfg.prob_swap}, "
          f"p_purify={env_cfg.prob_purify}", flush=True)
    print(f"  Coherence: {env_cfg.t_coherence} ms, "
          f"Init fidelity: {env_cfg.init_fidelity}", flush=True)

    # ── Training ─────────────────────────────────────────────────
    trainer = PPOTrainer(env, ppo_cfg, device=args.device)
    trainer.train(
        total_timesteps=args.steps,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
    )

    # ── Quick eval after training ────────────────────────────────
    print(f"\n{'─'*65}", flush=True)
    print(f"  Post-training evaluation (20 episodes)...", flush=True)
    print(f"{'─'*65}", flush=True)

    from qnbench.rl.networks import ActorCriticAgent
    from qnbench.evaluation import evaluate_agent

    agent = ActorCriticAgent(trainer.model, device=args.device)
    results = evaluate_agent(env, agent, n_episodes=20, seed=9999, verbose=True)
    print(
        f"\n  Result: reward={results['mean_reward']:+.2f} ± {results['std_reward']:.2f}  "
        f"delivered={results['total_delivered']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

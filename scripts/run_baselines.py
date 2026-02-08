#!/usr/bin/env python3
"""
scripts/run_baselines.py
========================

Evaluate all heuristic baselines on the quantum network environment
and print a comparison table.

Usage::

    python scripts/run_baselines.py
    python scripts/run_baselines.py --episodes 200 --config configs/default.yaml
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
    parser = argparse.ArgumentParser(description="Run QNBench baselines")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes per agent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (uses defaults if omitted)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # 基线评估不需要仿真细节日志
    setup_logging(level="INFO", sim_level="WARNING")

    # ── Build config ─────────────────────────────────────────────
    if args.config:
        cfg = EnvConfig.from_yaml(args.config)
    else:
        cfg = EnvConfig()

    print(f"\n{'='*70}", flush=True)
    print(f"  QNBench Baseline Comparison", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Topology  : {cfg.num_nodes} nodes, distances={cfg.node_distances} km",
          flush=True)
    print(f"  Physics   : p_gen={cfg.prob_gen}, p_swap={cfg.prob_swap}, "
          f"p_purify={cfg.prob_purify}", flush=True)
    print(f"  Coherence : {cfg.t_coherence} ms, Init fidelity: {cfg.init_fidelity}",
          flush=True)
    print(f"  Episodes  : {args.episodes}, Seed: {args.seed}", flush=True)
    print(f"{'='*70}", flush=True)

    # ── Build agents ─────────────────────────────────────────────
    n = cfg.num_nodes
    agents = {
        "Random":            RandomAgent(n, seed=args.seed),
        "Greedy":            GreedyAgent(n),
        "Swap-ASAP":         SwapASAPAgent(n, enable_purify=False),
        "Swap-ASAP+Purify":  SwapASAPAgent(n, enable_purify=True),
    }

    # ── Run evaluation ───────────────────────────────────────────
    results = compare_agents(
        cfg, agents,
        n_episodes=args.episodes,
        seed=args.seed,
        verbose=args.verbose,
    )

    # ── Print results table ──────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("  RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(format_results_table(results), flush=True)


if __name__ == "__main__":
    main()

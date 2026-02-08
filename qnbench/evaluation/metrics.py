"""
qnbench.evaluation.metrics
============================

Episode-level metrics and aggregation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics collected at the end of one episode."""
    reward: float           # total episode reward
    delivered: int          # number of end-to-end pairs delivered
    time_ms: float          # simulated time elapsed (ms)
    steps: int              # number of agent steps taken
    breakdown: dict = None  # per-component reward breakdown

    def __post_init__(self):
        if self.breakdown is None:
            self.breakdown = {}


def aggregate_metrics(episodes: List[EpisodeMetrics]) -> Dict[str, float]:
    """
    Aggregate a list of per-episode metrics into summary statistics.

    Returns
    -------
    dict with keys:
        mean_reward, std_reward, median_reward,
        mean_delivered, total_delivered,
        delivery_rate (deliveries per episode),
        mean_episode_time_ms, mean_steps,
        breakdown_* (mean of each reward component)
    """
    rewards = np.array([e.reward for e in episodes])
    delivered = np.array([e.delivered for e in episodes])
    times = np.array([e.time_ms for e in episodes])
    steps = np.array([e.steps for e in episodes])

    result = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "mean_delivered": float(np.mean(delivered)),
        "total_delivered": int(np.sum(delivered)),
        "delivery_rate": float(np.mean(delivered > 0)),  # fraction with ≥1
        "mean_episode_time_ms": float(np.mean(times)),
        "mean_steps": float(np.mean(steps)),
    }

    # 聚合奖励分项
    all_keys = set()
    for e in episodes:
        all_keys.update(e.breakdown.keys())
    for k in sorted(all_keys):
        vals = [e.breakdown.get(k, 0.0) for e in episodes]
        result[f"breakdown_{k}"] = float(np.mean(vals))

    return result


def format_results_table(results: List[Dict[str, float]]) -> str:
    """
    Format a list of agent results as a human-readable ASCII table.

    Parameters
    ----------
    results : list of dicts from ``evaluate_agent`` / ``compare_agents``

    Returns
    -------
    Multi-line string table.
    """
    header = (
        f"{'Agent':<22} │ {'Reward':>10} │ {'±Std':>8} │ "
        f"{'Delivered':>9} │ {'Del/Ep':>7} │ {'Steps':>6} │ "
        f"{'Wall(s)':>8}"
    )
    sep = "─" * len(header)
    lines = [sep, header, sep]

    for r in sorted(results, key=lambda x: x.get("mean_reward", 0), reverse=True):
        lines.append(
            f"{r.get('agent', '?'):<22} │ "
            f"{r.get('mean_reward', 0):>+10.2f} │ "
            f"{r.get('std_reward', 0):>8.2f} │ "
            f"{r.get('total_delivered', 0):>9d} │ "
            f"{r.get('mean_delivered', 0):>7.2f} │ "
            f"{r.get('mean_steps', 0):>6.0f} │ "
            f"{r.get('wall_time_s', 0):>8.1f}"
        )
    lines.append(sep)
    return "\n".join(lines)

"""
qnbench.evaluation
===================

Evaluation runner and metrics for the benchmark.
"""

from .runner import evaluate_agent, compare_agents
from .metrics import EpisodeMetrics, aggregate_metrics, format_results_table

__all__ = [
    "evaluate_agent",
    "compare_agents",
    "format_results_table",
]

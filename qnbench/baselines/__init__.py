"""
qnbench.baselines
==================

Heuristic baseline agents for the quantum network benchmark.
"""

from .base import BaseAgent
from .random_agent import RandomAgent
from .greedy_agent import GreedyAgent
from .swap_asap import SwapASAPAgent

ALL_BASELINES = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "swap_asap": SwapASAPAgent,
    "swap_asap_purify": lambda n: SwapASAPAgent(n, enable_purify=True),
}

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "GreedyAgent",
    "SwapASAPAgent",
    "ALL_BASELINES",
]

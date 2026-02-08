"""
qnbench.baselines.random_agent
===============================

Uniformly random valid-action agent.

At each step, every node independently picks a uniformly random action
from its set of valid actions.  This is the weakest meaningful baseline
and establishes a performance floor.
"""

from __future__ import annotations

import numpy as np
from .base import BaseAgent


class RandomAgent(BaseAgent):
    """Pick a uniformly random valid action per node."""

    def __init__(self, num_nodes: int, seed: int = 0):
        super().__init__(num_nodes)
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray,
            deterministic: bool = True) -> np.ndarray:
        actions = np.zeros(self.num_nodes, dtype=int)
        for i in range(self.num_nodes):
            valid = np.where(mask[i])[0]
            actions[i] = self.rng.choice(valid) if len(valid) > 0 else 0
        return actions

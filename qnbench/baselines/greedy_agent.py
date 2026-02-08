"""
qnbench.baselines.greedy_agent
===============================

Greedy heuristic that chooses the highest-priority valid action per node.

Priority order (highest first):
    1. Swap   — if both left and right links are available, bridge them.
    2. Gen_L  — generate towards the left if a free memory exists.
    3. Gen_R  — generate towards the right if a free memory exists.
    4. Purify_L / Purify_R — if ≥2 same-direction links, purify.
    5. Discard — drop lowest-fidelity link to free memory.
    6. Wait   — do nothing.

This mirrors a "generate-always, swap-ASAP, purify-opportunistically"
heuristic that is commonly used as a strong baseline in the quantum
networking literature.
"""

from __future__ import annotations

import numpy as np
from .base import BaseAgent


# Priority table: action_id → priority (higher = preferred)
# Swap > Gen > Purify > Discard > Wait
_PRIORITY = {
    3: 100,   # Swap
    1: 80,    # Gen_L
    2: 80,    # Gen_R
    4: 60,    # Purify_L
    5: 60,    # Purify_R
    6: 20,    # Discard
    0: 0,     # Wait
}


class GreedyAgent(BaseAgent):
    """
    Deterministic greedy agent with configurable priority.

    For each node, pick the valid action with the highest priority.
    Ties are broken by action ID (lower ID preferred).
    """

    def __init__(self, num_nodes: int, priority: dict | None = None):
        super().__init__(num_nodes)
        self.priority = priority or _PRIORITY

    def act(self, obs: np.ndarray, mask: np.ndarray,
            deterministic: bool = True) -> np.ndarray:
        actions = np.zeros(self.num_nodes, dtype=int)
        for i in range(self.num_nodes):
            best_action = 0
            best_prio = -1
            for a in range(self.num_actions):
                if mask[i, a] and self.priority.get(a, 0) > best_prio:
                    best_prio = self.priority[a]
                    best_action = a
            actions[i] = best_action
        return actions

"""
qnbench.baselines.swap_asap
============================

Swap-ASAP protocol: a role-aware heuristic commonly used as a strong
baseline in quantum repeater chain literature.

Policy per node role:
    **Endpoints** (Node 0, Node N-1):
        → Always generate towards the interior of the chain.

    **Repeaters** (interior nodes):
        → If both left and right links are available → **Swap**.
        → Otherwise → **Generate** on whichever side has no link.
        → If both sides have links but only one each → **Wait** for
          the other to complete (swap needs both).
        → If ≥2 links on one side → **Purify** (optional, off by default).

Parameters
----------
enable_purify : bool
    If True, repeaters will purify when they have ≥2 same-side links
    *and* cannot swap.  Default False (pure Swap-ASAP).
"""

from __future__ import annotations

import numpy as np
from .base import BaseAgent


class SwapASAPAgent(BaseAgent):
    """
    Implements the Swap-ASAP entanglement distribution protocol.

    This is typically the strongest simple heuristic: it always generates
    and swaps as soon as possible, minimising link age.
    """

    def __init__(self, num_nodes: int, enable_purify: bool = False):
        super().__init__(num_nodes)
        self.enable_purify = enable_purify

    def act(self, obs: np.ndarray, mask: np.ndarray,
            deterministic: bool = True) -> np.ndarray:
        actions = np.zeros(self.num_nodes, dtype=int)

        for i in range(self.num_nodes):
            actions[i] = self._decide_node(i, obs[i], mask[i])

        return actions

    def _decide_node(self, nid: int, node_obs: np.ndarray,
                     node_mask: np.ndarray) -> int:
        """
        Decide action for a single node based on its observation.

        Observation features used:
            [0] is_endpoint
            [1] is_repeater
            [3] num_available_left_links
            [6] num_available_right_links
        """
        is_endpoint_left = (nid == 0)
        is_endpoint_right = (nid == self.num_nodes - 1)
        n_left = int(node_obs[3])    # available left links
        n_right = int(node_obs[6])   # available right links

        # ── Endpoint nodes ───────────────────────────────────────
        if is_endpoint_left:
            # Left endpoint → always generate right
            if node_mask[2]:    # Gen_R
                return 2
            return 0  # Wait

        if is_endpoint_right:
            # Right endpoint → always generate left
            if node_mask[1]:    # Gen_L
                return 1
            return 0  # Wait

        # ── Repeater nodes ───────────────────────────────────────

        # Priority 1: Swap if possible
        if n_left >= 1 and n_right >= 1 and node_mask[3]:
            return 3  # Swap

        # Priority 2 (optional): Purify if ≥2 same-side links
        if self.enable_purify:
            if n_left >= 2 and node_mask[4]:
                return 4   # Purify_L
            if n_right >= 2 and node_mask[5]:
                return 5   # Purify_R

        # Priority 3: Generate on the side that needs links
        if n_left == 0 and node_mask[1]:
            return 1  # Gen_L
        if n_right == 0 and node_mask[2]:
            return 2  # Gen_R

        # If one side has a link but the other doesn't, gen the missing side
        if n_left > 0 and n_right == 0 and node_mask[2]:
            return 2  # Gen_R
        if n_right > 0 and n_left == 0 and node_mask[1]:
            return 1  # Gen_L

        # Both sides have links but can't swap (maybe still generating)
        # → generate more if memory is free
        if node_mask[2]:
            return 2  # Gen_R
        if node_mask[1]:
            return 1  # Gen_L

        return 0  # Wait

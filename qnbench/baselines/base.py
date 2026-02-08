"""
qnbench.baselines.base
=======================

Abstract base class for all baseline agents.

Every agent must implement :meth:`act`, which receives an observation
and an action mask and returns a joint action vector.
"""

from __future__ import annotations

import abc
import numpy as np


class BaseAgent(abc.ABC):
    """
    Interface that all baseline (and RL) agents must satisfy.

    Parameters
    ----------
    num_nodes : number of nodes in the network
    num_actions : number of actions per node (default 7)
    """

    def __init__(self, num_nodes: int, num_actions: int = 7):
        self.num_nodes = num_nodes
        self.num_actions = num_actions

    @abc.abstractmethod
    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Choose an action for every node.

        Parameters
        ----------
        obs  : ``(num_nodes, obs_dim)`` observation array
        mask : ``(num_nodes, num_actions)`` boolean mask of valid actions
        deterministic : if False, may add exploration noise

        Returns
        -------
        actions : ``(num_nodes,)`` integer action per node
        """
        ...

    def reset(self):
        """Called at the start of each episode (optional)."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

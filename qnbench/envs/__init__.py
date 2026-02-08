"""
qnbench.envs
=============

Quantum network Gymnasium environment.

Quick start::

    from qnbench.envs import QuantumNetworkEnv, EnvConfig

    env = QuantumNetworkEnv(EnvConfig(node_distances=[30, 30, 30]))
    obs, info = env.reset(seed=42)
    obs, reward, term, trunc, info = env.step([0, 2, 2, 0])
"""

from .config import EnvConfig, ACTION_NAMES, NUM_ACTIONS, FIDELITY_MIXED_STATE
from .env import QuantumNetworkEnv
from .engine import QuantumNetworkEngine
from .physics import (
    swap_fidelity,
    purify_fidelity_bbpssw,
    decoherence_fidelity,
)
from .structs import Link, Memory, MemoryState, Node, EventType

__all__ = [
    "EnvConfig",
    "QuantumNetworkEnv",
    "QuantumNetworkEngine",
    "ACTION_NAMES",
    "NUM_ACTIONS",
]

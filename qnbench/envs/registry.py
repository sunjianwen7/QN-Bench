"""
qnbench.envs.registry
======================

Register QNBench environments with Gymnasium so they can be created via
``gymnasium.make("QuantumNetwork-v7")``.
"""

import gymnasium as gym


def register_envs():
    """Register all QNBench environments."""
    gym.register(
        id="QuantumNetwork-v7",
        entry_point="qnbench.envs.env:QuantumNetworkEnv",
        max_episode_steps=500,
    )

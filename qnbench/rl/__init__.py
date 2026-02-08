"""
qnbench.rl
============

Reinforcement learning components for the benchmark.

Requires PyTorch (install with ``pip install torch``).
"""

from .utils import RolloutBuffer, Transition, compute_gae

__all__ = ["RolloutBuffer", "Transition", "compute_gae"]

# Lazy imports for torch-dependent modules
def get_ppo_trainer():
    from .masked_ppo import PPOTrainer, PPOConfig
    return PPOTrainer, PPOConfig

def get_actor_critic():
    from .networks import ActorCritic, ActorCriticAgent
    return ActorCritic, ActorCriticAgent

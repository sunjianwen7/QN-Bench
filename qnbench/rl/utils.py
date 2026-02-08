"""
qnbench.rl.utils
==================

Reinforcement learning utilities:
- ``RolloutBuffer``: stores transitions for on-policy algorithms.
- ``compute_gae``: Generalised Advantage Estimation.
- ``masked_categorical_sample``: sample from a categorical distribution
  with an action mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


# =================================================================
# GAE (Generalised Advantage Estimation)
# =================================================================

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute advantages and returns using GAE(λ).

    Parameters
    ----------
    rewards : (T,) step rewards
    values  : (T+1,) value estimates (last element = V(s_T+1))
    dones   : (T,) episode-done flags (1 = done)
    gamma   : discount factor
    lam     : GAE lambda

    Returns
    -------
    advantages : (T,)
    returns    : (T,)  (= advantages + values[:T])
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values[:T]
    return advantages, returns


# =================================================================
# Masked Categorical Sampling
# =================================================================

def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to logits and compute softmax.

    Invalid actions (mask=False) get probability 0.
    """
    # Set invalid logits to -inf
    masked_logits = np.where(mask, logits, -1e8)
    # Numerically stable softmax
    shifted = masked_logits - masked_logits.max()
    exp_logits = np.exp(shifted)
    probs = exp_logits / (exp_logits.sum() + 1e-10)
    return probs


# =================================================================
# Rollout Buffer
# =================================================================

@dataclass
class Transition:
    """A single environment transition."""
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    mask: np.ndarray
    log_prob: float = 0.0
    value: float = 0.0


class RolloutBuffer:
    """
    Fixed-size buffer for on-policy rollout collection.

    Stores transitions and computes GAE when the buffer is full.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.transitions: List[Transition] = []

    def add(self, t: Transition):
        self.transitions.append(t)

    def is_full(self) -> bool:
        return len(self.transitions) >= self.capacity

    def size(self) -> int:
        return len(self.transitions)

    def clear(self):
        self.transitions.clear()

    def compute_returns(
        self,
        last_value: float,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:
        """
        Finalise the buffer: compute GAE and pack arrays.

        Returns
        -------
        obs_batch      : (N, *obs_shape)
        action_batch   : (N, num_nodes)
        return_batch   : (N,)
        advantage_batch: (N,)
        log_prob_batch : (N,)
        mask_batch     : (N, num_nodes, num_actions)
        value_batch    : (N,)
        """
        N = len(self.transitions)
        rewards = np.array([t.reward for t in self.transitions], dtype=np.float32)
        dones = np.array([float(t.done) for t in self.transitions], dtype=np.float32)
        values = np.array(
            [t.value for t in self.transitions] + [last_value],
            dtype=np.float32,
        )

        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)

        # Normalise advantages
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        obs_batch = np.stack([t.obs for t in self.transitions])
        action_batch = np.stack([t.action for t in self.transitions])
        log_prob_batch = np.array([t.log_prob for t in self.transitions], dtype=np.float32)
        mask_batch = np.stack([t.mask for t in self.transitions])
        value_batch = values[:N]

        return (obs_batch, action_batch, returns, advantages,
                log_prob_batch, mask_batch, value_batch)

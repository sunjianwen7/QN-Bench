"""
qnbench.rl.networks
=====================

Actor-Critic network for the quantum network environment.

Architecture
------------
The observation is ``(num_nodes, obs_dim)``.  We use a **shared encoder**
that processes each node's features, then:
- A **per-node actor head** outputs masked action logits.
- A **global critic head** pools node embeddings → single value.

This design respects the multi-agent structure while sharing parameters.

::

    obs (N, obs_dim)  ──►  SharedEncoder  ──►  node_emb (N, hidden)
                                │                       │
                           CriticHead              ActorHead (per node)
                           pool → V(s)             logits → π(a|s)
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for RL training.  "
            "Install with: pip install torch"
        )


class ActorCritic(nn.Module):
    """
    Shared-encoder Actor-Critic for multi-node action selection.

    Parameters
    ----------
    obs_dim     : per-node observation dimension (default 18)
    num_actions : actions per node (default 7)
    hidden_dim  : hidden layer width
    n_layers    : number of hidden layers in the encoder
    num_nodes   : number of nodes in the network
    """

    def __init__(
        self,
        obs_dim: int = 18,
        num_actions: int = 7,
        hidden_dim: int = 128,
        n_layers: int = 2,
        num_nodes: int = 4,
    ):
        _check_torch()
        super().__init__()

        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_nodes = num_nodes

        # ── Shared node encoder ──────────────────────────────────
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        # ── Actor head: per-node action logits ───────────────────
        self.actor_head = nn.Linear(hidden_dim, num_actions)

        # ── Critic head: global value from pooled embeddings ─────
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass.

        Parameters
        ----------
        obs  : (batch, num_nodes, obs_dim)
        mask : (batch, num_nodes, num_actions) boolean

        Returns
        -------
        logits : (batch, num_nodes, num_actions) — masked
        value  : (batch,)
        """
        B, N, D = obs.shape

        # Encode each node independently (parameter sharing)
        flat = obs.reshape(B * N, D)                    # (B*N, obs_dim)
        emb = self.encoder(flat)                        # (B*N, hidden)
        emb = emb.reshape(B, N, -1)                     # (B, N, hidden)

        # Actor: per-node logits with mask
        logits = self.actor_head(emb)                   # (B, N, num_actions)
        # Mask: set invalid actions to very large negative
        logits = logits.masked_fill(~mask, -1e8)

        # Critic: mean-pool node embeddings → scalar value
        pooled = emb.mean(dim=1)                        # (B, hidden)
        value = self.critic_head(pooled).squeeze(-1)    # (B,)

        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        """
        Sample (or evaluate) actions.

        Parameters
        ----------
        obs  : (batch, num_nodes, obs_dim)
        mask : (batch, num_nodes, num_actions)
        action : if provided, compute log_prob of these actions
        deterministic : if True, pick argmax instead of sampling

        Returns
        -------
        actions    : (batch, num_nodes)
        log_probs  : (batch,) sum of per-node log probs
        entropy    : (batch,) sum of per-node entropies
        value      : (batch,)
        """
        logits, value = self.forward(obs, mask)
        B, N, A = logits.shape

        # Per-node categorical distributions
        all_actions = []
        all_log_probs = []
        all_entropy = []

        for n in range(N):
            dist = Categorical(logits=logits[:, n, :])
            if action is None:
                if deterministic:
                    a = logits[:, n, :].argmax(dim=-1)
                else:
                    a = dist.sample()
            else:
                a = action[:, n]
            all_actions.append(a)
            all_log_probs.append(dist.log_prob(a))
            all_entropy.append(dist.entropy())

        actions = torch.stack(all_actions, dim=1)              # (B, N)
        log_probs = torch.stack(all_log_probs, dim=1).sum(1)  # (B,)
        entropy = torch.stack(all_entropy, dim=1).sum(1)       # (B,)

        return actions, log_probs, entropy, value


class ActorCriticAgent:
    """
    Wraps ``ActorCritic`` to conform to the ``BaseAgent`` interface
    for evaluation.
    """

    def __init__(self, model: "ActorCritic", device: str = "cpu"):
        _check_torch()
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.num_nodes = model.num_nodes

    @property
    def name(self) -> str:
        return "PPO"

    def reset(self):
        pass

    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Select actions using the trained model."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
            actions, _, _, _ = self.model.get_action_and_value(
                obs_t, mask_t, deterministic=deterministic
            )
        return actions.squeeze(0).cpu().numpy()

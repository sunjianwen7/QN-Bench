"""
qnbench.rl.masked_ppo
=======================

Proximal Policy Optimisation (PPO) with invalid-action masking.

This is a clean, self-contained PPO implementation tailored for the
quantum network environment.  Key features:

- **Action masking**: invalid actions get logit = −∞, so the policy
  never selects them.  No wasted exploration on impossible actions.
- **Multi-node actions**: each node has an independent categorical
  distribution (parameter-shared through the encoder).
- **GAE(λ)**: Generalised Advantage Estimation for variance reduction.

Usage::

    from qnbench.rl.masked_ppo import PPOTrainer

    trainer = PPOTrainer(env, cfg)
    trainer.train(total_timesteps=200000)
    trainer.save("checkpoints/ppo_best.pt")
"""

from __future__ import annotations

import os
import sys
import time
import logging
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from qnbench.envs import QuantumNetworkEnv, EnvConfig
from qnbench.rl.networks import ActorCritic
from qnbench.rl.utils import RolloutBuffer, Transition
from qnbench.utils.logging import ensure_logging

logger = logging.getLogger("qnbench.ppo")


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _log(msg: str):
    """print + flush, 保证管道 / 文件重定向时日志立即可见。"""
    print(msg, flush=True)


def _format_time(seconds: float) -> str:
    """把秒数格式化为 HH:MM:SS 或 MM:SS。"""
    if seconds < 3600:
        return f"{int(seconds)//60:02d}:{int(seconds)%60:02d}"
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:d}:{m:02d}:{s:02d}"


# ─────────────────────────────────────────────────────────────────
# PPO Config
# ─────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 128          # rollout length before each update
    n_epochs: int = 4           # PPO epochs per update
    batch_size: int = 64
    hidden_dim: int = 128
    n_layers: int = 2
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> "PPOConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────────
# PPO Trainer
# ─────────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    PPO training loop for the quantum network environment.

    Parameters
    ----------
    env : QuantumNetworkEnv instance
    ppo_cfg : PPO hyperparameters
    device : "cpu" or "cuda"
    """

    def __init__(
        self,
        env: QuantumNetworkEnv,
        ppo_cfg: PPOConfig | None = None,
        device: str = "cpu",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required.  pip install torch")

        # 保证日志系统已初始化（即使用户没有调用 setup_logging）
        ensure_logging()

        self.env = env
        self.cfg = ppo_cfg or PPOConfig()
        self.device = torch.device(device)

        # ── Model ────────────────────────────────────────────────
        self.model = ActorCritic(
            obs_dim=env.obs_dim,
            num_actions=7,
            hidden_dim=self.cfg.hidden_dim,
            n_layers=self.cfg.n_layers,
            num_nodes=env.num_nodes,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.buffer = RolloutBuffer(self.cfg.n_steps)

        # ── Tracking ─────────────────────────────────────────────
        self.total_steps = 0
        self.episode_count = 0
        self.best_mean_reward = -float("inf")
        self.reward_history: list[float] = []
        self.delivery_history: list[int] = []

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("ActorCritic: %d parameters, device=%s", n_params, self.device)

    # =============================================================
    # Main Training Loop
    # =============================================================

    def train(
        self,
        total_timesteps: int = 200_000,
        log_interval: int = 5,
        save_dir: str = "checkpoints",
    ):
        """
        Train the agent for *total_timesteps* environment steps.

        Parameters
        ----------
        total_timesteps : total env steps to train
        log_interval    : print progress every N updates
        save_dir        : directory for model checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)

        # ── 打印训练配置 ─────────────────────────────────────────
        total_updates = total_timesteps // self.cfg.n_steps
        _log(f"\n{'='*65}")
        _log(f"  PPO Training — QNBench")
        _log(f"{'='*65}")
        _log(f"  Total timesteps  : {total_timesteps:,}")
        _log(f"  Total updates    : {total_updates:,}")
        _log(f"  Rollout length   : {self.cfg.n_steps}")
        _log(f"  Batch size       : {self.cfg.batch_size}")
        _log(f"  PPO epochs       : {self.cfg.n_epochs}")
        _log(f"  Learning rate    : {self.cfg.lr:.1e}")
        _log(f"  Hidden dim       : {self.cfg.hidden_dim}")
        _log(f"  Entropy coef     : {self.cfg.entropy_coef}")
        _log(f"  Device           : {self.device}")
        _log(f"{'='*65}\n")

        obs, _ = self.env.reset(seed=self.cfg.seed)
        ep_reward = 0.0
        ep_deliveries = 0
        ep_step_in_current = 0
        update_count = 0
        t_start = time.time()

        # 滑动窗口, 记录最近 50 个 episode 的指标
        window_size = 50
        ep_rewards_window: list[float] = []
        ep_deliveries_window: list[int] = []
        ep_lengths_window: list[int] = []

        while self.total_steps < total_timesteps:
            # ── 1. Collect rollout ───────────────────────────────
            self.buffer.clear()
            self.model.eval()
            rollout_reward = 0.0      # 本次 rollout 的累计 reward
            rollout_episodes_done = 0 # 本次 rollout 完成的 episode 数

            for _ in range(self.cfg.n_steps):
                mask = self.env.get_action_mask()

                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
                    actions, log_prob, _, value = self.model.get_action_and_value(
                        obs_t, mask_t
                    )

                action_np = actions.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated

                self.buffer.add(Transition(
                    obs=obs, action=action_np, reward=reward,
                    next_obs=next_obs, done=done, mask=mask,
                    log_prob=log_prob.item(), value=value.item(),
                ))

                obs = next_obs
                ep_reward += reward
                ep_deliveries = info.get("delivered", 0)
                ep_step_in_current += 1
                rollout_reward += reward
                self.total_steps += 1

                if done:
                    # ── Episode 结束 → 记录指标 ──────────────────
                    self.reward_history.append(ep_reward)
                    self.delivery_history.append(ep_deliveries)
                    ep_rewards_window.append(ep_reward)
                    ep_deliveries_window.append(ep_deliveries)
                    ep_lengths_window.append(ep_step_in_current)
                    # 保持窗口大小
                    if len(ep_rewards_window) > window_size:
                        ep_rewards_window.pop(0)
                        ep_deliveries_window.pop(0)
                        ep_lengths_window.pop(0)

                    self.episode_count += 1
                    rollout_episodes_done += 1
                    ep_reward = 0.0
                    ep_deliveries = 0
                    ep_step_in_current = 0
                    obs, _ = self.env.reset(
                        seed=self.cfg.seed + self.episode_count
                    )

            # ── 2. Compute returns ───────────────────────────────
            with torch.no_grad():
                last_obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                last_mask_t = torch.BoolTensor(
                    self.env.get_action_mask()
                ).unsqueeze(0).to(self.device)
                _, last_value = self.model(last_obs_t, last_mask_t)
                last_val = last_value.item()

            (obs_b, act_b, ret_b, adv_b,
             lp_b, mask_b, val_b) = self.buffer.compute_returns(
                last_val, self.cfg.gamma, self.cfg.gae_lambda,
            )

            # ── 3. PPO Update ────────────────────────────────────
            loss_info = self._update(obs_b, act_b, ret_b, adv_b, lp_b, mask_b)
            update_count += 1

            # ── 4. Progress output ───────────────────────────────
            if update_count % log_interval == 0:
                self._print_progress(
                    update_count=update_count,
                    total_updates=total_updates,
                    total_timesteps=total_timesteps,
                    t_start=t_start,
                    loss_info=loss_info,
                    rollout_reward=rollout_reward,
                    rollout_episodes_done=rollout_episodes_done,
                    ep_rewards_window=ep_rewards_window,
                    ep_deliveries_window=ep_deliveries_window,
                    ep_lengths_window=ep_lengths_window,
                    save_dir=save_dir,
                )

        # ── 训练完成 ─────────────────────────────────────────────
        self.save(os.path.join(save_dir, "ppo_final.pt"))
        elapsed = time.time() - t_start
        _log(f"\n{'='*65}")
        _log(f"  ✓ Training Complete")
        _log(f"{'='*65}")
        _log(f"  Total steps      : {self.total_steps:,}")
        _log(f"  Total episodes   : {self.episode_count}")
        _log(f"  Total deliveries : {sum(self.delivery_history)}")
        _log(f"  Best mean reward : {self.best_mean_reward:+.2f}")
        _log(f"  Wall time        : {_format_time(elapsed)}")
        _log(f"  Saved            : {save_dir}/ppo_final.pt")
        _log(f"{'='*65}")

    # =============================================================
    # Progress Printer
    # =============================================================

    def _print_progress(
        self,
        update_count: int,
        total_updates: int,
        total_timesteps: int,
        t_start: float,
        loss_info: dict,
        rollout_reward: float,
        rollout_episodes_done: int,
        ep_rewards_window: list,
        ep_deliveries_window: list,
        ep_lengths_window: list,
        save_dir: str,
    ):
        """
        输出当前训练进度。

        即使还没有完整 episode 完成也能输出 (使用 rollout 级别的统计)。
        """
        elapsed = time.time() - t_start
        fps = self.total_steps / max(elapsed, 1e-6)
        pct = self.total_steps / total_timesteps * 100
        eta = (total_timesteps - self.total_steps) / max(fps, 1)

        # ── 进度条 ───────────────────────────────────────────────
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        _log(
            f"  [{bar}] {pct:5.1f}%  "
            f"Step {self.total_steps:>8,}/{total_timesteps:,}  "
            f"Update {update_count}/{total_updates}"
        )

        # ── Episode 级别统计 (如果有已完成的 episode) ────────────
        if ep_rewards_window:
            n = len(ep_rewards_window)
            mean_r = np.mean(ep_rewards_window)
            std_r = np.std(ep_rewards_window)
            mean_del = np.mean(ep_deliveries_window)
            mean_len = np.mean(ep_lengths_window)

            _log(
                f"    Reward  : {mean_r:>+9.2f} ± {std_r:<7.2f}  "
                f"(over last {n} episodes)"
            )
            _log(
                f"    Deliver : {mean_del:>6.2f}/ep           "
                f"EpLen: {mean_len:>6.0f}"
            )
        else:
            # 还没有完成过一个 episode → 显示 rollout 级别的信息
            _log(
                f"    Rollout : reward={rollout_reward:>+9.2f}  "
                f"(no episode completed yet, ep_step={self.total_steps})"
            )

        # ── Loss + 速度 ──────────────────────────────────────────
        parts = [f"    Loss    : {loss_info['total_loss']:>8.4f}"]
        if "policy_loss" in loss_info:
            parts.append(f"  (π={loss_info['policy_loss']:.4f}")
            parts.append(f" V={loss_info['value_loss']:.4f}")
            parts.append(f" H={loss_info['entropy']:.4f})")
        _log("".join(parts))
        _log(
            f"    Speed   : {fps:>6.0f} step/s  "
            f"Elapsed: {_format_time(elapsed)}  "
            f"ETA: {_format_time(eta)}"
        )

        # ── 最优模型保存 ─────────────────────────────────────────
        if ep_rewards_window and len(ep_rewards_window) >= 10:
            mean_r = np.mean(ep_rewards_window)
            if mean_r > self.best_mean_reward:
                self.best_mean_reward = mean_r
                path = os.path.join(save_dir, "ppo_best.pt")
                self.save(path)
                _log(f"    ★ New best model saved → {path}  (reward={mean_r:+.2f})")

        _log("")   # 空行分隔

    # =============================================================
    # PPO Update
    # =============================================================

    def _update(
        self,
        obs_b: np.ndarray,
        act_b: np.ndarray,
        ret_b: np.ndarray,
        adv_b: np.ndarray,
        old_lp_b: np.ndarray,
        mask_b: np.ndarray,
    ) -> dict:
        """
        Run PPO clipped objective update for ``n_epochs``.

        Returns
        -------
        dict with keys: total_loss, policy_loss, value_loss, entropy
        """
        self.model.train()

        # Convert to tensors
        obs_t = torch.FloatTensor(obs_b).to(self.device)
        act_t = torch.LongTensor(act_b).to(self.device)
        ret_t = torch.FloatTensor(ret_b).to(self.device)
        adv_t = torch.FloatTensor(adv_b).to(self.device)
        old_lp_t = torch.FloatTensor(old_lp_b).to(self.device)
        mask_t = torch.BoolTensor(mask_b).to(self.device)

        N = len(obs_t)
        total_loss_acc = 0.0
        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        entropy_acc = 0.0
        n_updates = 0

        for _ in range(self.cfg.n_epochs):
            # Shuffle indices
            indices = np.arange(N)
            np.random.shuffle(indices)

            for start in range(0, N, self.cfg.batch_size):
                end = min(start + self.cfg.batch_size, N)
                idx = indices[start:end]

                b_obs = obs_t[idx]
                b_act = act_t[idx]
                b_ret = ret_t[idx]
                b_adv = adv_t[idx]
                b_old_lp = old_lp_t[idx]
                b_mask = mask_t[idx]

                # Forward pass
                _, new_lp, entropy, new_val = self.model.get_action_and_value(
                    b_obs, b_mask, action=b_act
                )

                # ── Policy loss (clipped surrogate) ──────────────
                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(
                    ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps
                ) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Value loss ───────────────────────────────────
                value_loss = nn.functional.mse_loss(new_val, b_ret)

                # ── Entropy bonus ────────────────────────────────
                entropy_mean = entropy.mean()
                entropy_loss = -entropy_mean

                # ── Total loss ───────────────────────────────────
                loss = (policy_loss
                        + self.cfg.value_coef * value_loss
                        + self.cfg.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                total_loss_acc += loss.item()
                policy_loss_acc += policy_loss.item()
                value_loss_acc += value_loss.item()
                entropy_acc += entropy_mean.item()
                n_updates += 1

        denom = max(n_updates, 1)
        return {
            "total_loss": total_loss_acc / denom,
            "policy_loss": policy_loss_acc / denom,
            "value_loss": value_loss_acc / denom,
            "entropy": entropy_acc / denom,
        }

    # =============================================================
    # Save / Load
    # =============================================================

    def save(self, path: str):
        """Save model weights and training state."""
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "best_mean_reward": self.best_mean_reward,
            "reward_history": self.reward_history,
            "delivery_history": self.delivery_history,
        }, path)
        logger.debug("Saved checkpoint → %s", path)

    def load(self, path: str):
        """Load model weights and training state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.episode_count = ckpt.get("episode_count", 0)
        self.best_mean_reward = ckpt.get("best_mean_reward", -float("inf"))
        self.reward_history = ckpt.get("reward_history", [])
        self.delivery_history = ckpt.get("delivery_history", [])
        logger.info("Loaded checkpoint from %s (step %d)", path, self.total_steps)

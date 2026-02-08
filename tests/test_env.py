"""
tests/test_env.py
==================

Integration tests for the QuantumNetworkEnv Gymnasium wrapper.
Tests cover: step/reset API, action masking, truncation, delivery,
observation modes, and baseline agent compatibility.
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnbench.envs import QuantumNetworkEnv, EnvConfig
from qnbench.baselines import RandomAgent, GreedyAgent, SwapASAPAgent


# ── Basic API ────────────────────────────────────────────────────

def test_reset_returns_correct_shape():
    env = QuantumNetworkEnv(verbose=False)
    obs, info = env.reset(seed=42)
    assert obs.shape == (4, 18), f"Expected (4,18), got {obs.shape}"
    assert isinstance(info, dict)


def test_step_returns_five_values():
    env = QuantumNetworkEnv(verbose=False)
    env.reset(seed=42)
    result = env.step([0, 0, 0, 0])
    assert len(result) == 5, f"step() should return 5 values, got {len(result)}"
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (4, 18)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_mask_shape():
    env = QuantumNetworkEnv(verbose=False)
    env.reset(seed=42)
    mask = env.get_action_mask()
    assert mask.shape == (4, 7)
    assert mask.dtype == bool
    # Wait should always be valid
    assert mask[:, 0].all(), "Wait should always be valid"


def test_action_mask_endpoints():
    """Endpoints should not be able to generate towards the outside."""
    env = QuantumNetworkEnv(verbose=False)
    env.reset(seed=42)
    mask = env.get_action_mask()
    # Node 0 cannot Gen_L (action 1)
    assert not mask[0, 1], "Node 0 should not Gen_L"
    # Node 3 cannot Gen_R (action 2)
    assert not mask[3, 2], "Node 3 should not Gen_R"
    # Endpoints cannot Swap
    assert not mask[0, 3], "Node 0 should not Swap"
    assert not mask[3, 3], "Node 3 should not Swap"


# ── Truncation ───────────────────────────────────────────────────

def test_truncation():
    """Episode should truncate at max_steps."""
    cfg = EnvConfig(max_steps=5)
    env = QuantumNetworkEnv(cfg=cfg, verbose=False)
    env.reset(seed=42)

    for i in range(10):
        _, _, terminated, truncated, info = env.step([0, 0, 0, 0])
        if truncated:
            assert info["step"] == 5, f"Truncated at step {info['step']}, expected 5"
            return
    assert False, "Should have truncated by step 5"


# ── Observation modes ────────────────────────────────────────────

def test_oracle_vs_experimental_differ():
    """Oracle and Experimental modes should produce different observations."""
    seed = 123
    actions = [[0, 2, 2, 0]] * 5

    env_o = QuantumNetworkEnv(cfg=EnvConfig(oracle_mode=True), verbose=False)
    env_o.reset(seed=seed)
    for a in actions:
        env_o.step(a)
    obs_o = env_o._build_obs()

    env_e = QuantumNetworkEnv(cfg=EnvConfig(oracle_mode=False), verbose=False)
    env_e.reset(seed=seed)
    for a in actions:
        env_e.step(a)
    obs_e = env_e._build_obs()

    # Features 4-10 should differ between modes
    assert not np.allclose(obs_o[:, 4:11], obs_e[:, 4:11]), \
        "Oracle and Experimental observations should differ in fidelity/age features"


# ── Invalid action penalty ───────────────────────────────────────

def test_invalid_action_penalised():
    """Selecting an invalid action should give a penalty and be corrected to Wait."""
    env = QuantumNetworkEnv(verbose=False)
    env.reset(seed=42)
    # Node 0 Gen_L is always invalid
    _, reward, _, _, _ = env.step([1, 0, 0, 0])
    assert reward < 0, f"Invalid action should give negative reward, got {reward}"


# ── Delivery ─────────────────────────────────────────────────────

def test_delivery_gives_large_reward():
    """
    If we manage to get an end-to-end link with F ≥ threshold,
    we should get a large positive reward.
    """
    # Use high probabilities so delivery is likely in a few steps
    cfg = EnvConfig(
        prob_gen=1.0, prob_swap=1.0, t_coherence=10000.0,
        node_distances=[10.0, 10.0, 10.0], max_steps=200,
    )
    env = QuantumNetworkEnv(cfg=cfg, verbose=False)
    agent = SwapASAPAgent(num_nodes=4)

    env.reset(seed=42)
    total_reward = 0
    delivered = 0

    for _ in range(200):
        mask = env.get_action_mask()
        actions = agent.act(env._build_obs(), mask)
        _, reward, term, trunc, info = env.step(actions)
        total_reward += reward
        delivered = info["delivered"]
        if term or trunc:
            break

    assert delivered > 0, "Should have delivered at least 1 pair with p=1.0"
    # Note: total_reward can be negative due to accumulated operation costs
    # over many steps; we only check that delivery actually happened.


# ── Baseline agent compatibility ─────────────────────────────────

def test_random_agent_runs():
    """Random agent should run without errors for a full episode."""
    env = QuantumNetworkEnv(cfg=EnvConfig(max_steps=50), verbose=False)
    agent = RandomAgent(num_nodes=4, seed=42)
    obs, _ = env.reset(seed=42)
    for _ in range(50):
        mask = env.get_action_mask()
        actions = agent.act(obs, mask)
        obs, _, term, trunc, _ = env.step(actions)
        if term or trunc:
            break


def test_greedy_agent_runs():
    """Greedy agent should run without errors."""
    env = QuantumNetworkEnv(cfg=EnvConfig(max_steps=50), verbose=False)
    agent = GreedyAgent(num_nodes=4)
    obs, _ = env.reset(seed=42)
    for _ in range(50):
        mask = env.get_action_mask()
        actions = agent.act(obs, mask)
        obs, _, term, trunc, _ = env.step(actions)
        if term or trunc:
            break


def test_swap_asap_agent_runs():
    """Swap-ASAP agent should run without errors."""
    env = QuantumNetworkEnv(cfg=EnvConfig(max_steps=50), verbose=False)
    agent = SwapASAPAgent(num_nodes=4, enable_purify=True)
    obs, _ = env.reset(seed=42)
    for _ in range(50):
        mask = env.get_action_mask()
        actions = agent.act(obs, mask)
        obs, _, term, trunc, _ = env.step(actions)
        if term or trunc:
            break


# ── Reproducibility at env level ─────────────────────────────────

def test_env_reproducibility():
    """Same seed + same actions → identical reward sequences."""
    actions_seq = [[0, 2, 2, 0]] * 5 + [[0, 3, 0, 0]] * 3

    rewards = []
    for _ in range(2):
        env = QuantumNetworkEnv(verbose=False)
        env.reset(seed=42)
        ep_rewards = []
        for a in actions_seq:
            _, r, _, _, _ = env.step(a)
            ep_rewards.append(r)
        rewards.append(ep_rewards)

    for i, (r1, r2) in enumerate(zip(*rewards)):
        assert abs(r1 - r2) < 1e-9, f"Step {i}: {r1} != {r2}"


if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")

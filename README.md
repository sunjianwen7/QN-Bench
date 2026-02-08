# Quantum Network Routing Benchmark (QNBench)

A Gymnasium-based benchmark for evaluating quantum network entanglement
distribution protocols over linear repeater chains.

## Project Structure

```
quantum_network_benchmark/
│
├── qnbench/                     # Core library
│   ├── envs/                    # Gymnasium environment
│   │   ├── config.py            # EnvConfig dataclass & reward constants
│   │   ├── physics.py           # Werner-state fidelity formulas
│   │   ├── structs.py           # Link, Memory, Node, Event data structures
│   │   ├── engine.py            # Discrete-event quantum network engine
│   │   ├── env.py               # Gym wrapper (QuantumNetworkEnv)
│   │   └── registry.py          # Gym environment registration
│   │
│   ├── baselines/               # Heuristic baselines
│   │   ├── base.py              # Abstract BaseAgent interface
│   │   ├── random_agent.py      # Uniformly random valid actions
│   │   ├── greedy_agent.py      # Greedy fidelity-maximising heuristic
│   │   └── swap_asap.py         # Swap-ASAP + Generate-Always protocol
│   │
│   ├── rl/                      # Reinforcement learning
│   │   ├── networks.py          # Actor-Critic network architectures
│   │   ├── masked_ppo.py        # PPO with invalid-action masking
│   │   └── utils.py             # Rollout buffer, advantage estimation
│   │
│   ├── evaluation/              # Evaluation & metrics
│   │   ├── runner.py            # Run agents on env, collect trajectories
│   │   └── metrics.py           # Delivery rate, fidelity, throughput
│   │
│   └── utils/
│       └── logging.py           # Centralised logging setup
│
├── scripts/                     # Entry-point scripts
│   ├── run_baselines.py         # Evaluate all baselines, print table
│   ├── train_ppo.py             # Train PPO agent
│   └── evaluate.py              # Load & evaluate a trained model
│
├── tests/                       # Unit tests (pytest)
│   ├── test_physics.py
│   ├── test_engine.py
│   └── test_env.py
│
├── configs/
│   └── default.yaml             # Default hyperparameters
│
├── pyproject.toml               # Build & dependency metadata
└── README.md                    # This file
```

## Quick Start

```bash
# Install
pip install -e ".[rl]"

# Run baselines comparison
python scripts/run_baselines.py

# Train PPO agent
python scripts/train_ppo.py --steps 200000

# Evaluate trained agent
python scripts/evaluate.py --checkpoint checkpoints/ppo_best.pt
```

## Environment

**Observation**: `(num_nodes, 18)` float array per node — fidelity/age of
left/right links, memory utilisation, distance features, swap counts.

**Action**: `MultiDiscrete([7] * num_nodes)` — each node independently
chooses Wait / Gen_L / Gen_R / Swap / Purify_L / Purify_R / Discard.

**Reward**: operation costs + time penalty + bonuses for generation,
swap, purification, and end-to-end delivery above fidelity threshold.

## Key Features

- **Werner-state physics**: accurate fidelity tracking through swap,
  purification, and decoherence.
- **Discrete-event engine**: geometric-distribution generation model,
  classical communication delays.
- **Action masking**: invalid actions are masked; agents receive the mask
  and can use it for safe exploration.
- **Oracle / Experimental modes**: expose true fidelity or only link age.
- **Configurable**: single `EnvConfig` dataclass controls all parameters.

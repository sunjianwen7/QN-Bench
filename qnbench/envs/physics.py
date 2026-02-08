"""
qnbench.envs.physics
=====================

Quantum-information formulas used by the simulation engine.

All functions operate on *fidelity* values in [0.25, 1.0] and use the
Werner-state depolarisation model internally.

Reference
---------
- Werner (1989): Quantum states with Einstein-Podolsky-Rosen correlations
  admitting a hidden-variable model.  Phys. Rev. A 40, 4277.
- Bennett et al. (1996): Purification of noisy entanglement and faithful
  teleportation via noisy channels (BBPSSW).  Phys. Rev. Lett. 76, 722.
"""

from __future__ import annotations

import math

from .config import FIDELITY_MIXED_STATE


# ── Werner parameter ↔ fidelity conversion ───────────────────────

def werner_parameter_from_fidelity(fidelity: float) -> float:
    """
    Convert Bell-state fidelity F to the Werner depolarisation parameter p.

    A Werner state is  ρ = p|Φ⁺⟩⟨Φ⁺| + (1−p)/4 · I₄ ,
    giving  F = ⟨Φ⁺|ρ|Φ⁺⟩ = 0.75p + 0.25.
    """
    fidelity = max(FIDELITY_MIXED_STATE, min(1.0, fidelity))
    return (fidelity - FIDELITY_MIXED_STATE) / 0.75


def fidelity_from_werner_parameter(p: float) -> float:
    """Inverse of :func:`werner_parameter_from_fidelity`."""
    p = max(0.0, min(1.0, p))
    return 0.75 * p + FIDELITY_MIXED_STATE


# ── Entanglement swap ────────────────────────────────────────────

def swap_fidelity(f1: float, f2: float) -> float:
    """
    Fidelity after entanglement swap of two Werner-state links.

    Under the Werner model, swap multiplies the depolarisation parameters:
    p_new = p1 × p2,  giving  F_new = 0.25 + (F1 − 0.25)(F2 − 0.25) × 4/3.
    """
    p1 = werner_parameter_from_fidelity(f1)
    p2 = werner_parameter_from_fidelity(f2)
    return fidelity_from_werner_parameter(p1 * p2)


# ── BBPSSW purification ─────────────────────────────────────────

def purify_fidelity_bbpssw(f_target: float, f_source: float) -> float:
    """
    Output fidelity of the BBPSSW entanglement purification protocol.

    Consumes *f_source* to improve *f_target*.  The protocol succeeds
    with some probability (handled by the engine); this function returns
    the fidelity *conditioned on success*.

    Formula:  F' = F_t · F_s / (F_t · F_s + (1−F_t)(1−F_s))
    """
    num = f_target * f_source
    den = f_target * f_source + (1.0 - f_target) * (1.0 - f_source)
    if den < 1e-12:
        return f_target
    return max(FIDELITY_MIXED_STATE, min(1.0, num / den))


# ── Decoherence ──────────────────────────────────────────────────

def decoherence_fidelity(f_initial: float, age: float, t_coh: float) -> float:
    """
    Fidelity after exponential decoherence.

    The Werner parameter decays as  p(t) = p(0) × exp(−t / T₂),
    so  F(t) = 0.25 + (F₀ − 0.25) × exp(−t / T₂).
    """
    if age < 0:
        age = 0
    decay = math.exp(-age / t_coh) if t_coh > 0 else 0.0
    return FIDELITY_MIXED_STATE + (f_initial - FIDELITY_MIXED_STATE) * decay

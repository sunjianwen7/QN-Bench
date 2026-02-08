"""
tests/test_physics.py
=====================

Unit tests for Werner-state fidelity formulas.
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnbench.envs.physics import (
    swap_fidelity,
    purify_fidelity_bbpssw,
    decoherence_fidelity,
    werner_parameter_from_fidelity,
    fidelity_from_werner_parameter,
)
from qnbench.envs.config import FIDELITY_MIXED_STATE


# ── Swap fidelity ────────────────────────────────────────────────

def test_swap_perfect():
    """Swapping two perfect Bell pairs gives a perfect pair."""
    assert abs(swap_fidelity(1.0, 1.0) - 1.0) < 1e-9

def test_swap_mixed():
    """Swapping two maximally mixed states stays mixed."""
    assert abs(swap_fidelity(0.25, 0.25) - 0.25) < 1e-9

def test_swap_symmetric():
    """swap(F1, F2) == swap(F2, F1)"""
    assert abs(swap_fidelity(0.9, 0.7) - swap_fidelity(0.7, 0.9)) < 1e-9

def test_swap_degrades():
    """Swap always degrades fidelity (unless both are 1.0)."""
    f1, f2 = 0.9, 0.8
    result = swap_fidelity(f1, f2)
    assert result < min(f1, f2)
    assert result >= FIDELITY_MIXED_STATE

def test_swap_known_values():
    """Check against hand-computed values."""
    cases = [
        (0.9, 0.9, 0.813),
        (0.5, 0.5, 0.333),
    ]
    for f1, f2, expected in cases:
        result = swap_fidelity(f1, f2)
        assert abs(result - expected) < 0.01, f"swap({f1},{f2})={result}, expected {expected}"


# ── Purification ─────────────────────────────────────────────────

def test_purify_improves():
    """Purification should increase fidelity when both inputs are > 0.5."""
    f_t, f_s = 0.8, 0.7
    result = purify_fidelity_bbpssw(f_t, f_s)
    assert result > f_t

def test_purify_bounds():
    """Result must be in [0.25, 1.0]."""
    for ft in [0.3, 0.5, 0.7, 0.95]:
        for fs in [0.3, 0.5, 0.7, 0.95]:
            r = purify_fidelity_bbpssw(ft, fs)
            assert 0.25 - 1e-9 <= r <= 1.0 + 1e-9


# ── Decoherence ──────────────────────────────────────────────────

def test_decoherence_no_age():
    """Zero age → no degradation."""
    assert abs(decoherence_fidelity(0.9, 0.0, 1000.0) - 0.9) < 1e-9

def test_decoherence_infinite_age():
    """Very long age → approaches mixed state."""
    result = decoherence_fidelity(0.9, 1e6, 100.0)
    assert abs(result - 0.25) < 0.01

def test_decoherence_half_life():
    """At t = t_coh, fidelity decays by exp(-1)."""
    f0, t_coh = 0.9, 100.0
    result = decoherence_fidelity(f0, t_coh, t_coh)
    expected = 0.25 + (f0 - 0.25) * math.exp(-1)
    assert abs(result - expected) < 1e-9


# ── Werner parameter roundtrip ───────────────────────────────────

def test_werner_roundtrip():
    """Converting F→p→F should be identity."""
    for f in [0.25, 0.5, 0.75, 1.0]:
        p = werner_parameter_from_fidelity(f)
        f2 = fidelity_from_werner_parameter(p)
        assert abs(f - f2) < 1e-9, f"Roundtrip failed for F={f}"


if __name__ == "__main__":
    # Simple test runner (works without pytest)
    import inspect
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

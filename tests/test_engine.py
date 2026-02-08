"""
tests/test_engine.py
====================

Unit tests for the QuantumNetworkEngine.
Tests cover: generation timing, swap TTL inheritance, stale event
cancellation, zombie link defense, and reproducibility.
"""

import sys
import os
import heapq

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnbench.envs.config import EnvConfig
from qnbench.envs.engine import QuantumNetworkEngine
from qnbench.envs.structs import EventType


def _make_engine(cfg: EnvConfig, seed: int = 42) -> QuantumNetworkEngine:
    """Helper: build a 4-node engine with the given config."""
    rng = np.random.default_rng(seed)
    engine = QuantumNetworkEngine(cfg, rng)
    engine.add_node(4, is_repeater=False)
    engine.add_node(4, is_repeater=True)
    engine.add_node(4, is_repeater=True)
    engine.add_node(4, is_repeater=False)
    return engine


def _drain_until_expire(engine: QuantumNetworkEngine):
    """Process all events until the first LINK_EXPIRE (exclusive)."""
    while engine.event_queue:
        evt = engine.event_queue[0]
        if evt.event_type == EventType.LINK_EXPIRE and not evt.cancelled:
            break
        heapq.heappop(engine.event_queue)
        engine._events_by_id.pop(evt.event_id, None)
        if not evt.cancelled:
            engine.current_time = evt.timestamp
            evt.callback(evt.data)


def _drain_all(engine: QuantumNetworkEngine):
    """Process all remaining events."""
    while engine.event_queue:
        evt = heapq.heappop(engine.event_queue)
        engine._events_by_id.pop(evt.event_id, None)
        if not evt.cancelled:
            engine.current_time = evt.timestamp
            evt.callback(evt.data)


# ── Generation timing ────────────────────────────────────────────

def test_generation_timing():
    """
    Verify generation times follow the expected geometric distribution.
    Mean should be ≈ heralding_delay + (1/prob_gen) * attempt_period.
    """
    cfg = EnvConfig(prob_gen=0.5, t_coherence=10000.0)
    gen_times = []

    for trial in range(200):
        engine = _make_engine(cfg, seed=trial)
        t0 = engine.current_time
        engine.req_entangle(1, 1)   # Node1 → Node2
        # Only drain gen attempts (stop at first non-GEN_ATTEMPT event)
        while engine.event_queue:
            evt = engine.event_queue[0]
            if evt.event_type != EventType.GEN_ATTEMPT:
                break
            heapq.heappop(engine.event_queue)
            engine._events_by_id.pop(evt.event_id, None)
            if not evt.cancelled:
                engine.current_time = evt.timestamp
                result = evt.callback(evt.data)
                if result:  # gen succeeded
                    break
        gen_times.append(engine.current_time - t0)

    mean_t = np.mean(gen_times)

    heralding = engine.get_cc_delay(1, 2, round_trip=False)
    period = cfg.gen_attempt_period
    theoretical = heralding + (1.0 / cfg.prob_gen) * period

    error = abs(mean_t - theoretical) / theoretical
    assert error < 0.15, (
        f"Gen timing off: measured={mean_t:.4f}, theoretical={theoretical:.4f}, "
        f"error={error*100:.1f}%"
    )


# ── Swap TTL inheritance ─────────────────────────────────────────

def test_swap_ttl_inheritance():
    """After swap, child link TTL ≤ min(parent TTLs), not a fresh t_coh."""
    cfg = EnvConfig(prob_gen=1.0, prob_swap=1.0, t_coherence=100.0,
                    swap_resets_coherence=False)
    engine = _make_engine(cfg, seed=42)

    engine.req_entangle(1, 0)  # left link
    engine.req_entangle(1, 1)  # right link
    _drain_until_expire(engine)

    # Age links to half-life
    engine.current_time = 50.0
    parent_ttls = [l.ttl(50.0) for l in engine.links.values()]
    assert all(t < 55 for t in parent_ttls), f"Parents should be ~50ms TTL, got {parent_ttls}"

    engine.req_swap(1)
    _drain_until_expire(engine)

    child_links = list(engine.links.values())
    assert len(child_links) == 1, f"Expected 1 child link, got {len(child_links)}"
    child_ttl = child_links[0].ttl(engine.current_time)
    assert child_ttl < 60, f"Child TTL={child_ttl:.1f}ms should be < 60 (inherited), not ~100"


def test_swap_ttl_reset_compat():
    """With swap_resets_coherence=True, child gets full t_coh (V6 compat)."""
    cfg = EnvConfig(prob_gen=1.0, prob_swap=1.0, t_coherence=100.0,
                    swap_resets_coherence=True)
    engine = _make_engine(cfg, seed=42)

    engine.req_entangle(1, 0)
    engine.req_entangle(1, 1)
    _drain_until_expire(engine)

    engine.current_time = 50.0
    engine.req_swap(1)
    _drain_until_expire(engine)

    child = list(engine.links.values())
    assert len(child) == 1
    assert child[0].ttl(engine.current_time) > 95, "With reset=True, should get ~100ms"


# ── Stale event cancellation ────────────────────────────────────

def test_stale_expire_events():
    """After purify, there should be exactly as many active expire events as links."""
    cfg = EnvConfig(prob_gen=1.0, prob_purify=1.0, t_coherence=100.0)
    engine = _make_engine(cfg, seed=42)

    engine.req_entangle(1, 0)
    engine.req_entangle(1, 0)
    _drain_until_expire(engine)

    engine.req_purify(1, "left")
    _drain_until_expire(engine)

    active_expires = sum(
        1 for e in engine.event_queue
        if e.event_type == EventType.LINK_EXPIRE and not e.cancelled
    )
    n_links = len(engine.links)
    assert active_expires == n_links, (
        f"Stale events: {active_expires} active expires for {n_links} links"
    )


# ── Zombie link defence ─────────────────────────────────────────

def test_no_zombie_after_failed_swap():
    """
    Scenario: links expire while locked by a swap that then fails.
    V7 should retry the expire event → no zombie.
    """
    cfg = EnvConfig(prob_gen=1.0, prob_swap=0.0, t_coherence=10.0,
                    swap_resets_coherence=False)
    engine = _make_engine(cfg, seed=42)

    engine.req_entangle(1, 0)
    engine.req_entangle(1, 1)
    _drain_until_expire(engine)

    # Advance to just before expiry, then swap (will fail: p=0)
    engine.current_time = 10.0
    engine.req_swap(1)

    _drain_all(engine)

    # Should have no links remaining
    zombies = sum(1 for l in engine.links.values()
                  if engine.current_time > l.expire_time)
    assert zombies == 0, f"Found {zombies} zombie links"
    assert len(engine.links) == 0, f"Expected 0 links, got {len(engine.links)}"


# ── Reproducibility ──────────────────────────────────────────────

def test_reproducibility():
    """Two runs with the same seed must produce identical results."""
    cfg = EnvConfig()
    results = []

    for _ in range(2):
        engine = _make_engine(cfg, seed=42)
        for _ in range(5):
            engine.req_entangle(1, 1)
            _drain_until_expire(engine)

        state = (
            engine.current_time,
            len(engine.links),
            engine.accumulated_step_reward,
        )
        results.append(state)

    assert results[0] == results[1], f"Reproducibility failed: {results}"


# ── Lock assertion ───────────────────────────────────────────────

def test_lock_assertion():
    """Locking an already-locked memory should raise AssertionError."""
    cfg = EnvConfig(prob_gen=1.0, t_coherence=10000.0)
    engine = _make_engine(cfg, seed=42)

    engine.req_entangle(1, 0)
    _drain_until_expire(engine)

    # Memory 0 on node 1 should be ENTANGLED
    # Trying to lock it should raise (it's ENTANGLED, not BUSY, but
    # _lock_memory checks is_locked which only covers BUSY_*)
    # Actually ENTANGLED is not locked, so this tests the BUSY case:
    engine.req_entangle(1, 1)   # locks memory 1
    # Now try to manually re-lock the same memory
    from qnbench.envs.structs import MemoryState
    try:
        # Find a BUSY memory
        for m in engine.nodes[1].memories:
            if m.is_locked():
                engine._lock_memory(1, m.idx, MemoryState.BUSY_SWAP, 999)
                assert False, "Should have raised AssertionError"
    except AssertionError as e:
        if "Double-lock" in str(e):
            pass  # Expected
        else:
            raise


if __name__ == "__main__":
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

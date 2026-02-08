"""
qnbench.envs.structs
=====================

Core data structures for the quantum network simulation.

- ``MemoryState`` / ``Memory``: quantum memory slots on each node.
- ``Link``: an entangled pair spanning two memories.
- ``Event``: a scheduled discrete-event callback.
- ``Node``: a network node holding several memories.
- ``OperationManager``: tracks in-flight async operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Tuple

from .physics import decoherence_fidelity
from .config import FIDELITY_MIXED_STATE


# =====================================================================
# Memory
# =====================================================================

class MemoryState(Enum):
    """Possible states of a single quantum memory slot."""
    FREE = 0            # Available for new entanglement
    BUSY_GEN = 1        # Locked by an ongoing generation attempt
    BUSY_SWAP = 2       # Locked by an ongoing swap operation
    BUSY_PURIFY = 3     # Locked by an ongoing purification
    ENTANGLED = 4       # Holds one half of an entangled pair

    def is_busy(self) -> bool:
        return self in (MemoryState.BUSY_GEN,
                        MemoryState.BUSY_SWAP,
                        MemoryState.BUSY_PURIFY)


class Memory:
    """
    A single quantum memory slot.

    Each memory is either FREE, locked by an operation (BUSY_*), or
    holding one half of an entangled pair (ENTANGLED).
    """
    __slots__ = ("idx", "state", "link_id", "entangled_node",
                 "locked_by_operation")

    def __init__(self, idx: int):
        self.idx = idx
        self.state = MemoryState.FREE
        self.link_id: Optional[int] = None
        self.entangled_node: int = -1
        self.locked_by_operation: Optional[int] = None

    # ── Queries ──────────────────────────────────────────────────
    def is_free(self) -> bool:
        return self.state == MemoryState.FREE

    def is_entangled_and_available(self) -> bool:
        return self.state == MemoryState.ENTANGLED

    def is_locked(self) -> bool:
        return self.state.is_busy()

    # ── Mutation ─────────────────────────────────────────────────
    def reset(self):
        """Return this memory to the FREE state."""
        self.state = MemoryState.FREE
        self.link_id = None
        self.entangled_node = -1
        self.locked_by_operation = None


# =====================================================================
# Link  (entangled pair)
# =====================================================================

@dataclass
class Link:
    """
    Represents an active entangled pair between two memories.

    ``fidelity`` is the fidelity *at* ``creation_time``; the current
    fidelity is computed on the fly by :meth:`current_fidelity` using
    an exponential-decay decoherence model.
    """
    link_id: int
    node_a: int
    mem_a: int
    node_b: int
    mem_b: int
    fidelity: float          # fidelity at creation_time
    creation_time: float
    expire_time: float
    span_distance: float = 0.0
    swap_count: int = 0
    purify_count: int = 0
    expire_event_id: int = -1   # tracks the scheduled LINK_EXPIRE event

    def __post_init__(self):
        if self.fidelity < FIDELITY_MIXED_STATE - 1e-9:
            self.fidelity = FIDELITY_MIXED_STATE

    def current_fidelity(self, current_time: float, t_coh: float) -> float:
        """Fidelity right now, accounting for decoherence."""
        return decoherence_fidelity(
            self.fidelity, current_time - self.creation_time, t_coh
        )

    def age(self, current_time: float) -> float:
        """How long this link has existed (ms)."""
        return max(0.0, current_time - self.creation_time)

    def ttl(self, current_time: float) -> float:
        """Remaining time-to-live (ms)."""
        return max(0.0, self.expire_time - current_time)

    def get_other_end(self, node_id: int) -> Tuple[int, int]:
        """Given one endpoint node, return (other_node, other_mem)."""
        if self.node_a == node_id:
            return (self.node_b, self.mem_b)
        return (self.node_a, self.mem_a)


# =====================================================================
# Event  (discrete-event scheduler entry)
# =====================================================================

class EventType(Enum):
    GEN_ATTEMPT = auto()
    SWAP_COMPLETE = auto()
    PURIFY_COMPLETE = auto()
    LINK_EXPIRE = auto()


@dataclass(order=True)
class Event:
    """
    A scheduled callback in the discrete-event simulation.

    Events are ordered by ``timestamp`` for the min-heap.  The
    ``cancelled`` flag supports lazy deletion: cancelled events are
    simply skipped when popped.
    """
    timestamp: float
    event_id: int = field(compare=False)
    event_type: EventType = field(compare=False)
    callback: object = field(compare=False)     # callable
    data: dict = field(compare=False)
    cancelled: bool = field(default=False, compare=False)


# =====================================================================
# Node
# =====================================================================

class Node:
    """A network node (endpoint or repeater) with quantum memories."""

    def __init__(self, node_id: int, num_memories: int,
                 is_repeater: bool = True, position: float = 0.0):
        self.node_id = node_id
        self.is_repeater = is_repeater
        self.memories = [Memory(i) for i in range(num_memories)]
        self.position = position    # absolute position along the chain (km)

    def get_first_free_memory(self) -> Optional[int]:
        """Return index of the first FREE memory, or None."""
        for i, m in enumerate(self.memories):
            if m.is_free():
                return i
        return None


# =====================================================================
# Operation Manager
# =====================================================================

class OperationManager:
    """
    Tracks in-flight asynchronous operations (gen / swap / purify).

    Each operation gets a unique integer ID.  The engine checks
    ``is_active(op_id)`` in callbacks to detect stale events whose
    resources have already been freed.
    """

    def __init__(self):
        self._counter: int = 0
        self._active: Dict[int, dict] = {}

    def reset(self):
        self._counter = 0
        self._active.clear()

    def new_operation(self, op_type: str, **kwargs) -> int:
        """Register a new operation; returns its unique ID."""
        self._counter += 1
        self._active[self._counter] = {"type": op_type, **kwargs}
        return self._counter

    def complete_operation(self, op_id: int) -> Optional[dict]:
        """Mark an operation as done; returns its metadata or None."""
        return self._active.pop(op_id, None)

    def is_active(self, op_id: int) -> bool:
        return op_id in self._active

    def get_active_count(self) -> int:
        return len(self._active)

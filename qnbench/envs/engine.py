"""
qnbench.envs.engine
====================

Discrete-event simulation engine for a linear quantum repeater chain.

The engine manages:
- Network topology (nodes, distances, classical communication delays).
- Entanglement generation with geometric-distribution attempt model.
- Entanglement swap with Werner-state fidelity tracking.
- BBPSSW purification.
- Exponential decoherence and link expiry.
- Operation lifecycle via ``OperationManager``.

**V7 improvements** (see README for full changelog):
1. Swap inherits parent TTL instead of resetting coherence clock.
2. Stale LINK_EXPIRE events are cancelled via event-ID tracking.
3. Zombie-link defence: deferred expire retry + periodic cleanup.
4. ``_lock_memory`` asserts that the target is not already locked.
"""

from __future__ import annotations

import heapq
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import EnvConfig, SPEED_OF_LIGHT_FIBER, FIDELITY_MIXED_STATE
from .structs import (
    Node, Link, Event, EventType, Memory, MemoryState, OperationManager,
)
from .physics import swap_fidelity, purify_fidelity_bbpssw

logger = logging.getLogger("qnbench.engine")


class QuantumNetworkEngine:
    """
    Core physics engine driven by a min-heap event queue.

    Typical lifecycle per Gym step:
    1. The Env calls ``req_*`` methods to enqueue operations.
    2. The Env pops events from ``event_queue`` until a *critical* event
       fires (one that changes the observation, indicated by returning
       ``True`` from its callback).
    """

    def __init__(self, cfg: EnvConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

        # ── State ────────────────────────────────────────────────
        self.nodes: List[Node] = []
        self.links: Dict[int, Link] = {}
        self.event_queue: List[Event] = []
        self._events_by_id: Dict[int, Event] = {}

        self.current_time: float = 0.0
        self.link_counter: int = 0
        self.event_counter: int = 0
        self.delivered_pairs: int = 0
        self.accumulated_step_reward: float = 0.0

        self.distance_matrix: Dict[Tuple[int, int], float] = {}
        self.ops = OperationManager()

    # ─────────────────────────────────────────────────────────────
    # Setup & Reset
    # ─────────────────────────────────────────────────────────────

    def set_rng(self, rng: np.random.Generator):
        self.rng = rng

    def reset(self):
        """Clear all state for a new episode."""
        self.nodes.clear()
        self.links.clear()
        self.event_queue.clear()
        self._events_by_id.clear()
        self.current_time = 0.0
        self.link_counter = 0
        self.event_counter = 0
        self.delivered_pairs = 0
        self.accumulated_step_reward = 0.0
        self.distance_matrix.clear()
        self.ops.reset()

    def add_node(self, num_memories: int, is_repeater: bool = True):
        """Append a node to the linear chain."""
        nid = len(self.nodes)
        pos = sum(self.cfg.node_distances[:nid]) if nid > 0 else 0.0
        node = Node(nid, num_memories, is_repeater, pos)
        self.nodes.append(node)
        # Update pairwise distance matrix
        for oid in range(nid):
            d = abs(node.position - self.nodes[oid].position)
            self.distance_matrix[(oid, nid)] = d
            self.distance_matrix[(nid, oid)] = d
        self.distance_matrix[(nid, nid)] = 0.0

    # ─────────────────────────────────────────────────────────────
    # Distance & Delay Helpers
    # ─────────────────────────────────────────────────────────────

    def get_distance(self, a: int, b: int) -> float:
        return self.distance_matrix.get((a, b), 0.0)

    def get_cc_delay(self, a: int, b: int, round_trip: bool = True) -> float:
        """Classical-communication delay between two nodes (ms)."""
        one_way = self.get_distance(a, b) / SPEED_OF_LIGHT_FIBER * 1000
        return one_way * 2 if round_trip else one_way

    # ─────────────────────────────────────────────────────────────
    # Event Scheduling
    # ─────────────────────────────────────────────────────────────

    def _schedule(self, delay: float, etype: EventType,
                  callback, data: dict) -> int:
        self.event_counter += 1
        eid = self.event_counter
        evt = Event(self.current_time + delay, eid, etype, callback, data)
        heapq.heappush(self.event_queue, evt)
        self._events_by_id[eid] = evt
        return eid

    def _cancel_event(self, event_id: int):
        """Mark an event as cancelled (lazy deletion from heap)."""
        evt = self._events_by_id.pop(event_id, None)
        if evt is not None:
            evt.cancelled = True

    def pop_event(self) -> Optional[Event]:
        """Pop the next non-cancelled event, or None if queue is empty."""
        while self.event_queue:
            evt = heapq.heappop(self.event_queue)
            self._events_by_id.pop(evt.event_id, None)
            if not evt.cancelled:
                return evt
        return None

    # ─────────────────────────────────────────────────────────────
    # Link Management
    # ─────────────────────────────────────────────────────────────

    def _schedule_link_expire(self, link: Link):
        """(Re)schedule the expiry event for *link*, cancelling any old one."""
        if link.expire_event_id >= 0:
            self._cancel_event(link.expire_event_id)
        ttl = link.expire_time - self.current_time
        if ttl > 0:
            eid = self._schedule(
                ttl, EventType.LINK_EXPIRE,
                self._handle_expire, {"lid": link.link_id},
            )
            link.expire_event_id = eid
        else:
            link.expire_event_id = -1

    def _destroy_link(self, link_id: int):
        """Remove a link completely: cancel event, free memories, delete."""
        link = self.links.pop(link_id, None)
        if link is None:
            return
        if link.expire_event_id >= 0:
            self._cancel_event(link.expire_event_id)
        self._force_release(link.node_a, link.mem_a)
        self._force_release(link.node_b, link.mem_b)

    def get_links_by_direction(
        self, node_id: int, direction: str, only_available: bool = False,
    ) -> List[Link]:
        """
        Return links connected to *node_id* on the given side.

        Parameters
        ----------
        direction : ``"left"`` or ``"right"``
        only_available : if True, skip links whose memories are busy.
        """
        result: List[Link] = []
        for mem in self.nodes[node_id].memories:
            if mem.link_id is None or mem.link_id not in self.links:
                continue
            link = self.links[mem.link_id]
            is_left = mem.entangled_node < node_id
            if (direction == "left") != is_left:
                continue
            if only_available:
                ma = self.nodes[link.node_a].memories[link.mem_a]
                mb = self.nodes[link.node_b].memories[link.mem_b]
                if not (ma.is_entangled_and_available()
                        and mb.is_entangled_and_available()):
                    continue
            result.append(link)
        return result

    # ─────────────────────────────────────────────────────────────
    # Action Validity
    # ─────────────────────────────────────────────────────────────

    def can_do_action(self, nidx: int, action: int) -> bool:
        """Check whether *action* is currently valid for node *nidx*."""
        node = self.nodes[nidx]
        nn = len(self.nodes)

        if action == 0:       # Wait
            return True
        elif action == 1:     # Gen_L
            return (nidx > 0
                    and node.get_first_free_memory() is not None
                    and self.nodes[nidx - 1].get_first_free_memory() is not None)
        elif action == 2:     # Gen_R
            return (nidx < nn - 1
                    and node.get_first_free_memory() is not None
                    and self.nodes[nidx + 1].get_first_free_memory() is not None)
        elif action == 3:     # Swap
            if nidx == 0 or nidx == nn - 1:
                return False
            return (len(self.get_links_by_direction(nidx, "left", True)) >= 1
                    and len(self.get_links_by_direction(nidx, "right", True)) >= 1)
        elif action == 4:     # Purify_L
            return (nidx > 0
                    and len(self.get_links_by_direction(nidx, "left", True)) >= 2)
        elif action == 5:     # Purify_R
            return (nidx < nn - 1
                    and len(self.get_links_by_direction(nidx, "right", True)) >= 2)
        elif action == 6:     # Discard
            return any(m.is_entangled_and_available() and m.link_id in self.links
                       for m in node.memories)
        return False

    # =================================================================
    # GENERATION  (geometric-distribution attempt model)
    # =================================================================

    def req_entangle(self, node_idx: int, direction: int) -> bool:
        """
        Start an entanglement generation attempt.

        ``direction``: 0 = left neighbour, 1 = right neighbour.

        The first attempt fires after the heralding delay (one-way CC);
        subsequent retries are spaced by ``gen_attempt_period``.
        """
        n1 = node_idx
        n2 = node_idx + (-1 if direction == 0 else 1)
        if n2 < 0 or n2 >= len(self.nodes):
            return False

        m1 = self.nodes[n1].get_first_free_memory()
        m2 = self.nodes[n2].get_first_free_memory()
        if m1 is None or m2 is None:
            return False

        op_id = self.ops.new_operation("gen", time=self.current_time)
        self._lock_memory(n1, m1, MemoryState.BUSY_GEN, op_id)
        self._lock_memory(n2, m2, MemoryState.BUSY_GEN, op_id)

        data = {
            "n1": n1, "m1": m1, "n2": n2, "m2": m2,
            "distance": self.get_distance(n1, n2),
            "op_id": op_id, "attempt_count": 0,
        }
        heralding = self.get_cc_delay(n1, n2, round_trip=False)
        self._schedule(heralding, EventType.GEN_ATTEMPT,
                       self._handle_gen_attempt, data)
        return True

    def _handle_gen_attempt(self, data: dict) -> bool:
        """Process one generation attempt; retry on failure."""
        op_id = data["op_id"]
        mem1 = self.nodes[data["n1"]].memories[data["m1"]]
        mem2 = self.nodes[data["n2"]].memories[data["m2"]]

        if not self.ops.is_active(op_id):
            return False
        if (mem1.locked_by_operation != op_id
                or mem2.locked_by_operation != op_id):
            self.ops.complete_operation(op_id)
            return False

        data["attempt_count"] += 1

        if self.rng.random() < self.cfg.prob_gen:
            # ── Success → create the entangled link ──────────────
            self.link_counter += 1
            link = Link(
                link_id=self.link_counter,
                node_a=data["n1"], mem_a=data["m1"],
                node_b=data["n2"], mem_b=data["m2"],
                fidelity=self.cfg.init_fidelity,
                creation_time=self.current_time,
                expire_time=self.current_time + self.cfg.t_coherence,
                span_distance=data["distance"],
            )
            self.links[link.link_id] = link
            self._bind_memory(data["n1"], data["m1"], link.link_id, data["n2"])
            self._bind_memory(data["n2"], data["m2"], link.link_id, data["n1"])
            self._schedule_link_expire(link)
            self.ops.complete_operation(op_id)
            self.accumulated_step_reward += self.cfg.gen_success_reward
            logger.info(
                "Gen OK  Link %d (%d↔%d, %.0fkm) F=%.3f att=%d",
                link.link_id, data["n1"], data["n2"],
                data["distance"], self.cfg.init_fidelity,
                data["attempt_count"],
            )
            return True
        else:
            # ── Failure → schedule next attempt ──────────────────
            self._schedule(self.cfg.gen_attempt_period,
                           EventType.GEN_ATTEMPT,
                           self._handle_gen_attempt, data)
            return False

    # =================================================================
    # SWAP
    # =================================================================

    def req_swap(self, node_id: int) -> bool:
        """Perform entanglement swap at a repeater node."""
        lefts = self.get_links_by_direction(node_id, "left", True)
        rights = self.get_links_by_direction(node_id, "right", True)
        if not lefts or not rights:
            return False

        tc = self.cfg.t_coherence
        ll = max(lefts, key=lambda l: l.current_fidelity(self.current_time, tc))
        rl = max(rights, key=lambda l: l.current_fidelity(self.current_time, tc))

        op_id = self.ops.new_operation("swap", time=self.current_time)
        self._lock_memory(ll.node_a, ll.mem_a, MemoryState.BUSY_SWAP, op_id)
        self._lock_memory(ll.node_b, ll.mem_b, MemoryState.BUSY_SWAP, op_id)
        self._lock_memory(rl.node_a, rl.mem_a, MemoryState.BUSY_SWAP, op_id)
        self._lock_memory(rl.node_b, rl.mem_b, MemoryState.BUSY_SWAP, op_id)

        le, lm = ll.get_other_end(node_id)
        re, rm = rl.get_other_end(node_id)
        cc = max(self.get_cc_delay(node_id, le, False),
                 self.get_cc_delay(node_id, re, False))
        delay = self.cfg.gate_time_swap + cc + self.cfg.classical_processing

        self._schedule(delay, EventType.SWAP_COMPLETE,
                       self._handle_swap_complete, {
            "nid": node_id,
            "lid1": ll.link_id, "lid2": rl.link_id,
            "le": le, "lm": lm, "re": re, "rm": rm,
            "span": ll.span_distance + rl.span_distance,
            "swaps": max(ll.swap_count, rl.swap_count) + 1,
            "purifs": ll.purify_count + rl.purify_count,
            "parent_min_ttl": min(ll.ttl(self.current_time),
                                  rl.ttl(self.current_time)),
            "op_id": op_id,
        })
        return True

    def _handle_swap_complete(self, data: dict) -> bool:
        op_id = data["op_id"]
        if not self.ops.is_active(op_id):
            return True

        nid = data["nid"]
        l1 = self.links.get(data["lid1"])
        l2 = self.links.get(data["lid2"])
        tc = self.cfg.t_coherence

        f1 = l1.current_fidelity(self.current_time, tc) if l1 else 0
        f2 = l2.current_fidelity(self.current_time, tc) if l2 else 0

        # Release middle-node memories & destroy parent links
        for parent in (l1, l2):
            if parent is None:
                continue
            mm = parent.mem_a if parent.node_a == nid else parent.mem_b
            self._release_memory(nid, mm, op_id)
            if parent.expire_event_id >= 0:
                self._cancel_event(parent.expire_event_id)
            self.links.pop(parent.link_id, None)

        self.ops.complete_operation(op_id)

        if not l1 or not l2:
            # One parent was already gone → clean up far endpoints
            self._release_memory(data["le"], data["lm"], op_id)
            self._release_memory(data["re"], data["rm"], op_id)
            logger.info("Swap FAIL @ %.2fms (link missing)", self.current_time)
            return True

        if self.rng.random() < self.cfg.prob_swap:
            # ── Success ──────────────────────────────────────────
            new_fid = swap_fidelity(f1, f2)
            self.link_counter += 1

            # V7: TTL inheritance
            if self.cfg.swap_resets_coherence:
                new_expire = self.current_time + tc
            else:
                gate_overhead = self.cfg.gate_time_swap + self.cfg.classical_processing
                inherited = max(1.0, data["parent_min_ttl"] - gate_overhead)
                new_expire = self.current_time + inherited

            nl = Link(
                link_id=self.link_counter,
                node_a=data["le"], mem_a=data["lm"],
                node_b=data["re"], mem_b=data["rm"],
                fidelity=new_fid, creation_time=self.current_time,
                expire_time=new_expire,
                span_distance=data["span"],
                swap_count=data["swaps"],
                purify_count=data["purifs"],
            )
            self.links[nl.link_id] = nl
            self._bind_memory(nl.node_a, nl.mem_a, nl.link_id, nl.node_b)
            self._bind_memory(nl.node_b, nl.mem_b, nl.link_id, nl.node_a)
            self._schedule_link_expire(nl)
            self.accumulated_step_reward += self.cfg.swap_success_reward
            # 链路跨度进度奖励: 归一化跨度 (0~1) × span_progress_reward
            max_span = sum(self.cfg.node_distances) or 1.0
            span_ratio = nl.span_distance / max_span
            self.accumulated_step_reward += span_ratio * self.cfg.span_progress_reward
            logger.info(
                "Swap OK  Node %d → Link %d (%d↔%d) F=%.4f TTL=%.1fms",
                nid, nl.link_id, nl.node_a, nl.node_b,
                new_fid, nl.ttl(self.current_time),
            )
        else:
            # ── Probabilistic failure ────────────────────────────
            self._release_memory(data["le"], data["lm"], op_id)
            self._release_memory(data["re"], data["rm"], op_id)
            logger.info("Swap FAIL @ %.2fms (probabilistic)", self.current_time)

        return True

    # =================================================================
    # PURIFICATION
    # =================================================================

    def req_purify(self, node_id: int, direction: str) -> bool:
        """Initiate BBPSSW purification between two same-direction links."""
        cands = self.get_links_by_direction(node_id, direction, True)
        if len(cands) < 2:
            return False

        tc = self.cfg.t_coherence
        cands.sort(key=lambda l: l.current_fidelity(self.current_time, tc),
                   reverse=True)
        target, source = cands[0], cands[1]

        op_id = self.ops.new_operation("purify", time=self.current_time)
        self._lock_memory(target.node_a, target.mem_a, MemoryState.BUSY_PURIFY, op_id)
        self._lock_memory(target.node_b, target.mem_b, MemoryState.BUSY_PURIFY, op_id)
        self._lock_memory(source.node_a, source.mem_a, MemoryState.BUSY_PURIFY, op_id)
        self._lock_memory(source.node_b, source.mem_b, MemoryState.BUSY_PURIFY, op_id)

        other, _ = target.get_other_end(node_id)
        cc = self.get_cc_delay(node_id, other, round_trip=True)
        delay = self.cfg.gate_time_purify + cc + self.cfg.classical_processing

        self._schedule(delay, EventType.PURIFY_COMPLETE,
                       self._handle_purify_complete, {
            "lid_t": target.link_id, "lid_s": source.link_id, "op_id": op_id,
        })
        return True

    def _handle_purify_complete(self, data: dict) -> bool:
        op_id = data["op_id"]
        if not self.ops.is_active(op_id):
            return True

        lt = self.links.get(data["lid_t"])
        ls = self.links.get(data["lid_s"])
        tc = self.cfg.t_coherence
        f_t = lt.current_fidelity(self.current_time, tc) if lt else 0
        f_s = ls.current_fidelity(self.current_time, tc) if ls else 0

        # Always destroy the *source* link
        if ls:
            if ls.expire_event_id >= 0:
                self._cancel_event(ls.expire_event_id)
            self._release_memory(ls.node_a, ls.mem_a, op_id)
            self._release_memory(ls.node_b, ls.mem_b, op_id)
            self.links.pop(data["lid_s"], None)

        self.ops.complete_operation(op_id)

        if not lt or not ls:
            if lt:
                self._destroy_link(data["lid_t"])
            logger.info("Purify ABORT @ %.2fms (link missing)", self.current_time)
            return True

        if self.rng.random() < self.cfg.prob_purify:
            new_fid = purify_fidelity_bbpssw(f_t, f_s)
            lt.fidelity = max(FIDELITY_MIXED_STATE, min(1.0, new_fid))
            lt.creation_time = self.current_time
            lt.expire_time = self.current_time + tc
            lt.purify_count += 1
            self._bind_memory(lt.node_a, lt.mem_a, lt.link_id, lt.node_b)
            self._bind_memory(lt.node_b, lt.mem_b, lt.link_id, lt.node_a)
            self._schedule_link_expire(lt)
            delta = new_fid - f_t
            bonus = max(0, delta) * self.cfg.purify_bonus_factor
            self.accumulated_step_reward += self.cfg.purify_base_reward + bonus
            logger.info("Purify OK  Link %d: F %.4f→%.4f (Δ=%+.4f)",
                         lt.link_id, f_t, new_fid, delta)
        else:
            self._destroy_link(data["lid_t"])
            logger.info("Purify FAIL @ %.2fms", self.current_time)

        return True

    # =================================================================
    # DISCARD
    # =================================================================

    def req_discard(self, node_id: int) -> bool:
        """Discard the lowest-fidelity link touching *node_id*."""
        tc = self.cfg.t_coherence
        cands = []
        for m in self.nodes[node_id].memories:
            if m.is_entangled_and_available() and m.link_id in self.links:
                link = self.links[m.link_id]
                cands.append((link.current_fidelity(self.current_time, tc),
                              link.link_id))
        if not cands:
            return False
        _, lid = min(cands)
        self._destroy_link(lid)
        return True

    # =================================================================
    # LINK EXPIRY
    # =================================================================

    def _handle_expire(self, data: dict) -> bool:
        lid = data["lid"]
        if lid not in self.links:
            return False
        link = self.links[lid]
        if self.current_time < link.expire_time - 1e-9:
            return False

        ma = self.nodes[link.node_a].memories[link.mem_a]
        mb = self.nodes[link.node_b].memories[link.mem_b]
        if ma.is_locked() or mb.is_locked():
            # V7: retry instead of silently dropping
            self._schedule(0.1, EventType.LINK_EXPIRE,
                           self._handle_expire, data)
            return False

        self._force_release(link.node_a, link.mem_a)
        self._force_release(link.node_b, link.mem_b)
        self.links.pop(lid, None)
        logger.debug("Link %d expired @ %.2fms", lid, self.current_time)
        return True

    # =================================================================
    # MEMORY MANAGEMENT
    # =================================================================

    def _lock_memory(self, nid: int, mid: int,
                     state: MemoryState, op_id: int):
        """Lock a memory for an operation.  Asserts it is not busy."""
        mem = self.nodes[nid].memories[mid]
        assert not mem.is_locked(), (
            f"Double-lock: Node {nid} Mem {mid} is {mem.state.name} "
            f"(op={mem.locked_by_operation}), new op={op_id}"
        )
        mem.state = state
        mem.locked_by_operation = op_id

    def _bind_memory(self, nid: int, mid: int, lid: int, target: int):
        """Bind a memory to an active entangled link."""
        mem = self.nodes[nid].memories[mid]
        mem.state = MemoryState.ENTANGLED
        mem.link_id = lid
        mem.entangled_node = target
        mem.locked_by_operation = None

    def _release_memory(self, nid: int, mid: int,
                        expected_op_id: Optional[int]):
        """
        Release a memory back to FREE.

        Only succeeds if ``locked_by_operation`` matches *expected_op_id*
        (pass ``None`` to skip the check).  After releasing, runs zombie
        cleanup.
        """
        mem = self.nodes[nid].memories[mid]
        if expected_op_id is not None and mem.locked_by_operation != expected_op_id:
            return
        mem.reset()
        self._cleanup_zombies()

    def _force_release(self, nid: int, mid: int):
        """Unconditionally reset a memory (used by _destroy_link)."""
        self.nodes[nid].memories[mid].reset()

    def _cleanup_zombies(self):
        """Remove expired links whose memories are no longer locked."""
        to_remove = []
        for lid, link in self.links.items():
            if self.current_time <= link.expire_time:
                continue
            ma = self.nodes[link.node_a].memories[link.mem_a]
            mb = self.nodes[link.node_b].memories[link.mem_b]
            if not ma.is_locked() and not mb.is_locked():
                to_remove.append(lid)
        for lid in to_remove:
            link = self.links.pop(lid, None)
            if link is None:
                continue
            if link.expire_event_id >= 0:
                self._cancel_event(link.expire_event_id)
            self._force_release(link.node_a, link.mem_a)
            self._force_release(link.node_b, link.mem_b)
            logger.warning("Zombie link %d cleaned @ %.2fms", lid, self.current_time)

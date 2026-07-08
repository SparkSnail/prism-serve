"""Request lifecycle state machine for prism-serve.

Tracks every request from arrival to completion on the serve (control-plane)
side.  The infer (data-plane) side has its own SequenceStatus; the two are
kept in sync through the interface contract (docs/03).

Borrowing:
  - SeqState enum  ← Ray GCS Actor state machine
  - RequestTracker ← Ray GCS Actor state tracking (explicit state, no inference)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator


# ---------------------------------------------------------------------------
# SeqState
# ---------------------------------------------------------------------------

class SeqState(Enum):
    """serve-side request lifecycle states (← Ray GCS Actor state machine).

    Distinct from infer-side SequenceStatus:
      - infer: GPU execution state (RUNNING / KV_TRANSFERRING / MIGRATING_*)
      - serve: scheduling phase (WAITING → PREFILLING → ... → FINISHED)

    The two state machines are maintained independently and synchronised
    through event notifications defined in interface contract [03].
    """
    WAITING    = auto()  # in waiting queue; P instance not yet assigned
    PREFILLING = auto()  # dispatched to P instance; prefill forward running
    KV_PENDING = auto()  # prefill done; KV transfer in-flight
    DECODING   = auto()  # KV arrived at D instance; decode running
    FINISHED   = auto()  # EOS or max_tokens reached
    ABORTED    = auto()  # max_recompute_attempts exhausted or fatal error


# ---------------------------------------------------------------------------
# RequestInfo
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RequestInfo:
    """Complete serve-side state for a single request (stored in RequestTracker).

    Invariant: state advances only along the SeqState transition graph,
               except KV_PENDING → WAITING (recompute fallback).
    """
    req_id: str
    state: SeqState = SeqState.WAITING

    # Instance assignment
    prefill_instance: str = ""   # e.g. "p-0"
    decode_instance:  str = ""   # e.g. "d-0"

    # Timestamps (stuck detection + SLA metrics)
    arrived_at:    float = field(default_factory=time.monotonic)
    prefill_start: float = 0.0   # when dispatch_prefill was sent
    kv_sent_at:    float = 0.0   # when TransferGovernor submitted transfer
    decode_start:  float = 0.0   # when D side started decoding
    finished_at:   float = 0.0   # when request completed

    # Recompute guard
    recompute_count: int = 0     # times recompute has been triggered

    # KV transfer metadata
    kv_size_bytes: int = 0
    block_table:   list[int] = field(default_factory=list)

    def ttft_ms(self) -> float:
        """Time To First Token: arrived_at → decode_start (ms).

        Returns -1.0 if decode has not started yet.
        """
        if self.decode_start == 0.0:
            return -1.0
        return (self.decode_start - self.arrived_at) * 1000.0

    def is_stuck(self, timeout_s: float) -> bool:
        """True if in KV_PENDING and transfer has exceeded timeout_s."""
        if self.state != SeqState.KV_PENDING:
            return False
        return (time.monotonic() - self.kv_sent_at) > timeout_s


# ---------------------------------------------------------------------------
# InstanceSlot
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class InstanceSlot:
    """One sequence occupancy unit on a decode instance.

    Semantics (← Ray Actor explicit resource holding):
      KV blocks are held exclusively for the entire sequence lifetime.
      Unlike a web connection pool (release after request), slots are
      held because attention needs historical KV at every decode step.
    """
    instance_id: str
    seq_id: str | None = None
    block_table: list[int] = field(default_factory=list)
    allocated_at: float = 0.0

    def is_idle(self) -> bool:
        return self.seq_id is None

    def is_stale(self, timeout_s: float = 300.0) -> bool:
        """Held beyond timeout with no active output — possible leak.

        300 s = 5× the estimated maximum generation time
        (max_tokens=2048, ~50 TPS → ~41 s; 300 s is a 5× safety factor).
        """
        if self.seq_id is None:
            return False
        return (time.monotonic() - self.allocated_at) > timeout_s

    @staticmethod
    def compute_max_slots(
        gpu_memory_gb: float,
        model_weight_gb: float,
        avg_seq_kv_gb: float,
        safety_margin: float = 0.85,
    ) -> int:
        """Compute max slots for one decode instance (← Celeborn slot_count).

        Formula: floor((gpu_memory_gb - model_weight_gb) * safety_margin
                       / avg_seq_kv_gb)

        Qwen3-7B example:
          gpu=80 GB, weight=14 GB, avg_seq_kv=0.44 GB (1024-token seq)
          → (80-14) * 0.85 / 0.44 ≈ 127 slots
        """
        usable = (gpu_memory_gb - model_weight_gb) * safety_margin
        return max(1, int(usable / avg_seq_kv_gb))


# ---------------------------------------------------------------------------
# TransferTask
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TransferTask:
    """A single KV transfer task inside TransferGovernor.

    Corresponds to TransferReq in interface contract [03], wrapped with
    scheduling metadata.
    """
    req_id:      str
    src:         str           # P instance ID
    dst:         str           # D instance ID
    kv_size:     int           # bytes to transfer (used for flow-control)
    priority:    int = 1       # 1 = normal PD; 0 = migration/replica (stricter cap)
    enqueued_at: float = field(default_factory=time.monotonic)
    on_complete: object = None  # Callable[[], None] | None


# ---------------------------------------------------------------------------
# RequestTracker
# ---------------------------------------------------------------------------

def _validate_transition(current: SeqState, new: SeqState) -> None:
    """Assert that current → new is a legal state transition.

    Legal transitions:
      WAITING     → PREFILLING
      PREFILLING  → KV_PENDING
      KV_PENDING  → DECODING | WAITING  (recompute)
      DECODING    → FINISHED | ABORTED
      any state   → ABORTED             (fatal error)
    """
    VALID: dict[SeqState, set[SeqState]] = {
        SeqState.WAITING:    {SeqState.PREFILLING},
        SeqState.PREFILLING: {SeqState.KV_PENDING},
        SeqState.KV_PENDING: {SeqState.DECODING, SeqState.WAITING},
        SeqState.DECODING:   {SeqState.FINISHED, SeqState.ABORTED},
    }
    allowed = VALID.get(current, set()) | {SeqState.ABORTED}
    assert new in allowed, (
        f"illegal transition {current.name} → {new.name}; "
        f"allowed: {[s.name for s in allowed]}"
    )


class RequestTracker:
    """Full request state tracker for the serve control plane.

    Mirrors Ray GCS Actor state tracking:
      - explicit state held per request (no implicit inference)
      - full transition log for debugging
      - timeout detection via timestamps (no heartbeat dependency)
    """

    def __init__(self, metrics) -> None:
        self.metrics = metrics
        self._requests: dict[str, RequestInfo] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, req: RequestInfo) -> None:
        """Register a new request (must start in WAITING state)."""
        assert req.req_id not in self._requests, (
            f"duplicate request id {req.req_id!r}"
        )
        self._requests[req.req_id] = req

    def remove(self, req_id: str) -> RequestInfo | None:
        """Remove a completed/aborted request and return it."""
        return self._requests.pop(req_id, None)

    def get(self, req_id: str) -> RequestInfo | None:
        return self._requests.get(req_id)

    # ------------------------------------------------------------------
    # State transition
    # ------------------------------------------------------------------

    def transition(self, req_id: str, new_state: SeqState, **kwargs) -> None:
        """Advance a request to new_state, updating timestamps and metrics.

        Args:
            req_id:    request identifier
            new_state: target state
            **kwargs:  extra field updates (prefill_instance, decode_instance, …)

        Raises:
            KeyError:       req_id not found
            AssertionError: illegal transition
        """
        req = self._requests[req_id]
        _validate_transition(req.state, new_state)

        now = time.monotonic()
        if new_state == SeqState.PREFILLING:
            req.prefill_start = now
        elif new_state == SeqState.KV_PENDING:
            req.kv_sent_at = now
        elif new_state == SeqState.DECODING:
            req.decode_start = now
        elif new_state in (SeqState.FINISHED, SeqState.ABORTED):
            req.finished_at = now
            self.metrics.observe(
                "request_ttft_ms",
                req.ttft_ms(),
                labels={"state": new_state.name},
            )

        for k, v in kwargs.items():
            setattr(req, k, v)

        req.state = new_state

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_stuck_requests(self, timeout_s: float) -> list[RequestInfo]:
        """Return all KV_PENDING requests that have exceeded timeout_s.

        Sorted by kv_sent_at ascending (oldest first).
        """
        stuck = [r for r in self._requests.values() if r.is_stuck(timeout_s)]
        return sorted(stuck, key=lambda r: r.kv_sent_at)

    def iter_by_state(self, state: SeqState) -> Iterator[RequestInfo]:
        """Iterate over all requests in a given state."""
        return (r for r in self._requests.values() if r.state == state)

    def __len__(self) -> int:
        return len(self._requests)

    def __contains__(self, req_id: str) -> bool:
        return req_id in self._requests

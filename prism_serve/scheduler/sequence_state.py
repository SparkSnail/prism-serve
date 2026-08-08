"""Serve-side request lifecycle state."""

from __future__ import annotations

import time
import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator


class SeqState(Enum):
    WAITING = auto()
    PREFILLING = auto()
    KV_PENDING = auto()
    RECOMPUTING = auto()
    AFFINITY_LOADING = auto()
    PREFIX_PREFILLING = auto()
    DECODING = auto()
    FINISHED = auto()
    ABORTED = auto()


@dataclass(slots=True)
class RequestInfo:
    """State for one request owned by a gateway.

    Recompute remains assigned to the original decode instance.
    """
    req_id: str
    state: SeqState = SeqState.WAITING
    fingerprint: object | None = None
    sampling_params: dict = field(default_factory=dict)
    token_ids: list[int] = field(default_factory=list)
    model: str = ""

    prefill_instance: str = ""
    decode_instance:  str = ""
    prefill_instance_epoch: str = ""
    decode_instance_epoch: str = ""

    arrived_at:    float = field(default_factory=time.monotonic)
    prefill_start: float = 0.0
    kv_sent_at:    float = 0.0
    recompute_start: float = 0.0
    decode_start:  float = 0.0
    first_token_at: float = 0.0
    last_output_query_at: float = 0.0
    finished_at:   float = 0.0


    recompute_count: int = 0
    dispatch_attempts: int = 0


    kv_size_bytes: int = 0
    block_table:   list[int] = field(default_factory=list)
    transfer_operation_id: str = ""
    active_operation_id: str = ""
    decode_slot_lease_id: str = ""
    cache_source: str = ""
    cached_prefix_tokens: int = 0
    correctness_path: str = ""
    correctness_source_instance: str = ""
    correctness_target_instance: str = ""
    correctness_cached_prefix_tokens: int = 0
    affinity_started_at: float = 0.0
    prefix_ready_at: float = 0.0
    suffix_prefill_started_at: float = 0.0
    command_id: str = ""
    publish_outcome: str = "NOT_STARTED"
    dispatch_operation_ref: object | None = None
    suffix_operation_ref: object | None = None
    target_request_ref: object | None = None
    target_request_commit_ref: object | None = None
    target_block_table: list[int] = field(default_factory=list)
    transfer_source_ref: object | None = None
    transfer_target_ref: object | None = None

    def ttft_ms(self) -> float:
        """Return first-token latency in milliseconds, or -1 if unknown."""
        if self.first_token_at == 0.0:
            return -1.0
        return (self.first_token_at - self.arrived_at) * 1000.0

    def is_stuck(self, timeout_s: float) -> bool:
        """True if in KV_PENDING and transfer has exceeded timeout_s."""
        if self.state != SeqState.KV_PENDING:
            return False
        return (time.monotonic() - self.kv_sent_at) > timeout_s

    def is_prefill_stuck(self, timeout_s: float) -> bool:
        if self.state != SeqState.PREFILLING:
            return False
        return (time.monotonic() - self.prefill_start) > timeout_s

    def is_recompute_stuck(self, timeout_s: float) -> bool:
        if self.state != SeqState.RECOMPUTING:
            return False
        return (time.monotonic() - self.recompute_start) > timeout_s

    def is_decode_stuck(self, timeout_s: float) -> bool:
        if self.state != SeqState.DECODING:
            return False
        return (time.monotonic() - self.decode_start) > timeout_s

    def is_suffix_prefill_stuck(self, timeout_s: float) -> bool:
        if self.state != SeqState.PREFIX_PREFILLING:
            return False
        return (time.monotonic() - self.suffix_prefill_started_at) > timeout_s


@dataclass(slots=True)
class InstanceSlot:
    """Hold KV blocks for one sequence across its full decode lifetime."""
    instance_id: str
    seq_id: str | None = None
    block_table: list[int] = field(default_factory=list)
    allocated_at: float = 0.0

    def is_idle(self) -> bool:
        return self.seq_id is None

    def is_stale(self, timeout_s: float = 300.0) -> bool:
        """Return whether an occupied slot has exceeded its lease."""
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
        """Compute slots from usable GPU memory and average sequence KV size."""
        usable = (gpu_memory_gb - model_weight_gb) * safety_margin
        return max(1, int(usable / avg_seq_kv_gb))


@dataclass(slots=True)
class TransferTask:
    """KV transfer work item with flow-control metadata."""
    req_id:      str
    src:         str
    dst:         str
    kv_size:     int
    operation_id: str = ""
    src_epoch: str = ""
    dst_epoch: str = ""
    src_block_ids: tuple[int, ...] = ()
    dst_block_ids: tuple[int, ...] = ()
    target_request_ref: object | None = None
    target_request_commit_ref: object | None = None
    token_ids: tuple[int, ...] = ()
    sampling_params: dict = field(default_factory=dict)
    first_token: int | None = None
    transfer_source_ref: object | None = None
    transfer_target_ref: object | None = None
    correctness_path: str = ""
    priority:    int = 1
    enqueued_at: float = field(default_factory=time.monotonic)
    on_complete: object = None  # Callable[[], None] | None
    operation_id: str = ""


def _validate_transition(current: SeqState, new: SeqState) -> None:
    """Reject transitions that skip a control-plane phase."""
    VALID: dict[SeqState, set[SeqState]] = {
        SeqState.WAITING:    {SeqState.PREFILLING, SeqState.AFFINITY_LOADING},
        SeqState.PREFILLING: {SeqState.KV_PENDING},
        SeqState.AFFINITY_LOADING: {SeqState.PREFIX_PREFILLING, SeqState.WAITING},
        SeqState.PREFIX_PREFILLING: {SeqState.DECODING, SeqState.WAITING},
        SeqState.KV_PENDING: {SeqState.DECODING, SeqState.RECOMPUTING},
        SeqState.RECOMPUTING: {SeqState.DECODING},
        SeqState.DECODING:   {SeqState.FINISHED, SeqState.ABORTED},
    }
    allowed = VALID.get(current, set()) | {SeqState.ABORTED}
    assert new in allowed, (
        f"illegal transition {current.name} to {new.name}; "
        f"allowed: {[s.name for s in allowed]}"
    )


class RequestTracker:
    """Track explicit request phases and their deadline timestamps."""

    def __init__(self, metrics) -> None:
        self.metrics = metrics
        self._requests: dict[str, RequestInfo] = {}
        self._cleanup_tasks: dict[str, asyncio.Task] = {}

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

    def set_publish_outcome(self, req_id: str, outcome: str) -> None:
        assert outcome in {"NOT_STARTED", "ACKED", "UNKNOWN"}
        self._requests[req_id].publish_outcome = outcome

    def get_or_create_cleanup_task(self, req_id: str, factory) -> asyncio.Task:
        """CAS-create one supervisor-owned cleanup task per request."""
        existing = self._cleanup_tasks.get(req_id)
        if existing is not None:
            return existing
        task = asyncio.create_task(factory())
        self._cleanup_tasks[req_id] = task

        def forget(done: asyncio.Task) -> None:
            if self._cleanup_tasks.get(req_id) is done:
                self._cleanup_tasks.pop(req_id, None)

        task.add_done_callback(forget)
        return task

    def cleanup_task(self, req_id: str) -> asyncio.Task | None:
        return self._cleanup_tasks.get(req_id)

    async def quiesce_cleanup_tasks(self) -> None:
        """Cancel and join every supervisor cleanup before replacement freeze."""
        tasks = list(self._cleanup_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def clear_cleanup_task(self, req_id: str, task: asyncio.Task) -> None:
        if self._cleanup_tasks.get(req_id) is task:
            self._cleanup_tasks.pop(req_id, None)

    def transition(self, req_id: str, new_state: SeqState, **kwargs) -> None:
        """Advance a request and update its phase timestamps."""
        req = self._requests[req_id]
        _validate_transition(req.state, new_state)

        now = time.monotonic()
        if new_state == SeqState.PREFILLING:
            req.prefill_start = now
            req.dispatch_attempts += 1
        elif new_state == SeqState.KV_PENDING:
            req.kv_sent_at = now
        elif new_state == SeqState.AFFINITY_LOADING:
            req.affinity_started_at = now
        elif new_state == SeqState.PREFIX_PREFILLING:
            req.prefix_ready_at = now
            req.suffix_prefill_started_at = now
        elif new_state == SeqState.RECOMPUTING:
            req.recompute_start = now
        elif new_state == SeqState.DECODING:
            req.decode_start = now
        elif new_state in (SeqState.FINISHED, SeqState.ABORTED):
            req.finished_at = now

        for k, v in kwargs.items():
            setattr(req, k, v)

        req.state = new_state

    def get_stuck_requests(self, timeout_s: float) -> list[RequestInfo]:
        """Return all KV_PENDING requests that have exceeded timeout_s.

        Sorted by kv_sent_at ascending (oldest first).
        """
        stuck = [r for r in self._requests.values() if r.is_stuck(timeout_s)]
        return sorted(stuck, key=lambda r: r.kv_sent_at)

    def get_stuck_prefills(self, timeout_s: float) -> list[RequestInfo]:
        stuck = [
            request for request in self._requests.values()
            if request.is_prefill_stuck(timeout_s)
        ]
        return sorted(stuck, key=lambda request: request.prefill_start)

    def get_stuck_recomputes(self, timeout_s: float) -> list[RequestInfo]:
        stuck = [
            request for request in self._requests.values()
            if request.is_recompute_stuck(timeout_s)
        ]
        return sorted(stuck, key=lambda request: request.recompute_start)

    def get_stuck_decodes(self, timeout_s: float) -> list[RequestInfo]:
        stuck = [
            request for request in self._requests.values()
            if request.is_decode_stuck(timeout_s)
        ]
        return sorted(stuck, key=lambda request: request.decode_start)

    def get_stuck_suffix_prefills(self, timeout_s: float) -> list[RequestInfo]:
        stuck = [
            request for request in self._requests.values()
            if request.is_suffix_prefill_stuck(timeout_s)
        ]
        return sorted(stuck, key=lambda request: request.suffix_prefill_started_at)

    def record_first_token(self, req_id: str) -> bool:
        """Record TTFT once for an actively decoding request."""
        req = self._requests.get(req_id)
        if req is None or req.state != SeqState.DECODING:
            return False
        if req.first_token_at != 0.0:
            return False
        req.first_token_at = time.monotonic()
        self.metrics.observe(
            "request_ttft_ms", req.ttft_ms(), labels={"state": "DECODING"}
        )
        self.metrics.observe(
            "gateway_first_token_stage_latency_ms",
            req.ttft_ms(),
            labels={
                "path": "affinity" if req.cached_prefix_tokens > 0 else "cold"
            },
        )
        return True

    def iter_by_state(self, state: SeqState) -> Iterator[RequestInfo]:
        """Iterate over all requests in a given state."""
        return (r for r in self._requests.values() if r.state == state)

    def all_requests(self) -> list[RequestInfo]:
        """Return a stable snapshot for shutdown cleanup."""
        return list(self._requests.values())

    def __len__(self) -> int:
        return len(self._requests)

    def __contains__(self, req_id: str) -> bool:
        return req_id in self._requests

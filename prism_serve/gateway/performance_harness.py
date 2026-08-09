"""Strict performance admission and a bounded request-evidence ledger."""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass


PERFORMANCE_REQUEST_SCHEMA = "prism.performance_request/v1"
PERFORMANCE_EVIDENCE_SCHEMA = "prism.performance_request_evidence/v1"
PERFORMANCE_TRACE_CAP = 8192
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$", re.ASCII)


class PerformanceTraceCapacity(RuntimeError):
    """The fixed performance trace ledger is full."""


class PerformanceTraceConflict(RuntimeError):
    """The request id already exists in this Gateway world."""


@dataclass(slots=True, frozen=True)
class PerformanceRequest:
    request_id: str
    expected_input_tokens: int
    expected_output_tokens: int
    input_token_ids: tuple[int, ...]
    raw_content_sha256: str
    input_token_ids_sha256: str
    model_profile: dict[str, object]


@dataclass(slots=True)
class _ActiveTrace:
    request: PerformanceRequest
    world_identity: dict[str, object]
    stream_terminal_observed: bool = False
    detached: bool = False
    stream_error: str | None = None


def _token_ids_sha256(token_ids: list[int] | tuple[int, ...]) -> str:
    payload = json.dumps(
        list(token_ids), ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _freeze_world_identity(value: dict[str, object]) -> dict[str, object]:
    """Validate and copy the 2P2D identity observed at admission."""
    if set(value) != {
        "gateway", "topology_generation", "affinity_enabled", "workers"
    }:
        raise ValueError("performance world identity has unexpected fields")
    gateway = value.get("gateway")
    if not isinstance(gateway, dict) or set(gateway) != {"pod_uid", "clock_epoch"}:
        raise ValueError("performance Gateway identity is incomplete")
    pod_uid = gateway.get("pod_uid")
    clock_epoch = gateway.get("clock_epoch")
    if not isinstance(pod_uid, str) or not pod_uid \
            or not isinstance(clock_epoch, str) or not clock_epoch:
        raise ValueError("performance Gateway identity fields must be non-empty")
    topology_generation = value.get("topology_generation")
    if not isinstance(topology_generation, str) or not topology_generation:
        raise ValueError("performance topology_generation must be non-empty")
    affinity_enabled = value.get("affinity_enabled")
    if type(affinity_enabled) is not bool:
        raise ValueError("performance affinity_enabled must be boolean")
    workers = value.get("workers")
    expected_ranks = {"p0": 0, "p1": 1, "d0": 2, "d1": 3}
    if not isinstance(workers, dict) or set(workers) != set(expected_ranks):
        raise ValueError("performance world must identify p0, p1, d0 and d1")
    frozen_workers: dict[str, dict[str, object]] = {}
    seen_pods: set[str] = set()
    seen_epochs: set[str] = set()
    for instance_id, expected_rank in expected_ranks.items():
        identity = workers.get(instance_id)
        if not isinstance(identity, dict) or set(identity) != {
            "pod_uid", "instance_epoch", "global_rank"
        }:
            raise ValueError(f"performance worker identity is incomplete: {instance_id}")
        worker_pod_uid = identity.get("pod_uid")
        instance_epoch = identity.get("instance_epoch")
        global_rank = identity.get("global_rank")
        if not isinstance(worker_pod_uid, str) or not worker_pod_uid \
                or not isinstance(instance_epoch, str) \
                or not instance_epoch.startswith(f"{worker_pod_uid}:") \
                or len(instance_epoch) == len(worker_pod_uid) + 1 \
                or type(global_rank) is not int or global_rank != expected_rank:
            raise ValueError(f"performance worker identity is invalid: {instance_id}")
        if worker_pod_uid in seen_pods or instance_epoch in seen_epochs:
            raise ValueError("performance worker identities must be unique")
        seen_pods.add(worker_pod_uid)
        seen_epochs.add(instance_epoch)
        frozen_workers[instance_id] = {
            "pod_uid": worker_pod_uid,
            "instance_epoch": instance_epoch,
            "global_rank": global_rank,
        }
    return {
        "gateway": {"pod_uid": pod_uid, "clock_epoch": clock_epoch},
        "topology_generation": topology_generation,
        "affinity_enabled": affinity_enabled,
        "workers": frozen_workers,
    }


def parse_performance_request(
    body: dict[str, object],
    *,
    encoder: object,
    runtime_model: str,
    model_profile: dict[str, object],
) -> PerformanceRequest:
    """Validate one raw user prompt and tokenize it with the pinned encoder."""
    request_id = body.get("request_id")
    if not isinstance(request_id, str) or _REQUEST_ID.fullmatch(request_id) is None:
        raise ValueError("request_id does not match the performance harness format")
    if body.get("model") != runtime_model:
        raise ValueError("performance model does not match runtime")
    if body.get("stream") is not True:
        raise ValueError("performance harness requires stream=true")
    temperature = body.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
            or float(temperature) != 0.0:
        raise ValueError("performance harness requires temperature=0")
    if body.get("ignore_eos") is not True:
        raise ValueError("performance harness requires ignore_eos=true")
    if "input_token_ids" in body:
        raise ValueError("performance harness accepts raw text, not input_token_ids")

    envelope = body.get("prism_performance")
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema_version", "expected_input_tokens", "expected_output_tokens"
    }:
        raise ValueError("prism_performance has unexpected fields")
    if envelope.get("schema_version") != PERFORMANCE_REQUEST_SCHEMA:
        raise ValueError("unsupported prism_performance schema_version")
    expected_input = _require_int(
        envelope.get("expected_input_tokens"),
        name="expected_input_tokens",
        minimum=1,
        maximum=4096,
    )
    expected_output = _require_int(
        envelope.get("expected_output_tokens"),
        name="expected_output_tokens",
        minimum=2,
        maximum=256,
    )
    if type(body.get("max_tokens")) is not int \
            or body.get("max_tokens") != expected_output:
        raise ValueError("max_tokens must equal expected_output_tokens")

    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("performance messages must contain exactly one user item")
    message = messages[0]
    if not isinstance(message, dict) or set(message) != {"role", "content"} \
            or message.get("role") != "user" \
            or not isinstance(message.get("content"), str):
        raise ValueError("performance prompt must be one raw user string")
    content = str(message["content"])
    token_ids = list(encoder.encode(content, add_special_tokens=False))
    if len(token_ids) != expected_input:
        raise ValueError("tokenized input length does not match expected_input_tokens")
    if not token_ids or not all(
        type(token) is int and 0 <= token < 2**64 for token in token_ids
    ):
        raise ValueError("tokenizer returned a non-uint64 token id")

    return PerformanceRequest(
        request_id=request_id,
        expected_input_tokens=expected_input,
        expected_output_tokens=expected_output,
        input_token_ids=tuple(token_ids),
        raw_content_sha256=(
            "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
        ),
        input_token_ids_sha256=_token_ids_sha256(token_ids),
        model_profile=dict(model_profile),
    )


class PerformanceTraceRegistry:
    """Retain active, terminal, and tombstone entries for one Gateway world."""

    def __init__(self, capacity: int = PERFORMANCE_TRACE_CAP) -> None:
        if capacity != PERFORMANCE_TRACE_CAP:
            raise ValueError(f"performance trace capacity must be {PERFORMANCE_TRACE_CAP}")
        self.capacity = capacity
        self._active: dict[str, _ActiveTrace] = {}
        self._terminal: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._tombstones: OrderedDict[str, None] = OrderedDict()

    def _contains(self, request_id: str) -> bool:
        return request_id in self._active \
            or request_id in self._terminal \
            or request_id in self._tombstones

    def reserve(
        self, request: PerformanceRequest, *, world_identity: dict[str, object]
    ) -> None:
        if self._contains(request.request_id):
            raise PerformanceTraceConflict("performance request id already exists")
        if self.counts()["total"] >= self.capacity:
            raise PerformanceTraceCapacity("performance trace capacity exhausted")
        self._active[request.request_id] = _ActiveTrace(
            request=request,
            world_identity=_freeze_world_identity(world_identity),
        )

    def rollback_uncommitted(self, request_id: str) -> None:
        self._active.pop(request_id, None)

    def observe_stream_terminal(self, request_id: str) -> None:
        trace = self._active.get(request_id)
        if trace is not None:
            trace.stream_terminal_observed = True

    def mark_detached(self, request_id: str) -> None:
        trace = self._active.get(request_id)
        if trace is not None:
            trace.detached = True

    def mark_stream_error(self, request_id: str, error: str) -> None:
        trace = self._active.get(request_id)
        if trace is not None:
            trace.stream_error = error

    def state(self, request_id: str) -> str:
        if request_id in self._active:
            return "active"
        if request_id in self._terminal:
            return "terminal"
        if request_id in self._tombstones:
            return "tombstone"
        return "unknown"

    def active_trace(self, request_id: str) -> _ActiveTrace | None:
        return self._active.get(request_id)

    def terminal_trace(self, request_id: str) -> dict[str, object] | None:
        value = self._terminal.get(request_id)
        return dict(value) if value is not None else None

    def finalize(
        self,
        request_id: str,
        *,
        output_token_ids: list[int],
        runtime_error: str | None,
        route_evidence: dict[str, object],
    ) -> dict[str, object]:
        existing = self._terminal.get(request_id)
        if existing is not None:
            return dict(existing)
        active = self._active.get(request_id)
        if active is None:
            raise KeyError(request_id)
        if not all(
            type(token) is int and 0 <= token < 2**64 for token in output_token_ids
        ):
            raise ValueError("output contains a non-uint64 token id")

        runtime_error = runtime_error or active.stream_error
        status = "PASS"
        error_code: str | None = None
        if active.detached:
            status = "CANCELLED"
            error_code = "CLIENT_DISCONNECTED"
        elif runtime_error is not None:
            status = "FAILED"
            error_code = "SERVER_ERROR"
        elif len(output_token_ids) != active.request.expected_output_tokens:
            status = "FAILED"
            error_code = "OUTPUT_TOKEN_COUNT_MISMATCH"

        transport = route_evidence["transport"]
        route = {
            "path": route_evidence["path"],
            "source": route_evidence["route"]["source"],
            "target": route_evidence["route"]["target"],
            "cached_prefix_tokens": route_evidence["cached_prefix_tokens"],
            "suffix_tokens": route_evidence["suffix_tokens"],
            "src_block_ids": list(route_evidence["src_block_ids"]),
            "dst_block_ids": list(route_evidence["dst_block_ids"]),
            "mapping": list(route_evidence["mapping"]),
        }
        completed_bytes = int(transport["completed_bytes"])
        started_ns = transport.get("transfer_started_ns")
        terminal_ns = transport.get("transfer_terminal_ns")
        elapsed_ns = (
            terminal_ns - started_ns
            if type(started_ns) is int and type(terminal_ns) is int
            else None
        )
        if elapsed_ns is not None and elapsed_ns <= 0:
            raise ValueError("Gateway transfer timing is not monotonic")
        transfer = {
            "selected_mode": transport["selected_mode"],
            "pair_id": transport.get("pair_id"),
            "completed_bytes": completed_bytes,
            "work_terminal": transport.get("work_terminal") is True,
            "cuda_terminal": transport.get("cuda_terminal") is True,
            "gateway_clock_epoch": transport.get("gateway_clock_epoch"),
            "transfer_started_ns": started_ns,
            "transfer_terminal_ns": terminal_ns,
            "gateway_observed_transfer_wall_ms": (
                elapsed_ns / 1_000_000 if elapsed_ns is not None else None
            ),
            "effective_bytes_per_s": (
                completed_bytes * 1_000_000_000 / elapsed_ns
                if elapsed_ns is not None else None
            ),
        }
        if transfer["gateway_clock_epoch"] \
                != active.world_identity["gateway"]["clock_epoch"]:
            raise ValueError("Gateway clock epoch changed within performance request")
        value: dict[str, object] = {
            "schema_version": PERFORMANCE_EVIDENCE_SCHEMA,
            "request_id": request_id,
            "status": status,
            "error_code": error_code,
            "runtime_error": runtime_error,
            "model_profile": dict(active.request.model_profile),
            "world_identity": active.world_identity,
            "request": {
                "prompt_format": "single_user_raw_text/v1",
                "chat_template_applied": False,
                "expected_input_tokens": active.request.expected_input_tokens,
                "expected_output_tokens": active.request.expected_output_tokens,
                "actual_input_tokens": len(active.request.input_token_ids),
                "actual_output_tokens": len(output_token_ids),
                "raw_content_sha256": active.request.raw_content_sha256,
                "input_token_ids_sha256": active.request.input_token_ids_sha256,
                "output_token_ids": list(output_token_ids),
                "output_token_ids_sha256": _token_ids_sha256(output_token_ids),
            },
            "route": route,
            "transfer": transfer,
            "resources": {
                "terminal": True,
                "request_active": False,
                "work_terminal": transfer["work_terminal"],
                "cuda_terminal": transfer["cuda_terminal"],
            },
            "stream": {
                "terminal_observed": active.stream_terminal_observed,
                "detached": active.detached,
            },
        }
        self._active.pop(request_id)
        self._terminal[request_id] = value
        return dict(value)

    def acknowledge(self, request_id: str) -> str:
        if request_id in self._active:
            return "active"
        if request_id in self._tombstones:
            return "acked"
        if request_id not in self._terminal:
            return "unknown"
        self._terminal.pop(request_id)
        self._tombstones[request_id] = None
        return "acked"

    def counts(self) -> dict[str, int]:
        active = len(self._active)
        terminal = len(self._terminal)
        tombstones = len(self._tombstones)
        return {
            "active": active,
            "unacked_terminal": terminal,
            "acked_tombstones": tombstones,
            "total": active + terminal + tombstones,
            "capacity": self.capacity,
        }

"""Gated fault injection and evidence routes for the fixed 2P2D snapshot."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import secrets
import time
import uuid


AUTH_HEADER = "x-prism-week12-token"
EXPECTED_INPUT_TOKENS = 769
EXPECTED_OUTPUT_TOKENS = 32
EXPECTED_CACHED_TOKENS = 512


@dataclass(slots=True, frozen=True)
class CorrectnessRoute:
    path: str
    source_instance: str
    target_instance: str
    cached_prefix_tokens: int
    include_in_packet: bool = True


ROUTES = {
    "cold": CorrectnessRoute("cold", "p0", "d1", 0),
    "same_instance": CorrectnessRoute(
        "same_instance", "d1", "d1", EXPECTED_CACHED_TOKENS
    ),
    "cross_instance": CorrectnessRoute(
        "cross_instance", "d0", "d1", EXPECTED_CACHED_TOKENS
    ),
    # Seed through p1 so p0 remains genuinely cold for the final acceptance
    # request.  Reusing p0 here would let its local prefix cache accelerate the
    # request that the packet labels as the cold p0 -> d1 path.
    "seed_d1": CorrectnessRoute("seed_d1", "p1", "d1", 0, False),
    # A private warm-up request creates the d0 source used by the cross case.
    # It is deliberately not a schema case and is never included in a packet.
    "seed_d0": CorrectnessRoute("seed_d0", "p1", "d0", 0, False),
}


FAULT_CHECKPOINTS = {
    "nats_disconnect": "before_nats_dispatch",
    "nats_drop": "before_nats_dispatch",
    "nats_duplicate": "before_nats_dispatch",
    "nats_publish_unknown": "before_nats_dispatch",
    "rpc_response_loss_source": "after_infer_success_before_control_observe",
    "rpc_response_loss_target": "after_infer_success_before_control_observe",
    "finalize_response_loss_source": (
        "after_infer_success_before_control_observe"
    ),
    "finalize_response_loss_target": (
        "after_infer_success_before_control_observe"
    ),
    "worker_crash": "before_nccl_source_start",
    "nccl_timeout": "before_nccl_source_start",
    "gateway_restart": "after_nccl_pair_mutations_accepted",
}


_POST_SUCCESS_FAULT_MATCH = {
    "rpc_response_loss_source": ("/v1/transfers/start", "source"),
    "rpc_response_loss_target": ("/v1/transfers/prepare-receive", "target"),
    "finalize_response_loss_source": ("/v1/cleanup/finalize", "source"),
    "finalize_response_loss_target": ("/v1/cleanup/finalize", "target"),
}


class FaultInjectionGate:
    """One-shot, authenticated correctness checkpoint.

    The gate pauses only a request explicitly routed through the correctness
    harness.  It records observed endpoint refs allocated by production code;
    it never manufactures release evidence or a terminal result.
    """

    def __init__(self, *, timeout_s: float = 300.0) -> None:
        if timeout_s <= 0:
            raise ValueError("fault gate timeout must be positive")
        self.timeout_s = timeout_s
        self._lock = asyncio.Lock()
        self._active: dict[str, object] | None = None
        self._release: asyncio.Event | None = None

    async def arm(self, fault_kind: str) -> dict[str, object]:
        checkpoint = FAULT_CHECKPOINTS.get(fault_kind)
        if checkpoint is None:
            raise ValueError("unsupported fault kind")
        async with self._lock:
            if self._active is not None and self._active["state"] in {
                "ARMED", "REACHED"
            }:
                raise RuntimeError("a fault gate is already active")
            self._release = asyncio.Event()
            self._active = {
                "fault_run_id": uuid.uuid4().hex,
                "fault_kind": fault_kind,
                "checkpoint": checkpoint,
                "timeout_s": self.timeout_s,
                "state": "ARMED",
                "armed_at_ns": time.monotonic_ns(),
                "reached_at_ns": None,
                "released_at_ns": None,
                "details": {},
                "events": [],
            }
            return dict(self._active)

    async def arrive(
        self, checkpoint: str, details: dict[str, object]
    ) -> dict[str, object] | None:
        async with self._lock:
            if (
                self._active is None
                or self._active["state"] != "ARMED"
                or self._active["checkpoint"] != checkpoint
            ):
                return None
            fault_kind = str(self._active["fault_kind"])
            match = _POST_SUCCESS_FAULT_MATCH.get(fault_kind)
            if match is not None and (
                details.get("path") != match[0]
                or details.get("route_role") != match[1]
                or not details.get("request_id")
                or not isinstance(details.get("endpoint_ref"), dict)
                or type(details.get("response_status")) is not int
                or not 200 <= int(details["response_status"]) < 300
                or not str(details.get("request_digest") or "").startswith("sha256:")
                or not str(details.get("response_digest") or "").startswith("sha256:")
            ):
                # Other correctness traffic can use the same shared HTTP
                # client while a fault is armed.  Only the exact mutation and
                # cross-path role selected by the fault kind may consume it.
                return None
            if fault_kind.startswith("finalize_response_loss_") and (
                not details.get("cleanup_id")
                or not details.get("cleanup_operation_id")
                or details.get("cleanup_operation_id")
                != details["endpoint_ref"].get("operation_id")
                or not details.get("cleanup_lease_id")
                or not str(
                    details.get("cleanup_payload_digest") or ""
                ).startswith("sha256:")
            ):
                return None
            release = self._release
            assert release is not None
            self._active["state"] = "REACHED"
            self._active["reached_at_ns"] = time.monotonic_ns()
            self._active["details"] = dict(details)
            fault_run_id = str(self._active["fault_run_id"])
        try:
            await asyncio.wait_for(release.wait(), timeout=self.timeout_s)
        except TimeoutError:
            async with self._lock:
                if self._active is not None and self._active.get(
                    "fault_run_id"
                ) == fault_run_id:
                    self._active["state"] = "EXPIRED"
            raise RuntimeError("fault gate expired before release")
        except asyncio.CancelledError:
            async with self._lock:
                if self._active is not None and self._active.get(
                    "fault_run_id"
                ) == fault_run_id:
                    self._active["state"] = "CANCELLED"
            raise
        async with self._lock:
            if self._active is not None and self._active.get(
                "fault_run_id"
            ) == fault_run_id and self._active.get("state") != "SEALED":
                self._active["state"] = "RELEASED"
                self._active["released_at_ns"] = time.monotonic_ns()
            if self._active is not None and self._active.get(
                "fault_run_id"
            ) == fault_run_id:
                return dict(self._active)
        return None

    async def release(self, fault_run_id: str) -> dict[str, object]:
        async with self._lock:
            if self._active is None or self._active.get(
                "fault_run_id"
            ) != fault_run_id:
                raise KeyError("unknown fault run")
            if self._active["state"] != "REACHED":
                raise RuntimeError("fault gate has not reached its checkpoint")
            assert self._release is not None
            self._release.set()
            return dict(self._active)

    async def snapshot(self) -> dict[str, object] | None:
        async with self._lock:
            if self._active is None:
                return None
            value = dict(self._active)
            value["details"] = dict(value.get("details") or {})
            value["events"] = [dict(item) for item in value.get("events") or []]
            return value

    def record_event(self, name: str, details: dict[str, object]) -> None:
        """Record an event observed by production cleanup/replacement code.

        Callers run on the Gateway event loop, so the append is atomic with
        respect to HTTP snapshot handling.  An inactive gate is deliberately a
        no-op; the harness cannot turn ordinary traffic into fault evidence.
        """
        active = self._active
        if active is None or active.get("state") not in {
            "REACHED", "RELEASED", "CANCELLED"
        }:
            return
        active_details = active.get("details")
        active_request_id = (
            str(active_details.get("request_id") or "")
            if isinstance(active_details, dict) else ""
        )
        active_operation_ids = {active_request_id} if active_request_id else set()
        if isinstance(active_details, dict):
            for role in ("source", "target"):
                ref = active_details.get(f"{role}_endpoint_ref")
                if isinstance(ref, dict) and ref.get("operation_id"):
                    active_operation_ids.add(str(ref["operation_id"]))
            ref = active_details.get("endpoint_ref")
            if isinstance(ref, dict) and ref.get("operation_id"):
                active_operation_ids.add(str(ref["operation_id"]))
        observed_operation_ids = {
            str(value) for value in details.get("operation_ids", ())
        } if isinstance(details.get("operation_ids"), (list, tuple)) else set()
        for key in ("operation_id", "request_id"):
            if details.get(key):
                observed_operation_ids.add(str(details[key]))
        if active_operation_ids and not active_operation_ids.intersection(
            observed_operation_ids
        ):
            return
        source_ref = (
            active_details.get("source_endpoint_ref")
            if isinstance(active_details, dict) else None
        )
        if not isinstance(source_ref, dict) and isinstance(active_details, dict):
            source_ref = active_details.get("endpoint_ref")
        producer_epoch = (
            str(source_ref.get("owner_generation") or "")
            if isinstance(source_ref, dict) else ""
        )
        if not producer_epoch:
            return
        events = active.get("events")
        if not isinstance(events, list):
            return
        events.append({
            "name": name,
            "value": time.monotonic_ns(),
            "clock": "gateway_monotonic",
            "unit": "ns",
            "producer_epoch": producer_epoch,
            "details": dict(details),
        })
        if name == "slot_released":
            active["state"] = "SEALED"
            active["sealed_at_ns"] = time.monotonic_ns()
            if self._release is not None:
                self._release.set()


def authorize(*, enabled: bool, configured_secret: str, supplied_secret: str) -> str:
    """Return ``ok``/``disabled``/``forbidden`` without leaking the token."""
    if not enabled:
        return "disabled"
    if not configured_secret or not supplied_secret:
        return "forbidden"
    return (
        "ok"
        if secrets.compare_digest(configured_secret, supplied_secret)
        else "forbidden"
    )


def parse_route(value: object) -> CorrectnessRoute:
    if not isinstance(value, dict) or set(value) != {"path"}:
        raise ValueError("week12_correctness must contain only path")
    path = value.get("path")
    if not isinstance(path, str) or path not in ROUTES:
        raise ValueError("unsupported correctness path")
    return ROUTES[path]


def validate_fixture(
    *, route: CorrectnessRoute, token_ids: list[int], sampling: dict[str, object]
) -> None:
    expected_input = 513 if route.path.startswith("seed_") else EXPECTED_INPUT_TOKENS
    expected_output = 1 if route.path.startswith("seed_") else EXPECTED_OUTPUT_TOKENS
    if len(token_ids) != expected_input:
        raise ValueError(
            f"{route.path} fixture requires {expected_input} input tokens"
        )
    if sampling != {
        "temperature": 0.0,
        "max_tokens": expected_output,
        "ignore_eos": True,
    }:
        raise ValueError(
            f"{route.path} fixture requires temperature=0, "
            f"max_tokens={expected_output} and ignore_eos=true"
        )

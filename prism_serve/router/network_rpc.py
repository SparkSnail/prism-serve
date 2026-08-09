"""Network-backed control protocol for disaggregated inference workers."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

from prism_serve.router.http_rpc import (
    AmbiguousRPCError,
    EndpointOperationRef,
    EndpointSequenceAllocator,
    HttpInferClient,
    InferRPCError,
    validate_release_snapshot,
)
from prism_serve.router.protocol import (
    CachedPrefixPlan,
    MappedTransferStatus,
    PreparedPrefix,
    PrefixOperationStatus,
    ResolvedPrefix,
)


@dataclass(slots=True, frozen=True)
class OwnerTakeoverListOperationEvidence:
    endpoint_ref: EndpointOperationRef
    state: str
    resources_held: bool
    held_resource_kinds: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class OwnerTakeoverListEvidence:
    instance_id: str
    instance_epoch: str
    owner_generation: str
    complete: bool
    endpoint_refs: tuple[EndpointOperationRef, ...]
    operations: tuple[OwnerTakeoverListOperationEvidence, ...]
    report_digest: str


@dataclass(slots=True, frozen=True)
class OwnerTakeoverQueryEvidence:
    instance_id: str
    instance_epoch: str
    owner_generation: str
    operation_id: str
    endpoint_ref: EndpointOperationRef
    state: str
    resources_held: bool
    held_resource_kinds: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class OwnerTakeoverFinalizeEvidence:
    instance_id: str
    cleanup_id: str
    operation_id: str
    lease_id: str
    endpoint_epoch: str
    request_endpoint_refs: tuple[EndpointOperationRef, ...]
    released_resource_kinds: tuple[str, ...]
    released_counts: tuple[tuple[str, int], ...]
    resources_held_after: bool
    payload_digest: str


@dataclass(slots=True, frozen=True)
class OwnerTakeoverOperationAudit:
    instance_id: str
    old_owner: str
    endpoint_ref: EndpointOperationRef
    listed_state: str
    query_confirmed: bool
    terminal_state: str
    abort_attempted: bool
    resources_held_before_finalize: bool
    held_resource_kinds: tuple[str, ...]
    finalize_acknowledged: bool
    query_evidence: OwnerTakeoverQueryEvidence
    finalize_evidence: OwnerTakeoverFinalizeEvidence | None


@dataclass(slots=True, frozen=True)
class OwnerTakeoverAudit:
    new_owner: str
    instances: tuple[str, ...]
    old_owners: tuple[tuple[str, str], ...]
    operation_list_evidence: tuple[OwnerTakeoverListEvidence, ...]
    operations: tuple[OwnerTakeoverOperationAudit, ...]
    finalized_operation_ids: tuple[str, ...]
    retired_owners: tuple[tuple[str, str], ...]
    activated_instances: tuple[str, ...]
    confirmed_active_owners: tuple[tuple[str, str], ...]


@dataclass(slots=True, frozen=True)
class _ParsedOperationSnapshot:
    endpoint_ref: EndpointOperationRef
    state: str
    resources_held: bool
    held_resource_kinds: tuple[str, ...]


_ENDPOINT_REF_FIELDS = frozenset({
    "topology_generation",
    "owner_generation",
    "operation_seq",
    "target_instance",
    "target_worker_epoch",
    "operation_id",
    "payload_digest",
})
_OPERATION_STATES = frozenset({
    "PREPARED", "RUNNING", "COMPLETED", "FENCED", "UNKNOWN"
})
_SOURCE_ENDPOINT_KINDS = frozenset({
    "prefix.resolve", "request.source", "transfer.source",
})
_TARGET_ENDPOINT_KINDS = frozenset({
    "prefix.commit",
    "prefix.prepare",
    "request.commit",
    "request.prepare",
    "suffix",
    "transfer.target",
})


def _parse_endpoint_ref(
    value: object, *, context: str
) -> EndpointOperationRef:
    if not isinstance(value, dict) or set(value) != _ENDPOINT_REF_FIELDS:
        raise RuntimeError(f"{context} endpoint ref is incomplete")
    for field in _ENDPOINT_REF_FIELDS - {"operation_seq"}:
        if not isinstance(value[field], str) or not value[field]:
            raise RuntimeError(f"{context} endpoint ref field {field} is invalid")
    operation_seq = value["operation_seq"]
    if isinstance(operation_seq, bool) or not isinstance(operation_seq, int) \
            or operation_seq <= 0:
        raise RuntimeError(f"{context} endpoint operation sequence is invalid")
    return EndpointOperationRef(**value)


def _parse_operation_snapshot(
    value: object,
    *,
    instance_id: str,
    owner_generation: str,
    instance_epoch: str,
    context: str,
    expected_ref: EndpointOperationRef | None = None,
) -> _ParsedOperationSnapshot:
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} operation entry is not an object")
    ref = _parse_endpoint_ref(value.get("endpoint_ref"), context=context)
    if expected_ref is not None and ref != expected_ref:
        raise RuntimeError(f"{context} returned another endpoint ref")
    if (
        ref.target_instance != instance_id
        or ref.owner_generation != owner_generation
        or ref.target_worker_epoch != instance_epoch
        or not ref.operation_id
    ):
        raise RuntimeError(f"{context} returned a foreign endpoint ref")
    state = value.get("state")
    resources_held = value.get("resources_held")
    kinds = value.get("held_resource_kinds")
    if not isinstance(state, str) or state not in _OPERATION_STATES:
        raise RuntimeError(f"{context} operation state is invalid")
    if not isinstance(resources_held, bool):
        raise RuntimeError(f"{context} resources_held is invalid")
    if (
        not isinstance(kinds, (list, tuple))
        or any(not isinstance(kind, str) or not kind for kind in kinds)
        or len(set(kinds)) != len(kinds)
    ):
        raise RuntimeError(f"{context} held resource kinds are invalid")
    held_resource_kinds = tuple(sorted(kinds))
    if resources_held != bool(held_resource_kinds):
        raise RuntimeError(f"{context} held resource state is inconsistent")
    return _ParsedOperationSnapshot(
        endpoint_ref=ref,
        state=state,
        resources_held=resources_held,
        held_resource_kinds=held_resource_kinds,
    )


def _parse_complete_operation_report(
    report: object,
    *,
    instance_id: str,
    owner_generation: str,
    instance_epoch: str,
    max_entries: int | None = None,
) -> tuple[_ParsedOperationSnapshot, ...]:
    if not isinstance(report, dict) or report.get("complete") is not True:
        raise RuntimeError("old-owner operation report is not complete")
    if report.get("instance_epoch") != instance_epoch:
        raise RuntimeError("old-owner operation report has another worker epoch")
    operations = report.get("operations")
    if not isinstance(operations, list):
        raise RuntimeError("complete old-owner operation report lacks operations")
    if max_entries is not None and len(operations) > max_entries:
        raise RuntimeError("old-owner operation audit exceeds configured cap")
    return tuple(
        _parse_operation_snapshot(
            value,
            instance_id=instance_id,
            owner_generation=owner_generation,
            instance_epoch=instance_epoch,
            context="old-owner operation list",
        )
        for value in operations
    )


def _operation_list_evidence(
    *,
    instance_id: str,
    owner_generation: str,
    instance_epoch: str,
    operations: tuple[_ParsedOperationSnapshot, ...],
) -> OwnerTakeoverListEvidence:
    ordered = tuple(sorted(
        operations,
        key=lambda value: (
            value.endpoint_ref.target_instance,
            value.endpoint_ref.operation_seq,
            value.endpoint_ref.operation_id,
            value.endpoint_ref.payload_digest,
            value.endpoint_ref.owner_generation,
            value.endpoint_ref.target_worker_epoch,
            value.endpoint_ref.topology_generation,
        ),
    ))
    snapshots = tuple(
        OwnerTakeoverListOperationEvidence(
            endpoint_ref=value.endpoint_ref,
            state=value.state,
            resources_held=value.resources_held,
            held_resource_kinds=value.held_resource_kinds,
        )
        for value in ordered
    )
    canonical_payload = {
        "instance_epoch": instance_epoch,
        "complete": True,
        "operations": [
            {
                "endpoint_ref": asdict(value.endpoint_ref),
                "state": value.state,
                "resources_held": value.resources_held,
                "held_resource_kinds": list(value.held_resource_kinds),
            }
            for value in snapshots
        ],
    }
    report_bytes = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return OwnerTakeoverListEvidence(
        instance_id=instance_id,
        instance_epoch=instance_epoch,
        owner_generation=owner_generation,
        complete=True,
        endpoint_refs=tuple(value.endpoint_ref for value in snapshots),
        operations=snapshots,
        report_digest="sha256:" + hashlib.sha256(report_bytes).hexdigest(),
    )


def _finalize_evidence(
    instance_id: str,
    endpoint_refs: tuple[EndpointOperationRef, ...],
    value: dict[str, object],
) -> OwnerTakeoverFinalizeEvidence:
    return OwnerTakeoverFinalizeEvidence(
        instance_id=instance_id,
        cleanup_id=value["cleanup_id"],
        operation_id=value["operation_id"],
        lease_id=value["lease_id"],
        endpoint_epoch=value["endpoint_epoch"],
        request_endpoint_refs=endpoint_refs,
        released_resource_kinds=tuple(value["released_resource_kinds"]),
        released_counts=tuple(
            (kind, count) for kind, count in value["released_counts"]
        ),
        resources_held_after=value["resources_held_after"],
        payload_digest=value["payload_digest"],
    )


class NetworkControlRPC:
    week12_network_control = True

    def __init__(
        self,
        client: HttpInferClient,
        allocator: EndpointSequenceAllocator,
        worker_epochs: dict[str, str],
        *,
        query_interval_s: float = 0.05,
        operation_timeout_s: float = 30.0,
        block_size: int = 256,
        block_bytes: int = 0,
        active_operation_cap: int = 512,
        terminal_snapshot_cap: int = 4096,
        gateway_clock_epoch: str = "",
        metrics=None,
    ) -> None:
        if active_operation_cap <= 0 or terminal_snapshot_cap <= 0:
            raise ValueError("network control caps must be positive")
        self.client = client
        self.allocator = allocator
        self.worker_epochs = dict(worker_epochs)
        self.query_interval_s = query_interval_s
        self.operation_timeout_s = operation_timeout_s
        self.block_size = block_size
        self.block_bytes = block_bytes
        self.active_operation_cap = active_operation_cap
        self.terminal_snapshot_cap = terminal_snapshot_cap
        self.gateway_clock_epoch = gateway_clock_epoch
        self.metrics = metrics
        self._refs: dict[tuple[str, str, str], EndpointOperationRef] = {}


        self._attempted_refs: set[EndpointOperationRef] = set()
        self._ambiguous_refs: set[EndpointOperationRef] = set()
        self._normal_tasks: dict[str, asyncio.Task] = {}
        self._request_metadata: dict[tuple[str, str], dict[str, object]] = {}
        self._cleanup_plans: dict[str, object] = {}
        self._retired_operations: OrderedDict[str, None] = OrderedDict()
        self._request_evidence: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._correctness_required: set[str] = set()
        # Kept through terminal output until resource cleanup retires the exact
        # operation.  Finalize response-loss injection happens after transfer
        # evidence has already been stored, so `_correctness_required` alone is
        # intentionally too short-lived for that checkpoint.
        self._correctness_operations: set[str] = set()
        # Transfer/prefix operation IDs and logical request IDs are separate
        # identity domains.  Response-loss injection resolves this explicit
        # ownership edge; it never assumes that the two strings are equal.
        self._correctness_operation_owners: dict[str, str] = {}
        self._correctness_fault_gate = None
        # A single active Gateway is the global launch sequencer for all five
        # overlapping NCCL pair groups.  Different ranks must never observe
        # concurrent group launches in different orders.
        self._nccl_sequence_lock = asyncio.Lock()
        self._nccl_sequence_poisoned = False
        self.release_evaluator = None

    def set_release_evaluator(self, evaluator) -> None:
        self.release_evaluator = evaluator

    def set_correctness_fault_gate(self, gate) -> None:
        self._correctness_fault_gate = gate
        install = getattr(self.client, "set_correctness_post_success_hook", None)
        if install is not None:
            install(self._correctness_post_success_hook if gate is not None else None)

    async def _correctness_post_success_hook(
        self, details: dict[str, object]
    ) -> bool:
        """Discard only an armed correctness mutation's successful response."""
        endpoint_ref = details.get("endpoint_ref")
        if not isinstance(endpoint_ref, dict):
            return False
        try:
            ref = _parse_endpoint_ref(
                endpoint_ref, context="correctness post-success hook"
            )
        except RuntimeError:
            return False
        path = str(details.get("path") or "")
        instance_id = str(details.get("instance_id") or "")
        if not instance_id or ref.target_instance != instance_id:
            return False
        operation_id = (
            str(details.get("cleanup_operation_id") or "")
            if path == "/v1/cleanup/finalize"
            else ref.operation_id
        )
        request_id = self._correctness_operation_owners.get(operation_id)
        if (
            request_id is None
            or request_id not in self._correctness_operations
        ):
            return False
        ref_kind = self._correctness_ref_kind(instance_id, ref)
        if path == "/v1/transfers/prepare-receive":
            if ref_kind != "transfer.target":
                return False
            route_role = "target"
        elif path == "/v1/transfers/start":
            if ref_kind != "transfer.source":
                return False
            route_role = "source"
        elif path == "/v1/cleanup/finalize":
            if ref_kind in _SOURCE_ENDPOINT_KINDS:
                route_role = "source"
            elif ref_kind in _TARGET_ENDPOINT_KINDS:
                route_role = "target"
            else:
                return False
        else:
            return False
        observed = dict(details)
        observed.update({
            "request_id": request_id,
            "route_role": route_role,
        })
        reached = await self.correctness_fault_checkpoint(
            "after_infer_success_before_control_observe", observed
        )
        if reached is None:
            return False
        self.record_correctness_fault_event(
            "response_loss_injected",
            {
                **observed,
                "fault_kind": str(reached.get("fault_kind") or ""),
            },
        )
        return True

    def _correctness_ref_kind(
        self, instance_id: str, ref: EndpointOperationRef
    ) -> str | None:
        """Return the role-bearing kind only for the exact observed ref."""
        kinds = {
            kind
            for (kind, instance, _), stored_ref in self._refs.items()
            if instance == instance_id and stored_ref == ref
        }
        for plan in self._cleanup_plans.values():
            for predicate in getattr(plan, "predicates", ()):
                if (
                    getattr(predicate, "endpoint_ref", None) == ref
                    and ref.target_instance == instance_id
                ):
                    kinds.add(str(getattr(predicate, "name", "")))
        return kinds.pop() if len(kinds) == 1 else None

    async def correctness_fault_checkpoint(
        self, checkpoint: str, details: dict[str, object]
    ) -> dict[str, object] | None:
        gate = self._correctness_fault_gate
        if gate is not None:
            return await gate.arrive(checkpoint, details)
        return None

    def record_correctness_fault_event(
        self, name: str, details: dict[str, object]
    ) -> None:
        gate = self._correctness_fault_gate
        if gate is not None:
            gate.record_event(name, details)

    async def wait_nats_command_fault_authority(
        self, endpoint_ref: dict[str, object], fault_kind: str
    ) -> dict[str, int]:
        """Wait until the worker has authoritatively observed the injected command.

        Cleanup must not race ahead of the NATS consumer for duplicate/unknown
        scenarios; otherwise the packet could claim exactly-once from publisher
        intent while the worker truth says the command never executed.
        """
        ref = EndpointOperationRef(**endpoint_ref)
        expected = {
            "nats_duplicate": (2, 1),
            "nats_publish_unknown": (1, 1),
        }.get(fault_kind)
        if expected is None:
            raise ValueError("worker authority wait is not valid for this fault")
        deadline = time.monotonic() + self.operation_timeout_s
        last: object = None
        while time.monotonic() < deadline:
            try:
                snapshot = await self.client.operation_ref_status(
                    ref.target_instance, ref
                )
            except asyncio.CancelledError:
                raise
            except AmbiguousRPCError as exc:
                last = {"error": f"{type(exc).__name__}: {exc}"}
                await asyncio.sleep(self.query_interval_s)
                continue
            except InferRPCError as exc:
                if exc.code != "PRECONDITION_FAILED":
                    raise
                last = {"error": f"{type(exc).__name__}: {exc}"}
                await asyncio.sleep(self.query_interval_s)
                continue
            last = snapshot
            if snapshot.get("endpoint_ref") != endpoint_ref:
                raise RuntimeError("worker returned another endpoint ref")
            delivery_count = snapshot.get("delivery_count")
            execution_count = snapshot.get("execution_count")
            if (delivery_count, execution_count) == expected:
                return {
                    "delivery_count": delivery_count,
                    "execution_count": execution_count,
                }
            if type(delivery_count) is int and delivery_count > expected[0]:
                raise RuntimeError("worker observed too many NATS deliveries")
            if type(execution_count) is int and execution_count > 1:
                raise RuntimeError("worker executed one endpoint ref more than once")
            await asyncio.sleep(self.query_interval_s)
        raise RuntimeError(
            f"{fault_kind} worker authority did not converge before timeout; "
            f"last={last}"
        )

    async def quiesce(self) -> tuple[BaseException, ...]:
        """Cancel and join every old-generation transfer/callback task."""
        tasks = tuple(self._normal_tasks.values())
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self._normal_tasks.clear()
        return tuple(
            result for result in results if isinstance(result, BaseException)
        )

    def _allocate(
        self, kind: str, instance: str, operation_id: str, payload: dict[str, object]
    ) -> EndpointOperationRef:
        self._ensure_operation_capacity(operation_id)
        ref = self.allocator.allocate(
            target_instance=instance,
            target_worker_epoch=self.worker_epochs[instance],
            operation_id=operation_id,
            payload=payload,
        )
        self._refs[(kind, instance, operation_id)] = ref
        return ref

    def remember_external_ref(
        self, kind: str, instance: str, operation_id: str, ref: EndpointOperationRef
    ) -> None:
        self._ensure_operation_capacity(operation_id)
        self._refs[(kind, instance, operation_id)] = ref

    def mark_external_ref_attempted(self, ref: EndpointOperationRef) -> None:
        if ref not in self._refs.values():
            raise RuntimeError("external endpoint ref was not remembered")
        self._mark_ref_attempted(ref)

    def mark_external_ref_ambiguous(self, ref: EndpointOperationRef) -> None:
        if ref not in self._refs.values():
            raise RuntimeError("external endpoint ref was not remembered")
        self._mark_ref_ambiguous(ref)

    def _mark_ref_attempted(self, ref: EndpointOperationRef) -> None:
        self._attempted_refs.add(ref)

    def _mark_ref_ambiguous(self, ref: EndpointOperationRef) -> None:
        self._attempted_refs.add(ref)
        self._ambiguous_refs.add(ref)

    def _ref_requires_cleanup(self, ref: EndpointOperationRef) -> bool:
        return ref in self._attempted_refs or ref in self._ambiguous_refs

    def _ensure_operation_capacity(self, operation_id: str) -> None:
        if operation_id in self._retired_operations:
            raise RuntimeError("network operation is already resource-free terminal")
        active = {key[2] for key in self._refs}
        if operation_id not in active and len(active) >= self.active_operation_cap:
            raise RuntimeError("active network operation capacity exhausted")

    def _bind_correctness_operation(
        self, operation_id: str, request_id: str
    ) -> None:
        if request_id not in self._correctness_operations:
            return
        owner = self._correctness_operation_owners.get(operation_id)
        if owner is not None and owner != request_id:
            raise RuntimeError("network operation has another correctness request owner")
        if (
            owner is None
            and len(self._correctness_operation_owners) >= self.active_operation_cap
        ):
            raise RuntimeError("correctness operation ownership capacity exhausted")
        self._correctness_operation_owners[operation_id] = request_id

    def _discard_correctness_request_if_unowned(self, request_id: str) -> None:
        if request_id in self._correctness_required:
            return
        if request_id in self._correctness_operation_owners.values():
            return
        if any(key[2] == request_id for key in self._refs):
            return
        self._correctness_operations.discard(request_id)

    def _retire_operation(self, operation_id: str) -> None:
        task = self._normal_tasks.get(operation_id)
        if task is not None and not task.done():
            raise RuntimeError("cannot evict an active network task")
        request_id = self._correctness_operation_owners.pop(operation_id, None)
        self._refs = {
            key: ref for key, ref in self._refs.items() if key[2] != operation_id
        }
        self._attempted_refs = {
            ref for ref in self._attempted_refs
            if ref.operation_id != operation_id
        }
        self._ambiguous_refs = {
            ref for ref in self._ambiguous_refs
            if ref.operation_id != operation_id
        }
        self._request_metadata = {
            key: value for key, value in self._request_metadata.items()
            if key[1] != operation_id
        }
        self._cleanup_plans = {
            cleanup_id: plan
            for cleanup_id, plan in self._cleanup_plans.items()
            if getattr(plan, "operation_id", None) != operation_id
        }
        self._normal_tasks.pop(operation_id, None)
        self._discard_correctness_request_if_unowned(
            request_id if request_id is not None else operation_id
        )
        self._retired_operations[operation_id] = None
        self._retired_operations.move_to_end(operation_id)
        while len(self._retired_operations) > self.terminal_snapshot_cap:
            self._retired_operations.popitem(last=False)

    def state_counts(self) -> dict[str, int]:
        return {
            "active_operations": len({key[2] for key in self._refs}),
            "refs": len(self._refs),
            "request_metadata": len(self._request_metadata),
            "cleanup_plans": len(self._cleanup_plans),
            "retired_operations": len(self._retired_operations),
        }

    def require_correctness_evidence(self, req_id: str) -> None:
        self.require_request_evidence(req_id)

    def require_request_evidence(self, req_id: str) -> None:
        if req_id not in self._correctness_operations \
                and len(self._correctness_operations) >= self.active_operation_cap:
            raise RuntimeError("request evidence capacity exhausted")
        self._correctness_required.add(req_id)
        self._correctness_operations.add(req_id)

    def cancel_correctness_evidence(self, req_id: str) -> None:
        self.cancel_request_evidence(req_id)

    def cancel_request_evidence(self, req_id: str) -> None:
        self._correctness_required.discard(req_id)
        self._discard_correctness_request_if_unowned(req_id)

    def _store_request_evidence(
        self, req_id: str, evidence: dict[str, object]
    ) -> None:
        self._request_evidence[req_id] = dict(evidence)
        self._correctness_required.discard(req_id)
        self._discard_correctness_request_if_unowned(req_id)
        self._request_evidence.move_to_end(req_id)
        while len(self._request_evidence) > self.terminal_snapshot_cap:
            self._request_evidence.popitem(last=False)

    def request_evidence(self, req_id: str) -> dict[str, object] | None:
        value = self._request_evidence.get(req_id)
        return dict(value) if value is not None else None

    @staticmethod
    def _terminal_transfer_facts(
        source: dict[str, object], target: dict[str, object]
    ) -> tuple[int, bool, bool]:
        if source.get("state") != "COMPLETED" or target.get("state") != "COMPLETED":
            raise RuntimeError("transfer evidence requires two COMPLETED endpoints")
        source_result = source.get("result")
        target_result = target.get("result")
        if not isinstance(source_result, dict) or not isinstance(target_result, dict):
            raise RuntimeError("transfer terminal result is missing")
        source_bytes = source_result.get("completed_bytes")
        target_bytes = target_result.get("completed_bytes")
        if type(source_bytes) is not int or source_bytes <= 0 or source_bytes != target_bytes:
            raise RuntimeError("transfer endpoint completed bytes disagree")
        work_terminal = (
            source_result.get("work_terminal") is True
            and target_result.get("work_terminal") is True
        )
        cuda_terminal = (
            source_result.get("cuda_terminal") is True
            and target_result.get("cuda_terminal") is True
        )
        if not work_terminal or not cuda_terminal:
            raise RuntimeError("transfer endpoint terminal proof is incomplete")
        return source_bytes, work_terminal, cuda_terminal

    async def prepare_normal_request(
        self, target: str, target_epoch: str, req_id: str,
        token_ids: list[int], sampling_params: dict,
        event_subjects: dict[str, str] | None = None,
        *,
        prefix_identity: dict[str, str] | None = None,
    ) -> tuple[EndpointOperationRef, tuple[int, ...]]:
        if self.worker_epochs[target] != target_epoch:
            raise ValueError("target worker epoch changed")
        payload: dict[str, object] = {
            "req_id": req_id,
            "mode": "remote_transfer",
            "block_count": max(1, math.ceil(len(token_ids) / self.block_size)),
            "token_ids": list(token_ids),
            "sampling_params": dict(sampling_params),
            "held_resource_kinds": ["TARGET_PENDING"],
            **(event_subjects or {}),
        }
        ref = self._allocate("request.prepare", target, req_id, payload)
        self._request_metadata[(target, req_id)] = {
            **dict(event_subjects or {}),
            **dict(prefix_identity or {}),
        }
        self._mark_ref_attempted(ref)
        try:
            response = await self.client.prepare_request(target, ref, payload)
        except AmbiguousRPCError:
            self._mark_ref_ambiguous(ref)
            raise
        result = response.get("result") or {}
        return ref, tuple(int(value) for value in result.get("dst_block_ids", ()))

    async def request_output(self, instance_id: str, req_id: str, after_seq: int = 0):
        return await self.client.request_output(instance_id, req_id, after_seq)

    async def get_kv_usage_all(self):
        values = await asyncio.gather(*(
            self.client.get_resources(instance) for instance in self.worker_epochs
        ))
        return {
            instance: {
                "ratio": float(value.get("kv_usage", 0.0)),
                "instance_epoch": value["instance_epoch"],
            }
            for instance, value in zip(self.worker_epochs, values)
        }

    def reset_to_waiting(self, instance_id: str, req_id: str) -> bool:

        # target allocation and explicitly adds no reset RPC.  Fail closed so
        # the caller enters canonical exact-ref cleanup instead of mutating the
        # already-owned target Sequence in place.
        return False

    def transfer_task(self, task, on_complete) -> None:
        if task.operation_id in self._normal_tasks:
            return
        future = asyncio.create_task(self._run_normal_transfer(task, on_complete))
        self._normal_tasks[task.operation_id] = future
        future.add_done_callback(
            lambda done, op=task.operation_id: self._normal_tasks.pop(op, None)
        )

    async def _run_normal_transfer(self, task, on_complete) -> None:
        transfer_started = time.monotonic()
        transfer_started_ns = time.monotonic_ns()
        mapping = {
            "req_id": task.req_id,
            "source_instance": task.src,
            "source_epoch": task.src_epoch,
            "target_instance": task.dst,
            "target_epoch": task.dst_epoch,
            "src_block_ids": list(task.src_block_ids),
            "dst_block_ids": list(task.dst_block_ids),
            "kv_size_bytes": task.kv_size,
        }
        target_payload = dict(mapping)
        source_payload = {
            **mapping,
            "held_resource_kinds": ["SOURCE_RETAIN", "TRANSFER_BYTES"],
        }
        target_ref = self._allocate(
            "transfer.target", task.dst, task.operation_id, target_payload,
        )
        source_ref = self._allocate(
            "transfer.source", task.src, task.operation_id, source_payload,
        )
        self._bind_correctness_operation(task.operation_id, task.req_id)
        task.transfer_target_ref = target_ref
        task.transfer_source_ref = source_ref
        target_terminal, source_terminal = await self._run_nccl_pair(
            source_instance=task.src,
            source_ref=source_ref,
            source_payload=source_payload,
            target_instance=task.dst,
            target_ref=target_ref,
            target_payload=target_payload,
        )
        transfer_terminal_ns = time.monotonic_ns()
        if self.metrics is not None:
            labels = {"pair": f"{task.src}--{task.dst}", "path": "normal"}
            self.metrics.observe(
                "nccl_transfer_latency_ms",
                (time.monotonic() - transfer_started) * 1000.0,
                labels=labels,
            )
            self.metrics.increment(
                "nccl_transfer_bytes", task.kv_size, labels=labels
            )
        commit_payload = {
            "req_id": task.req_id,
            "transfer_operation_id": task.operation_id,
            "namespace": "",
            "kv_compatibility_id": "",
            "request_context_digest": "",
            "cached_prefix_tokens": len(task.token_ids),
            "first_token": getattr(task, "first_token", None),
            "transfer_endpoint_ref": asdict(target_ref),
            "operation_id": task.operation_id,
            **self._request_metadata.get((task.dst, task.req_id), {}),
        }
        commit_ref = self._allocate("request.commit", task.dst, task.req_id, commit_payload)
        task.target_request_commit_ref = commit_ref
        self._mark_ref_attempted(commit_ref)
        try:
            await self.client._mutate(
                task.dst, "/v1/requests/commit", commit_ref, commit_payload
            )
        except AmbiguousRPCError:
            self._mark_ref_ambiguous(commit_ref)
            value = await self.client.operation_ref_status(task.dst, commit_ref)
            state = str(value.get("state", "UNKNOWN"))
            if state != "COMPLETED":
                if state in {"RUNNING", "PREPARED"}:
                    self._mark_ref_attempted(commit_ref)
                    await self.client._mutate(
                        task.dst, "/v1/requests/commit", commit_ref,
                        commit_payload,
                    )
                else:
                    raise
        correctness_path = getattr(task, "correctness_path", "")
        if correctness_path or task.req_id in self._correctness_required:
            completed_bytes, work_terminal, cuda_terminal = self._terminal_transfer_facts(
                source_terminal, target_terminal
            )
            self._store_request_evidence(task.req_id, {
                "request_id": task.req_id,
                "operation_id": task.operation_id,
                "path": correctness_path or "cold",
                "route": {"source": task.src, "target": task.dst},
                "src_block_ids": list(task.src_block_ids),
                "dst_block_ids": list(task.dst_block_ids),
                "cached_prefix_tokens": 0,
                "suffix_tokens": len(task.token_ids),
                "completed_bytes": completed_bytes,
                "work_terminal": work_terminal,
                "cuda_terminal": cuda_terminal,
                "gateway_clock_epoch": self.gateway_clock_epoch,
                "transfer_started_ns": transfer_started_ns,
                "transfer_terminal_ns": transfer_terminal_ns,
                "source_endpoint_ref": asdict(source_ref),
                "target_endpoint_ref": asdict(target_ref),
                "source_terminal": source_terminal,
                "target_terminal": target_terminal,
            })
        on_complete()

    async def _wait_terminal(
        self, instance: str, kind: str, ref: EndpointOperationRef
    ) -> dict[str, object]:
        deadline = asyncio.get_running_loop().time() + self.operation_timeout_s
        while True:
            value = await self.client.operation_ref_status(instance, ref)
            if value.get("state") in {"COMPLETED", "FENCED"}:
                if value.get("state") != "COMPLETED":
                    raise RuntimeError(f"remote operation fenced: {ref.operation_id}")
                return value
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"remote operation did not terminalize: {ref.operation_id}")
            await asyncio.sleep(self.query_interval_s)

    async def _reconcile_ambiguous_transfer_mutation(
        self,
        *,
        phase: str,
        instance: str,
        ref: EndpointOperationRef,
        error: AmbiguousRPCError,
    ) -> dict[str, object]:
        """Resolve response loss from the original transfer identity only."""
        if error.endpoint_ref != ref:
            raise RuntimeError(
                f"{phase} response loss referred to another endpoint ref"
            )
        value = await self.client.operation_ref_status(instance, ref)
        if value.get("endpoint_ref") != asdict(ref):
            raise RuntimeError(
                f"{phase} reconciliation returned another endpoint ref"
            )
        state = str(value.get("state", "UNKNOWN"))
        if state not in {"PREPARED", "RUNNING", "UNKNOWN", "COMPLETED"}:
            raise RuntimeError(
                f"{phase} reconciliation did not prove an accepted operation: {state}"
            )
        result = value.get("result")
        if (
            not isinstance(result, dict)
            or not str(result.get("pair_id") or "")
            or type(result.get("completed_bytes")) is not int
            or int(result["completed_bytes"]) <= 0
            or type(result.get("work_terminal")) is not bool
            or type(result.get("cuda_terminal")) is not bool
        ):
            raise RuntimeError(
                f"{phase} reconciliation lacks transfer launch authority"
            )
        self.record_correctness_fault_event(
            "rpc_response_loss_reconciled",
            {
                "phase": phase,
                "instance_id": instance,
                "operation_id": ref.operation_id,
                "endpoint_ref": asdict(ref),
                "response": value,
            },
        )
        return value

    async def _run_nccl_pair(
        self,
        *,
        source_instance: str,
        source_ref: EndpointOperationRef,
        source_payload: dict[str, object],
        target_instance: str,
        target_ref: EndpointOperationRef,
        target_payload: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Launch one pair globally, retaining the gate through both terminals."""
        async with self._nccl_sequence_lock:
            if self._nccl_sequence_poisoned:
                raise RuntimeError(
                    "NCCL launch sequencer is poisoned by an ambiguous prior transfer"
                )
            try:
                target_acceptance: dict[str, object]
                self._mark_ref_attempted(target_ref)
                try:
                    target_acceptance = await self.client.prepare_receive(
                        target_instance, target_ref, target_payload
                    )
                except AmbiguousRPCError as exc:
                    self._mark_ref_ambiguous(target_ref)
                    target_acceptance = await self._reconcile_ambiguous_transfer_mutation(
                        phase="target_prepare_receive",
                        instance=target_instance,
                        ref=target_ref,
                        error=exc,
                    )
                if str(source_payload.get("req_id") or "") in self._correctness_required:
                    await self.correctness_fault_checkpoint(
                        "before_nccl_source_start",
                        {
                            "request_id": str(source_payload.get("req_id") or ""),
                            "source_instance": source_instance,
                            "target_instance": target_instance,
                            "source_endpoint_ref": asdict(source_ref),
                            "target_endpoint_ref": asdict(target_ref),
                            "source_payload": dict(source_payload),
                            "target_payload": dict(target_payload),
                        },
                    )
                source_acceptance: dict[str, object]
                self._mark_ref_attempted(source_ref)
                try:
                    source_acceptance = await self.client.start_transfer(
                        source_instance, source_ref, source_payload
                    )
                except AmbiguousRPCError as exc:
                    self._mark_ref_ambiguous(source_ref)
                    source_acceptance = await self._reconcile_ambiguous_transfer_mutation(
                        phase="source_start",
                        instance=source_instance,
                        ref=source_ref,
                        error=exc,
                    )
                if str(source_payload.get("req_id") or "") in self._correctness_required:
                    await self.correctness_fault_checkpoint(
                        "after_nccl_pair_mutations_accepted",
                        {
                            "request_id": str(source_payload.get("req_id") or ""),
                            "source_instance": source_instance,
                            "target_instance": target_instance,
                            "source_endpoint_ref": asdict(source_ref),
                            "target_endpoint_ref": asdict(target_ref),
                            # Formal restart evidence must bind the accepted
                            # refs to the actual mapped-transfer payload.  The
                            # harness consumes these observed payloads; it does
                            # not reconstruct block IDs from the fixture.
                            "source_payload": dict(source_payload),
                            "target_payload": dict(target_payload),
                            "source_acceptance": dict(source_acceptance),
                            "target_acceptance": dict(target_acceptance),
                        },
                    )
                target_terminal, source_terminal = await asyncio.gather(
                    self._wait_terminal(target_instance, "transfers", target_ref),
                    self._wait_terminal(source_instance, "transfers", source_ref),
                )
                # COMPLETED is accepted only with explicit Work/CUDA facts from
                # both worker endpoints.  This also keeps the next pair behind
                # the gate until the prior pair is genuinely terminal.
                self._terminal_transfer_facts(source_terminal, target_terminal)
                return target_terminal, source_terminal
            except BaseException:
                self._nccl_sequence_poisoned = True
                raise

    async def resolve_prefix(
        self, source, source_epoch, operation_id, expected_blocks, **identity
    ):
        if self.worker_epochs[source] != source_epoch:
            raise ValueError("source worker epoch changed")
        payload = {
            **identity,
            "expected_blocks": [asdict(block) for block in expected_blocks],
            "held_resource_kinds": ["SOURCE_PIN"],
        }
        ref = self._allocate("prefix.resolve", source, operation_id, payload)
        self._mark_ref_attempted(ref)
        try:
            value = await self.client.prefix_mutation(source, "resolve", ref, payload)
        except AmbiguousRPCError:
            self._mark_ref_ambiguous(ref)
            raise
        result = value.get("result") or {}
        if result.get("miss"):
            return None
        return ResolvedPrefix(operation_id, source_epoch, tuple(result["src_block_ids"]))

    async def prepare_prefix(
        self, target, target_epoch, operation_id, req_id, **kwargs
    ):
        if self.worker_epochs[target] != target_epoch:
            raise ValueError("target worker epoch changed")
        mode = str(kwargs["mode"])
        payload = {
            "req_id": req_id,
            **kwargs,
            "token_ids": list(kwargs["token_ids"]),
            # local_reuse allocates no destination blocks.  The resolve ref is
            # the sole SOURCE_PIN owner until commit atomically migrates that
            # pin into the commit ref's TARGET_SEQUENCE ownership.
            "held_resource_kinds": (
                ["TARGET_PENDING"] if mode == "remote_transfer" else []
            ),
        }
        ref = self._allocate("prefix.prepare", target, operation_id, payload)
        self._mark_ref_attempted(ref)
        try:
            value = await self.client.prefix_mutation(target, "prepare", ref, payload)
        except AmbiguousRPCError:
            self._mark_ref_ambiguous(ref)
            raise
        result = value.get("result") or {}
        if value.get("state") == "FENCED":
            return None
        return PreparedPrefix(operation_id, str(result["mode"]), tuple(result["dst_block_ids"]))

    async def transfer_cached_prefix(self, plan: CachedPrefixPlan):
        mapping = {
            "req_id": plan.req_id, "source_instance": plan.source_instance,
            "source_epoch": plan.source_epoch, "target_instance": plan.target_instance,
            "target_epoch": plan.target_epoch, "src_block_ids": list(plan.src_block_ids),
            "dst_block_ids": list(plan.dst_block_ids), "namespace": plan.namespace,
            "kv_compatibility_id": plan.kv_compatibility_id,
            "request_context_digest": plan.request_context_digest,
            "kv_size_bytes": len(plan.src_block_ids) * self.block_bytes,
        }
        target_ref = self._allocate("transfer.target", plan.target_instance, plan.operation_id, mapping)
        source_ref = self._allocate("transfer.source", plan.source_instance, plan.operation_id, mapping)
        self._bind_correctness_operation(plan.operation_id, plan.req_id)
        try:
            transfer_started_ns = time.monotonic_ns()
            target_terminal, source_terminal = await self._run_nccl_pair(
                source_instance=plan.source_instance,
                source_ref=source_ref,
                source_payload=mapping,
                target_instance=plan.target_instance,
                target_ref=target_ref,
                target_payload=mapping,
            )
            transfer_terminal_ns = time.monotonic_ns()
            if plan.req_id in self._correctness_required:
                completed_bytes, work_terminal, cuda_terminal = self._terminal_transfer_facts(
                    source_terminal, target_terminal
                )
                self._store_request_evidence(plan.req_id, {
                    "request_id": plan.req_id,
                    "operation_id": plan.operation_id,
                    "path": "cross_instance",
                    "route": {
                        "source": plan.source_instance,
                        "target": plan.target_instance,
                    },
                    "src_block_ids": list(plan.src_block_ids),
                    "dst_block_ids": list(plan.dst_block_ids),
                    "cached_prefix_tokens": plan.cached_prefix_tokens,
                    "suffix_tokens": len(plan.token_ids) - plan.cached_prefix_tokens,
                    "completed_bytes": completed_bytes,
                    "work_terminal": work_terminal,
                    "cuda_terminal": cuda_terminal,
                    "gateway_clock_epoch": self.gateway_clock_epoch,
                    "transfer_started_ns": transfer_started_ns,
                    "transfer_terminal_ns": transfer_terminal_ns,
                    "source_endpoint_ref": asdict(source_ref),
                    "target_endpoint_ref": asdict(target_ref),
                    "source_terminal": source_terminal,
                    "target_terminal": target_terminal,
                })
            return MappedTransferStatus.COMPLETED
        except AmbiguousRPCError:
            return MappedTransferStatus.UNKNOWN

    async def commit_cached_prefix(self, target, operation_id, plan):
        payload = {
            "namespace": plan.namespace,
            "kv_compatibility_id": plan.kv_compatibility_id,
            "request_context_digest": plan.request_context_digest,
            "cached_prefix_tokens": plan.cached_prefix_tokens,
            "mode": plan.mode,
        }
        if plan.mode == "remote_transfer":
            transfer_ref = self._refs.get(("transfer.target", target, operation_id))
            if transfer_ref is None:
                raise RuntimeError("remote prefix commit lost target transfer identity")
            payload.update({
                "transfer_operation_id": transfer_ref.operation_id,
                "transfer_endpoint_ref": asdict(transfer_ref),
            })
        ref = self._allocate("prefix.commit", target, operation_id, payload)
        self._mark_ref_attempted(ref)
        try:
            await self.client.prefix_mutation(target, "commit", ref, payload)
        except AmbiguousRPCError:
            self._mark_ref_ambiguous(ref)
            raise
        if plan.mode == "local_reuse" and plan.req_id in self._correctness_required:
            self._store_request_evidence(plan.req_id, {
                "request_id": plan.req_id,
                "operation_id": plan.operation_id,
                "path": "same_instance",
                "route": {
                    "source": plan.source_instance,
                    "target": plan.target_instance,
                },
                "src_block_ids": [],
                "dst_block_ids": [],
                "cached_prefix_tokens": plan.cached_prefix_tokens,
                "suffix_tokens": len(plan.token_ids) - plan.cached_prefix_tokens,
                "completed_bytes": 0,
                "work_terminal": True,
                "cuda_terminal": True,
                "gateway_clock_epoch": self.gateway_clock_epoch,
                "transfer_started_ns": None,
                "transfer_terminal_ns": None,
                "commit_endpoint_ref": asdict(ref),
            })

    async def abort_mapped_prefix(self, source, target, operation_id):
        refs = [
            (source, self._refs.get(("transfer.source", source, operation_id))),
            (target, self._refs.get(("transfer.target", target, operation_id))),
        ]
        if any(ref is None for _, ref in refs):
            return MappedTransferStatus.UNKNOWN
        values = await asyncio.gather(*(
            self.client.abort_transfer(instance, ref, reason="mapped prefix abort")
            for instance, ref in refs if ref is not None
        ), return_exceptions=True)
        if any(isinstance(value, BaseException) for value in values):
            return MappedTransferStatus.UNKNOWN
        states = {value.get("state") for value in values}
        return (MappedTransferStatus.COMPLETED if states == {"COMPLETED"}
                else MappedTransferStatus.FENCED if states <= {"COMPLETED", "FENCED"}
                else MappedTransferStatus.UNKNOWN)

    async def get_prefix_operation(self, target, operation_id):
        value = await self.client._request(
            "GET", target, f"/v1/prefix/status/{operation_id}"
        )
        state = str(value.get("state"))
        if state == "FENCED":
            return PrefixOperationStatus.ABORTED
        result = value.get("result") or {}
        if state == "COMPLETED" and "seq_id" in result:
            return PrefixOperationStatus.COMMITTED
        if state in {"PREPARED", "RUNNING", "COMPLETED"}:
            return PrefixOperationStatus.PREPARED
        return PrefixOperationStatus.UNKNOWN

    async def abort_cached_prefix(self, target, operation_id):
        ref = self._refs.get(("prefix.prepare", target, operation_id))
        if ref is None:
            return
        await self.client._abort(target, "/v1/prefix/abort", ref, "prefix abort")

    async def unpin_prefix(self, source, operation_id):
        ref = self._refs.get(("prefix.resolve", source, operation_id))
        if ref is None:
            return
        try:
            await self.client.finalize_release(
                source,
                cleanup_id=f"prefix-unpin:{operation_id}",
                operation_id=operation_id,
                lease_id=f"prefix:{operation_id}",
                endpoint_refs=(ref,),
                resource_kinds=("SOURCE_PIN",),
            )
        except AmbiguousRPCError:
            # Query the exact resolve ref.  A lost finalize response may have
            # already released the pin; otherwise the caller replays the same
            # cleanup id and payload rather than allocating a new identity.
            snapshot = await self.client.operation_ref_status(source, ref)
            if snapshot.get("state") in {"COMPLETED", "FENCED"} \
                    and snapshot.get("resources_held") is False:
                return
            raise

    async def abort_suffix_prefill(self, target, operation_id):
        ref = self._refs.get(("suffix", target, operation_id))
        if ref is None:
            return False
        value = await self.client.abort_request(target, ref, reason="suffix timeout")
        return value.get("state") in {"COMPLETED", "FENCED"}

    async def get_prefix_resource_counts(self, instance):
        value = await self.client.get_resources(instance)
        resources = value.get("resources") or {}
        return {
            "transfer_pins": int(resources.get("SOURCE_PIN", 0)),
            "pending_allocations": int(resources.get("TARGET_PENDING", 0)),
        }

    async def full_report_and_register(self, instance_id, consumer_id, generation):
        import time
        from prism_serve.router.prefix_index import CacheLocation
        from prism_serve.router.reconciler import PrefixReport

        value = await self.client._request(
            "POST", instance_id, "/v1/prefix/reports/register",
            json_body={"consumer_id": consumer_id, "generation": generation},
        )
        received_at = time.monotonic()
        return PrefixReport(
            str(value["instance_id"]), str(value["instance_epoch"]),
            int(value["snapshot_seq_no"]),
            tuple(self._cache_location(item, received_at) for item in value["locations"]),
        )

    async def peek_events(self, instance_id, consumer_id, generation, after_seq, limit):
        value = await self.client._request(
            "GET", instance_id,
            f"/v1/prefix/events?consumer_id={consumer_id}&generation={generation}&after_seq={after_seq}&limit={limit}",
        )
        import time
        from prism_serve.router.prefix_index import PrefixEvent

        received_at = time.monotonic()
        return [
            PrefixEvent(
                str(item["kind"]), self._cache_location(item, received_at),
                int(item["seq_no"]),
            )
            for item in value.get("events", [])
        ]

    async def ack_events(self, instance_id, consumer_id, generation, up_to_seq):
        await self.client._request(
            "POST", instance_id, "/v1/prefix/events/ack",
            json_body={"consumer_id": consumer_id, "generation": generation, "up_to_seq": up_to_seq},
        )

    @staticmethod
    def _cache_location(value, received_at):
        from prism_serve.router.prefix_index import CacheLocation

        return CacheLocation(
            str(value["instance_id"]), str(value["instance_epoch"]),
            str(value["namespace"]), str(value["kv_compatibility_id"]),
            str(value["request_context_digest"]), int(value["chain_hash"]),
            int(value["block_index"]), int(value["block_id"]),
            int(value["prefix_tokens"]), received_at,
        )

    async def cleanup_prefix_context(self, scheduler, context):
        """Fence original refs, query every predicate, then invoke sole evaluator."""
        from prism_serve.router.protocol import CleanupOutcome, PrefixOperationStatus
        from prism_serve.scheduler.resource_release import (
            CleanupPlan,
            EndpointFinalizePlan,
            PredicateSnapshot,
        )

        if self.release_evaluator is None:
            raise RuntimeError("resource release evaluator is not installed")
        lease = scheduler.decode_slot_lease(context.operation_id)
        if lease is None:
            raise ValueError("prefix cleanup lost decode slot lease")
        if lease.state != "QUARANTINED":
            scheduler.quarantine_decode_slot(context.operation_id)
        candidates = [
            (kind, instance, ref)
            for (kind, instance, operation_id), ref in self._refs.items()
            if operation_id == context.operation_id
            and self._ref_requires_cleanup(ref)
        ]
        for kind, instance, ref in candidates:
            try:
                if kind.startswith("transfer."):
                    await self.client.abort_transfer(instance, ref, reason="prefix cleanup")
                elif kind == "suffix":
                    await self.client.abort_request(instance, ref, reason="prefix cleanup")
                else:
                    await self.client._abort(
                        instance, "/v1/prefix/abort", ref, "prefix cleanup"
                    )
            except Exception:
                # Ambiguous abort is repaired only by original-ref query below.
                pass

        snapshots = []
        for kind, instance, ref in candidates:
            try:
                value = await self.client.operation_ref_status(instance, ref)
            except Exception:
                return CleanupOutcome("QUARANTINED", PrefixOperationStatus.UNKNOWN)
            status = str(value.get("state", "UNKNOWN"))
            if status not in {"COMPLETED", "FENCED"}:
                return CleanupOutcome("QUARANTINED", PrefixOperationStatus.UNKNOWN)
            snapshots.append((kind, instance, ref, value))

        cleanup_id = f"prefix-cleanup:{context.operation_id}"
        plan = self._cleanup_plans.get(cleanup_id)
        if plan is None:
            grouped: dict[str, list[tuple[EndpointOperationRef, tuple[str, ...]]]] = {}
            predicates = []
            for kind, instance, ref, value in snapshots:
                held = bool(value.get("resources_held"))
                predicates.append(PredicateSnapshot(
                    kind, str(value["state"]), ref, held
                ))
                if held:
                    grouped.setdefault(instance, []).append((
                        ref, tuple(str(item) for item in value.get(
                            "held_resource_kinds", ()
                        ))
                    ))
            endpoints = tuple(
                EndpointFinalizePlan(
                    instance,
                    tuple(item[0] for item in items),
                    tuple(sorted({kind for _, kinds in items for kind in kinds})),
                )
                for instance, items in grouped.items()
            )
            plan = CleanupPlan(
                cleanup_id=cleanup_id,
                operation_id=context.operation_id,
                lease_id=lease.lease_id,
                predicates=tuple(predicates),
                endpoints=endpoints,
            )
            self._cleanup_plans[cleanup_id] = plan
        result = await self.release_evaluator.release_endpoint_terminal(plan)
        if result is None:
            return CleanupOutcome("QUARANTINED", PrefixOperationStatus.UNKNOWN)
        self._retire_operation(context.operation_id)
        return CleanupOutcome("ABORTED", PrefixOperationStatus.ABORTED)

    async def cleanup_request(self, scheduler, req, *, abort: bool) -> bool:
        """Terminalize/query every original normal-PD ref, then release slot-last."""
        from prism_serve.scheduler.resource_release import (
            CleanupPlan,
            EndpointFinalizePlan,
            PredicateSnapshot,
        )

        if self.release_evaluator is None:
            raise RuntimeError("resource release evaluator is not installed")
        operation_id = req.active_operation_id or req.req_id
        lease = scheduler.decode_slot_lease(operation_id)
        if lease is None:
            raise ValueError("normal request lost decode slot lease")
        if lease.state == "RELEASED":
            return True
        if lease.state != "QUARANTINED":
            scheduler.quarantine_decode_slot(operation_id)

        candidates: list[tuple[str, str, EndpointOperationRef]] = []
        if req.dispatch_operation_ref is not None:
            candidates.append((
                "request.source", req.prefill_instance, req.dispatch_operation_ref
            ))
        candidates.extend(
            (kind, instance, ref)
            for (kind, instance, correlated_id), ref in self._refs.items()
            if correlated_id == operation_id
            and self._ref_requires_cleanup(ref)
        )
        # Stable endpoint sequence identity prevents accidental duplicate plans.
        candidates = list({
            (instance, ref.operation_seq): (kind, instance, ref)
            for kind, instance, ref in candidates
        }.values())
        if not candidates:
            return False

        if abort:
            for kind, instance, ref in candidates:
                try:
                    if kind.startswith("transfer."):
                        abort_value = await self.client.abort_transfer(
                            instance, ref, reason="normal request cleanup"
                        )
                    else:
                        abort_value = await self.client.abort_request(
                            instance, ref, reason="normal request cleanup"
                        )
                    self.record_correctness_fault_event(
                        "endpoint_abort_observed",
                        {
                            "operation_id": operation_id,
                            "instance_id": instance,
                            "endpoint_ref": asdict(ref),
                            "response": abort_value,
                        },
                    )
                except Exception as exc:
                    self.record_correctness_fault_event(
                        "endpoint_abort_ambiguous",
                        {
                            "operation_id": operation_id,
                            "instance_id": instance,
                            "endpoint_ref": asdict(ref),
                            "error_type": type(exc).__name__,
                        },
                    )
                    # Only the exact-ref query below can resolve response loss.
                    pass

        snapshots = []
        for kind, instance, ref in candidates:
            try:
                value = await self.client.operation_ref_status(instance, ref)
            except Exception:
                return False
            self.record_correctness_fault_event(
                "endpoint_query_observed",
                {
                    "operation_id": operation_id,
                    "instance_id": instance,
                    "endpoint_ref": asdict(ref),
                    "response": value,
                },
            )
            if str(value.get("state", "UNKNOWN")) not in {"COMPLETED", "FENCED"}:
                return False
            snapshots.append((kind, instance, ref, value))

        cleanup_id = f"normal-cleanup:{operation_id}"
        plan = self._cleanup_plans.get(cleanup_id)
        if plan is None:
            grouped: dict[str, list[tuple[EndpointOperationRef, tuple[str, ...]]]] = {}
            predicates = []
            for kind, instance, ref, value in snapshots:
                held = bool(value.get("resources_held"))
                predicates.append(PredicateSnapshot(
                    kind, str(value["state"]), ref, held
                ))
                if held:
                    grouped.setdefault(instance, []).append((
                        ref,
                        tuple(str(item) for item in value.get(
                            "held_resource_kinds", ()
                        )),
                    ))
            endpoints = tuple(
                EndpointFinalizePlan(
                    instance,
                    tuple(item[0] for item in items),
                    tuple(sorted({kind for _, kinds in items for kind in kinds})),
                )
                for instance, items in sorted(grouped.items())
            )
            plan = CleanupPlan(
                cleanup_id=cleanup_id,
                operation_id=operation_id,
                lease_id=lease.lease_id,
                predicates=tuple(predicates),
                endpoints=endpoints,
            )
            self._cleanup_plans[cleanup_id] = plan
        try:
            result = await self.release_evaluator.release_endpoint_terminal(plan)
        except Exception:
            return False
        if result is None:
            return False
        self._retire_operation(operation_id)
        return True


async def activate_replacement_owner(
    client: HttpInferClient,
    instances: tuple[str, ...],
    new_owner: str,
    *,
    max_audit_entries: int = 18432,
    reconcile_deadline: float | None = None,
    retry_interval_s: float = 0.2,
) -> OwnerTakeoverAudit:
    """Exact-ref sweep, generic-finalize, retire-all, then activate-all.

    Admission remains closed until this function returns.  In particular, a
    held-terminal orphan is not treated as safe merely because its writer has
    stopped: all refs are queried, the sole evaluator obtains remote finalize
    ACKs, and only then may the old owner retire.
    """
    from prism_serve.scheduler.resource_release import (
        CleanupPlan,
        EndpointFinalizePlan,
        PredicateSnapshot,
        ResourceReleaseEvaluator,
    )

    class _NoLocalSlot:
        def release_quarantined_decode_slot(self, *args, **kwargs):
            return False

    if not instances or len(set(instances)) != len(instances):
        raise ValueError("takeover instances must be non-empty and unique")
    if max_audit_entries <= 0:
        raise ValueError("takeover audit cap must be positive")
    if retry_interval_s <= 0:
        raise ValueError("takeover retry interval must be positive")
    ordered_instances = tuple(sorted(instances))
    statuses = {
        instance: await client.owner_status(instance)
        for instance in ordered_instances
    }
    old_by_instance = {
        instance: str(value["active_owner"])
        for instance, value in statuses.items()
        if value.get("active_owner") not in {None, new_owner}
    }
    if len(set(old_by_instance.values())) > 1:
        raise RuntimeError("workers disagree on the prior active owner")
    identity_epochs: dict[str, str] = {}
    for instance in old_by_instance:
        identity = await client.get_identity(instance)
        if (
            not isinstance(identity, dict)
            or identity.get("instance_id") != instance
            or not isinstance(identity.get("instance_epoch"), str)
            or not identity["instance_epoch"]
        ):
            raise RuntimeError("worker identity is incomplete during owner takeover")
        identity_epochs[instance] = identity["instance_epoch"]
    reports: dict[str, tuple[_ParsedOperationSnapshot, ...]] = {}
    operation_list_evidence: list[OwnerTakeoverListEvidence] = []
    remaining_audit_entries = max_audit_entries
    for instance, owner in old_by_instance.items():
        raw_report = await client.list_operations(instance, owner)
        report = _parse_complete_operation_report(
            raw_report,
            instance_id=instance,
            owner_generation=owner,
            instance_epoch=identity_epochs[instance],
            max_entries=remaining_audit_entries,
        )
        reports[instance] = report
        operation_list_evidence.append(_operation_list_evidence(
            instance_id=instance,
            owner_generation=owner,
            instance_epoch=identity_epochs[instance],
            operations=report,
        ))
        remaining_audit_entries -= len(report)
    snapshots_by_operation: dict[
        str, list[tuple[str, EndpointOperationRef, _ParsedOperationSnapshot]]
    ] = {}
    listed_state_by_ref: dict[EndpointOperationRef, str] = {}
    query_evidence_by_ref: dict[
        EndpointOperationRef, OwnerTakeoverQueryEvidence
    ] = {}
    finalize_evidence_by_ref: dict[
        EndpointOperationRef, OwnerTakeoverFinalizeEvidence
    ] = {}
    abort_attempted_refs: set[EndpointOperationRef] = set()
    parsed_entries: list[tuple[str, _ParsedOperationSnapshot]] = []
    seen_refs: set[EndpointOperationRef] = set()
    for instance, report in reports.items():
        for listed in report:
            ref = listed.endpoint_ref
            if ref in seen_refs:
                raise RuntimeError("old-owner report returned a duplicate endpoint ref")
            seen_refs.add(ref)
            listed_state_by_ref[ref] = listed.state
            parsed_entries.append((instance, listed))
    for instance, listed in parsed_entries:
        ref = listed.endpoint_ref
        if listed.state not in {"COMPLETED", "FENCED"}:
            abort_attempted_refs.add(ref)
            try:
                await client.abort_request(
                    instance, ref, reason="replacement orphan sweep"
                )
            except Exception:
                # Response loss is resolved only by the exact query below.
                pass
        try:
            value = await client.operation_ref_status(instance, ref)
        except Exception as exc:
            raise RuntimeError(
                f"old owner operation UNKNOWN on {instance}"
            ) from exc
        try:
            snapshot = _parse_operation_snapshot(
                value,
                instance_id=instance,
                owner_generation=old_by_instance[instance],
                instance_epoch=identity_epochs[instance],
                context="old-owner exact query",
                expected_ref=ref,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"old-owner exact query is unbound on {instance}"
            ) from exc
        if snapshot.state not in {"COMPLETED", "FENCED"}:
            raise RuntimeError(
                f"old owner operation UNKNOWN on {instance}"
            )
        query_evidence_by_ref[ref] = OwnerTakeoverQueryEvidence(
            instance_id=instance,
            instance_epoch=identity_epochs[instance],
            owner_generation=ref.owner_generation,
            operation_id=ref.operation_id,
            endpoint_ref=ref,
            state=snapshot.state,
            resources_held=snapshot.resources_held,
            held_resource_kinds=snapshot.held_resource_kinds,
        )
        snapshots_by_operation.setdefault(ref.operation_id, []).append(
            (instance, ref, snapshot)
        )

    async def finalize_remote(instance: str, **kwargs) -> dict[str, object]:
        value = await client.finalize_release(instance, **kwargs)
        return validate_release_snapshot(
            value,
            instance_id=instance,
            cleanup_id=kwargs["cleanup_id"],
            operation_id=kwargs["operation_id"],
            lease_id=kwargs["lease_id"],
            endpoint_refs=kwargs["endpoint_refs"],
            resource_kinds=kwargs["resource_kinds"],
            expected_endpoint_epoch=identity_epochs[instance],
        )

    evaluator = ResourceReleaseEvaluator(
        _NoLocalSlot(), finalize_remote
    )
    owner_key = "+".join(sorted(set(old_by_instance.values()))) or "none"
    finalized_operation_ids: set[str] = set()
    for operation_id, snapshots in snapshots_by_operation.items():
        grouped: dict[str, list[tuple[EndpointOperationRef, tuple[str, ...]]]] = {}
        predicates = []
        for instance, ref, value in snapshots:
            held = value.resources_held
            predicates.append(PredicateSnapshot(
                instance, value.state, ref, held
            ))
            if held:
                grouped.setdefault(instance, []).append((
                    ref,
                    value.held_resource_kinds,
                ))
        if not grouped:
            continue
        plan = CleanupPlan(
            cleanup_id=f"orphan:{owner_key}:{operation_id}",
            operation_id=operation_id,
            lease_id=f"orphan:{owner_key}:{operation_id}",
            predicates=tuple(predicates),
            endpoints=tuple(
                EndpointFinalizePlan(
                    instance,
                    tuple(ref for ref, _ in values),
                    tuple(sorted({
                        kind for _, kinds in values for kind in kinds
                    })),
                )
                for instance, values in sorted(grouped.items())
            ),
        )
        last_error = None
        for _ in range(3):
            try:
                result = await evaluator.release_endpoint_terminal(plan)
                if result is None or len(result) != len(plan.endpoints):
                    raise RuntimeError("finalize ACK set is incomplete")
                for endpoint, value in zip(plan.endpoints, result):
                    evidence = _finalize_evidence(
                        endpoint.instance_id, endpoint.endpoint_refs, value
                    )
                    for ref in endpoint.endpoint_refs:
                        finalize_evidence_by_ref[ref] = evidence
                last_error = None
                finalized_operation_ids.add(operation_id)
                break
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise RuntimeError(
                f"orphan finalize did not converge for {operation_id}"
            ) from last_error

    # Verify held ownership is actually gone before changing any owner.
    for instance, owner in old_by_instance.items():
        operations = _parse_complete_operation_report(
            await client.list_operations(instance, owner),
            instance_id=instance,
            owner_generation=owner,
            instance_epoch=identity_epochs[instance],
        )
        if {value.endpoint_ref for value in operations} != {
            value.endpoint_ref for value in reports[instance]
        }:
            raise RuntimeError("post-finalize complete report changed endpoint set")
        if any(value.resources_held for value in operations):
            raise RuntimeError(
                f"old owner {owner} still holds resources on {instance}"
            )
        if any(value.state not in {"COMPLETED", "FENCED"}
               for value in operations):
            raise RuntimeError(f"old owner {owner} still active on {instance}")



    loop = asyncio.get_running_loop()
    if reconcile_deadline is None:
        reconcile_deadline = loop.time() + 120.0

    def active_owner(value: object, *, instance: str) -> str | None:
        if not isinstance(value, dict) or "active_owner" not in value:
            raise RuntimeError(
                f"owner status for {instance} must contain active_owner"
            )
        owner = value["active_owner"]
        if owner is None:
            return None
        if not isinstance(owner, str) or not owner:
            raise RuntimeError(
                f"owner status for {instance} has malformed active_owner"
            )
        return owner

    def retryable_owner_error(exc: BaseException) -> bool:
        pending: BaseException | None = exc
        visited: set[int] = set()
        while pending is not None and id(pending) not in visited:
            visited.add(id(pending))
            if isinstance(
                pending,
                (AmbiguousRPCError, ConnectionError, TimeoutError),
            ):
                return True
            if isinstance(pending, InferRPCError):
                return pending.status_code in {409, 503}
            pending = pending.__cause__ or pending.__context__
        return False

    def require_remaining(*, action: str) -> float:
        remaining = reconcile_deadline - loop.time()
        if remaining <= 0:
            raise RuntimeError(
                f"owner reconciliation deadline exceeded during {action}"
            )
        return remaining

    async def retry_pause(exc: BaseException, *, action: str) -> None:
        try:
            remaining = require_remaining(action=action)
        except RuntimeError as deadline_error:
            raise deadline_error from exc
        await asyncio.sleep(min(retry_interval_s, remaining))

    async def read_owner(instance: str) -> str | None:
        while True:
            require_remaining(action=f"owner status {instance}")
            try:
                return active_owner(
                    await client.owner_status(instance),
                    instance=instance,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not retryable_owner_error(exc):
                    raise
                await retry_pause(exc, action=f"owner status {instance}")

    async def mutate_owner(
        operation,
        *,
        action: str,
    ) -> None:
        require_remaining(action=action)
        try:
            await operation()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not retryable_owner_error(exc):
                raise


            await retry_pause(exc, action=action)

    retired_instances: set[str] = set()
    for instance, owner in sorted(old_by_instance.items()):
        while True:
            observed = await read_owner(instance)
            if observed in {None, new_owner}:
                retired_instances.add(instance)
                break
            if observed != owner:
                raise RuntimeError(
                    f"foreign owner observed during retire: "
                    f"{instance}={observed}"
                )
            await mutate_owner(
                lambda instance=instance, owner=owner: client.retire_owner(
                    instance, owner
                ),
                action=f"retire owner {instance}",
            )

    activated_instances: set[str] = set()
    for instance in ordered_instances:
        expected_old = old_by_instance.get(instance)
        while True:
            observed = await read_owner(instance)
            if observed == new_owner:
                break
            if observed is None:
                activated_instances.add(instance)
                await mutate_owner(
                    lambda instance=instance: client.activate_owner(
                        instance, new_owner
                    ),
                    action=f"activate owner {instance}",
                )
                continue
            if expected_old is not None and observed == expected_old:
                await mutate_owner(
                    lambda instance=instance, owner=expected_old:
                        client.retire_owner(instance, owner),
                    action=f"close retire before activate {instance}",
                )
                continue
            raise RuntimeError(
                f"foreign owner observed during activate: "
                f"{instance}={observed}"
            )

    confirmed = []
    for instance in ordered_instances:
        confirmed.append((instance, (await read_owner(instance)) or ""))
    confirmed_active_owners = tuple(confirmed)
    if any(owner != new_owner for _, owner in confirmed_active_owners):
        raise RuntimeError("replacement owner activation was not confirmed")

    operations = tuple(
        OwnerTakeoverOperationAudit(
            instance_id=instance,
            old_owner=old_by_instance[instance],
            endpoint_ref=ref,
            listed_state=listed_state_by_ref[ref],
            query_confirmed=(
                query_evidence_by_ref[ref].endpoint_ref == ref
                and query_evidence_by_ref[ref].operation_id == ref.operation_id
                and query_evidence_by_ref[ref].owner_generation
                == ref.owner_generation
            ),
            terminal_state=value.state,
            abort_attempted=ref in abort_attempted_refs,
            resources_held_before_finalize=value.resources_held,
            held_resource_kinds=value.held_resource_kinds,
            finalize_acknowledged=ref in finalize_evidence_by_ref,
            query_evidence=query_evidence_by_ref[ref],
            finalize_evidence=finalize_evidence_by_ref.get(ref),
        )
        for operation_id in sorted(snapshots_by_operation)
        for instance, ref, value in sorted(
            snapshots_by_operation[operation_id], key=lambda item: item[0]
        )
    )
    return OwnerTakeoverAudit(
        new_owner=new_owner,
        instances=ordered_instances,
        old_owners=tuple(sorted(old_by_instance.items())),
        operation_list_evidence=tuple(sorted(
            operation_list_evidence, key=lambda value: value.instance_id
        )),
        operations=operations,
        finalized_operation_ids=tuple(sorted(finalized_operation_ids)),
        retired_owners=tuple(sorted(
            (instance, old_by_instance[instance])
            for instance in retired_instances
        )),
        activated_instances=tuple(sorted(activated_instances)),
        confirmed_active_owners=confirmed_active_owners,
    )

"""Fence resource release on durable endpoint or replacement evidence."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable

from prism_serve.router.http_rpc import EndpointOperationRef
from prism_serve.scheduler.replacement_store import (
    ReplacementConflict,
    ReplacementDecisionStore,
    ReplacementReleaseRecord,
    ReplacementRunSeal,
    ReplacementStoreUnavailable,
)


TERMINAL_PREDICATES = {"COMPLETED", "FENCED"}


@dataclass(slots=True, frozen=True)
class PredicateSnapshot:
    name: str
    status: str
    endpoint_ref: EndpointOperationRef
    resources_held: bool = True


@dataclass(slots=True, frozen=True)
class EndpointFinalizePlan:
    instance_id: str
    endpoint_refs: tuple[EndpointOperationRef, ...]
    resource_kinds: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class CleanupPlan:
    cleanup_id: str
    operation_id: str
    lease_id: str
    predicates: tuple[PredicateSnapshot, ...]
    endpoints: tuple[EndpointFinalizePlan, ...]


@dataclass(slots=True, frozen=True)
class ReplacementEvidence:
    restart_run_id: str
    old_topology_generation: str
    new_topology_generation: str
    old_termination_proof_digests: tuple[str, ...]
    fresh_resource_report_digests: tuple[str, ...]
    excluded_old_operation_digest: str
    accepted: bool
    decision_digest: str


class NoLiveReplacementLease(ValueError):
    code = "NO_LIVE_REPLACEMENT_LEASE"
    http_status = 409


class ResourceReleaseEvaluator:
    """Release resources only after endpoint or replacement evidence is durable."""

    def __init__(
        self,
        scheduler,
        finalize_remote: Callable[..., Awaitable[dict[str, object]]],
        metrics=None,
        *,
        active_operation_cap: int = 512,
        terminal_snapshot_cap: int = 4096,
        replacement_store: ReplacementDecisionStore | None = None,
    ) -> None:
        if active_operation_cap <= 0 or terminal_snapshot_cap <= 0:
            raise ValueError("release evaluator caps must be positive")
        self.scheduler = scheduler
        self._finalize_remote = finalize_remote
        self.metrics = metrics
        self._endpoint_acks: dict[tuple[str, str], dict[str, object]] = {}
        self.active_operation_cap = active_operation_cap
        self.terminal_snapshot_cap = terminal_snapshot_cap
        self._active_endpoint: dict[str, CleanupPlan] = {}
        self._completed_endpoint: OrderedDict[
            str, tuple[dict[str, object], ...]
        ] = OrderedDict()
        self.replacement_store = replacement_store
        self.correctness_fault_gate = None

    def set_correctness_fault_gate(self, gate) -> None:
        self.correctness_fault_gate = gate

    def _record_correctness_event(
        self, name: str, details: dict[str, object]
    ) -> None:
        gate = self.correctness_fault_gate
        if gate is not None:
            gate.record_event(name, details)

    async def release_endpoint_terminal(
        self, plan: CleanupPlan
    ) -> tuple[dict[str, object], ...] | None:
        completed = self._completed_endpoint.get(plan.cleanup_id)
        if completed is not None:
            self._completed_endpoint.move_to_end(plan.cleanup_id)
            if self.metrics is not None:
                self.metrics.increment(
                    "cleanup_finalize_replay_total", labels={"endpoint": "all"}
                )
            return completed
        if not plan.predicates or any(
            predicate.status not in TERMINAL_PREDICATES
            for predicate in plan.predicates
        ):

            return None
        expected_refs = {
            predicate.endpoint_ref for predicate in plan.predicates
            if predicate.resources_held
        }
        planned_refs = {
            endpoint_ref
            for endpoint in plan.endpoints
            for endpoint_ref in endpoint.endpoint_refs
        }
        if not expected_refs.issubset(planned_refs):
            raise ValueError("endpoint finalize plan omits predicate refs")
        active = self._active_endpoint.get(plan.cleanup_id)
        if active is None:
            if len(self._active_endpoint) >= self.active_operation_cap:
                raise RuntimeError("active cleanup capacity exhausted")
            self._active_endpoint[plan.cleanup_id] = plan
        elif active != plan:
            raise ValueError("cleanup id reused with different plan")
        self._record_correctness_event(
            "release_predicates_satisfied",
            {
                "cleanup_id": plan.cleanup_id,
                "operation_id": plan.operation_id,
                "lease_id": plan.lease_id,
                "predicates": [
                    {
                        "name": value.name,
                        "status": value.status,
                        "resources_held": value.resources_held,
                        "endpoint_ref": asdict(value.endpoint_ref),
                    }
                    for value in plan.predicates
                ],
            },
        )
        results = []
        for endpoint in plan.endpoints:
            key = (plan.cleanup_id, endpoint.instance_id)
            snapshot = self._endpoint_acks.get(key)
            if snapshot is None:
                snapshot = await self._finalize_remote(
                    endpoint.instance_id,
                    cleanup_id=plan.cleanup_id,
                    operation_id=plan.operation_id,
                    lease_id=plan.lease_id,
                    endpoint_refs=endpoint.endpoint_refs,
                    resource_kinds=endpoint.resource_kinds,
                )
                self._endpoint_acks[key] = snapshot
                if self.metrics is not None:
                    self.metrics.increment(
                        "cleanup_finalize_total",
                        labels={"endpoint": endpoint.instance_id, "status": "ACKED"},
                    )
            results.append(snapshot)
        result = tuple(results)
        self._record_correctness_event(
            "endpoint_finalize_acked",
            {
                "cleanup_id": plan.cleanup_id,
                "operation_id": plan.operation_id,
                "acks": [dict(value) for value in result],
            },
        )

        self.scheduler.release_quarantined_decode_slot(
            plan.operation_id, plan.lease_id, plan.cleanup_id
        )
        self._record_correctness_event(
            "slot_released",
            {
                "cleanup_id": plan.cleanup_id,
                "operation_id": plan.operation_id,
                "lease_id": plan.lease_id,
            },
        )
        self._completed_endpoint[plan.cleanup_id] = result
        self._completed_endpoint.move_to_end(plan.cleanup_id)
        self._active_endpoint.pop(plan.cleanup_id, None)
        for key in tuple(self._endpoint_acks):
            if key[0] == plan.cleanup_id:
                self._endpoint_acks.pop(key, None)
        while len(self._completed_endpoint) > self.terminal_snapshot_cap:
            self._completed_endpoint.popitem(last=False)
        return result

    def release_whole_world_replaced(
        self,
        *,
        cleanup_id: str,
        operation_id: str,
        lease_id: str,
        old_resource_kinds: tuple[str, ...],
        evidence: ReplacementEvidence,
    ) -> ReplacementReleaseRecord:
        store = self._require_replacement_store()
        record = self._replacement_record(
            cleanup_id=cleanup_id,
            operation_id=operation_id,
            lease_id=lease_id,
            old_resource_kinds=old_resource_kinds,
            evidence=evidence,
        )
        existing = store.lookup(
            record,
            old_topology_generation=evidence.old_topology_generation,
        )
        if existing is not None:
            if self.metrics is not None:
                self.metrics.increment(
                    "cleanup_replacement_record_replay_total",
                    labels={"outcome": "replay"},
                )


            self._release_local_record(existing, allow_missing=True)
            return existing

        self._validate_replacement_evidence(evidence)
        self._require_live_quarantined_lease(record)
        stored, created = store.persist_record(
            record,
            old_topology_generation=evidence.old_topology_generation,
        )
        if created and self.metrics is not None:
            self.metrics.increment(
                "cleanup_replacement_record_total",
                labels={"outcome": "accepted"},
            )

        self._release_local_record(stored, allow_missing=False)
        return stored

    def persist_whole_world_replaced_batch(
        self,
        entries: tuple[tuple[str, str, str, tuple[str, ...]], ...],
        *,
        evidence: ReplacementEvidence,
    ) -> tuple[ReplacementReleaseRecord, ...]:
        store = self._require_replacement_store()
        self._validate_replacement_evidence(evidence)
        records = tuple(
            self._replacement_record(
                cleanup_id=cleanup_id,
                operation_id=operation_id,
                lease_id=lease_id,
                old_resource_kinds=old_resource_kinds,
                evidence=evidence,
            )
            for cleanup_id, operation_id, lease_id, old_resource_kinds in entries
        )
        for record in records:
            existing = store.lookup(
                record,
                old_topology_generation=evidence.old_topology_generation,
            )
            if existing is None:
                self._require_live_quarantined_lease(record)
        before = store.active_record_count
        stored = store.persist_records(
            records,
            restart_run_id=evidence.restart_run_id,
            old_topology_generation=evidence.old_topology_generation,
            decision_digest=evidence.decision_digest,
        )
        created = store.active_record_count - before
        if created and self.metrics is not None:
            for _ in range(created):
                self.metrics.increment(
                    "cleanup_replacement_record_total",
                    labels={"outcome": "accepted"},
                )
        return stored

    def release_persisted_replacement_batch(
        self, records: tuple[ReplacementReleaseRecord, ...]
    ) -> None:
        for record in records:
            self._release_local_record(record, allow_missing=False)

    def seal_whole_world_replacement(
        self, evidence: ReplacementEvidence
    ) -> ReplacementRunSeal:
        store = self._require_replacement_store()
        self._validate_replacement_evidence(evidence)
        live = self.scheduler.replacement_decode_leases()
        if live:
            raise NoLiveReplacementLease(
                "replacement run still has live local leases"
            )
        return store.seal_run(
            restart_run_id=evidence.restart_run_id,
            old_topology_generation=evidence.old_topology_generation,
            new_topology_generation=evidence.new_topology_generation,
            decision_digest=evidence.decision_digest,
        )

    def rollback_whole_world_replaced(
        self, *, cleanup_id: str, evidence: ReplacementEvidence
    ) -> None:
        raise RuntimeError(
            "durable replacement release decisions cannot be rolled back"
        )

    @property
    def _replacement_records(
        self,
    ) -> dict[tuple[str, str, str], ReplacementReleaseRecord]:
        if self.replacement_store is None:
            return {}
        return {
            record.key: record for record in self.replacement_store.records()
        }

    def _require_replacement_store(self) -> ReplacementDecisionStore:
        if self.replacement_store is None:
            raise ReplacementStoreUnavailable(
                "whole-world release requires a durable replacement store"
            )
        if self.replacement_store.last_error is not None:
            raise ReplacementStoreUnavailable(
                "durable replacement store is not ready: "
                f"{self.replacement_store.last_error}"
            )
        return self.replacement_store

    @staticmethod
    def _validate_replacement_evidence(evidence: ReplacementEvidence) -> None:
        if not evidence.accepted:
            raise ValueError("restart run is not accepted")
        if evidence.old_topology_generation == evidence.new_topology_generation:
            raise ValueError("replacement requires a fresh topology generation")
        if len(evidence.old_termination_proof_digests) != 4:
            raise ValueError("replacement requires four old termination proofs")
        if len(set(evidence.old_termination_proof_digests)) != 4:
            raise ValueError("old termination proofs must identify four members")
        if len(evidence.fresh_resource_report_digests) != 4:
            raise ValueError("replacement requires four fresh complete reports")
        if len(set(evidence.fresh_resource_report_digests)) != 4:
            raise ValueError("resource reports must identify four members")
        if not evidence.excluded_old_operation_digest or not evidence.decision_digest:
            raise ValueError("replacement evidence digests are required")

    @staticmethod
    def _replacement_record(
        *,
        cleanup_id: str,
        operation_id: str,
        lease_id: str,
        old_resource_kinds: tuple[str, ...],
        evidence: ReplacementEvidence,
    ) -> ReplacementReleaseRecord:
        if not cleanup_id or not operation_id or not lease_id:
            raise ValueError("replacement cleanup/operation/lease ids are required")
        return ReplacementReleaseRecord(
            cleanup_id=cleanup_id,
            restart_run_id=evidence.restart_run_id,
            old_operation_digest=evidence.excluded_old_operation_digest,
            operation_id=operation_id,
            lease_id=lease_id,
            old_resource_kinds=tuple(sorted(set(old_resource_kinds))),
            decision_digest=evidence.decision_digest,
        )

    def _require_live_quarantined_lease(
        self, record: ReplacementReleaseRecord
    ) -> None:
        lease = self.scheduler.decode_slot_lease(record.operation_id)
        if (
            lease is None
            or lease.lease_id != record.lease_id
            or lease.state != "QUARANTINED"
        ):
            raise NoLiveReplacementLease(
                "NO_LIVE_REPLACEMENT_LEASE: exact quarantined lease required"
            )

    def _release_local_record(
        self,
        record: ReplacementReleaseRecord,
        *,
        allow_missing: bool,
    ) -> None:
        lease = self.scheduler.decode_slot_lease(record.operation_id)
        if lease is None:
            if allow_missing:
                return
            raise NoLiveReplacementLease(
                "NO_LIVE_REPLACEMENT_LEASE: local lease disappeared after record"
            )
        if lease.lease_id != record.lease_id:
            raise NoLiveReplacementLease(
                "NO_LIVE_REPLACEMENT_LEASE: local lease id changed"
            )
        if lease.state == "RELEASED":
            if lease.release_cleanup_id != record.cleanup_id:
                raise ReplacementConflict(
                    "released replacement lease used another cleanup id"
                )
            return
        if lease.state != "QUARANTINED":
            raise NoLiveReplacementLease(
                "NO_LIVE_REPLACEMENT_LEASE: lease is not quarantined"
            )
        released = self.scheduler.release_quarantined_decode_slot(
            record.operation_id, record.lease_id, record.cleanup_id
        )
        if not released:
            raise RuntimeError(
                "durable replacement record did not release its local lease"
            )

    def state_counts(self) -> dict[str, int]:
        return {
            "active_endpoint": len(self._active_endpoint),
            "endpoint_acks": len(self._endpoint_acks),
            "completed_endpoint": len(self._completed_endpoint),
            "replacement_records": (
                0
                if self.replacement_store is None
                else self.replacement_store.active_record_count
            ),
        }

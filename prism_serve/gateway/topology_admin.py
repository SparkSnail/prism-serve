"""Fail-closed whole-world topology acceptance authority."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json

from prism_serve.gateway.termination_authority import (
    validate_termination_records,
)
from prism_serve.router.worker_registry import (
    EXPECTED_MEMBERS,
    EXPECTED_PAIRS,
    PairCapability,
    WorkerIdentity,
    WorkerRegistry,
)


_WORKER_IDENTITY_FIELDS = (
    "instance_id",
    "role",
    "topology_generation",
    "pod_uid",
    "process_generation",
    "rpc_endpoint",
    "global_rank",
    "topology_digest",
    "kv_compatibility_id",
)
_WORKER_IDENTITY_STRING_FIELDS = tuple(
    field for field in _WORKER_IDENTITY_FIELDS if field != "global_rank"
)
_WORKER_IDENTITY_WIRE_FIELDS = frozenset(
    (*_WORKER_IDENTITY_FIELDS, "instance_epoch")
)


def parse_worker_identity(value: object) -> WorkerIdentity:
    """Parse one strict wire identity without duplicating derived state."""
    if not isinstance(value, dict):
        raise ValueError("worker identity must be an object")

    fields = set(value)
    missing = set(_WORKER_IDENTITY_FIELDS) - fields
    if missing:
        raise ValueError(
            "worker identity is missing required fields: "
            + ", ".join(sorted(missing))
        )
    unknown = fields - _WORKER_IDENTITY_WIRE_FIELDS
    if unknown:
        raise ValueError(
            "worker identity contains unknown fields: "
            + ", ".join(sorted(str(field) for field in unknown))
        )
    if any(
        not isinstance(value[field], str)
        for field in _WORKER_IDENTITY_STRING_FIELDS
    ):
        raise ValueError("worker identity string fields must be strings")
    global_rank = value["global_rank"]
    if isinstance(global_rank, bool) or not isinstance(global_rank, int):
        raise ValueError("worker identity global_rank must be an integer")

    expected_epoch = f"{value['pod_uid']}:{value['process_generation']}"
    if "instance_epoch" in value:
        instance_epoch = value["instance_epoch"]
        if not isinstance(instance_epoch, str):
            raise ValueError("worker identity instance_epoch must be a string")
        if instance_epoch != expected_epoch:
            raise ValueError(
                "worker identity instance_epoch does not match pod/process identity"
            )

    return WorkerIdentity(**{field: value[field] for field in _WORKER_IDENTITY_FIELDS})


def worker_identity_wire(identity: WorkerIdentity) -> dict[str, object]:
    """Serialize canonical fields plus the validated derived wire convenience."""
    value = {
        field: getattr(identity, field) for field in _WORKER_IDENTITY_FIELDS
    }
    value["instance_epoch"] = identity.instance_epoch
    return value


@dataclass(slots=True, frozen=True)
class RestartRunRecord:
    restart_run_id: str
    old_topology_generation: str
    new_topology_generation: str
    termination_proof_digests: tuple[str, str, str, str]
    fresh_resource_report_digests: tuple[str, str, str, str]
    pair_probe_digests: tuple[str, str, str, str, str]
    decision_digest: str
    accepted: bool = True


class TopologyAcceptanceLedger:
    def __init__(
        self, registry: WorkerRegistry, *, terminal_snapshot_cap: int = 4096
    ) -> None:
        if terminal_snapshot_cap <= 0:
            raise ValueError("topology ledger cap must be positive")
        self.registry = registry
        self.terminal_snapshot_cap = terminal_snapshot_cap
        self.records: OrderedDict[str, RestartRunRecord] = OrderedDict()

    @staticmethod
    def _digest(value: object) -> str:
        return "sha256:" + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def validate_physical_termination_records(
        body: dict[str, object],
    ) -> list[dict[str, object]]:
        values = body.get("termination_records")
        if not isinstance(values, list):
            raise ValueError("termination records must be a list")
        old_generation = body.get("old_topology_generation")
        if not isinstance(old_generation, str) or not old_generation:
            raise ValueError("old topology generation is required")
        return validate_termination_records(
            values,
            expected_generation=old_generation,
            expected_members=EXPECTED_MEMBERS,
        )

    def stage(
        self, body: dict[str, object]
    ) -> tuple[RestartRunRecord, WorkerRegistry, bool]:
        """Validate and build a replacement without publishing it.

        The runtime uses this phase to rebuild every generation-bound consumer
        and close old quarantines before one final state publication.
        """
        run_id = str(body["restart_run_id"])
        existing = self.records.get(run_id)
        decision_digest = self._digest(body)
        if existing is not None:
            if existing.decision_digest != decision_digest:
                raise ValueError("restart run id reused with different evidence")
            self.records.move_to_end(run_id)
            return existing, self.registry, True
        old_generation = str(body["old_topology_generation"])
        if old_generation != self.registry.expected_topology_generation:
            raise ValueError("old topology generation does not match active world")
        new_generation = str(body["new_topology_generation"])
        if new_generation == old_generation:
            raise ValueError("restart requires a fresh topology generation")

        terminations = self.validate_physical_termination_records(body)
        old_members = self.registry.members
        for value in terminations:
            instance = str(value["logical_instance_id"])
            if str(value.get("pod_uid")) != old_members[instance].pod_uid:
                raise ValueError("termination proof does not bind old pod uid")
            if str(value.get("process_generation")) != old_members[instance].process_generation:
                raise ValueError("termination proof does not bind old process generation")

        identity_values = body["identities"]
        if not isinstance(identity_values, list):
            raise ValueError("replacement identities must be a list")
        if len(identity_values) != 4:
            raise ValueError("replacement requires four unique worker identities")
        identities = [parse_worker_identity(value) for value in identity_values]
        if {identity.instance_id for identity in identities} != set(
            EXPECTED_MEMBERS
        ):
            raise ValueError("replacement requires four unique worker identities")
        if any(identity.topology_generation != new_generation for identity in identities):
            raise ValueError("new identity generation mismatch")
        for identity in identities:
            if identity.pod_uid == old_members[identity.instance_id].pod_uid:
                raise ValueError("replacement pod uid must be fresh")
        capability_values = list(body["pair_capabilities"])
        capabilities = [PairCapability(**value) for value in capability_values]
        if len(capabilities) != 5 or {
            value.pair_id for value in capabilities
        } != EXPECTED_PAIRS:
            raise ValueError("replacement requires five pair probes")
        reports = dict(body["resource_reports"])
        if set(reports) != set(EXPECTED_MEMBERS):
            raise ValueError("replacement requires four resource reports")
        if "old_operation_ids" not in body or "old_resource_ids" not in body:
            raise ValueError(
                "replacement evidence must include exhaustive old id snapshots"
            )
        operation_values = body["old_operation_ids"]
        resource_values = body["old_resource_ids"]
        if not isinstance(operation_values, list) or not isinstance(
            resource_values, list
        ):
            raise ValueError("replacement old id snapshots must be lists")
        excluded_operations = {str(value) for value in operation_values}
        excluded_resources = {str(value) for value in resource_values}
        if len(excluded_operations) != len(operation_values) or len(
            excluded_resources
        ) != len(resource_values):
            raise ValueError("replacement old id snapshots must be unique")

        candidate = WorkerRegistry(
            expected_topology_generation=new_generation,
            expected_kv_compatibility_id=(
                self.registry.expected_kv_compatibility_id
            ),
            expected_model_profile=self.registry.expected_model_profile,
            require_gpudirect_rdma=self.registry.require_gpudirect_rdma,
            resource_report_stale_after_s=self.registry.resource_report_stale_after_s,
        )
        if not candidate.install_world(identities, capabilities):
            raise ValueError("new worker world failed identity/capability validation")
        for instance, report in reports.items():
            if not candidate.update_resource_report(instance, report):
                raise ValueError("new resource report is incomplete or epoch-mismatched")
            operation_ids = {str(value) for value in report.get("operation_ids", ())}
            resource_ids = {str(value) for value in report.get("resource_ids", ())}
            if operation_ids & excluded_operations or resource_ids & excluded_resources:
                raise ValueError("fresh report still contains old operation/resource")
            if report.get("excluded_operation_ids") != sorted(excluded_operations):
                raise ValueError("report does not explicitly exclude every old operation")
            if report.get("excluded_resource_ids") != sorted(excluded_resources):
                raise ValueError("report does not explicitly exclude every old resource")

        termination_digests = tuple(
            self._digest(value) for value in terminations
        )
        report_digests = tuple(self._digest(reports[name]) for name in sorted(reports))
        pair_digests = tuple(
            self._digest(asdict(value)) for value in sorted(capabilities, key=lambda item: item.pair_id)
        )
        record = RestartRunRecord(
            restart_run_id=run_id,
            old_topology_generation=old_generation,
            new_topology_generation=new_generation,
            termination_proof_digests=termination_digests,
            fresh_resource_report_digests=report_digests,
            pair_probe_digests=pair_digests,
            decision_digest=decision_digest,
        )
        # Neither ledger nor active registry changes during staging.
        return record, candidate, False

    def commit(self, record: RestartRunRecord, candidate: WorkerRegistry) -> None:
        existing = self.records.get(record.restart_run_id)
        if existing is not None:
            if existing != record or self.registry is not candidate:
                raise ValueError("restart run publication conflict")
            return
        if self.registry.expected_topology_generation != record.old_topology_generation:
            raise ValueError("active topology changed during replacement staging")
        if candidate.expected_topology_generation != record.new_topology_generation:
            raise ValueError("candidate topology generation changed")
        self.records[record.restart_run_id] = record
        self.records.move_to_end(record.restart_run_id)
        while len(self.records) > self.terminal_snapshot_cap:
            self.records.popitem(last=False)
        self.registry = candidate

    def is_committed(self, record: RestartRunRecord) -> bool:
        return (
            self.records.get(record.restart_run_id) == record
            and self.registry.expected_topology_generation
            == record.new_topology_generation
        )

    def recover_committed_run(self, record: RestartRunRecord) -> None:
        """Rebuild the process-local ledger after a Gateway crash.

        The durable store and a fresh already-installed new worker world are
        the recovery authorities; this method never changes the registry.
        """
        existing = self.records.get(record.restart_run_id)
        if existing is not None:
            if existing != record:
                raise ValueError("restart recovery conflicts with accepted run")
            self.records.move_to_end(record.restart_run_id)
            return
        if self.registry.expected_topology_generation != record.new_topology_generation:
            raise ValueError("recovery world is not the accepted new generation")
        if record.old_topology_generation == record.new_topology_generation:
            raise ValueError("replacement recovery requires distinct generations")
        self.records[record.restart_run_id] = record
        self.records.move_to_end(record.restart_run_id)
        while len(self.records) > self.terminal_snapshot_cap:
            self.records.popitem(last=False)

    def rollback(
        self,
        record: RestartRunRecord,
        candidate: WorkerRegistry,
        old_registry: WorkerRegistry,
    ) -> None:
        if self.records.get(record.restart_run_id) != record:
            return
        if self.registry is not candidate:
            raise ValueError("cannot rollback a topology published by another transaction")
        self.records.pop(record.restart_run_id, None)
        self.registry = old_registry

    def accept(self, body: dict[str, object]) -> tuple[RestartRunRecord, WorkerRegistry]:
        record, candidate, replay = self.stage(body)
        if not replay:
            self.commit(record, candidate)
        return record, candidate

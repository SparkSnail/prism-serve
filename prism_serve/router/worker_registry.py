"""Track worker identity, capability, and freshness for a fixed 2P2D topology."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Callable


EXPECTED_MEMBERS = {"p0": ("prefill", 0), "p1": ("prefill", 1),
                    "d0": ("decode", 2), "d1": ("decode", 3)}
EXPECTED_PAIRS = {"p0--d0", "p0--d1", "p1--d0", "p1--d1", "d0--d1"}


class TopologyState(str, Enum):
    STARTING = "STARTING"
    READY = "READY"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


class ResourceSignalState(str, Enum):
    UNKNOWN = "UNKNOWN"
    FRESH = "FRESH"
    STALE = "STALE"


@dataclass(slots=True, frozen=True)
class WorkerIdentity:
    instance_id: str
    role: str
    topology_generation: str
    pod_uid: str
    process_generation: str
    rpc_endpoint: str
    global_rank: int
    topology_digest: str
    kv_compatibility_id: str

    @property
    def instance_epoch(self) -> str:
        return f"{self.pod_uid}:{self.process_generation}"


@dataclass(slots=True, frozen=True)
class PairCapability:
    pair_id: str
    source_epoch: str
    target_epoch: str
    transport: str
    probe_generation: str
    probe_passed: bool
    evidence_path: str


@dataclass(slots=True, frozen=True)
class ResourceSignal:
    state: ResourceSignalState
    instance_epoch: str | None = None
    received_at: float | None = None
    report: dict[str, object] | None = None
    age_s: float | None = None


class WorkerRegistry:
    """Publish complete fixed-topology worker snapshots atomically."""

    def __init__(
        self,
        *,
        expected_topology_generation: str,
        expected_kv_compatibility_id: str | None = None,
        expected_model_profile: dict[str, object] | None = None,
        require_gpudirect_rdma: bool = False,
        resource_report_stale_after_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        assert expected_topology_generation
        assert resource_report_stale_after_s > 0
        self.expected_topology_generation = expected_topology_generation
        self.expected_kv_compatibility_id = expected_kv_compatibility_id
        self.expected_model_profile = (
            dict(expected_model_profile) if expected_model_profile is not None else None
        )
        self.require_gpudirect_rdma = require_gpudirect_rdma
        self.resource_report_stale_after_s = resource_report_stale_after_s
        self._clock = clock
        self.state = TopologyState.STARTING
        self._members: dict[str, WorkerIdentity] = {}
        self._capabilities: dict[str, PairCapability] = {}
        self._resources: dict[str, ResourceSignal] = {}

    @property
    def members(self) -> dict[str, WorkerIdentity]:
        return dict(self._members)

    @property
    def capabilities(self) -> dict[str, PairCapability]:
        return dict(self._capabilities)

    def install_world(
        self,
        identities: list[WorkerIdentity],
        capabilities: list[PairCapability],
    ) -> bool:
        members = {identity.instance_id: identity for identity in identities}
        caps = {capability.pair_id: capability for capability in capabilities}
        valid = set(members) == set(EXPECTED_MEMBERS) and set(caps) == EXPECTED_PAIRS
        if valid:
            generations = {identity.topology_generation for identity in members.values()}
            digests = {identity.topology_digest for identity in members.values()}
            pod_uids = {identity.pod_uid for identity in members.values()}
            compatibility_ids = {
                identity.kv_compatibility_id for identity in members.values()
            }
            valid = (
                generations == {self.expected_topology_generation}
                and len(digests) == 1
                and len(pod_uids) == 4
                and len(compatibility_ids) == 1
                and "" not in compatibility_ids
                and (
                    self.expected_kv_compatibility_id is None
                    or compatibility_ids == {self.expected_kv_compatibility_id}
                )
                and all(
                    (identity.role, identity.global_rank) == EXPECTED_MEMBERS[name]
                    for name, identity in members.items()
                )
                and all(self._admit_capability(cap, members) for cap in caps.values())
            )
        if not valid:
            self._fail()
            return False
        self._members = members
        self._capabilities = caps
        self._resources.clear()
        self.state = TopologyState.READY
        return True

    def _admit_capability(
        self,
        capability: PairCapability,
        members: dict[str, WorkerIdentity],
    ) -> bool:
        if not capability.probe_passed or not capability.probe_generation:
            return False
        source, target = capability.pair_id.split("--", maxsplit=1)
        if source not in members or target not in members:
            return False
        if capability.source_epoch != members[source].instance_epoch:
            return False
        if capability.target_epoch != members[target].instance_epoch:
            return False
        if self.require_gpudirect_rdma:
            return capability.transport == "NCCL_GDR"
        return capability.transport in {"NCCL_GDR", "NCCL_SOCKET", "CUDA_IPC"}

    def observe_identity(self, identity: WorkerIdentity) -> bool:
        current = self._members.get(identity.instance_id)
        if current is None or current != identity:
            self._fail()
            return False
        return True

    def observe_capabilities(self, capabilities: list[PairCapability]) -> bool:
        observed = {value.pair_id: value for value in capabilities}
        if observed != self._capabilities:
            self._fail()
            return False
        return True

    def _fail(self) -> None:
        self.state = TopologyState.FAILED
        self._resources.clear()
        self._capabilities.clear()

    def capture_resource_report_received_at(self) -> float:
        """Capture the Gateway-local monotonic time at HTTP completion."""
        return self._clock()

    def update_resource_report(
        self,
        instance_id: str,
        report: dict[str, object],
        *,
        received_at: float | None = None,
    ) -> bool:
        identity = self._members.get(instance_id)
        valid = (
            identity is not None
            and report.get("instance_epoch") == identity.instance_epoch
            and report.get("complete") is True
            and isinstance(report.get("resources"), dict)
            and self._resource_report_schema_valid(report, identity)
        )
        if not valid:



            return False
        now = self._clock()
        gateway_received_at = now if received_at is None else received_at
        if (
            isinstance(gateway_received_at, bool)
            or not isinstance(gateway_received_at, (int, float))
            or not math.isfinite(gateway_received_at)
            or gateway_received_at > now
        ):
            return False
        age_s = max(0.0, now - float(gateway_received_at))
        state = (
            ResourceSignalState.FRESH
            if age_s <= self.resource_report_stale_after_s
            else ResourceSignalState.STALE
        )



        self._resources[instance_id] = ResourceSignal(
            state=state,
            instance_epoch=identity.instance_epoch,
            received_at=float(gateway_received_at),
            report=dict(report),
            age_s=age_s,
        )
        return True

    def _resource_report_schema_valid(
        self, report: dict[str, object], identity: WorkerIdentity
    ) -> bool:
        profile = report.get("model_profile")
        if self.expected_model_profile is not None:
            if not isinstance(profile, dict):
                return False
            if any(
                profile.get(key) != value
                for key, value in self.expected_model_profile.items()
            ):
                return False
            if profile.get("kv_compatibility_id") != identity.kv_compatibility_id:
                return False

        block_fields = {
            "num_gpu_blocks", "free_blocks", "block_buckets",
            "block_conservation_valid",
        }
        if not (block_fields & report.keys()):
            return self.expected_model_profile is None
        if not block_fields.issubset(report):
            return False
        total = report.get("num_gpu_blocks")
        free = report.get("free_blocks")
        buckets = report.get("block_buckets")
        if (
            isinstance(total, bool) or not isinstance(total, int) or total < 0
            or isinstance(free, bool) or not isinstance(free, int) or free < 0
            or not isinstance(buckets, dict)
            or report.get("block_conservation_valid") is not True
        ):
            return False
        expected_buckets = {"free", "pending", "sequence", "evictable", "quarantined"}
        if set(buckets) != expected_buckets:
            return False
        values = list(buckets.values())
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in values):
            return False
        return buckets["free"] == free and sum(values) == total

    def resource_report_failed(self, instance_id: str) -> None:
        pass

    def observe_unreachable(self, instance_id: str) -> None:


        return

    def resource_signal(self, instance_id: str) -> ResourceSignal:
        signal = self._resources.get(instance_id)
        if signal is None or signal.received_at is None:
            return ResourceSignal(ResourceSignalState.UNKNOWN)
        age = max(0.0, self._clock() - signal.received_at)
        state = (
            ResourceSignalState.FRESH
            if age <= self.resource_report_stale_after_s
            else ResourceSignalState.STALE
        )
        return ResourceSignal(
            state=state,
            instance_epoch=signal.instance_epoch,
            received_at=signal.received_at,
            report=signal.report,
            age_s=age,
        )

    def can_admit(self, instance_id: str) -> bool:
        return (
            self.state == TopologyState.READY
            and self.resource_signal(instance_id).state == ResourceSignalState.FRESH
        )

    can_send = can_admit
    can_route = can_admit
    can_govern = can_admit

    def world_fresh(self) -> bool:
        return (
            self.state == TopologyState.READY
            and set(self._members) == set(EXPECTED_MEMBERS)
            and all(self.can_admit(instance) for instance in EXPECTED_MEMBERS)
        )

    def can_reconcile_finalize(self, instance_id: str) -> bool:


        return self.can_admit(instance_id)

"""Cluster-level PD orchestration scheduler.

Decides which Prefill instance handles each request and which Decode instance
receives the resulting KV cache.  Does NOT handle KV transfer flow control
(see TransferGovernor) or per-request state tracking (see RequestTracker).

Borrowing:
  - pick_prefill_instance  ← shortest-queue strategy (standard load balancing)
  - pick_decode_instance   ← most-free-slots strategy (KV-pressure-aware)
  - decide_decode_instance_count ← Flink AdaptiveBatch (runtime parallelism)
"""

from __future__ import annotations

import math
import secrets
import time
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class QuarantineRecord:
    instance_id: str
    instance_epoch: str
    role: str
    reconciliation_token: str
    quarantined_at: float
    uncertain_transfer_operations: tuple[tuple[str, str, str, str], ...] = ()
    uncertain_dispatch_commands: tuple[tuple[str, str, str, str], ...] = ()


@dataclass(slots=True, frozen=True)
class KVUsageSample:
    ratio: float
    instance_epoch: str
    sampled_at: float


@dataclass(slots=True)
class DecodeSlotLease:
    lease_id: str
    operation_id: str
    req_id: str
    instance_id: str
    state: str
    reserved_at: float


class QuarantinedInstanceError(ValueError):
    def __init__(self, record: QuarantineRecord) -> None:
        super().__init__(f"instance {record.instance_id!r} is quarantined")
        self.record = record

class PDScheduler:
    """Cluster-level PD orchestration decision maker.

    Responsibilities:
      1. Maintain a slot-view of all P/D instances.
      2. For each incoming request: select the optimal P instance and
         pre-select a D instance.
      3. Dynamically adjust the recommended P:D ratio at runtime
         (← Flink AdaptiveBatch).

    Not responsible for: KV transfer flow control (TransferGovernor),
    per-request state tracking (RequestTracker).
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        # instance_id → queue depth (prefill instances)
        self._prefill_load: dict[str, int] = {}
        # instance_id → free slot count (decode instances)
        self._decode_free_slots: dict[str, int] = {}
        self._decode_max_slots: dict[str, int] = {}
        self._slot_leases: dict[str, DecodeSlotLease] = {}
        # Samples are epoch-scoped so a restarted instance cannot reuse stale capacity.
        self._kv_usage: dict[str, KVUsageSample] = {}
        self._kv_usage_stale_after_s: float = config.get(
            "kv_usage_stale_after_s", 10.0
        )
        self._instance_epochs: dict[str, str] = {}
        self._instance_roles: dict[str, str] = {}
        self._quarantined: dict[str, QuarantineRecord] = {}

    # Registration

    def register_instance(
        self,
        instance_id: str,
        role: str,
        max_slots: int = 0,
        instance_epoch: str = "",
        active_request_ids: list[str] | None = None,
    ) -> None:
        """Register a new instance when a K8S Pod comes up.

        Args:
            instance_id: unique instance ID, e.g. "prefill-0", "decode-1"
            role:        "prefill" | "decode"
            max_slots:   decode instances must supply this (> 0)
        """
        if instance_id in self._quarantined:
            raise QuarantinedInstanceError(self._quarantined[instance_id])
        if active_request_ids:
            raise ValueError(
                f"instance still owns active requests: {active_request_ids!r}"
            )
        epoch = instance_epoch or f"legacy:{instance_id}"
        existing_epoch = self._instance_epochs.get(instance_id)
        if existing_epoch is not None:
            if existing_epoch != epoch:
                raise ValueError(
                    "instance epoch changed without reconciliation: "
                    f"{instance_id=} {existing_epoch=} {epoch=}"
                )
            return

        if role == "prefill":
            self._prefill_load[instance_id] = 0
        elif role == "decode":
            assert max_slots > 0, (
                f"decode instance must specify max_slots > 0, got {max_slots=}"
            )
            self._decode_free_slots[instance_id] = max_slots
            self._decode_max_slots[instance_id] = max_slots
        else:
            raise ValueError(f"unknown role {role!r}; expected 'prefill' or 'decode'")
        self._instance_epochs[instance_id] = epoch
        self._instance_roles[instance_id] = role

    def deregister_instance(self, instance_id: str) -> None:
        """Remove an instance that is going offline."""
        self._prefill_load.pop(instance_id, None)
        self._decode_free_slots.pop(instance_id, None)
        self._decode_max_slots.pop(instance_id, None)
        self._kv_usage.pop(instance_id, None)
        self._instance_epochs.pop(instance_id, None)
        self._instance_roles.pop(instance_id, None)

    def quarantine_instance(
        self,
        instance_id: str,
        *,
        uncertain_transfer: tuple[str, str, str, str] | None = None,
        uncertain_dispatch: tuple[str, str, str, str] | None = None,
    ) -> QuarantineRecord:
        """Remove untrusted capacity while retaining its reconciliation fence."""
        existing = self._quarantined.get(instance_id)
        if existing is not None:
            transfer_known = uncertain_transfer is None or uncertain_transfer in existing.uncertain_transfer_operations
            dispatch_known = uncertain_dispatch is None or uncertain_dispatch in existing.uncertain_dispatch_commands
            if transfer_known and dispatch_known:
                return existing
            record = QuarantineRecord(
                instance_id=existing.instance_id,
                instance_epoch=existing.instance_epoch,
                role=existing.role,
                reconciliation_token=existing.reconciliation_token,
                quarantined_at=existing.quarantined_at,
                uncertain_transfer_operations=(
                    *existing.uncertain_transfer_operations,
                    *((uncertain_transfer,) if not transfer_known else ()),
                ),
                uncertain_dispatch_commands=(
                    *existing.uncertain_dispatch_commands,
                    *((uncertain_dispatch,) if not dispatch_known else ()),
                ),
            )
            self._quarantined[instance_id] = record
            return record
        epoch = self._instance_epochs.get(instance_id, "unknown")
        role = self._instance_roles.get(instance_id, "unknown")
        record = QuarantineRecord(
            instance_id=instance_id,
            instance_epoch=epoch,
            role=role,
            reconciliation_token=secrets.token_urlsafe(24),
            quarantined_at=time.monotonic(),
            uncertain_transfer_operations=(
                (uncertain_transfer,) if uncertain_transfer is not None else ()
            ),
            uncertain_dispatch_commands=(
                (uncertain_dispatch,) if uncertain_dispatch is not None else ()
            ),
        )
        self._prefill_load.pop(instance_id, None)
        self._decode_free_slots.pop(instance_id, None)
        self._decode_max_slots.pop(instance_id, None)
        self._kv_usage.pop(instance_id, None)
        self._quarantined[instance_id] = record
        return record

    def reconcile_instance(
        self,
        instance_id: str,
        instance_epoch: str,
        reconciliation_token: str,
        role: str,
        max_slots: int,
        active_request_ids: list[str],
        active_transfer_operation_ids: list[str] | None = None,
        pending_dispatch_command_ids: list[str] | None = None,
    ) -> None:
        """Restore quarantined capacity after the instance reports no requests."""
        record = self._quarantined.get(instance_id)
        if record is None:
            raise ValueError(f"instance {instance_id!r} is not quarantined")
        if reconciliation_token != record.reconciliation_token:
            raise ValueError("reconciliation token mismatch")
        if active_request_ids:
            raise ValueError(
                f"instance still has active requests: {active_request_ids!r}"
            )
        if active_transfer_operation_ids:
            raise ValueError(
                "instance still has active transfers: "
                f"{active_transfer_operation_ids!r}"
            )
        if pending_dispatch_command_ids:
            raise ValueError(
                "instance still has pending dispatches: "
                f"{pending_dispatch_command_ids!r}"
            )
        if not instance_epoch:
            raise ValueError("instance_epoch must not be empty")
        if role not in {"prefill", "decode"}:
            raise ValueError(f"unknown role {role!r}")
        if record.role != "unknown" and role != record.role:
            raise ValueError(
                f"reconciliation role changed: expected={record.role!r}, got={role!r}"
            )
        if role == "decode" and max_slots <= 0:
            raise ValueError(
                f"decode instance must specify max_slots > 0, got {max_slots=}"
            )

        # No validation or other fallible work may follow quarantine removal.
        if role == "prefill":
            self._prefill_load[instance_id] = 0
        else:
            self._decode_free_slots[instance_id] = max_slots
            self._decode_max_slots[instance_id] = max_slots
        self._instance_epochs[instance_id] = instance_epoch
        self._instance_roles[instance_id] = role
        self._quarantined.pop(instance_id)

    def quarantine_record(self, instance_id: str) -> QuarantineRecord | None:
        return self._quarantined.get(instance_id)

    # Selection

    def pick_prefill_instance(self, req_id: str) -> str | None:
        """Select the P instance with the shortest queue (least loaded).

        Returns None if no prefill instance is registered.

        Why not round-robin: queue depth is proportional to actual GPU
        load for prefill.  A short-queue instance is more likely to start
        the new request promptly.
        """
        if not self._prefill_load:
            return None
        # O(n_prefill_instances); n is typically < 10 in a single cluster
        best = min(self._prefill_load, key=lambda k: self._prefill_load[k])
        self._prefill_load[best] += 1
        return best

    def pick_decode_instance(
        self,
        req_id: str,
        kv_size_bytes: int,
    ) -> str | None:
        """Pre-select the D instance with the most free slots.

        Filters out instances whose KV usage is at or above HIGH_WATERMARK
        (transmission flow control is already congested there).

        Returns None if all D instances are full or congested.

        Why not round-robin: KV size varies wildly (short prompt = 0.11 GB,
        long prompt = 3.5 GB).  Round-robin ignores KV pressure; most-free-slots
        naturally balances it.
        """
        high_wm: float = self.config.get("HIGH_WATERMARK", 0.85)
        candidates = [
            (iid, slots)
            for iid, slots in self._decode_free_slots.items()
            if slots > 0 and self._sample_is_fresh(iid) and self._kv_usage[iid].ratio < high_wm
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda x: x[1])[0]
        self._decode_free_slots[best] -= 1
        return best

    def reserve_decode_slot(
        self, instance_id: str, req_id: str, operation_id: str
    ) -> DecodeSlotLease | None:
        """Reserve one process-local decode slot for an affinity operation."""
        assert operation_id, "decode slot reservation requires operation_id"
        existing = self._slot_leases.get(operation_id)
        if existing is not None:
            return existing if existing.state != "RELEASED" else None
        if self._decode_free_slots.get(instance_id, 0) <= 0:
            return None
        self._decode_free_slots[instance_id] -= 1
        lease = DecodeSlotLease(
            secrets.token_hex(16), operation_id, req_id, instance_id,
            "RESERVED", time.monotonic(),
        )
        self._slot_leases[operation_id] = lease
        return lease

    def commit_decode_slot(self, operation_id: str) -> None:
        lease = self._slot_leases[operation_id]
        if lease.state == "RESERVED":
            lease.state = "ACTIVE"

    def release_decode_slot(self, operation_id: str) -> bool:
        lease = self._slot_leases.get(operation_id)
        if lease is None or lease.state in {"RELEASED", "QUARANTINED"}:
            return False
        if lease.state == "ACTIVE":
            raise ValueError("cannot rollback an active decode slot")
        lease.state = "RELEASED"
        if lease.instance_id in self._decode_free_slots:
            maximum = self._decode_max_slots[lease.instance_id]
            self._decode_free_slots[lease.instance_id] = min(
                maximum, self._decode_free_slots[lease.instance_id] + 1
            )
        return True

    def quarantine_decode_slot(self, operation_id: str) -> None:
        lease = self._slot_leases[operation_id]
        if lease.state == "RELEASED":
            raise ValueError("cannot quarantine a released decode slot")
        lease.state = "QUARANTINED"

    def release_quarantined_decode_slot(self, operation_id: str) -> bool:
        """Release after an authoritative remote abort proves the slot safe."""
        lease = self._slot_leases.get(operation_id)
        if lease is None or lease.state != "QUARANTINED":
            return False
        lease.state = "RELEASED"
        if lease.instance_id in self._decode_free_slots:
            maximum = self._decode_max_slots[lease.instance_id]
            self._decode_free_slots[lease.instance_id] = min(
                maximum, self._decode_free_slots[lease.instance_id] + 1
            )
        return True

    def decode_slot_lease(self, operation_id: str) -> DecodeSlotLease | None:
        return self._slot_leases.get(operation_id)

    def decode_slot_lease_counts(self) -> dict[str, int]:
        counts = {"RESERVED": 0, "ACTIVE": 0, "QUARANTINED": 0}
        for lease in self._slot_leases.values():
            if lease.state in counts:
                counts[lease.state] += 1
        return counts

    # Feedback from infer layer

    def on_prefill_done(
        self, instance_id: str, assigned_epoch: str | None = None
    ) -> bool:
        """Release one prefill slot if the assigned epoch is still current."""
        if assigned_epoch is not None and not self.epoch_matches(instance_id, assigned_epoch):
            return False
        if instance_id in self._prefill_load:
            self._prefill_load[instance_id] = max(
                0, self._prefill_load[instance_id] - 1
            )
            return True
        return False

    def on_decode_finished(
        self,
        instance_id: str,
        assigned_epoch: str | None = None,
        operation_id: str | None = None,
    ) -> bool:
        """Release one decode slot if the assigned epoch is still current."""
        if assigned_epoch is not None and not self.epoch_matches(instance_id, assigned_epoch):
            return False
        if operation_id:
            lease = self._slot_leases.get(operation_id)
            if lease is None or lease.instance_id != instance_id \
                    or lease.state in {"RELEASED", "QUARANTINED"}:
                return False
            lease.state = "RELEASED"
        if instance_id in self._decode_free_slots:
            maximum = self._decode_max_slots[instance_id]
            if self._decode_free_slots[instance_id] >= maximum:
                return False
            self._decode_free_slots[instance_id] += 1
            return True
        return False

    def update_kv_usage(
        self, instance_id: str, sample: KVUsageSample | None
    ) -> None:
        """Replace or invalidate an epoch-scoped decode usage sample."""
        if sample is None:
            self._kv_usage.pop(instance_id, None)
            return
        self._kv_usage[instance_id] = KVUsageSample(
            ratio=max(0.0, min(1.0, sample.ratio)),
            instance_epoch=sample.instance_epoch,
            sampled_at=sample.sampled_at,
        )

    def _sample_is_fresh(self, instance_id: str, now: float | None = None) -> bool:
        sample = self._kv_usage.get(instance_id)
        return bool(
            sample is not None
            and sample.instance_epoch == self._instance_epochs.get(instance_id)
            and (time.monotonic() if now is None else now) - sample.sampled_at
            <= self._kv_usage_stale_after_s
        )

    def decode_instance_epochs(self) -> dict[str, str]:
        return {
            instance_id: self._instance_epochs[instance_id]
            for instance_id in self._decode_free_slots
        }

    def instance_epoch(self, instance_id: str) -> str:
        return self._instance_epochs.get(instance_id, "unknown")

    def epoch_matches(self, instance_id: str, assigned_epoch: str) -> bool:
        return bool(assigned_epoch) and self._instance_epochs.get(instance_id) == assigned_epoch

    # Adaptive decode instance count

    def decide_decode_instance_count(self, active_kv_bytes: int) -> int:
        """Compute recommended number of decode instances at runtime.

        Formula: ceil(active_kv_bytes / kv_per_instance), clamped to
                 [min_decode_instances, max_decode_instances].

        Flink AdaptiveBatch analogy:
          Flink : parallelism = ceil(totalBytes / avgDataVolumePerTask)
          prism : decode_count = ceil(active_kv_bytes / kv_per_instance)

        Args:
            active_kv_bytes: total KV size across all active sequences (bytes)

        Returns:
            recommended decode instance count (for autoscaler)
        """
        kv_per = self.config.get("kv_per_instance_bytes", 56 * 1024 ** 3)
        n = math.ceil(active_kv_bytes / kv_per) if active_kv_bytes > 0 else 1
        return max(
            self.config.get("min_decode_instances", 1),
            min(n, self.config.get("max_decode_instances", 64)),
        )

    # Introspection (for tests / monitoring)

    def prefill_queue_depths(self) -> dict[str, int]:
        return dict(self._prefill_load)

    def decode_free_slots(self) -> dict[str, int]:
        return dict(self._decode_free_slots)

    def kv_usage(self) -> dict[str, KVUsageSample]:
        return dict(self._kv_usage)

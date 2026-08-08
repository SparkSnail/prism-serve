"""Persist release decisions for whole-topology replacement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid


STORE_FORMAT_VERSION = 2
STORE_FILE_NAME = "replacement-store.v1.json"


@dataclass(slots=True, frozen=True)
class ReplacementReleaseRecord:
    cleanup_id: str
    restart_run_id: str
    old_operation_digest: str
    operation_id: str
    lease_id: str
    old_resource_kinds: tuple[str, ...]
    decision_digest: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (
            self.cleanup_id,
            self.restart_run_id,
            self.old_operation_digest,
        )


@dataclass(slots=True, frozen=True)
class ReplacementRunSeal:
    seal_sequence: int
    restart_run_id: str
    old_topology_generation_digest: str
    new_topology_generation: str
    decision_digest: str
    record_root_digest: str
    record_count: int

    @property
    def seal_digest(self) -> str:
        return _digest(asdict(self))


class ReplacementStoreError(RuntimeError):
    pass


class ReplacementStoreUnavailable(ReplacementStoreError):
    code = "REPLACEMENT_STORE_UNAVAILABLE"
    http_status = 503


class ReplacementStoreCapacity(ReplacementStoreError):
    code = "REPLACEMENT_STORE_CAPACITY"
    http_status = 503


class ReplacementConflict(ValueError):
    code = "CONFLICT"
    http_status = 409


class RetiredReplacementRun(ReplacementStoreError):
    code = "RETIRED_REPLACEMENT_RUN"
    http_status = 410

    def __init__(self, seal: ReplacementRunSeal) -> None:
        super().__init__(
            "RETIRED_REPLACEMENT_RUN: "
            f"{seal.restart_run_id} ({seal.seal_digest})"
        )
        self.seal = seal


class UnknownReplacementRun(ReplacementStoreError):
    code = "UNKNOWN_REPLACEMENT_RUN"
    http_status = 409


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _record_value(record: ReplacementReleaseRecord) -> dict[str, object]:
    value = asdict(record)
    value["old_resource_kinds"] = list(record.old_resource_kinds)
    return value


def _seal_value(seal: ReplacementRunSeal) -> dict[str, object]:
    return asdict(seal)


class ReplacementDecisionStore:
    """Persist immutable replacement decisions before releasing resources."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_records_per_run: int = 1024,
        seal_retention: int = 2,
    ) -> None:
        if max_records_per_run <= 0:
            raise ValueError("replacement record cap must be positive")
        if seal_retention <= 0:
            raise ValueError("replacement seal retention must be positive")
        self.root = Path(root)
        self.path = self.root / STORE_FILE_NAME
        self.max_records_per_run = max_records_per_run
        self.seal_retention = seal_retention
        self._lock = threading.RLock()
        self._fatal_error: str | None = None
        self._active_run: dict[str, str] | None = None
        self._records: dict[
            tuple[str, str, str], ReplacementReleaseRecord
        ] = {}
        self._seals: list[ReplacementRunSeal] = []
        self._next_seal_sequence = 1
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._load()
            self._finish_sealed_gc_after_restart()
        except Exception as exc:
            self._fatal_error = f"{type(exc).__name__}: {exc}"

    @property
    def ready(self) -> bool:
        return (
            self._fatal_error is None
            and len(self._records) < self.max_records_per_run
        )

    @property
    def last_error(self) -> str | None:
        return self._fatal_error

    @property
    def active_record_count(self) -> int:
        return len(self._records)

    @property
    def seal_count(self) -> int:
        return len(self._seals)

    @property
    def active_restart_run_id(self) -> str | None:
        if self._active_run is None:
            return None
        return self._active_run["restart_run_id"]

    @property
    def active_old_topology_generation_digest(self) -> str | None:
        if self._active_run is None:
            return None
        return self._active_run["old_topology_generation_digest"]

    @property
    def active_decision_digest(self) -> str | None:
        if self._active_run is None:
            return None
        return self._active_run["decision_digest"]

    @property
    def transition_closed(self) -> bool:
        return self._fatal_error is None and self._active_run is None

    @property
    def active_record_root_digest(self) -> str:
        return self._record_root(self._records.values())

    def seals(self) -> tuple[ReplacementRunSeal, ...]:
        return tuple(self._seals)

    def exact_completed_run(
        self,
        *,
        restart_run_id: str,
        old_topology_generation: str,
        new_topology_generation: str,
        decision_digest: str,
    ) -> ReplacementRunSeal | None:
        """Classify an exact replay of a retained, already sealed run.

        A seal is only a bounded completion receipt.  It never authorizes a new
        release; callers must still validate the current new worker world.
        """
        with self._lock:
            self._require_integrity()
            seal = self._seal_for_run(restart_run_id)
            if seal is None:
                return None
            if seal.old_topology_generation_digest != _digest(
                old_topology_generation
            ):
                raise ReplacementConflict(
                    "sealed replacement run reused with different old generation"
                )
            if seal.new_topology_generation != new_topology_generation:
                raise ReplacementConflict(
                    "sealed replacement run reused with different new generation"
                )
            if seal.decision_digest != decision_digest:
                raise ReplacementConflict(
                    "sealed replacement run reused with different decision evidence"
                )
            return seal

    def validate_cold_recovery_base(
        self, *, old_topology_generation: str
    ) -> None:
        """Validate the predecessor chain before a zero-local-lease recovery."""
        with self._lock:
            self._require_integrity()
            if self._active_run is not None or self._records:
                raise ReplacementConflict(
                    "cold replacement recovery requires a closed local segment"
                )
            if (
                self._seals
                and self._seals[-1].new_topology_generation
                != old_topology_generation
            ):
                raise ReplacementConflict(
                    "cold replacement old generation does not follow retained seal"
                )

    def records(self) -> tuple[ReplacementReleaseRecord, ...]:
        return tuple(
            sorted(self._records.values(), key=lambda record: record.key)
        )

    def validate_active_run(
        self,
        *,
        restart_run_id: str,
        old_topology_generation: str,
        decision_digest: str,
    ) -> None:
        """Bind crash recovery evidence to the one durable unsealed run."""
        with self._lock:
            self._require_integrity()
            if self._active_run is None:
                raise UnknownReplacementRun(
                    "UNKNOWN_REPLACEMENT_RUN: no active replacement segment"
                )
            self._validate_requested_run(
                restart_run_id, old_topology_generation, decision_digest
            )

    def state_summary(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "last_error": self.last_error,
            "active_restart_run_id": self.active_restart_run_id,
            "active_old_topology_generation_digest": (
                self.active_old_topology_generation_digest
            ),
            "active_decision_digest": self.active_decision_digest,
            "transition_closed": self.transition_closed,
            "record_count": self.active_record_count,
            "seal_count": self.seal_count,
            "record_root_digest": self.active_record_root_digest,
        }

    def lookup(
        self,
        record: ReplacementReleaseRecord,
        *,
        old_topology_generation: str,
    ) -> ReplacementReleaseRecord | None:
        with self._lock:
            self._require_integrity()
            existing = self._records.get(record.key)
            if existing is not None:
                if existing != record:
                    raise ReplacementConflict(
                        "replacement key reused with different record/digest"
                    )
                return existing
            seal = self._seal_for_run(record.restart_run_id)
            if seal is not None:
                raise RetiredReplacementRun(seal)
            self._validate_requested_run(
                record.restart_run_id,
                old_topology_generation,
                record.decision_digest,
            )
            return None

    def persist_record(
        self,
        record: ReplacementReleaseRecord,
        *,
        old_topology_generation: str,
    ) -> tuple[ReplacementReleaseRecord, bool]:
        with self._lock:
            existing = self.lookup(
                record, old_topology_generation=old_topology_generation
            )
            if existing is not None:
                return existing, False
            if len(self._records) >= self.max_records_per_run:
                raise ReplacementStoreCapacity(
                    "replacement record capacity exhausted"
                )
            for other in self._records.values():
                if (
                    other.operation_id == record.operation_id
                    and other.lease_id == record.lease_id
                    and other.key != record.key
                ):
                    raise ReplacementConflict(
                        "replacement lease already has a different decision key"
                    )
            active_run = self._active_run or {
                "restart_run_id": record.restart_run_id,
                "old_topology_generation_digest": _digest(
                    old_topology_generation
                ),
                "decision_digest": record.decision_digest,
            }
            records = dict(self._records)
            records[record.key] = record
            self._commit_state(
                active_run=active_run,
                records=records,
                seals=self._seals,
                next_seal_sequence=self._next_seal_sequence,
            )
            return record, True

    def persist_records(
        self,
        records: tuple[ReplacementReleaseRecord, ...],
        *,
        restart_run_id: str,
        old_topology_generation: str,
        decision_digest: str | None = None,
    ) -> tuple[ReplacementReleaseRecord, ...]:
        with self._lock:
            self._require_integrity()
            seal = self._seal_for_run(restart_run_id)
            if seal is not None:
                raise RetiredReplacementRun(seal)
            requested_decision = decision_digest or (
                records[0].decision_digest if records else ""
            )
            if not requested_decision:
                raise ReplacementConflict(
                    "replacement batch requires a decision digest"
                )
            if any(record.decision_digest != requested_decision for record in records):
                raise ReplacementConflict("replacement batch mixes decisions")
            self._validate_requested_run(
                restart_run_id, old_topology_generation, requested_decision
            )
            if any(record.restart_run_id != restart_run_id for record in records):
                raise ReplacementConflict("replacement batch mixes restart runs")
            candidate = dict(self._records)
            leases = {
                (record.operation_id, record.lease_id): record.key
                for record in candidate.values()
            }
            changed = False
            for record in records:
                existing = candidate.get(record.key)
                if existing is not None:
                    if existing != record:
                        raise ReplacementConflict(
                            "replacement key reused with different record/digest"
                        )
                    continue
                lease_key = (record.operation_id, record.lease_id)
                if lease_key in leases and leases[lease_key] != record.key:
                    raise ReplacementConflict(
                        "replacement lease already has a different decision key"
                    )
                candidate[record.key] = record
                leases[lease_key] = record.key
                changed = True
            if len(candidate) > self.max_records_per_run:
                raise ReplacementStoreCapacity(
                    "replacement record capacity exhausted"
                )
            if changed or self._active_run is None:
                self._commit_state(
                    active_run={
                        "restart_run_id": restart_run_id,
                        "old_topology_generation_digest": _digest(
                            old_topology_generation
                        ),
                        "decision_digest": requested_decision,
                    },
                    records=candidate,
                    seals=self._seals,
                    next_seal_sequence=self._next_seal_sequence,
                )
            return tuple(candidate[record.key] for record in records)

    def seal_run(
        self,
        *,
        restart_run_id: str,
        old_topology_generation: str,
        new_topology_generation: str,
        decision_digest: str,
    ) -> ReplacementRunSeal:
        with self._lock:
            self._require_integrity()
            existing = self._seal_for_run(restart_run_id)
            if existing is not None:
                if (
                    existing.old_topology_generation_digest
                    != _digest(old_topology_generation)
                    or existing.new_topology_generation
                    != new_topology_generation
                    or existing.decision_digest != decision_digest
                ):
                    raise ReplacementConflict(
                        "sealed replacement run reused with different evidence"
                    )
                return existing
            self._validate_requested_run(
                restart_run_id, old_topology_generation, decision_digest
            )
            if self._active_run is None:
                raise UnknownReplacementRun(
                    "UNKNOWN_REPLACEMENT_RUN: no active replacement segment"
                )
            seal = ReplacementRunSeal(
                seal_sequence=self._next_seal_sequence,
                restart_run_id=restart_run_id,
                old_topology_generation_digest=_digest(
                    old_topology_generation
                ),
                new_topology_generation=new_topology_generation,
                decision_digest=decision_digest,
                record_root_digest=self._record_root(self._records.values()),
                record_count=len(self._records),
            )
            seals = [*self._seals, seal][-self.seal_retention :]

            self._commit_state(
                active_run=self._active_run,
                records=self._records,
                seals=seals,
                next_seal_sequence=seal.seal_sequence + 1,
            )

            self._commit_state(
                active_run=None,
                records={},
                seals=seals,
                next_seal_sequence=seal.seal_sequence + 1,
            )
            return seal

    def _load(self) -> None:
        temp_files = tuple(self.root.glob(f".{STORE_FILE_NAME}.*.tmp"))
        if temp_files:
            raise ReplacementStoreError(
                "orphan replacement-store temp file requires operator review"
            )
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if set(raw) != {"format_version", "payload", "payload_digest"}:
            raise ReplacementStoreError("replacement store envelope is invalid")
        if raw["format_version"] != STORE_FORMAT_VERSION:
            raise ReplacementStoreError("unsupported replacement store version")
        payload = raw["payload"]
        if raw["payload_digest"] != _digest(payload):
            raise ReplacementStoreError("replacement store digest mismatch")
        if set(payload) != {
            "active_run",
            "records",
            "seals",
            "next_seal_sequence",
        }:
            raise ReplacementStoreError("replacement store payload is invalid")
        active_run = payload["active_run"]
        if active_run is not None:
            if set(active_run) != {
                "restart_run_id",
                "old_topology_generation_digest",
                "decision_digest",
            }:
                raise ReplacementStoreError("active replacement run is invalid")
            active_run = {
                "restart_run_id": str(active_run["restart_run_id"]),
                "old_topology_generation_digest": str(
                    active_run["old_topology_generation_digest"]
                ),
                "decision_digest": str(active_run["decision_digest"]),
            }
        records: dict[tuple[str, str, str], ReplacementReleaseRecord] = {}
        lease_keys: set[tuple[str, str]] = set()
        for value in payload["records"]:
            record = self._parse_record(value)
            if record.key in records:
                raise ReplacementStoreError("duplicate replacement record key")
            lease_key = (record.operation_id, record.lease_id)
            if lease_key in lease_keys:
                raise ReplacementStoreError("duplicate replacement lease decision")
            records[record.key] = record
            lease_keys.add(lease_key)
        if len(records) > self.max_records_per_run:
            raise ReplacementStoreError("replacement record cap exceeded")
        if records and active_run is None:
            raise ReplacementStoreError("orphan records have no active run")
        if active_run is not None and any(
            record.restart_run_id != active_run["restart_run_id"]
            for record in records.values()
        ):
            raise ReplacementStoreError("active segment mixes replacement runs")
        if active_run is not None and (
            not active_run["decision_digest"]
            or any(
                record.decision_digest != active_run["decision_digest"]
                for record in records.values()
            )
        ):
            raise ReplacementStoreError("active segment mixes replacement decisions")
        seals = [self._parse_seal(value) for value in payload["seals"]]
        if len(seals) > self.seal_retention:
            raise ReplacementStoreError("replacement seal retention exceeded")
        if any(
            left.seal_sequence >= right.seal_sequence
            for left, right in zip(seals, seals[1:])
        ):
            raise ReplacementStoreError("replacement seal sequence is not monotonic")
        if len({seal.restart_run_id for seal in seals}) != len(seals):
            raise ReplacementStoreError("duplicate retained replacement seal")
        next_sequence = int(payload["next_seal_sequence"])
        if next_sequence <= 0 or (
            seals and next_sequence <= seals[-1].seal_sequence
        ):
            raise ReplacementStoreError("invalid next replacement seal sequence")
        self._active_run = active_run
        self._records = records
        self._seals = seals
        self._next_seal_sequence = next_sequence

    def _finish_sealed_gc_after_restart(self) -> None:
        if self._active_run is None:
            return
        seal = self._seal_for_run(self._active_run["restart_run_id"])
        if seal is None:
            return
        if (
            seal.record_count != len(self._records)
            or seal.record_root_digest
            != self._record_root(self._records.values())
            or seal.old_topology_generation_digest
            != self._active_run["old_topology_generation_digest"]
            or seal.decision_digest != self._active_run["decision_digest"]
        ):
            raise ReplacementStoreError(
                "sealed replacement segment root/count mismatch"
            )
        self._commit_state(
            active_run=None,
            records={},
            seals=self._seals,
            next_seal_sequence=self._next_seal_sequence,
        )

    def _validate_requested_run(
        self,
        restart_run_id: str,
        old_topology_generation: str,
        decision_digest: str | None = None,
    ) -> None:
        if not restart_run_id or not old_topology_generation:
            raise UnknownReplacementRun(
                "UNKNOWN_REPLACEMENT_RUN: run and old generation are required"
            )
        if self._active_run is None:
            return
        if self._active_run["restart_run_id"] != restart_run_id:
            raise UnknownReplacementRun(
                "UNKNOWN_REPLACEMENT_RUN: another run is still unsealed"
            )
        if self._active_run["old_topology_generation_digest"] != _digest(
            old_topology_generation
        ):
            raise ReplacementConflict(
                "replacement run reused with different old generation"
            )
        if (
            decision_digest is not None
            and self._active_run["decision_digest"] != decision_digest
        ):
            raise ReplacementConflict(
                "replacement run reused with different decision evidence"
            )

    def _seal_for_run(self, restart_run_id: str) -> ReplacementRunSeal | None:
        return next(
            (
                seal
                for seal in self._seals
                if seal.restart_run_id == restart_run_id
            ),
            None,
        )

    def _commit_state(
        self,
        *,
        active_run: dict[str, str] | None,
        records: dict[tuple[str, str, str], ReplacementReleaseRecord],
        seals: list[ReplacementRunSeal],
        next_seal_sequence: int,
    ) -> None:
        payload = self._payload(
            active_run=active_run,
            records=records,
            seals=seals,
            next_seal_sequence=next_seal_sequence,
        )
        envelope = {
            "format_version": STORE_FORMAT_VERSION,
            "payload": payload,
            "payload_digest": _digest(payload),
        }
        try:
            self._atomic_write(_canonical_json(envelope) + b"\n")
        except Exception as exc:
            self._fatal_error = f"{type(exc).__name__}: {exc}"
            raise ReplacementStoreUnavailable(
                "replacement store persistence failed closed"
            ) from exc
        self._active_run = None if active_run is None else dict(active_run)
        self._records = dict(records)
        self._seals = list(seals)
        self._next_seal_sequence = next_seal_sequence

    def _payload(
        self,
        *,
        active_run: dict[str, str] | None,
        records: dict[tuple[str, str, str], ReplacementReleaseRecord],
        seals: list[ReplacementRunSeal],
        next_seal_sequence: int,
    ) -> dict[str, object]:
        ordered_records = sorted(records.values(), key=lambda record: record.key)
        return {
            "active_run": active_run,
            "records": [_record_value(record) for record in ordered_records],
            "seals": [_seal_value(seal) for seal in seals],
            "next_seal_sequence": next_seal_sequence,
        }

    def _atomic_write(self, data: bytes) -> None:
        temp_path = self.root / (
            f".{STORE_FILE_NAME}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        with temp_path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, self.path)
        directory_fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _require_integrity(self) -> None:
        if self._fatal_error is not None:
            raise ReplacementStoreUnavailable(
                f"replacement store is not ready: {self._fatal_error}"
            )

    @staticmethod
    def _record_root(
        records: object,
    ) -> str:
        values = sorted(
            (_record_value(record) for record in records),
            key=lambda value: (
                value["cleanup_id"],
                value["restart_run_id"],
                value["old_operation_digest"],
            ),
        )
        return _digest(values)

    @staticmethod
    def _parse_record(value: object) -> ReplacementReleaseRecord:
        if not isinstance(value, dict) or set(value) != {
            "cleanup_id",
            "restart_run_id",
            "old_operation_digest",
            "operation_id",
            "lease_id",
            "old_resource_kinds",
            "decision_digest",
        }:
            raise ReplacementStoreError("replacement record is invalid")
        kinds = value["old_resource_kinds"]
        if not isinstance(kinds, list) or any(
            not isinstance(kind, str) for kind in kinds
        ):
            raise ReplacementStoreError("replacement resource kinds are invalid")
        record = ReplacementReleaseRecord(
            cleanup_id=str(value["cleanup_id"]),
            restart_run_id=str(value["restart_run_id"]),
            old_operation_digest=str(value["old_operation_digest"]),
            operation_id=str(value["operation_id"]),
            lease_id=str(value["lease_id"]),
            old_resource_kinds=tuple(kinds),
            decision_digest=str(value["decision_digest"]),
        )
        if not all(
            (
                record.cleanup_id,
                record.restart_run_id,
                record.old_operation_digest,
                record.operation_id,
                record.lease_id,
                record.decision_digest,
            )
        ):
            raise ReplacementStoreError("replacement record fields are empty")
        return record

    @staticmethod
    def _parse_seal(value: object) -> ReplacementRunSeal:
        if not isinstance(value, dict) or set(value) != {
            "seal_sequence",
            "restart_run_id",
            "old_topology_generation_digest",
            "new_topology_generation",
            "decision_digest",
            "record_root_digest",
            "record_count",
        }:
            raise ReplacementStoreError("replacement seal is invalid")
        seal = ReplacementRunSeal(
            seal_sequence=int(value["seal_sequence"]),
            restart_run_id=str(value["restart_run_id"]),
            old_topology_generation_digest=str(
                value["old_topology_generation_digest"]
            ),
            new_topology_generation=str(value["new_topology_generation"]),
            decision_digest=str(value["decision_digest"]),
            record_root_digest=str(value["record_root_digest"]),
            record_count=int(value["record_count"]),
        )
        if (
            seal.seal_sequence <= 0
            or seal.record_count < 0
            or not seal.restart_run_id
            or not seal.old_topology_generation_digest
            or not seal.new_topology_generation
            or not seal.decision_digest
            or not seal.record_root_digest
        ):
            raise ReplacementStoreError("replacement seal fields are invalid")
        return seal

from __future__ import annotations

import json
from dataclasses import replace
import stat

import pytest

from prism_serve.scheduler.replacement_store import (
    ReplacementConflict,
    ReplacementDecisionStore,
    ReplacementReleaseRecord,
    ReplacementStoreCapacity,
    ReplacementStoreUnavailable,
    RetiredReplacementRun,
    UnknownReplacementRun,
)
from prism_serve.scheduler.resource_release import (
    NoLiveReplacementLease,
    ReplacementEvidence,
    ResourceReleaseEvaluator,
)
from prism_serve.scheduler.scheduler import PDScheduler


def _record(
    index: int = 1,
    *,
    run: str = "run-1",
    decision: str = "sha256:decision-1",
) -> ReplacementReleaseRecord:
    return ReplacementReleaseRecord(
        cleanup_id=f"cleanup-{index}",
        restart_run_id=run,
        old_operation_digest=f"sha256:old-operation-{index}",
        operation_id=f"operation-{index}",
        lease_id=f"lease-{index}",
        old_resource_kinds=("DECODE_SLOT",),
        decision_digest=decision,
    )


def _evidence(
    *,
    run: str = "run-1",
    old: str = "world-a",
    new: str = "world-b",
    decision: str = "sha256:decision-1",
    accepted: bool = True,
) -> ReplacementEvidence:
    return ReplacementEvidence(
        restart_run_id=run,
        old_topology_generation=old,
        new_topology_generation=new,
        old_termination_proof_digests=("t0", "t1", "t2", "t3"),
        fresh_resource_report_digests=("r0", "r1", "r2", "r3"),
        excluded_old_operation_digest="sha256:excluded-old-operations",
        accepted=accepted,
        decision_digest=decision,
    )


def _scheduler_with_quarantine(operation_id: str = "operation-1"):
    scheduler = PDScheduler({})
    scheduler.register_instance(
        "d0", "decode", max_slots=1, instance_epoch="pod-d0:boot-a"
    )
    lease = scheduler.reserve_decode_slot("d0", "request-1", operation_id)
    scheduler.quarantine_decode_slot(operation_id)
    return scheduler, lease


def test_replacement_gc_crash_before_seal_replays_records_and_slot_last(
    tmp_path, monkeypatch
):
    store_path = tmp_path / "replacement-store"
    store = ReplacementDecisionStore(store_path)
    scheduler, lease = _scheduler_with_quarantine()
    evaluator = ResourceReleaseEvaluator(
        scheduler,
        lambda *args, **kwargs: None,
        replacement_store=store,
    )
    original_release = scheduler.release_quarantined_decode_slot

    def checked_release(operation_id, lease_id, cleanup_id):
        restarted = ReplacementDecisionStore(store_path)
        assert restarted.ready is True
        assert restarted.active_record_count == 1
        assert restarted.records()[0].cleanup_id == cleanup_id
        return original_release(operation_id, lease_id, cleanup_id)

    monkeypatch.setattr(
        scheduler, "release_quarantined_decode_slot", checked_release
    )
    record = evaluator.release_whole_world_replaced(
        cleanup_id="cleanup-1",
        operation_id="operation-1",
        lease_id=lease.lease_id,
        old_resource_kinds=("DECODE_SLOT",),
        evidence=_evidence(),
    )

    assert lease.state == "RELEASED"
    assert store.records() == (record,)



    empty_scheduler = PDScheduler({})
    restarted_evaluator = ResourceReleaseEvaluator(
        empty_scheduler,
        lambda *args, **kwargs: None,
        replacement_store=ReplacementDecisionStore(store_path),
    )
    assert restarted_evaluator.release_whole_world_replaced(
        cleanup_id="cleanup-1",
        operation_id="operation-1",
        lease_id=lease.lease_id,
        old_resource_kinds=("DECODE_SLOT",),
        evidence=_evidence(accepted=False),
    ) == record


def test_atomic_writer_orders_file_fsync_replace_then_directory_fsync(
    tmp_path, monkeypatch
):
    store = ReplacementDecisionStore(tmp_path / "store")
    events = []
    original_fsync = __import__("os").fsync
    original_replace = __import__("os").replace

    def tracked_fsync(fd):
        mode = __import__("os").fstat(fd).st_mode
        events.append("directory_fsync" if stat.S_ISDIR(mode) else "file_fsync")
        return original_fsync(fd)

    def tracked_replace(source, target):
        events.append("atomic_replace")
        return original_replace(source, target)

    monkeypatch.setattr("prism_serve.scheduler.replacement_store.os.fsync", tracked_fsync)
    monkeypatch.setattr("prism_serve.scheduler.replacement_store.os.replace", tracked_replace)
    store.persist_record(_record(), old_topology_generation="world-a")

    assert events == ["file_fsync", "atomic_replace", "directory_fsync"]


def test_same_key_different_record_conflicts_without_mutating_store(tmp_path):
    store = ReplacementDecisionStore(tmp_path / "store")
    first = _record()
    store.persist_record(first, old_topology_generation="world-a")
    changed = replace(first, decision_digest="sha256:different")

    with pytest.raises(ReplacementConflict):
        store.persist_record(changed, old_topology_generation="world-a")

    assert store.records() == (first,)


def test_store_allows_one_unsealed_run_and_enforces_small_record_cap(tmp_path):
    store = ReplacementDecisionStore(
        tmp_path / "store", max_records_per_run=2
    )
    store.persist_record(_record(1), old_topology_generation="world-a")
    store.persist_record(_record(2), old_topology_generation="world-a")

    assert store.active_record_count == 2
    assert store.ready is False
    with pytest.raises(ReplacementStoreCapacity):
        store.persist_record(_record(3), old_topology_generation="world-a")
    with pytest.raises(UnknownReplacementRun):
        store.persist_record(
            _record(3, run="run-2"),
            old_topology_generation="world-b",
        )
    assert store.active_record_count == 2


def test_replacement_sealed_retry_returns_retired(tmp_path):
    store = ReplacementDecisionStore(tmp_path / "store")
    seals = []
    old_generation = "world-a"
    for index in range(1, 4):
        run = f"run-{index}"
        store.persist_record(
            _record(index, run=run, decision=f"sha256:decision-{index}"),
            old_topology_generation=old_generation,
        )
        seals.append(
            store.seal_run(
                restart_run_id=run,
                old_topology_generation=old_generation,
                new_topology_generation=f"world-{index + 1}",
                decision_digest=f"sha256:decision-{index}",
            )
        )
        old_generation = f"world-{index + 1}"

    assert store.active_record_count == 0
    assert store.seals() == tuple(seals[-2:])
    with pytest.raises(RetiredReplacementRun) as retired:
        store.lookup(
            _record(2, run="run-2", decision="sha256:decision-2"),
            old_topology_generation="world-2",
        )
    assert retired.value.seal == seals[1]
    assert retired.value.seal.seal_digest in str(retired.value)
    assert retired.value.http_status == 410

    retained = store.exact_completed_run(
        restart_run_id="run-2",
        old_topology_generation="world-2",
        new_topology_generation="world-3",
        decision_digest="sha256:decision-2",
    )
    assert retained == seals[1]
    assert retained.decision_digest == "sha256:decision-2"
    with pytest.raises(ReplacementConflict, match="different decision"):
        store.exact_completed_run(
            restart_run_id="run-2",
            old_topology_generation="world-2",
            new_topology_generation="world-3",
            decision_digest="sha256:changed",
        )


def test_replacement_gc_seal_before_delete_is_atomic(
    tmp_path, monkeypatch
):
    store_path = tmp_path / "store"
    store = ReplacementDecisionStore(store_path)
    store.persist_record(_record(), old_topology_generation="world-a")
    original = store._atomic_write
    writes = 0

    def fail_second_write(data):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("directory full during record GC")
        return original(data)

    monkeypatch.setattr(store, "_atomic_write", fail_second_write)
    with pytest.raises(ReplacementStoreUnavailable):
        store.seal_run(
            restart_run_id="run-1",
            old_topology_generation="world-a",
            new_topology_generation="world-b",
            decision_digest="sha256:decision-1",
        )
    assert store.ready is False

    restarted = ReplacementDecisionStore(store_path)
    assert restarted.ready is True
    assert restarted.active_record_count == 0
    assert restarted.seal_count == 1


def test_replacement_store_corruption_or_disk_full_fail_closed(
    tmp_path, monkeypatch
):
    store_path = tmp_path / "store"
    store = ReplacementDecisionStore(store_path)
    store.persist_record(_record(), old_topology_generation="world-a")
    state_path = store.path
    value = json.loads(state_path.read_text(encoding="utf-8"))
    value["payload"]["records"][0]["operation_id"] = "tampered"
    state_path.write_text(json.dumps(value), encoding="utf-8")

    corrupt = ReplacementDecisionStore(store_path)
    assert corrupt.ready is False
    assert "digest mismatch" in corrupt.last_error
    with pytest.raises(ReplacementStoreUnavailable):
        corrupt.persist_record(_record(2), old_topology_generation="world-a")

    clean_path = tmp_path / "temp-store"
    clean_path.mkdir()
    (clean_path / ".replacement-store.v1.json.crash.tmp").write_text(
        "partial", encoding="utf-8"
    )
    orphan = ReplacementDecisionStore(clean_path)
    assert orphan.ready is False
    assert "temp file" in orphan.last_error

    io_store = ReplacementDecisionStore(tmp_path / "disk-full-store")

    def fail_write(_data):
        raise OSError("disk full")

    monkeypatch.setattr(io_store, "_atomic_write", fail_write)
    with pytest.raises(ReplacementStoreUnavailable):
        io_store.persist_record(_record(), old_topology_generation="world-a")
    assert io_store.ready is False


def test_io_failure_keeps_lease_quarantined_and_does_not_rollback_record(
    tmp_path, monkeypatch
):
    store = ReplacementDecisionStore(tmp_path / "store")
    scheduler, lease = _scheduler_with_quarantine()
    evaluator = ResourceReleaseEvaluator(
        scheduler,
        lambda *args, **kwargs: None,
        replacement_store=store,
    )

    def fail_write(_data):
        raise OSError("disk full")

    monkeypatch.setattr(store, "_atomic_write", fail_write)
    with pytest.raises(ReplacementStoreUnavailable):
        evaluator.release_whole_world_replaced(
            cleanup_id="cleanup-1",
            operation_id="operation-1",
            lease_id=lease.lease_id,
            old_resource_kinds=("DECODE_SLOT",),
            evidence=_evidence(),
        )

    assert lease.state == "QUARANTINED"
    assert store.ready is False
    assert store.active_record_count == 0


def test_replacement_store_cap_rejects_1025_without_release(tmp_path):
    store = ReplacementDecisionStore(tmp_path / "store")
    records = tuple(_record(index) for index in range(1, 1025))
    store.persist_records(
        records,
        restart_run_id="run-1",
        old_topology_generation="world-a",
    )

    assert store.active_record_count == 1024
    assert store.ready is False
    with pytest.raises(ReplacementStoreCapacity):
        store.persist_record(
            _record(1025), old_topology_generation="world-a"
        )
    assert store.active_record_count == 1024


def test_replacement_store_bounds_1024_records_two_seals(tmp_path):
    store = ReplacementDecisionStore(tmp_path / "store")
    records = tuple(_record(index) for index in range(1, 1025))
    store.persist_records(
        records,
        restart_run_id="run-1",
        old_topology_generation="world-a",
    )
    first = store.seal_run(
        restart_run_id="run-1",
        old_topology_generation="world-a",
        new_topology_generation="world-b",
        decision_digest="sha256:decision-1",
    )
    for index, (run, old) in enumerate(
        (("run-2", "world-b"), ("run-3", "world-c")), 2
    ):
        store.persist_record(
            _record(index, run=run, decision=f"sha256:decision-{index}"),
            old_topology_generation=old,
        )
        store.seal_run(
            restart_run_id=run,
            old_topology_generation=old,
            new_topology_generation=f"world-{chr(ord('a') + index)}",
            decision_digest=f"sha256:decision-{index}",
        )

    assert first.record_count == 1024
    assert store.active_record_count == 0
    assert store.seal_count == 2
    assert [seal.restart_run_id for seal in store.seals()] == ["run-2", "run-3"]


def test_replacement_unknown_old_run_rejected(tmp_path):
    store = ReplacementDecisionStore(tmp_path / "store")
    store.persist_record(_record(), old_topology_generation="world-a")

    with pytest.raises(UnknownReplacementRun, match="UNKNOWN") as unknown:
        store.persist_record(
            _record(2, run="unknown-old-run"),
            old_topology_generation="world-old",
        )
    assert store.active_restart_run_id == "run-1"
    assert unknown.value.http_status == 409


def test_store_miss_requires_exact_live_quarantined_lease(tmp_path):
    store = ReplacementDecisionStore(tmp_path / "store")
    evaluator = ResourceReleaseEvaluator(
        PDScheduler({}),
        lambda *args, **kwargs: None,
        replacement_store=store,
    )
    with pytest.raises(NoLiveReplacementLease, match="NO_LIVE"):
        evaluator.release_whole_world_replaced(
            cleanup_id="cleanup-1",
            operation_id="operation-1",
            lease_id="missing-lease",
            old_resource_kinds=("DECODE_SLOT",),
            evidence=_evidence(),
        )
    assert store.active_record_count == 0


def test_whole_world_release_without_durable_store_fails_closed():
    scheduler, lease = _scheduler_with_quarantine()
    evaluator = ResourceReleaseEvaluator(
        scheduler, lambda *args, **kwargs: None
    )
    with pytest.raises(ReplacementStoreUnavailable, match="durable"):
        evaluator.release_whole_world_replaced(
            cleanup_id="cleanup-1",
            operation_id="operation-1",
            lease_id=lease.lease_id,
            old_resource_kinds=("DECODE_SLOT",),
            evidence=_evidence(),
        )
    assert lease.state == "QUARANTINED"


def test_batch_is_one_durable_barrier_before_any_slot_release(tmp_path):
    store = ReplacementDecisionStore(tmp_path / "store")
    scheduler = PDScheduler({})
    scheduler.register_instance(
        "d0", "decode", max_slots=2, instance_epoch="pod-d0:boot-a"
    )
    leases = []
    for index in (1, 2):
        lease = scheduler.reserve_decode_slot(
            "d0", f"request-{index}", f"operation-{index}"
        )
        scheduler.quarantine_decode_slot(f"operation-{index}")
        leases.append(lease)
    evaluator = ResourceReleaseEvaluator(
        scheduler,
        lambda *args, **kwargs: None,
        replacement_store=store,
    )
    entries = tuple(
        (
            f"cleanup-{index}",
            f"operation-{index}",
            leases[index - 1].lease_id,
            ("DECODE_SLOT",),
        )
        for index in (1, 2)
    )

    records = evaluator.persist_whole_world_replaced_batch(
        entries, evidence=_evidence()
    )
    assert store.active_record_count == 2
    assert store.transition_closed is False
    assert all(lease.state == "QUARANTINED" for lease in leases)

    evaluator.release_persisted_replacement_batch(records)
    assert all(lease.state == "RELEASED" for lease in leases)
    seal = evaluator.seal_whole_world_replacement(_evidence())
    assert seal.record_count == 2
    assert store.active_record_count == 0
    assert store.seal_count == 1
    assert store.transition_closed is True


def test_durable_record_cannot_be_rolled_back(tmp_path):
    store = ReplacementDecisionStore(tmp_path / "store")
    scheduler, lease = _scheduler_with_quarantine()
    evaluator = ResourceReleaseEvaluator(
        scheduler,
        lambda *args, **kwargs: None,
        replacement_store=store,
    )
    evaluator.release_whole_world_replaced(
        cleanup_id="cleanup-1",
        operation_id="operation-1",
        lease_id=lease.lease_id,
        old_resource_kinds=("DECODE_SLOT",),
        evidence=_evidence(),
    )

    with pytest.raises(RuntimeError, match="cannot be rolled back"):
        evaluator.rollback_whole_world_replaced(
            cleanup_id="cleanup-1", evidence=_evidence()
        )
    assert store.active_record_count == 1

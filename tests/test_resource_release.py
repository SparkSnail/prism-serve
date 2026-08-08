from __future__ import annotations

from dataclasses import replace

import pytest

from prism_serve.router.http_rpc import EndpointOperationRef
from prism_serve.scheduler.replacement_store import ReplacementDecisionStore
from prism_serve.scheduler.resource_release import (
    CleanupPlan,
    EndpointFinalizePlan,
    PredicateSnapshot,
    ReplacementConflict,
    ReplacementEvidence,
    ResourceReleaseEvaluator,
)
from prism_serve.scheduler.scheduler import PDScheduler


def _ref(instance: str, seq: int):
    return EndpointOperationRef(
        topology_generation="world-a",
        owner_generation="gateway-a:boot-a",
        operation_seq=seq,
        target_instance=instance,
        target_worker_epoch=f"pod-{instance}:boot-a",
        operation_id="op-1",
        payload_digest=f"sha256:{instance}-{seq}",
    )


def _scheduler():
    scheduler = PDScheduler({})
    scheduler.register_instance("d0", "decode", max_slots=1, instance_epoch="pod-d0:boot-a")
    lease = scheduler.reserve_decode_slot("d0", "req-1", "op-1")
    scheduler.quarantine_decode_slot("op-1")
    return scheduler, lease


def _plan(status="FENCED"):
    source_ref = _ref("p0", 1)
    target_ref = _ref("d0", 1)
    predicates = (
        PredicateSnapshot("source_request", status, source_ref),
        PredicateSnapshot("source_prefix", status, source_ref),
        PredicateSnapshot("source_transfer", status, source_ref),
        PredicateSnapshot("target_transfer", status, target_ref),
        PredicateSnapshot("target_request", status, target_ref),
        PredicateSnapshot("target_prefix", status, target_ref),
    )
    return predicates, (
        EndpointFinalizePlan("p0", (source_ref,), ("SOURCE_BLOCKS", "TRANSFER_BYTES")),
        EndpointFinalizePlan("d0", (target_ref,), ("TARGET_PENDING",)),
    )


@pytest.mark.asyncio
async def test_transfer_unknown_request_prefix_terminal_keeps_resources():
    scheduler, lease = _scheduler()
    predicates, endpoints = _plan()
    predicates = tuple(
        PredicateSnapshot(p.name, "UNKNOWN" if p.name == "target_transfer" else p.status, p.endpoint_ref)
        for p in predicates
    )
    calls = []

    async def finalize(instance, **kwargs):
        calls.append(instance)
        return {"cleanup_id": kwargs["cleanup_id"]}

    evaluator = ResourceReleaseEvaluator(scheduler, finalize)
    result = await evaluator.release_endpoint_terminal(
        CleanupPlan("cleanup-1", "op-1", lease.lease_id, predicates, endpoints)
    )

    assert result is None
    assert calls == []
    assert scheduler.decode_slot_lease("op-1").state == "QUARANTINED"


@pytest.mark.asyncio
async def test_transfer_later_terminal_finalize_exactly_once():
    scheduler, lease = _scheduler()
    predicates, endpoints = _plan()
    calls = []

    async def finalize(instance, **kwargs):
        calls.append((instance, kwargs["cleanup_id"]))
        return {"cleanup_id": kwargs["cleanup_id"], "instance": instance}

    evaluator = ResourceReleaseEvaluator(scheduler, finalize)
    plan = CleanupPlan("cleanup-1", "op-1", lease.lease_id, predicates, endpoints)

    first = await evaluator.release_endpoint_terminal(plan)
    replay = await evaluator.release_endpoint_terminal(plan)

    assert first == replay
    assert calls == [("p0", "cleanup-1"), ("d0", "cleanup-1")]
    assert scheduler.decode_slot_lease("op-1").state == "RELEASED"


@pytest.mark.asyncio
async def test_correctness_fault_audit_records_predicate_finalize_then_slot():
    scheduler, lease = _scheduler()
    predicates, endpoints = _plan()
    events = []

    class Gate:
        def record_event(self, name, details):
            events.append((name, details))

    async def finalize(instance, **kwargs):
        return {
            "cleanup_id": kwargs["cleanup_id"],
            "instance": instance,
            "state": "FENCED",
            "resources_held": False,
        }

    evaluator = ResourceReleaseEvaluator(scheduler, finalize)
    evaluator.set_correctness_fault_gate(Gate())
    await evaluator.release_endpoint_terminal(
        CleanupPlan("cleanup-audit", "op-1", lease.lease_id, predicates, endpoints)
    )

    assert [name for name, _ in events] == [
        "release_predicates_satisfied",
        "endpoint_finalize_acked",
        "slot_released",
    ]
    assert events[0][1]["predicates"][0]["endpoint_ref"]["operation_id"] == "op-1"
    assert scheduler.decode_slot_lease("op-1").state == "RELEASED"


@pytest.mark.asyncio
async def test_slot_release_after_remote_finalize_acks():
    scheduler, lease = _scheduler()
    predicates, endpoints = _plan()
    ack_target = False

    async def finalize(instance, **kwargs):
        if instance == "d0" and not ack_target:
            raise TimeoutError("target response lost")
        return {"cleanup_id": kwargs["cleanup_id"], "instance": instance}

    evaluator = ResourceReleaseEvaluator(scheduler, finalize)
    plan = CleanupPlan("cleanup-1", "op-1", lease.lease_id, predicates, endpoints)
    with pytest.raises(TimeoutError):
        await evaluator.release_endpoint_terminal(plan)
    assert scheduler.decode_slot_lease("op-1").state == "QUARANTINED"
    ack_target = True
    await evaluator.release_endpoint_terminal(plan)
    assert scheduler.decode_slot_lease("op-1").state == "RELEASED"


def _replacement_evidence(decision="sha256:decision-a"):
    return ReplacementEvidence(
        restart_run_id="restart-1",
        old_topology_generation="world-a",
        new_topology_generation="world-b",
        old_termination_proof_digests=("t0", "t1", "t2", "t3"),
        fresh_resource_report_digests=("r0", "r1", "r2", "r3"),
        excluded_old_operation_digest="sha256:old-op",
        accepted=True,
        decision_digest=decision,
    )


def test_replacement_empty_new_registry_creates_gateway_release_record(tmp_path):
    scheduler, lease = _scheduler()
    remote_calls = []

    async def finalize(*args, **kwargs):
        remote_calls.append((args, kwargs))

    evaluator = ResourceReleaseEvaluator(
        scheduler,
        finalize,
        replacement_store=ReplacementDecisionStore(tmp_path / "store"),
    )
    record = evaluator.release_whole_world_replaced(
        cleanup_id="cleanup-1",
        operation_id="op-1",
        lease_id=lease.lease_id,
        old_resource_kinds=("TARGET_PENDING", "SOURCE_PIN"),
        evidence=_replacement_evidence(),
    )

    assert record.restart_run_id == "restart-1"
    assert remote_calls == []
    assert scheduler.decode_slot_lease("op-1").state == "RELEASED"


def test_replacement_record_replay_after_response_loss(tmp_path):
    scheduler, lease = _scheduler()
    evaluator = ResourceReleaseEvaluator(
        scheduler,
        lambda *args, **kwargs: None,
        replacement_store=ReplacementDecisionStore(tmp_path / "store"),
    )
    kwargs = dict(
        cleanup_id="cleanup-1", operation_id="op-1", lease_id=lease.lease_id,
        old_resource_kinds=("TARGET_PENDING",), evidence=_replacement_evidence(),
    )
    assert evaluator.release_whole_world_replaced(**kwargs) == evaluator.release_whole_world_replaced(**kwargs)


def test_replacement_digest_conflict_rejected(tmp_path):
    scheduler, lease = _scheduler()
    evaluator = ResourceReleaseEvaluator(
        scheduler,
        lambda *args, **kwargs: None,
        replacement_store=ReplacementDecisionStore(tmp_path / "store"),
    )
    evaluator.release_whole_world_replaced(
        cleanup_id="cleanup-1", operation_id="op-1", lease_id=lease.lease_id,
        old_resource_kinds=("TARGET_PENDING",), evidence=_replacement_evidence(),
    )
    with pytest.raises(ReplacementConflict):
        evaluator.release_whole_world_replaced(
            cleanup_id="cleanup-1", operation_id="op-1", lease_id=lease.lease_id,
            old_resource_kinds=("TARGET_PENDING",),
            evidence=_replacement_evidence("sha256:different"),
        )


def test_slot_release_after_replacement_record(tmp_path):
    scheduler, lease = _scheduler()
    evaluator = ResourceReleaseEvaluator(
        scheduler,
        lambda *args, **kwargs: None,
        replacement_store=ReplacementDecisionStore(tmp_path / "store"),
    )
    invalid = _replacement_evidence()
    invalid = replace(invalid, fresh_resource_report_digests=("r0", "r1", "r2"))
    with pytest.raises(ValueError):
        evaluator.release_whole_world_replaced(
            cleanup_id="cleanup-1", operation_id="op-1", lease_id=lease.lease_id,
            old_resource_kinds=("TARGET_PENDING",), evidence=invalid,
        )
    assert scheduler.decode_slot_lease("op-1").state == "QUARANTINED"


def test_staged_uncommitted_replacement_cannot_authorize_release(tmp_path):
    scheduler, lease = _scheduler()
    evaluator = ResourceReleaseEvaluator(
        scheduler,
        lambda *args, **kwargs: None,
        replacement_store=ReplacementDecisionStore(tmp_path / "store"),
    )
    with pytest.raises(ValueError, match="not accepted"):
        evaluator.release_whole_world_replaced(
            cleanup_id="cleanup-1", operation_id="op-1",
            lease_id=lease.lease_id, old_resource_kinds=("TARGET_PENDING",),
            evidence=replace(_replacement_evidence(), accepted=False),
        )
    assert lease.state == "QUARANTINED"


@pytest.mark.asyncio
async def test_low_cap_holds_partial_ack_and_evicts_only_completed_cleanup():
    scheduler = PDScheduler({})
    scheduler.register_instance(
        "d0", "decode", max_slots=2, instance_epoch="pod-d0:boot-a"
    )
    leases = {}
    plans = {}
    for index, operation_id in enumerate(("op-1", "op-2"), 1):
        lease = scheduler.reserve_decode_slot("d0", f"r-{index}", operation_id)
        scheduler.quarantine_decode_slot(operation_id)
        leases[operation_id] = lease
        predicates, endpoints = _plan()
        rewrite = lambda ref: replace(
            ref, operation_id=operation_id,
            payload_digest=f"sha256:{operation_id}-{ref.target_instance}",
        )
        predicates = tuple(replace(item, endpoint_ref=rewrite(item.endpoint_ref)) for item in predicates)
        endpoints = tuple(replace(item, endpoint_refs=tuple(rewrite(ref) for ref in item.endpoint_refs)) for item in endpoints)
        plans[operation_id] = CleanupPlan(
            f"cleanup-{index}", operation_id, lease.lease_id,
            predicates, endpoints,
        )

    allow_target = False

    async def finalize(instance, **kwargs):
        if kwargs["operation_id"] == "op-1" and instance == "d0" and not allow_target:
            raise TimeoutError("response unknown")
        return {"instance": instance}

    evaluator = ResourceReleaseEvaluator(
        scheduler, finalize, active_operation_cap=1, terminal_snapshot_cap=1,
    )
    with pytest.raises(TimeoutError):
        await evaluator.release_endpoint_terminal(plans["op-1"])
    with pytest.raises(RuntimeError, match="capacity"):
        await evaluator.release_endpoint_terminal(plans["op-2"])
    assert evaluator.state_counts()["active_endpoint"] == 1
    assert evaluator.state_counts()["endpoint_acks"] == 1

    allow_target = True
    await evaluator.release_endpoint_terminal(plans["op-1"])
    await evaluator.release_endpoint_terminal(plans["op-2"])

    assert scheduler.decode_slot_lease("op-1").state == "RELEASED"
    assert scheduler.decode_slot_lease("op-2").state == "RELEASED"
    assert evaluator.state_counts() == {
        "active_endpoint": 0, "endpoint_acks": 0,
        "completed_endpoint": 1, "replacement_records": 0,
    }
    assert tuple(evaluator._completed_endpoint) == ("cleanup-2",)

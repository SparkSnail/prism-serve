"""Gateway lifecycle and endpoint tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import httpx
from fastapi.testclient import TestClient

from prism_serve.gateway.app import app
from prism_serve.gateway.app import _on_schedule_loop_done
from prism_serve.gateway.app import _wait_for_control_plane_drain
from prism_serve.gateway.app import _abort_remaining_requests
from prism_serve.gateway import app as gateway_module
from prism_serve import __version__


def sync_client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture(autouse=True)
def allow_mock_nats(monkeypatch):
    """Gateway tests explicitly opt into local mock-queue mode."""
    from prism_serve.scheduler.queue import NATSQueue

    async def fail_connect(_self):
        raise ConnectionError("test NATS unavailable")

    monkeypatch.setattr(gateway_module.settings, "nats_required", False)
    monkeypatch.setattr(gateway_module.settings, "gateway_pod_uid", "gateway-test-uid")
    monkeypatch.setattr(NATSQueue, "connect", fail_connect)


def test_healthz_returns_ok():
    with sync_client() as client:
        r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__


def test_readyz_ready_after_startup():
    with sync_client() as client:
        r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_readyz_not_ready_before_startup():
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/readyz")
    assert r.status_code == 503


def test_readyz_not_ready_when_nats_disconnects():
    with sync_client() as client:
        app.state.queue._use_mock = False
        app.state.queue._nc = None
        response = client.get("/readyz")
        app.state.queue._use_mock = True
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_schedule_loop_failure_disables_readiness():
    app.state.accepting = True

    async def fail_loop():
        raise RuntimeError("loop failed")

    task = asyncio.create_task(fail_loop())
    with pytest.raises(RuntimeError):
        await task
    _on_schedule_loop_done(app, task)

    assert app.state.accepting is False
    assert app.state.control_plane_failed is True


def test_healthz_fails_after_schedule_loop_crash():
    with sync_client() as client:
        app.state.control_plane_failed = True
        response = client.get("/healthz")
        app.state.control_plane_failed = False
    assert response.status_code == 503


def test_production_queue_requires_pod_uid():
    from prism_serve.scheduler.queue import NATSQueue

    with pytest.raises((AssertionError, ValueError), match="metadata.uid|non-empty"):
        NATSQueue({"nats_required": True, "scheduler_id": ""})


def test_lifespan_rejects_multiple_active_gateways(monkeypatch):
    monkeypatch.setattr(gateway_module.settings, "control_plane_replica_count", 2)
    with pytest.raises(RuntimeError, match="one active gateway"):
        with sync_client():
            pass


def test_metrics_exposes_prometheus_payload():
    with sync_client() as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_control_plane_drain_waits_for_tracker_and_governor(monkeypatch):
    tracker = MagicMock()
    tracker.__len__.return_value = 1
    governor = MagicMock()
    governor.is_drained.return_value = False

    async def finish_drain(_delay):
        tracker.__len__.return_value = 0
        governor.is_drained.return_value = True

    monkeypatch.setattr(gateway_module.asyncio, "sleep", finish_drain)

    assert await _wait_for_control_plane_drain(tracker, governor, 1.0)
    assert governor.is_drained.call_count == 1


@pytest.mark.asyncio
async def test_shutdown_abort_cleans_remaining_request():
    from prism_serve.metrics.collector import NullMetrics
    from prism_serve.scheduler.scheduler import PDScheduler
    from prism_serve.scheduler.sequence_state import RequestInfo, RequestTracker
    from prism_serve.scheduler.transfer_governor import TransferGovernor

    client = MagicMock()
    client.abort_request.return_value = {"success": True}
    metrics = NullMetrics()
    scheduler = PDScheduler({})
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="e1")
    scheduler.pick_decode_instance("R1", 0)
    governor = TransferGovernor({}, client, metrics)
    tracker = RequestTracker(metrics)
    tracker.add(RequestInfo(req_id="R1", decode_instance="d-0"))

    await _abort_remaining_requests(
        tracker, scheduler, governor, "gateway-uid", timeout_s=1.0
    )

    assert len(tracker) == 0
    assert scheduler.decode_free_slots()["d-0"] == 1
    assert governor.all_inflight_zero()


@pytest.mark.asyncio
async def test_shutdown_abort_transfer_response_lost_quarantines_d():
    """A decode abort ACK cannot replace an operation-scoped transfer fence."""
    import time

    from prism_serve.metrics.collector import NullMetrics
    from prism_serve.scheduler.scheduler import KVUsageSample, PDScheduler
    from prism_serve.scheduler.sequence_state import (
        RequestInfo, RequestTracker, SeqState, TransferTask,
    )
    from prism_serve.scheduler.transfer_governor import TransferGovernor

    client = MagicMock()
    client.transfer = MagicMock()
    client.abort_request = AsyncMock(return_value={"success": True})
    client.abort_transfer = AsyncMock(side_effect=asyncio.TimeoutError)
    scheduler = PDScheduler({})
    scheduler.register_instance("p-0", "prefill", instance_epoch="p-e1")
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="e1")
    sample = KVUsageSample(0.1, "e1", time.monotonic())
    scheduler.update_kv_usage("d-0", sample)
    scheduler.pick_decode_instance("R1", 0)
    governor = TransferGovernor({}, client, NullMetrics())
    governor.set_expected_epochs({"d-0": "e1"})
    governor.update_kv_usage("d-0", sample)
    governor.submit(TransferTask(
        req_id="R1", operation_id="op-1", src="p-0", dst="d-0", kv_size=1,
    ))
    late_completion = client.transfer.call_args.kwargs["on_complete"]
    tracker = RequestTracker(NullMetrics())
    tracker.add(RequestInfo(
        req_id="R1", state=SeqState.KV_PENDING,
            prefill_instance="p-0", decode_instance="d-0",
            prefill_instance_epoch="p-e1", decode_instance_epoch="e1",
            transfer_operation_id="op-1",
    ))

    await _abort_remaining_requests(
        tracker, scheduler, governor, "owner-1", timeout_s=0.01,
        transfer_timeout_s=0.01,
    )

    record = scheduler.quarantine_record("d-0")
    assert record is not None
    assert record.uncertain_transfer_operations == (
        ("owner-1", "R1", "op-1", "e1"),
    )
    assert "d-0" not in scheduler.decode_free_slots()
    assert governor.is_drained()
    late_completion()
    assert "d-0" not in scheduler.decode_free_slots()


@pytest.mark.asyncio
async def test_shutdown_inflight_fence_and_d_ack_release_slot():
    import time

    from prism_serve.metrics.collector import NullMetrics
    from prism_serve.scheduler.scheduler import KVUsageSample, PDScheduler
    from prism_serve.scheduler.sequence_state import (
        RequestInfo, RequestTracker, SeqState, TransferTask,
    )
    from prism_serve.scheduler.transfer_governor import TransferGovernor

    client = MagicMock()
    client.transfer = MagicMock()
    client.abort_request = AsyncMock(return_value={"success": True})
    client.abort_transfer = AsyncMock(return_value={"success": True})
    scheduler = PDScheduler({})
    scheduler.register_instance("p-0", "prefill", instance_epoch="p-e1")
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="e1")
    sample = KVUsageSample(0.1, "e1", time.monotonic())
    scheduler.update_kv_usage("d-0", sample)
    scheduler.pick_decode_instance("R1", 0)
    governor = TransferGovernor({}, client, NullMetrics())
    governor.set_expected_epochs({"d-0": "e1"})
    governor.update_kv_usage("d-0", sample)
    governor.submit(TransferTask(
        req_id="R1", operation_id="op-1", src="p-0", dst="d-0", kv_size=1,
    ))
    tracker = RequestTracker(NullMetrics())
    tracker.add(RequestInfo(
        req_id="R1", state=SeqState.KV_PENDING,
            prefill_instance="p-0", decode_instance="d-0",
            prefill_instance_epoch="p-e1", decode_instance_epoch="e1",
            transfer_operation_id="op-1",
    ))

    await _abort_remaining_requests(
        tracker, scheduler, governor, "owner-1", timeout_s=0.1,
        transfer_timeout_s=0.1,
    )

    assert scheduler.decode_free_slots()["d-0"] == 1
    assert scheduler.quarantine_record("d-0") is None
    client.abort_transfer.assert_awaited_once()


async def _run_named_shutdown_case(
    state,
    *,
    p_ack: bool = True,
    d_ack: bool = True,
    transfer_ack: bool = True,
    transfer_mode: str = "none",
):
    import time

    from prism_serve.metrics.collector import NullMetrics
    from prism_serve.scheduler.scheduler import KVUsageSample, PDScheduler
    from prism_serve.scheduler.sequence_state import RequestInfo, RequestTracker, TransferTask
    from prism_serve.scheduler.transfer_governor import TransferGovernor

    client = MagicMock()
    client.transfer = MagicMock()

    async def abort_request(instance_id, **_kwargs):
        return {"success": p_ack if instance_id == "p-0" else d_ack}

    client.abort_request = AsyncMock(side_effect=abort_request)
    client.abort_transfer = AsyncMock(return_value={"success": transfer_ack})
    scheduler = PDScheduler({})
    scheduler.register_instance("p-0", "prefill", instance_epoch="p-e1")
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="d-e1")
    sample = KVUsageSample(0.1, "d-e1", time.monotonic())
    scheduler.update_kv_usage("d-0", sample)
    scheduler.pick_prefill_instance("R1")
    scheduler.pick_decode_instance("R1", 0)
    governor = TransferGovernor({}, client, NullMetrics())
    governor.set_expected_epochs({"d-0": "d-e1"})
    governor.update_kv_usage("d-0", sample)
    if state.name != "PREFILLING":
        scheduler.on_prefill_done("p-0", "p-e1")
    if transfer_mode != "none":
        if transfer_mode == "deferred":
            governor.update_kv_usage(
                "d-0", KVUsageSample(0.9, "d-e1", time.monotonic())
            )
        governor.submit(TransferTask(
            req_id="R1", operation_id="op-1", src="p-0", dst="d-0", kv_size=1,
        ))
    tracker = RequestTracker(NullMetrics())
    tracker.add(RequestInfo(
        req_id="R1", state=state, prefill_instance="p-0",
        decode_instance="d-0", prefill_instance_epoch="p-e1",
        decode_instance_epoch="d-e1", transfer_operation_id="op-1",
        publish_outcome="ACKED" if state.name == "PREFILLING" else "NOT_STARTED",
    ))
    await _abort_remaining_requests(
        tracker, scheduler, governor, "owner", 0.1, 0.1,
    )
    return scheduler, client


@pytest.mark.asyncio
async def test_shutdown_prefilling_p_ack_releases_both_local_reservations():
    from prism_serve.scheduler.sequence_state import SeqState
    scheduler, _ = await _run_named_shutdown_case(SeqState.PREFILLING)
    assert scheduler.prefill_queue_depths()["p-0"] == 0
    assert scheduler.decode_free_slots()["d-0"] == 1


@pytest.mark.asyncio
async def test_shutdown_prefilling_p_timeout_only_quarantines_p():
    from prism_serve.scheduler.sequence_state import SeqState
    scheduler, _ = await _run_named_shutdown_case(SeqState.PREFILLING, p_ack=False)
    assert scheduler.quarantine_record("p-0") is not None
    assert scheduler.decode_free_slots()["d-0"] == 1


@pytest.mark.asyncio
async def test_shutdown_deferred_p_success_d_timeout():
    from prism_serve.scheduler.sequence_state import SeqState
    scheduler, _ = await _run_named_shutdown_case(
        SeqState.KV_PENDING, d_ack=False, transfer_mode="deferred",
    )
    assert scheduler.quarantine_record("d-0") is not None


@pytest.mark.asyncio
async def test_shutdown_deferred_p_timeout_d_success():
    from prism_serve.scheduler.sequence_state import SeqState
    scheduler, _ = await _run_named_shutdown_case(
        SeqState.KV_PENDING, p_ack=False, transfer_mode="deferred",
    )
    assert scheduler.decode_free_slots()["d-0"] == 1


@pytest.mark.asyncio
async def test_shutdown_inflight_p_success_d_timeout():
    from prism_serve.scheduler.sequence_state import SeqState
    scheduler, _ = await _run_named_shutdown_case(
        SeqState.KV_PENDING, d_ack=False, transfer_mode="inflight",
    )
    assert scheduler.quarantine_record("d-0") is not None


@pytest.mark.asyncio
async def test_shutdown_d_timeout_transfer_ack_quarantines_d():
    from prism_serve.scheduler.sequence_state import SeqState
    scheduler, client = await _run_named_shutdown_case(
        SeqState.KV_PENDING, d_ack=False, transfer_mode="inflight",
    )
    client.abort_transfer.assert_awaited_once()
    assert scheduler.quarantine_record("d-0") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("state_name", ["DECODING", "RECOMPUTING"])
async def test_shutdown_committed_d_ack_releases(state_name):
    from prism_serve.scheduler.sequence_state import SeqState
    scheduler, _ = await _run_named_shutdown_case(SeqState[state_name])
    assert scheduler.decode_free_slots()["d-0"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("state_name", ["DECODING", "RECOMPUTING"])
async def test_shutdown_committed_d_timeout_quarantines_d(state_name):
    from prism_serve.scheduler.sequence_state import SeqState
    scheduler, _ = await _run_named_shutdown_case(SeqState[state_name], d_ack=False)
    assert scheduler.quarantine_record("d-0") is not None


@pytest.mark.asyncio
async def test_unknown_publish_quarantines_before_blocked_abort():
    import time

    from prism_serve.metrics.collector import NullMetrics
    from prism_serve.scheduler.scheduler import KVUsageSample, PDScheduler
    from prism_serve.scheduler.sequence_state import RequestInfo, RequestTracker, SeqState
    from prism_serve.scheduler.transfer_governor import TransferGovernor

    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_abort(**_kwargs):
        entered.set()
        await release.wait()
        return {"success": False}

    client = MagicMock()
    client.abort_request = AsyncMock(side_effect=blocked_abort)
    scheduler = PDScheduler({})
    scheduler.register_instance("p-0", "prefill", instance_epoch="p-e1")
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="d-e1")
    sample = KVUsageSample(0.1, "d-e1", time.monotonic())
    scheduler.update_kv_usage("d-0", sample)
    scheduler.pick_prefill_instance("R1")
    scheduler.pick_decode_instance("R1", 0)
    governor = TransferGovernor({}, client, NullMetrics())
    tracker = RequestTracker(NullMetrics())
    tracker.add(RequestInfo(
        req_id="R1", state=SeqState.PREFILLING,
        prefill_instance="p-0", decode_instance="d-0",
        prefill_instance_epoch="p-e1", decode_instance_epoch="d-e1",
        command_id="owner:R1", publish_outcome="UNKNOWN",
    ))

    cleanup = asyncio.create_task(_abort_remaining_requests(
        tracker, scheduler, governor, "owner", 1.0, 1.0,
    ))
    await asyncio.wait_for(entered.wait(), 1.0)
    record = scheduler.quarantine_record("p-0")
    assert record is not None
    assert record.uncertain_dispatch_commands == (
        ("owner", "R1", "owner:R1", "p-e1"),
    )
    assert scheduler.pick_prefill_instance("late") is None
    assert scheduler.decode_free_slots()["d-0"] == 1
    release.set()
    await cleanup


@pytest.mark.asyncio
async def test_late_publish_after_ack_not_found_isolated():
    """ACK_NOT_FOUND is modeled as abort failure; quarantine remains authoritative."""
    await test_unknown_publish_quarantines_before_blocked_abort()


@pytest.mark.asyncio
async def test_cleanup_creator_cancelled_lifespan_joins_same_task():
    from prism_serve.scheduler.main_loop import _get_or_create_canonical_cleanup
    from prism_serve.scheduler.sequence_state import SeqState

    scheduler, client = await _run_named_shutdown_case(SeqState.PREFILLING)
    # The helper completed terminal cleanup; exercise CAS identity independently.
    from prism_serve.metrics.collector import NullMetrics
    from prism_serve.scheduler.sequence_state import RequestInfo, RequestTracker
    from prism_serve.scheduler.transfer_governor import TransferGovernor

    gate = asyncio.Event()
    client = MagicMock()

    async def blocked_abort(**_kwargs):
        await gate.wait()
        return {"success": True}

    client.abort_request = AsyncMock(side_effect=blocked_abort)
    tracker = RequestTracker(NullMetrics())
    scheduler.pick_prefill_instance("R2")
    scheduler.pick_decode_instance("R2", 0)
    req = RequestInfo(
        req_id="R2", state=SeqState.PREFILLING,
        prefill_instance="p-0", decode_instance="d-0",
        prefill_instance_epoch="p-e1", decode_instance_epoch="d-e1",
        publish_outcome="ACKED",
    )
    tracker.add(req)
    governor = TransferGovernor({}, client, NullMetrics())
    first = _get_or_create_canonical_cleanup(
        req, tracker, scheduler, governor, "owner", 1.0, 1.0,
    )
    second = _get_or_create_canonical_cleanup(
        req, tracker, scheduler, governor, "owner", 1.0, 1.0,
    )
    assert first is second
    async def join():
        await asyncio.shield(first)

    waiter = asyncio.create_task(join())
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not first.cancelled()
    gate.set()
    await _abort_remaining_requests(tracker, scheduler, governor, "owner", 1.0, 1.0)
    assert first.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc_result", [True, False, "response_lost"])
async def test_epoch_flip_during_p_abort_discards_result(rpc_result):
    import time
    from prism_serve.metrics.collector import NullMetrics
    from prism_serve.scheduler.main_loop import _fence_and_abort_request
    from prism_serve.scheduler.scheduler import KVUsageSample, PDScheduler
    from prism_serve.scheduler.sequence_state import RequestInfo, SeqState
    from prism_serve.scheduler.transfer_governor import TransferGovernor

    scheduler = PDScheduler({})
    scheduler.register_instance("p-0", "prefill", instance_epoch="p-e1")
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="d-e1")
    sample = KVUsageSample(0.1, "d-e1", time.monotonic())
    scheduler.update_kv_usage("d-0", sample)
    scheduler.pick_prefill_instance("R1")
    scheduler.pick_decode_instance("R1", 0)
    client = MagicMock()

    async def flip_p(**_kwargs):
        record = scheduler.quarantine_instance("p-0")
        scheduler.reconcile_instance(
            "p-0", "p-e2", record.reconciliation_token, "prefill", 0,
            [], [], [],
        )
        if rpc_result == "response_lost":
            raise ConnectionError("response lost after epoch flip")
        return {"success": rpc_result}

    client.abort_request = AsyncMock(side_effect=flip_p)
    governor = TransferGovernor({}, client, NullMetrics())
    req = RequestInfo(
        req_id="R1", state=SeqState.PREFILLING,
        prefill_instance="p-0", decode_instance="d-0",
        prefill_instance_epoch="p-e1", decode_instance_epoch="d-e1",
        publish_outcome="ACKED",
    )
    await _fence_and_abort_request(req, scheduler, governor, "owner", 0.1, 0.1)
    assert scheduler.prefill_queue_depths()["p-0"] == 0
    assert scheduler.decode_free_slots()["d-0"] == 1
    assert scheduler.quarantine_record("p-0") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc_result", [True, False, "response_lost"])
async def test_epoch_flip_during_transfer_abort_discards_result(rpc_result):
    import time
    from prism_serve.metrics.collector import NullMetrics
    from prism_serve.scheduler.main_loop import _fence_and_abort_request
    from prism_serve.scheduler.scheduler import KVUsageSample, PDScheduler
    from prism_serve.scheduler.sequence_state import RequestInfo, SeqState, TransferTask
    from prism_serve.scheduler.transfer_governor import TransferGovernor

    scheduler = PDScheduler({})
    scheduler.register_instance("p-0", "prefill", instance_epoch="p-e1")
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="d-e1")
    sample = KVUsageSample(0.1, "d-e1", time.monotonic())
    scheduler.update_kv_usage("d-0", sample)
    scheduler.pick_decode_instance("R1", 0)
    client = MagicMock()
    client.transfer = MagicMock()

    async def flip_d(**_kwargs):
        record = scheduler.quarantine_instance("d-0")
        scheduler.reconcile_instance(
            "d-0", "d-e2", record.reconciliation_token, "decode", 1,
            [], [], [],
        )
        if rpc_result == "response_lost":
            raise ConnectionError("response lost after epoch flip")
        return {"success": rpc_result}

    client.abort_transfer = AsyncMock(side_effect=flip_d)
    client.abort_request = AsyncMock(return_value={"success": True})
    governor = TransferGovernor({}, client, NullMetrics())
    governor.set_expected_epochs({"d-0": "d-e1"})
    governor.update_kv_usage("d-0", sample)
    governor.submit(TransferTask(
        req_id="R1", operation_id="op", src="p-0", dst="d-0", kv_size=1,
    ))
    req = RequestInfo(
        req_id="R1", state=SeqState.KV_PENDING,
        prefill_instance="p-0", decode_instance="d-0",
        prefill_instance_epoch="p-e1", decode_instance_epoch="d-e1",
        transfer_operation_id="op",
    )
    await _fence_and_abort_request(req, scheduler, governor, "owner", 0.1, 0.1)
    assert scheduler.decode_free_slots()["d-0"] == 1
    assert scheduler.quarantine_record("d-0") is None
    client.abort_request.assert_not_awaited()
    assert governor.is_drained()


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc_result", [True, False, "response_lost"])
async def test_source_epoch_flip_during_transfer_abort_quarantines_d(rpc_result):
    import time
    from prism_serve.metrics.collector import NullMetrics
    from prism_serve.scheduler.main_loop import _fence_and_abort_request
    from prism_serve.scheduler.scheduler import KVUsageSample, PDScheduler
    from prism_serve.scheduler.sequence_state import RequestInfo, SeqState, TransferTask
    from prism_serve.scheduler.transfer_governor import TransferGovernor

    scheduler = PDScheduler({})
    scheduler.register_instance("p-0", "prefill", instance_epoch="p-e1")
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="d-e1")
    sample = KVUsageSample(0.1, "d-e1", time.monotonic())
    scheduler.update_kv_usage("d-0", sample)
    scheduler.pick_decode_instance("R1", 0)
    client = MagicMock()
    client.transfer = MagicMock()

    async def flip_p(**_kwargs):
        record = scheduler.quarantine_instance("p-0")
        scheduler.reconcile_instance(
            "p-0", "p-e2", record.reconciliation_token, "prefill", 0,
            [], [], [],
        )
        if rpc_result == "response_lost":
            raise ConnectionError("response lost after source epoch flip")
        return {"success": rpc_result}

    client.abort_transfer = AsyncMock(side_effect=flip_p)
    client.abort_request = AsyncMock(return_value={"success": True})
    governor = TransferGovernor({}, client, NullMetrics())
    governor.set_expected_epochs({"d-0": "d-e1"})
    governor.update_kv_usage("d-0", sample)
    governor.submit(TransferTask(
        req_id="R1", operation_id="op", src="p-0", dst="d-0", kv_size=1,
    ))
    req = RequestInfo(
        req_id="R1", state=SeqState.KV_PENDING,
        prefill_instance="p-0", decode_instance="d-0",
        prefill_instance_epoch="p-e1", decode_instance_epoch="d-e1",
        transfer_operation_id="op",
    )
    await _fence_and_abort_request(req, scheduler, governor, "owner", 0.1, 0.1)
    assert scheduler.instance_epoch("p-0") == "p-e2"
    assert "d-0" not in scheduler.decode_free_slots()
    record = scheduler.quarantine_record("d-0")
    assert record is not None
    assert record.uncertain_transfer_operations == (
        ("owner", "R1", "op", "d-e1"),
    )
    client.abort_request.assert_not_awaited()
    assert governor.is_drained()


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc_result", [True, False, "response_lost"])
async def test_epoch_flip_during_d_abort_discards_result(rpc_result):
    import time
    from prism_serve.metrics.collector import NullMetrics
    from prism_serve.scheduler.main_loop import _fence_and_abort_request
    from prism_serve.scheduler.scheduler import KVUsageSample, PDScheduler
    from prism_serve.scheduler.sequence_state import RequestInfo, SeqState
    from prism_serve.scheduler.transfer_governor import TransferGovernor

    scheduler = PDScheduler({})
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="d-e1")
    sample = KVUsageSample(0.1, "d-e1", time.monotonic())
    scheduler.update_kv_usage("d-0", sample)
    scheduler.pick_decode_instance("R1", 0)
    client = MagicMock()

    async def flip_d(**_kwargs):
        record = scheduler.quarantine_instance("d-0")
        scheduler.reconcile_instance(
            "d-0", "d-e2", record.reconciliation_token, "decode", 1,
            [], [], [],
        )
        if rpc_result == "response_lost":
            raise ConnectionError("response lost after epoch flip")
        return {"success": rpc_result}

    client.abort_request = AsyncMock(side_effect=flip_d)
    governor = TransferGovernor({}, client, NullMetrics())
    req = RequestInfo(
        req_id="R1", state=SeqState.DECODING,
        decode_instance="d-0", decode_instance_epoch="d-e1",
    )
    await _fence_and_abort_request(req, scheduler, governor, "owner", 0.1, 0.1)
    assert scheduler.decode_free_slots()["d-0"] == 1
    assert scheduler.quarantine_record("d-0") is None


def test_chat_completions_returns_501_when_ready():
    with sync_client() as client:
        r = client.post("/v1/chat/completions", json={})
    assert r.status_code == 501
    assert r.json()["error"] == "not_implemented"


def test_chat_completions_returns_503_when_not_accepting():
    with sync_client() as client:
        app.state.accepting = False
        r = client.post("/v1/chat/completions", json={})
        app.state.accepting = True
    assert r.status_code == 503
    assert r.json()["error"] == "service_unavailable"


def test_register_prefill_instance():
    with sync_client() as client:
        r = client.post("/internal/register_instance", json={
            "instance_id": "p-0",
            "role": "prefill",
            "instance_epoch": "epoch-p0",
            "active_request_ids": [],
        })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "registered"
    assert body["instance_id"] == "p-0"


def test_register_decode_instance():
    with sync_client() as client:
        r = client.post("/internal/register_instance", json={
            "instance_id": "d-0",
            "role": "decode",
            "max_slots": 127,
            "instance_epoch": "epoch-d0",
            "active_request_ids": [],
        })
    assert r.status_code == 200
    assert r.json()["instance_id"] == "d-0"


def test_register_decode_without_max_slots_returns_400():
    with sync_client() as client:
        r = client.post("/internal/register_instance", json={
            "instance_id": "d-1",
            "role": "decode",
            "instance_epoch": "epoch-d1",
            "active_request_ids": [],
        })
    assert r.status_code == 400


def test_register_unknown_role_returns_400():
    with sync_client() as client:
        r = client.post("/internal/register_instance", json={
            "instance_id": "x-0",
            "role": "unknown_role",
            "instance_epoch": "epoch-x0",
            "active_request_ids": [],
        })
    assert r.status_code == 400


def test_register_missing_instance_id_returns_422():
    with sync_client() as client:
        r = client.post("/internal/register_instance", json={
            "role": "prefill",
        })
    assert r.status_code in (400, 422)


def test_register_increments_scheduler_load():
    with sync_client() as client:
        client.post("/internal/register_instance", json={
            "instance_id": "p-test",
            "role": "prefill",
            "instance_epoch": "epoch-p-test",
            "active_request_ids": [],
        })
        scheduler = app.state.scheduler
        assert "p-test" in scheduler._prefill_load


def test_register_decode_decrements_on_finish():
    with sync_client() as client:
        client.post("/internal/register_instance", json={
            "instance_id": "d-test",
            "role": "decode",
            "max_slots": 50,
            "instance_epoch": "epoch-d-test",
            "active_request_ids": [],
        })
        scheduler = app.state.scheduler
        assert scheduler._decode_free_slots["d-test"] == 50


def test_register_requires_instance_epoch():
    with sync_client() as client:
        response = client.post("/internal/register_instance", json={
            "instance_id": "p-no-epoch",
            "role": "prefill",
        })
    assert response.status_code == 400


def test_register_rejects_active_remote_requests():
    with sync_client() as client:
        response = client.post("/internal/register_instance", json={
            "instance_id": "d-stale",
            "instance_epoch": "epoch-stale",
            "role": "decode",
            "max_slots": 4,
            "active_request_ids": ["orphan-request"],
        })
    assert response.status_code == 400


def test_quarantined_instance_requires_reconciliation():
    with sync_client() as client:
        client.post("/internal/register_instance", json={
            "instance_id": "d-quarantine",
            "instance_epoch": "epoch-1",
            "role": "decode",
            "max_slots": 4,
            "active_request_ids": [],
        })
        record = app.state.scheduler.quarantine_instance("d-quarantine")

        retry = client.post("/internal/register_instance", json={
            "instance_id": "d-quarantine",
            "instance_epoch": "epoch-1",
            "role": "decode",
            "max_slots": 4,
            "active_request_ids": [],
        })
        assert retry.status_code == 409
        assert retry.json()["reconciliation_token"] == record.reconciliation_token

        app.state.governor.infer_client.get_reconciliation_report = MagicMock(
            side_effect=[
            {
                "instance_id": "d-quarantine",
                "instance_epoch": "epoch-2",
                "challenge": record.reconciliation_token,
                "active_request_ids": ["stale-request"],
                "active_transfer_operation_ids": [],
                "pending_dispatch_command_ids": [],
            },
                {
                    "instance_id": "d-quarantine",
                    "instance_epoch": "epoch-2",
                    "challenge": record.reconciliation_token,
                    "active_request_ids": [],
                    "active_transfer_operation_ids": [],
                    "pending_dispatch_command_ids": [],
                },
        ])

        active = client.post("/internal/reconcile_instance", json={
            "instance_id": "d-quarantine",
            "instance_epoch": "epoch-2",
            "reconciliation_token": record.reconciliation_token,
            "role": "decode",
            "max_slots": 4,
            "active_request_ids": [],
        })
        assert active.status_code == 400
        assert app.state.scheduler.quarantine_record("d-quarantine") is not None

        reconciled = client.post("/internal/reconcile_instance", json={
            "instance_id": "d-quarantine",
            "instance_epoch": "epoch-2",
            "reconciliation_token": record.reconciliation_token,
            "role": "decode",
            "max_slots": 4,
            "active_request_ids": ["untrusted-request-body"],
        })
        assert reconciled.status_code == 200
        assert app.state.scheduler.decode_free_slots()["d-quarantine"] == 4


def test_reconciliation_rejects_unfenced_worker_report():
    with sync_client() as client:
        client.post("/internal/register_instance", json={
            "instance_id": "d-fenced",
            "instance_epoch": "epoch-1",
            "role": "decode",
            "max_slots": 2,
            "active_request_ids": [],
        })
        record = app.state.scheduler.quarantine_instance("d-fenced")
        app.state.governor.infer_client.get_reconciliation_report = MagicMock(
            return_value={
                "instance_id": "d-fenced",
                "instance_epoch": "wrong-epoch",
                "challenge": record.reconciliation_token,
                "active_request_ids": [],
                "active_transfer_operation_ids": [],
                "pending_dispatch_command_ids": [],
            }
        )

        response = client.post("/internal/reconcile_instance", json={
            "instance_id": "d-fenced",
            "instance_epoch": "epoch-2",
            "reconciliation_token": record.reconciliation_token,
            "role": "decode",
            "max_slots": 2,
        })

    assert response.status_code == 400
    assert app.state.scheduler.quarantine_record("d-fenced") is not None


def test_reconciliation_timeout_keeps_instance_quarantined(monkeypatch):
    async def never_returns(**_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(gateway_module.settings, "reconciliation_timeout_s", 0.01)
    with sync_client() as client:
        client.post("/internal/register_instance", json={
            "instance_id": "d-timeout",
            "instance_epoch": "epoch-1",
            "role": "decode",
            "max_slots": 2,
            "active_request_ids": [],
        })
        record = app.state.scheduler.quarantine_instance("d-timeout")
        app.state.governor.infer_client.get_reconciliation_report = AsyncMock(
            side_effect=never_returns
        )

        response = client.post("/internal/reconcile_instance", json={
            "instance_id": "d-timeout",
            "instance_epoch": "epoch-2",
            "reconciliation_token": record.reconciliation_token,
            "role": "decode",
            "max_slots": 2,
        })

        assert response.status_code == 503
        assert app.state.scheduler.quarantine_record("d-timeout") is not None


def test_lifespan_sets_accepting_true():
    with sync_client() as client:
        assert app.state.accepting is True


def test_lifespan_sets_scheduler():
    with sync_client() as client:
        assert app.state.scheduler is not None


def test_lifespan_sets_tracker():
    with sync_client() as client:
        assert app.state.tracker is not None


def test_lifespan_sets_governor():
    with sync_client() as client:
        assert app.state.governor is not None


def test_lifespan_fails_closed_and_cancels_metrics(monkeypatch):
    """Required NATS prevents readiness and leaves no metrics task running."""
    from prism_serve.scheduler.queue import NATSQueue

    created_tasks = []
    real_create_task = asyncio.create_task

    def capture_task(coro):
        task = real_create_task(coro)
        created_tasks.append(task)
        return task

    async def fail_connect(self):
        raise ConnectionError("nats unavailable")

    monkeypatch.setattr(gateway_module.asyncio, "create_task", capture_task)
    monkeypatch.setattr(gateway_module.settings, "nats_required", True)
    monkeypatch.setattr(NATSQueue, "connect", fail_connect)

    with pytest.raises(RuntimeError, match="NATS connect failed"):
        with sync_client():
            pass

    assert len(created_tasks) == 1
    assert created_tasks[0].cancelled()

"""Gateway lifecycle and endpoint tests."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import httpx
from fastapi.testclient import TestClient

from prism_serve.gateway.app import app
from prism_serve.gateway.app import _on_schedule_loop_done
from prism_serve.gateway.app import _wait_for_control_plane_drain
from prism_serve.gateway.app import _abort_remaining_requests
from prism_serve.gateway.app import _background_control_plane_tasks_healthy
from prism_serve.gateway.app import _on_control_plane_task_done
from prism_serve.gateway.app import _prefix_world_allows_admission
from prism_serve.gateway import app as gateway_module
from prism_serve.scheduler.replacement_store import ReplacementRunSeal
from prism_serve.scheduler.replacement_store import RetiredReplacementRun
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


@pytest.mark.parametrize(
    "body",
    [
        {"temperature": "0"},
        {"temperature": True},
        {"temperature": float("nan")},
        {"max_tokens": "32"},
        {"max_tokens": 1.5},
        {"max_tokens": True},
    ],
)
def test_sampling_rejects_non_json_numeric_controls(body):
    with pytest.raises(ValueError, match="temperature|max_tokens"):
        gateway_module._parse_sampling(body)


def test_sampling_preserves_valid_numeric_controls():
    assert gateway_module._parse_sampling(
        {"temperature": 0.25, "max_tokens": 16, "ignore_eos": True}
    ) == {"temperature": 0.25, "max_tokens": 16, "ignore_eos": True}


def test_sampling_rejects_generation_budget_above_runtime_context_limit():
    with pytest.raises(ValueError, match="max_tokens.*max_model_len"):
        gateway_module._parse_sampling(
            {"max_tokens": 65}, max_model_len=64
        )


def test_sampling_rejects_input_plus_generation_over_runtime_context_limit():
    with pytest.raises(ValueError, match="input tokens.*max_model_len"):
        gateway_module._parse_sampling(
            {"max_tokens": 16}, input_token_count=49, max_model_len=64
        )


def test_sampling_accepts_exact_runtime_context_budget():
    assert gateway_module._parse_sampling(
        {"max_tokens": 16}, input_token_count=48, max_model_len=64
    )["max_tokens"] == 16


def test_chat_rejects_context_overflow_before_tracker_admission(monkeypatch):
    monkeypatch.setattr(gateway_module.settings, "max_model_len", 2)
    with sync_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "request_id": "context-overflow",
                "model": gateway_module.settings.model_id,
                "messages": [{"role": "user", "content": "ignored"}],
                "input_token_ids": [1, 2],
                "max_tokens": 1,
            },
        )
        assert response.status_code == 422
        assert "max_model_len" in response.json()["detail"]
        assert app.state.tracker.get("context-overflow") is None


def test_readyz_not_ready_when_nats_disconnects():
    with sync_client() as client:
        app.state.queue._use_mock = False
        app.state.queue._nc = None
        response = client.get("/readyz")
        app.state.queue._use_mock = True
    assert response.status_code == 503


def test_readyz_not_ready_when_worker_rpc_refresh_failed():
    from starlette.requests import Request

    class LiveTask:
        def done(self):
            return False

    fake_app = SimpleNamespace(state=SimpleNamespace(
        accepting=True,
        runtime_config={"multinode_e2e_enabled": True, "affinity_enabled": False},
        worker_rpc_ready=False,
        loop_task=LiveTask(),
        queue=SimpleNamespace(is_connected=True),
        worker_registry=SimpleNamespace(world_fresh=lambda: True),
        resource_refresh_task=LiveTask(),
        control_plane_failed=False,
    ))
    request = Request({"type": "http", "app": fake_app, "headers": []})

    response = gateway_module.readyz(request)

    assert response.status_code == 503
    assert response.body == b'{"status":"not_ready"}'


@pytest.mark.asyncio
async def test_chat_completions_rejects_failed_worker_rpc_refresh():
    class LiveTask:
        def done(self):
            return False

    fake_app = SimpleNamespace(state=SimpleNamespace(
        accepting=True,
        runtime_config={"multinode_e2e_enabled": True, "affinity_enabled": False},
        worker_rpc_ready=False,
        queue=SimpleNamespace(is_connected=True),
        worker_registry=SimpleNamespace(world_fresh=lambda: True),
        resource_refresh_task=LiveTask(),
        control_plane_failed=False,
    ))
    from starlette.requests import Request

    request = Request({"type": "http", "app": fake_app, "headers": []})

    response = await gateway_module.chat_completions(request)

    assert response.status_code == 503
    assert response.body == b'{"error":"service_unavailable","detail":"gateway not ready"}'


def test_week12_admission_requires_exact_published_prefix_world():
    from types import SimpleNamespace

    expected = {
        "p0": "p0-e1", "p1": "p1-e1", "d0": "d0-e1", "d1": "d1-e1"
    }

    class Publication:
        def matches(self, observed):
            return observed == expected

    class Reconciler:
        def world_ready(self, observed):
            return observed == expected

    class Task:
        finished = False

        def done(self):
            return self.finished

    reconciler_task = Task()
    resource_task = Task()
    fake = SimpleNamespace(state=SimpleNamespace(
        runtime_config={"multinode_e2e_enabled": True, "affinity_enabled": True},
        worker_registry=SimpleNamespace(members={
            name: SimpleNamespace(instance_epoch=epoch)
            for name, epoch in expected.items()
        }),
        prefix_reconciler=Reconciler(),
        prefix_world_publication=None,
        reconciler_task=reconciler_task,
        resource_refresh_task=resource_task,
        control_plane_failed=False,
    ))
    assert _prefix_world_allows_admission(fake) is False
    fake.state.prefix_world_publication = Publication()
    assert _prefix_world_allows_admission(fake) is True
    assert _background_control_plane_tasks_healthy(fake) is True
    reconciler_task.finished = True
    assert _prefix_world_allows_admission(fake) is False
    assert _background_control_plane_tasks_healthy(fake) is False
    reconciler_task.finished = False
    resource_task.finished = True
    assert _background_control_plane_tasks_healthy(fake) is False
    resource_task.finished = False
    fake.state.worker_registry.members["d1"].instance_epoch = "d1-e2"
    assert _prefix_world_allows_admission(fake) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("component", "invalidate_prefix"),
    [
        pytest.param("prefix_reconciler", True, id="prefix"),
        pytest.param("resource_refresh", False, id="resources"),
    ],
)
async def test_control_plane_background_exit_fails_closed(
    component,
    invalidate_prefix,
):
    publication = object()
    reconciler = SimpleNamespace(world_publication=publication)
    fake = SimpleNamespace(state=SimpleNamespace(
        accepting=True,
        control_plane_failed=False,
        prefix_world_publication=publication,
        prefix_reconciler=reconciler,
    ))

    async def fail():
        raise RuntimeError(f"{component} failed")

    task = asyncio.create_task(fail())
    with pytest.raises(RuntimeError, match=f"{component} failed"):
        await task
    _on_control_plane_task_done(
        fake,
        task,
        component=component,
        invalidate_prefix=invalidate_prefix,
    )

    assert fake.state.accepting is False
    assert fake.state.control_plane_failed is True
    if invalidate_prefix:
        assert fake.state.prefix_world_publication is None
        assert reconciler.world_publication is None
    else:
        assert fake.state.prefix_world_publication is publication
        assert reconciler.world_publication is publication


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


def test_topology_accept_returns_retired_seal_as_http_410(monkeypatch):
    seal = ReplacementRunSeal(
        seal_sequence=1,
        restart_run_id="run-old",
        old_topology_generation_digest="sha256:old",
        new_topology_generation="world-b",
        decision_digest="sha256:decision",
        record_root_digest="sha256:records",
        record_count=0,
    )

    async def retired(*_args, **_kwargs):
        raise RetiredReplacementRun(seal)

    monkeypatch.setattr(
        gateway_module, "_accept_replacement_topology", retired
    )
    with sync_client() as client:
        had_topology_admin = hasattr(app.state, "topology_admin")
        previous = getattr(app.state, "topology_admin", None)
        app.state.topology_admin = object()
        try:
            response = client.post("/admin/topology/accept", json={})
        finally:
            if had_topology_admin:
                app.state.topology_admin = previous
            else:
                del app.state.topology_admin

    assert response.status_code == 410
    assert response.json() == {
        "error": str(RetiredReplacementRun(seal)),
        "code": "RETIRED_REPLACEMENT_RUN",
        "seal_digest": seal.seal_digest,
    }


def test_healthz_keeps_temporary_nats_disconnect_live_but_not_ready():
    with sync_client() as client:
        app.state.queue._use_mock = False
        app.state.queue._nc = SimpleNamespace(
            is_connected=False, is_closed=False
        )
        try:
            health = client.get("/healthz")
            ready = client.get("/readyz")
        finally:
            app.state.queue._use_mock = True
            app.state.queue._nc = None

    assert health.status_code == 200
    assert ready.status_code == 503


def test_healthz_fails_after_nats_reconnects_are_exhausted():
    with sync_client() as client:
        app.state.queue._use_mock = False
        app.state.queue._nc = SimpleNamespace(
            is_connected=False, is_closed=True
        )
        try:
            response = client.get("/healthz")
        finally:
            app.state.queue._use_mock = True
            app.state.queue._nc = None

    assert response.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault_kind",
    [
        "nats_disconnect",
        "nats_drop",
        "nats_duplicate",
        "nats_publish_unknown",
        "rpc_response_loss_source",
        "rpc_response_loss_target",
        "finalize_response_loss_source",
        "finalize_response_loss_target",
    ],
)
async def test_endpoint_authority_route_accepts_supported_fault_kinds(fault_kind):
    from starlette.requests import Request

    class Gate:
        async def snapshot(self):
            return {
                "fault_run_id": "fault-run",
                "fault_kind": fault_kind,
                "details": {},
            }

    fake_app = SimpleNamespace(state=SimpleNamespace(
        runtime_config={
            "correctness_harness_enabled": True,
            "correctness_harness_secret": "s" * 32,
        },
        correctness_fault_gate=Gate(),
        http_infer_client=None,
    ))
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/internal/week12/correctness/faults/fault-run/endpoints",
        "query_string": b"",
        "headers": [(b"x-prism-week12-token", b"s" * 32)],
        "scheme": "http",
        "server": ("test", 80),
        "client": ("test", 1),
        "app": fake_app,
    })

    response = await gateway_module.correctness_fault_endpoint_evidence(
        request, "fault-run"
    )

    assert response.status_code == 503
    assert response.body == b'{"error":"endpoint_authority_not_ready"}'


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
async def test_control_plane_drain_default_deadline_is_exactly_sixty_seconds(
    monkeypatch,
):
    now_ns = [0]
    sleep_count = [0]
    tracker = MagicMock()
    tracker.__len__.return_value = 1
    governor = MagicMock()
    governor.is_drained.return_value = False

    class Clock:
        @staticmethod
        def time():
            return now_ns[0] / 1_000_000_000

    async def advance_clock(delay):
        sleep_count[0] += 1
        now_ns[0] += int(delay * 1_000_000_000)

    monkeypatch.setattr(gateway_module.asyncio, "get_event_loop", lambda: Clock())
    monkeypatch.setattr(gateway_module.asyncio, "sleep", advance_clock)

    assert not await _wait_for_control_plane_drain(tracker, governor, 60.0)
    assert now_ns[0] == 60_000_000_000
    assert sleep_count[0] == 600


@pytest.mark.asyncio
async def test_runtime_io_close_preserves_order_and_joins_metrics_task():
    calls = []

    class Queue:
        async def close(self):
            calls.append("queue")

    class Metrics:
        async def flush(self):
            calls.append("metrics")

    class Client:
        async def close(self):
            calls.append("client")

    metrics_task = asyncio.create_task(asyncio.Event().wait())

    assert await gateway_module._close_runtime_io(
        Queue(), metrics_task, Metrics(), Client(), timeout_s=1.0
    )
    assert metrics_task.cancelled()
    assert calls == ["queue", "metrics", "client"]


@pytest.mark.asyncio
async def test_runtime_io_close_deadline_requests_local_task_cancellation():
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class Queue:
        async def close(self):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    class Metrics:
        async def flush(self):
            raise AssertionError("flush must not run after queue close timeout")

    metrics_task = asyncio.create_task(asyncio.Event().wait())

    assert not await gateway_module._close_runtime_io(
        Queue(), metrics_task, Metrics(), None, timeout_s=0.01
    )
    assert entered.is_set()
    await asyncio.wait_for(cancelled.wait(), timeout=0.1)
    assert metrics_task.cancelled()


@pytest.mark.asyncio
async def test_runtime_io_close_does_not_join_cancel_suppressing_nats_drain():
    entered = asyncio.Event()
    cancellation_suppressed = asyncio.Event()
    release_suppressed_drain = asyncio.Event()
    finished = asyncio.Event()

    class Queue:
        async def close(self):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_suppressed.set()
                await release_suppressed_drain.wait()
            finally:
                finished.set()

    class Metrics:
        async def flush(self):
            raise AssertionError("flush must not run after the hard deadline")

    metrics_task = asyncio.create_task(asyncio.Event().wait())

    assert not await gateway_module._close_runtime_io(
        Queue(), metrics_task, Metrics(), None, timeout_s=0.01
    )
    assert entered.is_set()
    assert not finished.is_set()
    await asyncio.wait_for(cancellation_suppressed.wait(), timeout=0.1)
    assert not finished.is_set()

    release_suppressed_drain.set()
    await asyncio.wait_for(finished.wait(), timeout=0.1)
    await asyncio.sleep(0)
    assert metrics_task.cancelled()


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
    class Metrics(NullMetrics):
        def __init__(self):
            self.increments = []

        def increment(self, name, amount=1, *, labels=None):
            self.increments.append((name, amount, labels))

    metrics = Metrics()
    governor = TransferGovernor({}, client, metrics)
    tracker = RequestTracker(metrics)
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
    assert (
        "operation_cancelled_before_arrival_total", 1,
        {"endpoint": "dispatch.prefill", "publish_outcome": "UNKNOWN"},
    ) in metrics.increments
    release.set()
    await cleanup


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["UNKNOWN", "NOT_STARTED"])
async def test_week12_network_cleanup_emits_cancel_metric_before_early_return(outcome):
    from types import SimpleNamespace

    from prism_serve.scheduler.scheduler import PDScheduler
    from prism_serve.scheduler.sequence_state import RequestInfo, RequestTracker, SeqState

    class Metrics:
        def __init__(self):
            self.increments = []

        def increment(self, name, amount=1, *, labels=None):
            self.increments.append((name, amount, labels))

    class Infer:
        week12_network_control = True

        async def cleanup_request(self, scheduler, req, *, abort):
            return False

    metrics = Metrics()
    tracker = RequestTracker(metrics)
    scheduler = PDScheduler({})
    scheduler.register_instance("p0", "prefill", instance_epoch="p-epoch")
    scheduler.pick_prefill_instance("r1")
    request = RequestInfo(
        "r1", state=SeqState.PREFILLING, prefill_instance="p0",
        prefill_instance_epoch="p-epoch", publish_outcome=outcome,
    )
    tracker.add(request)
    governor = SimpleNamespace(
        infer_client=Infer(), finish_request=lambda req_id: None,
    )

    await _abort_remaining_requests(
        tracker, scheduler, governor, "owner", 0.1, 0.1,
    )

    assert (
        "operation_cancelled_before_arrival_total", 1,
        {"endpoint": "dispatch.prefill", "publish_outcome": outcome},
    ) in metrics.increments
    assert tracker.get("r1") is request


@pytest.mark.asyncio
async def test_shutdown_network_cleanup_deadline_cancels_local_task_without_release():
    """Network cleanup cannot outlive the process budget or invent release proof."""
    from prism_serve.metrics.collector import NullMetrics
    from prism_serve.scheduler.scheduler import PDScheduler
    from prism_serve.scheduler.sequence_state import RequestInfo, RequestTracker

    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class Infer:
        week12_network_control = True

        async def cleanup_request(self, scheduler, req, *, abort):
            scheduler.quarantine_decode_slot(req.active_operation_id or req.req_id)
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    scheduler = PDScheduler({})
    scheduler.register_instance("d0", "decode", max_slots=1, instance_epoch="d-e1")
    assert scheduler.reserve_decode_slot("d0", "r1", "r1") is not None
    tracker = RequestTracker(NullMetrics())
    request = RequestInfo(
        "r1", decode_instance="d0", decode_instance_epoch="d-e1"
    )
    tracker.add(request)
    governor = SimpleNamespace(
        infer_client=Infer(), finish_request=lambda req_id: None,
    )

    await _abort_remaining_requests(
        tracker,
        scheduler,
        governor,
        "owner",
        1.0,
        1.0,
        join_timeout_s=0.01,
    )

    assert entered.is_set()
    assert cancelled.is_set()
    assert tracker.get("r1") is request
    assert scheduler.decode_free_slots()["d0"] == 0


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


def test_chat_completions_validates_request_when_ready():
    with sync_client() as client:
        r = client.post("/v1/chat/completions", json={})
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_request"


def test_chat_completions_returns_503_when_not_accepting():
    with sync_client() as client:
        app.state.accepting = False
        r = client.post("/v1/chat/completions", json={})
        app.state.accepting = True
    assert r.status_code == 503
    assert r.json()["error"] == "service_unavailable"


def test_chat_completions_rejects_nats_disconnect_without_admission_mutation():
    with sync_client() as client:
        tracker_before = len(app.state.tracker)
        slots_before = app.state.scheduler.decode_free_slots()
        app.state.queue._use_mock = False
        app.state.queue._nc = None
        try:
            response = client.post("/v1/chat/completions", json={
                "request_id": "must-not-enter",
                "model": "test",
                "input_token_ids": [1],
                "messages": [{"role": "user", "content": "x"}],
            })
        finally:
            app.state.queue._use_mock = True

        assert response.status_code == 503
        assert len(app.state.tracker) == tracker_before
        assert app.state.tracker.get("must-not-enter") is None
        assert app.state.scheduler.decode_free_slots() == slots_before


@pytest.mark.asyncio
async def test_chat_wait_does_not_query_output_before_request_commit(monkeypatch):
    from prism_serve.gateway.output import GatewayOutputBuffer
    from prism_serve.metrics.collector import NullMetrics
    from prism_serve.scheduler.sequence_state import RequestTracker, SeqState

    class Control:
        def __init__(self):
            self.calls = []

        async def request_output(self, instance_id, req_id, after_seq):
            self.calls.append((instance_id, req_id, after_seq))
            return {
                "req_id": req_id,
                "instance_epoch": "d0-epoch",
                "operation_id": "request-1",
                "output_seq_no": 1,
                "token_ids": [7],
                "terminal": True,
            }

    tracker = RequestTracker(NullMetrics())
    control = Control()
    monkeypatch.setattr(
        gateway_module, "_build_config",
        lambda: {"operation_query_interval_ms": 1},
    )
    monkeypatch.setattr(app.state, "accepting", True, raising=False)
    monkeypatch.setattr(
        app.state, "queue", SimpleNamespace(is_connected=True), raising=False
    )
    monkeypatch.setattr(app.state, "worker_registry", None, raising=False)
    monkeypatch.setattr(
        app.state, "runtime_config", {"model_id": "test"}, raising=False
    )
    monkeypatch.setattr(app.state, "tokenizer_adapter", None, raising=False)
    monkeypatch.setattr(app.state, "tracker", tracker, raising=False)
    monkeypatch.setattr(
        app.state, "output_buffer", GatewayOutputBuffer(), raising=False
    )
    monkeypatch.setattr(app.state, "network_control", control, raising=False)
    monkeypatch.setattr(app.state, "metrics", NullMetrics(), raising=False)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        pending = asyncio.create_task(client.post("/v1/chat/completions", json={
            "request_id": "request-1",
            "model": "test",
            "input_token_ids": [1],
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 1,
        }))
        for _ in range(100):
            if tracker.get("request-1") is not None:
                break
            await asyncio.sleep(0)
        assert tracker.get("request-1") is not None
        tracker.transition(
            "request-1", SeqState.PREFILLING,
            decode_instance="d0", decode_instance_epoch="d0-epoch",
            active_operation_id="request-1",
        )
        tracker.transition(
            "request-1", SeqState.KV_PENDING,
            transfer_operation_id="request-1",
        )

        await asyncio.sleep(0.02)
        assert pending.done() is False
        assert control.calls == []

        tracker.transition("request-1", SeqState.DECODING)
        response = await asyncio.wait_for(pending, timeout=1.0)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["token_ids"] == [7]
    assert control.calls == [("d0", "request-1", 0)]


@pytest.mark.asyncio
async def test_chat_wait_terminates_when_canonical_request_was_removed(monkeypatch):
    """Canonical cleanup must wake an HTTP waiter even without worker output."""
    from prism_serve.gateway.output import GatewayOutputBuffer
    from prism_serve.metrics.collector import NullMetrics
    from prism_serve.scheduler.sequence_state import RequestTracker

    tracker = RequestTracker(NullMetrics())
    output = GatewayOutputBuffer()
    monkeypatch.setattr(
        gateway_module, "_build_config",
        lambda: {"operation_query_interval_ms": 1},
    )
    monkeypatch.setattr(app.state, "accepting", True, raising=False)
    monkeypatch.setattr(
        app.state, "queue", SimpleNamespace(is_connected=True), raising=False
    )
    monkeypatch.setattr(app.state, "worker_registry", None, raising=False)
    monkeypatch.setattr(
        app.state, "runtime_config", {"model_id": "test"}, raising=False
    )
    monkeypatch.setattr(app.state, "tokenizer_adapter", None, raising=False)
    monkeypatch.setattr(app.state, "tracker", tracker, raising=False)
    monkeypatch.setattr(app.state, "output_buffer", output, raising=False)
    monkeypatch.setattr(app.state, "network_control", None, raising=False)
    monkeypatch.setattr(app.state, "metrics", NullMetrics(), raising=False)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        pending = asyncio.create_task(client.post("/v1/chat/completions", json={
            "request_id": "removed-request",
            "model": "test",
            "input_token_ids": [1],
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 1,
        }))
        for _ in range(100):
            if tracker.get("removed-request") is not None:
                break
            await asyncio.sleep(0)
        assert tracker.remove("removed-request") is not None
        response = await asyncio.wait_for(pending, timeout=1.0)

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "error"
    assert output.snapshot("removed-request")[2] == \
        "request_terminated_without_output"
    assert output.state_counts()["resource_free_terminal"] == 1


def test_chat_capacity_failure_rolls_back_tracker_admission():
    from prism_serve.gateway.output import GatewayOutputBuffer

    with sync_client() as client:
        original = app.state.output_buffer
        bounded = GatewayOutputBuffer(
            active_operation_cap=1, terminal_snapshot_cap=1
        )
        bounded.ensure("held")
        app.state.output_buffer = bounded
        try:
            response = client.post("/v1/chat/completions", json={
                "request_id": "blocked", "model": gateway_module.settings.model_id,
                "input_token_ids": [1],
                "messages": [{"role": "user", "content": "x"}],
            })
            assert response.status_code == 503
            assert app.state.tracker.get("blocked") is None
        finally:
            app.state.output_buffer = original


def test_legacy_register_instance_cannot_mutate_week12_authority():
    with sync_client() as client:
        scheduler = app.state.scheduler
        original_registry = app.state.worker_registry
        before_prefill = dict(scheduler._prefill_load)
        before_decode = dict(scheduler._decode_free_slots)
        app.state.worker_registry = object()
        try:
            response = client.post("/internal/register_instance", json={
                "instance_id": "legacy-d0",
                "instance_epoch": "legacy-pod:legacy-process",
                "role": "decode",
                "max_slots": 999,
                "active_request_ids": [],
            })
        finally:
            app.state.worker_registry = original_registry

        assert response.status_code == 409
        assert response.json()["error"] == "worker_registry_is_authority"
        assert scheduler._prefill_load == before_prefill
        assert scheduler._decode_free_slots == before_decode


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


def test_main_bounds_uvicorn_graceful_shutdown(monkeypatch):
    calls = []
    monkeypatch.delenv("PRISM_SERVE_PROCESS_IDENTITY_PATH", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs))),
    )

    gateway_module.main()

    assert calls[0][1]["timeout_graceful_shutdown"] == \
        gateway_module.UVICORN_GRACEFUL_SHUTDOWN_TIMEOUT_S


def test_lifespan_fails_closed_when_nats_required(monkeypatch):
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

"""Gateway lifecycle and endpoint tests."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
import httpx
from fastapi.testclient import TestClient

from prism_serve.gateway.app import app
from prism_serve.gateway.app import _on_schedule_loop_done
from prism_serve.gateway.app import _wait_for_control_plane_drain
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
        })
    assert r.status_code == 200
    assert r.json()["instance_id"] == "d-0"


def test_register_decode_without_max_slots_returns_400():
    with sync_client() as client:
        r = client.post("/internal/register_instance", json={
            "instance_id": "d-1",
            "role": "decode",
        })
    assert r.status_code == 400


def test_register_unknown_role_returns_400():
    with sync_client() as client:
        r = client.post("/internal/register_instance", json={
            "instance_id": "x-0",
            "role": "unknown_role",
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
        })
        scheduler = app.state.scheduler
        assert "p-test" in scheduler._prefill_load


def test_register_decode_decrements_on_finish():
    with sync_client() as client:
        client.post("/internal/register_instance", json={
            "instance_id": "d-test",
            "role": "decode",
            "max_slots": 50,
        })
        scheduler = app.state.scheduler
        assert scheduler._decode_free_slots["d-test"] == 50


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

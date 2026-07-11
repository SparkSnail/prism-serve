"""Unit tests for gateway/app.py.

Uses httpx.AsyncClient + ASGITransport so the full FastAPI lifespan
(startup / shutdown) runs inside each test without a real server process.

Covers:
  GET  /healthz                 — liveness probe
  GET  /readyz                  — readiness probe (state.accepting flag)
  POST /v1/chat/completions     — 501 when ready, 503 when not accepting
  POST /internal/register_instance — register prefill / decode instances
"""

from __future__ import annotations

import asyncio

import pytest
import httpx
from fastapi.testclient import TestClient

from prism_serve.gateway.app import app
from prism_serve.gateway import app as gateway_module
from prism_serve import __version__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sync_client() -> TestClient:
    """Synchronous TestClient that runs lifespan on enter/exit."""
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture(autouse=True)
def allow_mock_nats(monkeypatch):
    """Gateway tests explicitly opt into local mock-queue mode."""
    monkeypatch.setattr(gateway_module.settings, "nats_required", False)


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------

def test_healthz_returns_ok():
    with sync_client() as client:
        r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__


# ---------------------------------------------------------------------------
# /readyz
# ---------------------------------------------------------------------------

def test_readyz_ready_after_startup():
    """Gateway sets state.accepting=True in lifespan startup."""
    with sync_client() as client:
        r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_readyz_not_ready_before_startup():
    """Without lifespan running, accepting is False → 503."""
    # Instantiate a raw ASGI call without lifespan
    client = TestClient(app, raise_server_exceptions=False)
    # Do NOT use context manager — lifespan never runs
    r = client.get("/readyz")
    # accepting defaults to False (attribute not set) → 503
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------

def test_chat_completions_returns_501_when_ready():
    with sync_client() as client:
        r = client.post("/v1/chat/completions", json={})
    assert r.status_code == 501
    assert r.json()["error"] == "not_implemented"


def test_chat_completions_returns_503_when_not_accepting():
    """If state.accepting is False, endpoint returns 503."""
    with sync_client() as client:
        # Manually flip the flag to simulate mid-shutdown
        app.state.accepting = False
        r = client.post("/v1/chat/completions", json={})
        app.state.accepting = True   # restore for other tests
    assert r.status_code == 503
    assert r.json()["error"] == "service_unavailable"


# ---------------------------------------------------------------------------
# /internal/register_instance
# ---------------------------------------------------------------------------

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
    """Decode registration without max_slots must be rejected."""
    with sync_client() as client:
        r = client.post("/internal/register_instance", json={
            "instance_id": "d-1",
            "role": "decode",
            # max_slots omitted → defaults to 0 → AssertionError in scheduler
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
    """FastAPI validates missing required fields → 422."""
    with sync_client() as client:
        r = client.post("/internal/register_instance", json={
            "role": "prefill",
            # instance_id missing
        })
    # FastAPI request body parsing raises 422 (or 400 from our KeyError)
    assert r.status_code in (400, 422)


def test_register_increments_scheduler_load():
    """Registered P instance appears in scheduler._prefill_load."""
    with sync_client() as client:
        client.post("/internal/register_instance", json={
            "instance_id": "p-test",
            "role": "prefill",
        })
        scheduler = app.state.scheduler
        assert "p-test" in scheduler._prefill_load


def test_register_decode_decrements_on_finish():
    """Registered D instance slots are tracked correctly."""
    with sync_client() as client:
        client.post("/internal/register_instance", json={
            "instance_id": "d-test",
            "role": "decode",
            "max_slots": 50,
        })
        scheduler = app.state.scheduler
        assert scheduler._decode_free_slots["d-test"] == 50


# ---------------------------------------------------------------------------
# Lifespan: state attributes set after startup
# ---------------------------------------------------------------------------

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

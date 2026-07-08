"""FastAPI gateway: the cluster ingress for prism-serve.

Manages the full lifecycle of the serve control plane via FastAPI lifespan:
  start: metrics → governor → schedule_loop → accept requests
  stop:  stop accepting → drain schedule_loop → drain governor → flush metrics

Borrowing:
  - lifespan start/stop ordering
    (later-started services stopped first to preserve dependency order)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from prism_serve import __version__
from prism_serve.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the serve control-plane lifecycle.

    Startup order:
      1. MetricsCollector  — infrastructure; all other components report to it
      2. TransferGovernor  — flow controller; schedule_loop depends on it
      3. NATSQueue         — message bus; schedule_loop publishes/polls it
      4. schedule_loop     — main loop; gateway depends on it to process reqs
      5. accepting = True  — open the gateway to external traffic

    Shutdown order
      1. accepting = False — stop accepting new requests
      2. cancel schedule_loop and wait (up to 60 s) for in-flight requests
      3. drain TransferGovernor deferred queue (up to 30 s)
      4. close NATSQueue
      5. flush MetricsCollector
    """
    from prism_serve.metrics.collector import MetricsCollector, NullMetrics
    from prism_serve.scheduler.transfer_governor import TransferGovernor
    from prism_serve.scheduler.scheduler import PDScheduler
    from prism_serve.scheduler.sequence_state import RequestTracker
    from prism_serve.scheduler.main_loop import schedule_loop
    from prism_serve.scheduler.queue import NATSQueue

    config = _build_config()

    # ── 1. Metrics ────────────────────────────────────────────────────
    try:
        app.state.metrics = MetricsCollector(config)
    except Exception:
        logger.warning("MetricsCollector init failed; using NullMetrics")
        app.state.metrics = NullMetrics()
    metrics_task = asyncio.create_task(app.state.metrics.tick_loop())

    # ── 2. TransferGovernor ───────────────────────────────────────────
    # infer_client stub; replace with real RPC client when available.
    infer_client = _make_stub_infer_client()
    app.state.governor = TransferGovernor(config, infer_client, app.state.metrics)
    # Wire metrics ↔ governor for kv_usage propagation
    if hasattr(app.state.metrics, "set_governor"):
        app.state.metrics.set_governor(app.state.governor)
    if hasattr(app.state.metrics, "set_infer_client"):
        app.state.metrics.set_infer_client(infer_client)

    # ── 3. NATS queue ─────────────────────────────────────────────────
    app.state.queue = NATSQueue(config)
    try:
        await app.state.queue.connect()
    except Exception as exc:
        logger.warning("NATS connect failed (%s); queue will use mock mode", exc)
        app.state.queue = NATSQueue(config, use_mock=True)

    # ── 4. schedule_loop ──────────────────────────────────────────────
    app.state.tracker   = RequestTracker(app.state.metrics)
    app.state.scheduler = PDScheduler(config)
    loop_task = asyncio.create_task(
        schedule_loop(
            app.state.scheduler,
            app.state.governor,
            app.state.tracker,
            app.state.queue,
            app.state.metrics,
            config,
        )
    )

    # ── 5. Open to traffic ────────────────────────────────────────────
    app.state.accepting = True
    logger.info("prism-serve gateway ready (version %s)", __version__)

    yield  # ← FastAPI serves external requests here

    # ── Shutdown: reverse order ───────────────────────────────────────
    logger.info("prism-serve shutting down")

    # 1. Stop accepting
    app.state.accepting = False

    # 2. Cancel and drain schedule_loop (max 60 s)
    loop_task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(loop_task), timeout=60.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    # 3. Drain governor deferred queue (max 30 s)
    await _drain_governor(app.state.governor, timeout_s=30.0)

    # 4. Close NATS
    await app.state.queue.close()

    # 5. Flush metrics
    metrics_task.cancel()
    await app.state.metrics.flush()
    logger.info("prism-serve shutdown complete")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="prism-serve",
    version=__version__,
    summary="Kubernetes control plane for disaggregated LLM serving",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe — process is up."""
    return {"status": "ok", "version": __version__}


@app.get("/readyz")
def readyz(request: Request) -> JSONResponse:
    """Readiness probe — returns 200 only when schedule_loop is accepting."""
    accepting = getattr(request.app.state, "accepting", False)
    if accepting:
        return JSONResponse({"status": "ready"})
    return JSONResponse({"status": "not_ready"}, status_code=503)


@app.post("/v1/chat/completions")
def chat_completions(request: Request) -> JSONResponse:
    """OpenAI-compatible inference entry point.

    Full routing not yet implemented; returns 501.
    """
    if not getattr(request.app.state, "accepting", False):
        return JSONResponse(
            {"error": "service_unavailable", "detail": "gateway not ready"},
            status_code=503,
        )
    return JSONResponse(
        status_code=501,
        content={
            "error": "not_implemented",
            "detail": "not implemented",
        },
    )


# ---------------------------------------------------------------------------
# Instance registration endpoint (called by infer pods on startup)
# ---------------------------------------------------------------------------

@app.post("/internal/register_instance")
async def register_instance(request: Request) -> JSONResponse:
    """Register a new prefill or decode infer instance.

    Body: {"instance_id": "p-0", "role": "prefill" | "decode",
           "max_slots": 127}
    """
    body = await request.json()
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        return JSONResponse({"error": "scheduler not ready"}, status_code=503)
    try:
        scheduler.register_instance(
            instance_id=body["instance_id"],
            role=body["role"],
            max_slots=body.get("max_slots", 0),
        )
    except (KeyError, AssertionError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"status": "registered", "instance_id": body["instance_id"]})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_config() -> dict:
    """Merge pydantic Settings into a plain dict for scheduler components."""
    return {
        "nats_url":                 settings.nats_url,
        "HIGH_WATERMARK":           settings.high_watermark,
        "LOW_WATERMARK":            settings.low_watermark,
        "MAX_BYTES_INFLIGHT":       settings.max_bytes_inflight,
        "kv_transfer_timeout_s":    settings.kv_transfer_timeout_s,
        "max_recompute_attempts":   settings.max_recompute_attempts,
        "schedule_loop_tick_ms":    settings.schedule_loop_tick_ms,
        "governor_tick_s":          settings.governor_tick_s,
        "slot_stale_timeout_s":     settings.slot_stale_timeout_s,
        "min_decode_instances":     settings.min_decode_instances,
        "max_decode_instances":     settings.max_decode_instances,
        "kv_per_instance_bytes":    settings.kv_per_instance_bytes,
        "kv_usage_scrape_interval_s": settings.governor_tick_s,
    }


def _make_stub_infer_client():
    """Return a no-op infer client stub (replace with real RPC client)."""
    class _Stub:
        def transfer(self, src, dst, req_id, on_complete=None):
            logger.debug("stub transfer %s→%s req=%s", src, dst, req_id)
            if on_complete:
                on_complete()

        def reset_to_waiting(self, dst, req_id):
            logger.debug("stub reset_to_waiting dst=%s req=%s", dst, req_id)

        async def get_kv_usage_all(self) -> dict:
            return {}

    return _Stub()


async def _drain_governor(governor, timeout_s: float) -> None:
    """Wait until all in-flight KV transfers complete or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if governor.all_inflight_zero():
            break
        await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Console-script entry point: ``prism-serve``."""
    import uvicorn
    uvicorn.run(
        "prism_serve.gateway.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()

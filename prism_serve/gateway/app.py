"""FastAPI gateway and control-plane lifecycle."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from prism_serve import __version__
from prism_serve.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open readiness after dependencies start and reverse that order on exit."""
    from prism_serve.metrics.collector import MetricsCollector, NullMetrics
    from prism_serve.scheduler.transfer_governor import TransferGovernor
    from prism_serve.scheduler.scheduler import PDScheduler
    from prism_serve.scheduler.sequence_state import RequestTracker
    from prism_serve.scheduler.main_loop import schedule_loop
    from prism_serve.scheduler.queue import NATSQueue

    config = _build_config()

    # Start metrics before components that report through it.
    try:
        app.state.metrics = MetricsCollector(config)
    except Exception:
        logger.warning("MetricsCollector init failed; using NullMetrics")
        app.state.metrics = NullMetrics()
    metrics_task = asyncio.create_task(app.state.metrics.tick_loop())

    infer_client = _make_stub_infer_client()
    app.state.governor = TransferGovernor(config, infer_client, app.state.metrics)
    if hasattr(app.state.metrics, "set_governor"):
        app.state.metrics.set_governor(app.state.governor)
    if hasattr(app.state.metrics, "set_infer_client"):
        app.state.metrics.set_infer_client(infer_client)

    # Connect NATS before accepting traffic; production fails closed.
    app.state.queue = NATSQueue(config)
    try:
        await app.state.queue.connect()
    except Exception as exc:
        if config.get("nats_required", True):
            metrics_task.cancel()
            try:
                await metrics_task
            except asyncio.CancelledError:
                pass
            raise RuntimeError(
                "NATS connect failed; gateway refusing to become ready"
            ) from exc
        logger.warning(
            "NATS connect failed (%s); using mock queue (nats_required=false)", exc
        )
        app.state.queue = NATSQueue(config, use_mock=True)

    # Start scheduling after its dependencies are ready.
    app.state.tracker   = RequestTracker(app.state.metrics)
    app.state.scheduler = PDScheduler(config)
    if hasattr(app.state.metrics, "set_scheduler"):
        app.state.metrics.set_scheduler(app.state.scheduler)
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
    app.state.loop_task = loop_task
    loop_task.add_done_callback(lambda task: _on_schedule_loop_done(app, task))

    app.state.accepting = True
    logger.info("prism-serve gateway ready (version %s)", __version__)

    yield

    logger.info("prism-serve shutting down")

    app.state.accepting = False

    # Keep scheduling while existing requests and transfer ledgers drain.
    await _wait_for_control_plane_drain(
        app.state.tracker,
        app.state.governor,
        timeout_s=config["shutdown_drain_timeout_s"],
    )
    loop_task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(loop_task), timeout=60.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    await _drain_governor(app.state.governor, timeout_s=30.0)

    await app.state.queue.close()

    metrics_task.cancel()
    await app.state.metrics.flush()
    logger.info("prism-serve shutdown complete")


app = FastAPI(
    title="prism-serve",
    version=__version__,
    summary="Kubernetes control plane for disaggregated LLM serving",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz() -> dict:
    """Report process liveness."""
    return {"status": "ok", "version": __version__}


@app.get("/readyz")
def readyz(request: Request) -> JSONResponse:
    """Return ready only while the scheduler and NATS transport are healthy."""
    accepting = getattr(request.app.state, "accepting", False)
    loop_task = getattr(request.app.state, "loop_task", None)
    queue = getattr(request.app.state, "queue", None)
    loop_healthy = loop_task is not None and not loop_task.done()
    queue_healthy = queue is not None and queue.is_connected
    if accepting and loop_healthy and queue_healthy:
        return JSONResponse({"status": "ready"})
    return JSONResponse({"status": "not_ready"}, status_code=503)


@app.get("/metrics")
def metrics() -> Response:
    """Expose metrics in the Prometheus text format."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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


@app.post("/internal/register_instance")
async def register_instance(request: Request) -> JSONResponse:
    """Register a prefill or decode infer instance."""
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


def _build_config() -> dict:
    """Merge pydantic Settings into a plain dict for scheduler components."""
    return {
        "nats_url":                 settings.nats_url,
        "nats_connect_timeout_s":   settings.nats_connect_timeout_s,
        "nats_max_reconnect_attempts": settings.nats_max_reconnect_attempts,
        "nats_required":            settings.nats_required,
        "HIGH_WATERMARK":           settings.high_watermark,
        "LOW_WATERMARK":            settings.low_watermark,
        "MAX_BYTES_INFLIGHT":       settings.max_bytes_inflight,
        "kv_transfer_timeout_s":    settings.kv_transfer_timeout_s,
        "prefill_timeout_s":        settings.prefill_timeout_s,
        "max_dispatch_attempts":    settings.max_dispatch_attempts,
        "recompute_timeout_s":      settings.recompute_timeout_s,
        "decode_timeout_s":         settings.decode_timeout_s,
        "abort_request_timeout_s":  settings.abort_request_timeout_s,
        "max_recompute_attempts":   settings.max_recompute_attempts,
        "schedule_loop_tick_ms":    settings.schedule_loop_tick_ms,
        "governor_tick_s":          settings.governor_tick_s,
        "shutdown_drain_timeout_s": settings.shutdown_drain_timeout_s,
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
            logger.debug("stub transfer %s to %s req=%s", src, dst, req_id)
            if on_complete:
                on_complete()

        def reset_to_waiting(self, dst, req_id):
            logger.debug("stub reset_to_waiting dst=%s req=%s", dst, req_id)

        def abort_request(self, instance_id, owner_id, req_id):
            logger.debug(
                "stub abort instance=%s owner=%s req=%s",
                instance_id, owner_id, req_id,
            )
            return {"success": True}

        async def get_kv_usage_all(self) -> dict:
            return {}

    return _Stub()


async def _drain_governor(governor, timeout_s: float) -> None:
    """Wait until all in-flight KV transfers complete or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if governor.is_drained():
            break
        await asyncio.sleep(0.1)


async def _wait_for_control_plane_drain(tracker, governor, timeout_s: float) -> bool:
    """Wait for both request state and transfer bookkeeping to drain."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if len(tracker) == 0 and governor.is_drained():
            return True
        await asyncio.sleep(0.1)
    return False


def _on_schedule_loop_done(app: FastAPI, task: asyncio.Task) -> None:
    """Fail readiness when the scheduler exits outside normal shutdown."""
    if task.cancelled():
        return
    app.state.accepting = False
    error = task.exception()
    if error is not None:
        logger.error(
            "schedule_loop stopped; readiness disabled",
            exc_info=(type(error), error, error.__traceback__),
        )


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

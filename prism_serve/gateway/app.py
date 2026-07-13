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
    if config["control_plane_replica_count"] != 1:
        raise RuntimeError(
            "process-local control-plane state requires one active gateway"
        )

    # Start metrics before components that report through it.
    try:
        app.state.metrics = MetricsCollector(config)
    except Exception:
        logger.warning("MetricsCollector init failed; using NullMetrics")
        app.state.metrics = NullMetrics()
    metrics_task = asyncio.create_task(app.state.metrics.tick_loop())

    infer_client = getattr(app.state, "infer_client", None) or _make_stub_infer_client()
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
    affinity_coordinator = None
    reconciler_task = None
    if config["affinity_enabled"]:
        from prism_serve.router.coordinator import AffinityCoordinator
        from prism_serve.router.prefix_index import PrefixIndex
        from prism_serve.router.reconciler import PrefixReconciler
        from prism_serve.router.router import AffinityRouter
        from prism_serve.router.topology import TopologyMatrix

        required = (
            "resolve_prefix", "prepare_prefix", "transfer_cached_prefix",
            "commit_cached_prefix", "abort_mapped_prefix", "get_prefix_operation",
            "abort_cached_prefix", "unpin_prefix", "abort_suffix_prefill",
            "get_prefix_resource_counts", "full_report_and_register",
            "peek_events", "ack_events",
        )
        missing = [name for name in required if not hasattr(infer_client, name)]
        if missing:
            raise RuntimeError(f"affinity RPC client missing capabilities: {missing}")
        if config["prefix_block_bytes"] <= 0:
            raise RuntimeError("affinity requires PRISM_SERVE_PREFIX_BLOCK_BYTES > 0")
        tokenizer_adapter = getattr(app.state, "tokenizer_adapter", None)
        if tokenizer_adapter is None or not hasattr(
            tokenizer_adapter, "fingerprint_request"
        ):
            raise RuntimeError("affinity requires a process-local TokenizerAdapter")
        app.state.tokenizer_adapter = tokenizer_adapter
        app.state.prefix_index = PrefixIndex(config["prefix_location_max_age_s"])
        topology = getattr(app.state, "topology_matrix", None) or TopologyMatrix()
        router = AffinityRouter(
            app.state.prefix_index, topology,
            block_bytes=config["prefix_block_bytes"],
            safety_margin_ms=config["affinity_safety_margin_ms"],
        )
        affinity_coordinator = AffinityCoordinator(
            router, infer_client, app.state.queue, config, app.state.metrics
        )
        app.state.affinity_coordinator = affinity_coordinator
        app.state.prefix_reconciler = PrefixReconciler(
            app.state.prefix_index, infer_client, app.state.queue.owner_id,
            config["scheduler_generation"], app.state.metrics,
        )
        reconciler_task = asyncio.create_task(app.state.prefix_reconciler.run(
            lambda: app.state.scheduler.decode_instance_epochs().keys(),
            config["prefix_event_poll_interval_ms"] / 1000.0,
        ))
        app.state.reconciler_task = reconciler_task
    loop_task = asyncio.create_task(
        schedule_loop(
            app.state.scheduler,
            app.state.governor,
            app.state.tracker,
            app.state.queue,
            app.state.metrics,
            config,
            affinity_coordinator,
        )
    )
    app.state.loop_task = loop_task
    app.state.control_plane_failed = False
    loop_task.add_done_callback(lambda task: _on_schedule_loop_done(app, task))

    app.state.accepting = True
    logger.info("prism-serve gateway ready (version %s)", __version__)

    yield

    logger.info("prism-serve shutting down")

    app.state.accepting = False

    if affinity_coordinator is not None:
        await affinity_coordinator.shutdown()
    if reconciler_task is not None:
        reconciler_task.cancel()
        try:
            await reconciler_task
        except asyncio.CancelledError:
            pass

    # Keep scheduling while existing requests and transfer ledgers drain.
    drained = await _wait_for_control_plane_drain(
        app.state.tracker,
        app.state.governor,
        timeout_s=config["shutdown_drain_timeout_s"],
    )
    loop_task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(loop_task), timeout=60.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    if not drained:
        await _abort_remaining_requests(
            app.state.tracker,
            app.state.scheduler,
            app.state.governor,
            app.state.queue.owner_id,
            timeout_s=config["abort_request_timeout_s"],
            transfer_timeout_s=config["abort_transfer_timeout_s"],
        )

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
def healthz(request: Request) -> JSONResponse:
    """Fail liveness after an unexpected scheduler exit."""
    if getattr(request.app.state, "control_plane_failed", False):
        return JSONResponse(
            {"status": "failed", "version": __version__}, status_code=503
        )
    return JSONResponse({"status": "ok", "version": __version__})


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
            instance_epoch=body["instance_epoch"],
            active_request_ids=body["active_request_ids"],
        )
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        from prism_serve.scheduler.scheduler import QuarantinedInstanceError
        if isinstance(exc, QuarantinedInstanceError):
            record = exc.record
            return JSONResponse(
                {
                    "error": "instance_quarantined",
                    "instance_id": record.instance_id,
                    "instance_epoch": record.instance_epoch,
                    "reconciliation_token": record.reconciliation_token,
                },
                status_code=409,
            )
        if not isinstance(exc, (AssertionError, ValueError)):
            raise
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"status": "registered", "instance_id": body["instance_id"]})


@app.post("/internal/reconcile_instance")
async def reconcile_instance(request: Request) -> JSONResponse:
    """Fetch a fenced worker report before restoring quarantined capacity."""
    body = await request.json()
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        return JSONResponse({"error": "scheduler not ready"}, status_code=503)
    try:
        report = await _get_reconciliation_report(
            request.app.state.governor.infer_client,
            instance_id=body["instance_id"],
            instance_epoch=body["instance_epoch"],
            challenge=body["reconciliation_token"],
            timeout_s=_build_config()["reconciliation_timeout_s"],
        )
        scheduler.reconcile_instance(
            instance_id=body["instance_id"],
            instance_epoch=body["instance_epoch"],
            reconciliation_token=body["reconciliation_token"],
            role=body["role"],
            max_slots=body.get("max_slots", 0),
            active_request_ids=report["active_request_ids"],
            active_transfer_operation_ids=report["active_transfer_operation_ids"],
            pending_dispatch_command_ids=report["pending_dispatch_command_ids"],
        )
    except (KeyError, AssertionError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except (TimeoutError, ConnectionError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    return JSONResponse({"status": "reconciled", "instance_id": body["instance_id"]})


def _build_config() -> dict:
    """Merge pydantic Settings into a plain dict for scheduler components."""
    return {
        "nats_url":                 settings.nats_url,
        "nats_connect_timeout_s":   settings.nats_connect_timeout_s,
        "nats_max_reconnect_attempts": settings.nats_max_reconnect_attempts,
        "nats_required":            settings.nats_required,
        "scheduler_id":             settings.gateway_pod_uid,
        "scheduler_generation":     settings.gateway_process_generation,
        "control_plane_replica_count": settings.control_plane_replica_count,
        "HIGH_WATERMARK":             settings.high_watermark,
        "LOW_WATERMARK":              settings.low_watermark,
        "MAX_BYTES_INFLIGHT":         settings.max_bytes_inflight,
        "kv_transfer_timeout_s":      settings.kv_transfer_timeout_s,
        "abort_transfer_timeout_s":   settings.abort_transfer_timeout_s,
        "kv_usage_stale_after_s":     settings.kv_usage_stale_after_s,
        "prefill_timeout_s":          settings.prefill_timeout_s,
        "max_dispatch_attempts":      settings.max_dispatch_attempts,
        "recompute_timeout_s":        settings.recompute_timeout_s,
        "decode_timeout_s":           settings.decode_timeout_s,
        "abort_request_timeout_s":    settings.abort_request_timeout_s,
        "reconciliation_timeout_s":   settings.reconciliation_timeout_s,
        "max_recompute_attempts":     settings.max_recompute_attempts,
        "schedule_loop_tick_ms":      settings.schedule_loop_tick_ms,
        "governor_tick_s":            settings.governor_tick_s,
        "shutdown_drain_timeout_s":   settings.shutdown_drain_timeout_s,
        "slot_stale_timeout_s":       settings.slot_stale_timeout_s,
        "affinity_enabled":           settings.affinity_enabled,
        "locality_wait_ms":           settings.locality_wait_ms,
        "max_affinity_wait_ms":       settings.max_affinity_wait_ms,
        "affinity_safety_margin_ms":  settings.affinity_safety_margin_ms,
        "affinity_decode_candidate_limit": settings.affinity_decode_candidate_limit,
        "decode_slot_lease_timeout_s": settings.decode_slot_lease_timeout_s,
        "prefix_event_log_capacity":  settings.prefix_event_log_capacity,
        "prefix_event_poll_interval_ms": settings.prefix_event_poll_interval_ms,
        "prefix_consumer_lease_s":    settings.prefix_consumer_lease_s,
        "prefix_full_report_interval_s": settings.prefix_full_report_interval_s,
        "prefix_location_max_age_s":  2 * settings.prefix_full_report_interval_s,
        "prefix_load_timeout_s":      settings.prefix_load_timeout_s,
        "suffix_prefill_timeout_s":   settings.suffix_prefill_timeout_s,
        "prefix_operation_watchdog_s": settings.prefix_operation_watchdog_s,
        "prefix_block_bytes":         settings.prefix_block_bytes,
        "prefill_ms_per_token":       settings.prefill_ms_per_token,
        "min_decode_instances":       settings.min_decode_instances,
        "max_decode_instances":       settings.max_decode_instances,
        "kv_per_instance_bytes":      settings.kv_per_instance_bytes,
        "kv_usage_scrape_interval_s": settings.governor_tick_s,
    }


def _make_stub_infer_client():
    """Return a no-op infer client stub (replace with real RPC client)."""
    class _Stub:
        def transfer(self, src, dst, req_id, operation_id, on_complete=None):
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

        def abort_transfer(
            self, src_instance, dst_instance, owner_id, req_id, operation_id
        ):
            logger.debug(
                "stub abort transfer src=%s dst=%s owner=%s req=%s operation=%s",
                src_instance, dst_instance, owner_id, req_id, operation_id,
            )
            return {"success": True}

        def get_reconciliation_report(
            self, instance_id, instance_epoch, challenge
        ):
            return {
                "instance_id": instance_id,
                "instance_epoch": instance_epoch,
                "challenge": challenge,
                "active_request_ids": [],
                "active_transfer_operation_ids": [],
                "pending_dispatch_command_ids": [],
            }

        async def get_kv_usage_all(self) -> dict:
            return {}

    return _Stub()


async def _get_reconciliation_report(
    infer_client,
    instance_id: str,
    instance_epoch: str,
    challenge: str,
    timeout_s: float,
) -> dict:
    """Fetch and validate an epoch- and challenge-fenced worker report."""
    method = infer_client.get_reconciliation_report
    kwargs = {
        "instance_id": instance_id,
        "instance_epoch": instance_epoch,
        "challenge": challenge,
    }

    async def invoke():
        if asyncio.iscoroutinefunction(method):
            return await method(**kwargs)
        return await asyncio.to_thread(method, **kwargs)

    report = await asyncio.wait_for(invoke(), timeout=timeout_s)
    if not isinstance(report, dict):
        raise ValueError("invalid reconciliation report")
    if report.get("instance_id") != instance_id:
        raise ValueError("reconciliation instance mismatch")
    if report.get("instance_epoch") != instance_epoch:
        raise ValueError("reconciliation epoch mismatch")
    if report.get("challenge") != challenge:
        raise ValueError("reconciliation challenge mismatch")
    active = report.get("active_request_ids")
    if not isinstance(active, list):
        raise ValueError("reconciliation active_request_ids must be a list")
    active_transfers = report.get("active_transfer_operation_ids")
    if not isinstance(active_transfers, list):
        raise ValueError(
            "reconciliation active_transfer_operation_ids must be a list"
        )
    pending_dispatches = report.get("pending_dispatch_command_ids")
    if not isinstance(pending_dispatches, list):
        raise ValueError(
            "reconciliation pending_dispatch_command_ids must be a list"
        )
    return report


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
    app.state.control_plane_failed = True
    error = task.exception()
    if error is not None:
        logger.error(
            "schedule_loop stopped; readiness disabled",
            exc_info=(type(error), error, error.__traceback__),
        )


async def _abort_remaining_requests(
    tracker,
    scheduler,
    governor,
    owner_id: str,
    timeout_s: float,
    transfer_timeout_s: float | None = None,
) -> None:
    """Abort all requests left after graceful drain before closing NATS."""
    from prism_serve.scheduler.main_loop import _get_or_create_canonical_cleanup

    tasks = [
        _get_or_create_canonical_cleanup(
            req, tracker, scheduler, governor, owner_id, timeout_s,
            transfer_timeout_s if transfer_timeout_s is not None else timeout_s,
        )
        for req in tracker.all_requests()
    ]
    for task in tasks:
        await asyncio.shield(task)


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

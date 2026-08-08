"""FastAPI gateway and control-plane lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from nats.errors import (
    ConnectionClosedError as NATSConnectionClosedError,
    NoServersError as NATSNoServersError,
    StaleConnectionError as NATSStaleConnectionError,
    TimeoutError as NATSTimeoutError,
)

from prism_serve import __version__
from prism_serve.config import settings

logger = logging.getLogger(__name__)

WEEK12_WORKER_IDS = ("p0", "p1", "d0", "d1")
GATEWAY_BOOTSTRAP_TIMEOUT_S = 570.0
GATEWAY_STARTUP_PROBE_MARGIN_S = 30.0
WORKER_BOOTSTRAP_RETRY_INTERVAL_S = 0.2
LOOP_SHUTDOWN_TIMEOUT_S = 60.0
GOVERNOR_SHUTDOWN_TIMEOUT_S = 30.0
UVICORN_GRACEFUL_SHUTDOWN_TIMEOUT_S = 30.0
NETWORK_CLEANUP_SHUTDOWN_TIMEOUT_S = 30.0
RUNTIME_IO_CLOSE_SHUTDOWN_TIMEOUT_S = 15.0
CORRECTNESS_ENDPOINT_AUTHORITY_FAULTS = frozenset({
    "nats_disconnect",
    "nats_drop",
    "nats_duplicate",
    "nats_publish_unknown",
    "rpc_response_loss_source",
    "rpc_response_loss_target",
    "finalize_response_loss_source",
    "finalize_response_loss_target",
})
_BootstrapResult = TypeVar("_BootstrapResult")


class _GatewayBootstrapNotReady(RuntimeError):
    pass


def _gateway_bootstrap_retryable(error: BaseException) -> bool:
    from prism_serve.router.http_rpc import AmbiguousRPCError, InferRPCError

    pending: BaseException | None = error
    visited: set[int] = set()
    while pending is not None and id(pending) not in visited:
        visited.add(id(pending))
        if isinstance(
            pending,
            (
                _GatewayBootstrapNotReady,
                AmbiguousRPCError,
                ConnectionError,
                TimeoutError,
                NATSNoServersError,
                NATSConnectionClosedError,
                NATSStaleConnectionError,
                NATSTimeoutError,
            ),
        ):
            return True
        if isinstance(pending, InferRPCError):
            return pending.status_code == 503
        pending = pending.__cause__ or pending.__context__
    return False


async def _run_gateway_bootstrap_stage(
    operation: Callable[[], Awaitable[_BootstrapResult]],
    *,
    stage: str,
    deadline: float,
    retry_interval_s: float,
) -> _BootstrapResult:
    if retry_interval_s <= 0:
        raise ValueError("gateway bootstrap retry budget must be positive")
    from prism_serve.router.http_rpc import _rpc_error_chain

    def log_diagnostic(
        level: int,
        *,
        event: str,
        retry_index: int,
        remaining: float,
        error: BaseException | None,
    ) -> None:
        value: dict[str, object] = {
            "event": event,
            "stage": stage,
            "retry_index": retry_index,
            "remaining_ms": max(0, int(remaining * 1000)),
            "error_chain": [] if error is None else _rpc_error_chain(error),
        }
        logger.log(
            level,
            "%s",
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ),
        )

    loop = asyncio.get_running_loop()
    retry_index = 0
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            log_diagnostic(
                logging.ERROR,
                event="gateway_bootstrap.deadline_exceeded",
                retry_index=retry_index,
                remaining=remaining,
                error=None,
            )
            raise RuntimeError(
                f"gateway bootstrap deadline exceeded during {stage}"
            )
        try:
            return await asyncio.wait_for(operation(), timeout=remaining)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not _gateway_bootstrap_retryable(exc):
                raise
            remaining = deadline - loop.time()
            if remaining <= 0:
                log_diagnostic(
                    logging.ERROR,
                    event="gateway_bootstrap.deadline_exceeded",
                    retry_index=retry_index,
                    remaining=remaining,
                    error=exc,
                )
                raise RuntimeError(
                    f"gateway bootstrap deadline exceeded during {stage}"
                ) from exc
            log_diagnostic(
                logging.WARNING,
                event="gateway_bootstrap.retry",
                retry_index=retry_index,
                remaining=remaining,
                error=exc,
            )
            retry_index += 1
            await asyncio.sleep(min(retry_interval_s, remaining))


def _bootstrap_retry_interval_s(config: dict[str, object]) -> float:
    value = config.get("operation_query_interval_ms")
    if value is None:
        return WORKER_BOOTSTRAP_RETRY_INTERVAL_S
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("operation_query_interval_ms must be numeric")
    interval = float(value) / 1000.0
    if interval <= 0:
        raise ValueError("operation_query_interval_ms must be positive")
    return interval


def _prefix_poll_interval_s(config: dict[str, object]) -> float:
    value = config.get("prefix_event_poll_interval_ms")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("prefix_event_poll_interval_ms must be numeric")
    interval = float(value) / 1000.0
    if interval <= 0:
        raise ValueError("prefix_event_poll_interval_ms must be positive")
    return interval


def _canonical_worker_identity_map(
    values: object,
) -> dict[str, dict[str, object]]:
    """Normalize a replacement wire set without trusting derived fields."""
    from prism_serve.gateway.topology_admin import (
        parse_worker_identity,
        worker_identity_wire,
    )

    if not isinstance(values, list):
        raise ValueError("replacement identities must be a list")
    identities = [parse_worker_identity(value) for value in values]
    result = {
        identity.instance_id: worker_identity_wire(identity)
        for identity in identities
    }
    if len(identities) != len(WEEK12_WORKER_IDS) or set(result) != set(
        WEEK12_WORKER_IDS
    ):
        raise ValueError("replacement requires four unique worker identities")
    return result


def _bootstrap_active_owner(
    value: object,
    *,
    instance_id: str,
) -> str | None:
    if not isinstance(value, dict) or "active_owner" not in value:
        raise ValueError(
            f"owner status for {instance_id} must contain active_owner"
        )
    active_owner = value["active_owner"]
    if active_owner is None:
        return None
    if not isinstance(active_owner, str) or not active_owner:
        raise ValueError(
            f"owner status for {instance_id} has malformed active_owner"
        )
    return active_owner


class _BootstrapOwnerClient:

    def __init__(
        self,
        client: object,
        *,
        new_owner: str,
        prior_owners: frozenset[str],
    ) -> None:
        self._client = client
        self._new_owner = new_owner
        self._prior_owners = prior_owners

    def __getattr__(self, name: str) -> object:
        return getattr(self._client, name)

    async def owner_status(self, instance_id: str) -> dict[str, object]:
        method = getattr(self._client, "owner_status")
        value = await method(instance_id)
        active_owner = _bootstrap_active_owner(
            value, instance_id=instance_id,
        )
        if active_owner is not None and active_owner != self._new_owner \
                and active_owner not in self._prior_owners:
            raise RuntimeError(
                f"foreign owner observed during gateway bootstrap: "
                f"{instance_id}={active_owner}"
            )
        return value


async def _bootstrap_replacement_owner(
    client: object,
    instances: tuple[str, ...],
    new_owner: str,
    *,
    max_audit_entries: int = 18432,
    deadline: float | None = None,
    retry_interval_s: float = WORKER_BOOTSTRAP_RETRY_INTERVAL_S,
) -> object:
    if not instances or len(set(instances)) != len(instances):
        raise ValueError("gateway bootstrap instances must be unique")
    if not new_owner:
        raise ValueError("gateway bootstrap new owner must be non-empty")
    if deadline is None:
        deadline = (
            asyncio.get_running_loop().time() + GATEWAY_BOOTSTRAP_TIMEOUT_S
        )
    ordered_instances = tuple(sorted(instances))

    async def read_initial_owners() -> frozenset[str]:
        method = getattr(client, "owner_status")
        values = await asyncio.gather(*(
            method(instance_id) for instance_id in ordered_instances
        ))
        active = {
            owner
            for instance_id, value in zip(ordered_instances, values)
            if (
                owner := _bootstrap_active_owner(
                    value, instance_id=instance_id,
                )
            ) not in {None, new_owner}
        }
        if len(active) > 1:
            raise RuntimeError(
                "workers disagree on the prior active owner during bootstrap"
            )
        return frozenset(active)

    prior_owners = await _run_gateway_bootstrap_stage(
        read_initial_owners,
        stage="owner status preflight",
        deadline=deadline,
        retry_interval_s=retry_interval_s,
    )
    fenced_client = _BootstrapOwnerClient(
        client,
        new_owner=new_owner,
        prior_owners=prior_owners,
    )

    async def activate() -> object:
        from prism_serve.router.network_rpc import activate_replacement_owner

        return await activate_replacement_owner(
            fenced_client,
            ordered_instances,
            new_owner,
            max_audit_entries=max_audit_entries,
            reconcile_deadline=deadline,
            retry_interval_s=retry_interval_s,
        )

    return await _run_gateway_bootstrap_stage(
        activate,
        stage="owner reconciliation",
        deadline=deadline,
        retry_interval_s=retry_interval_s,
    )


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
    app.state.accepting = False
    app.state.runtime_config = config
    app.state.control_plane_failed = False
    app.state.correctness_fault_gate = None
    app.state.owner_takeover_audit = None
    app.state.prefix_world_publication = None
    app.state.prefix_reconciler = None
    app.state.reconciler_task = None
    app.state.topology_acceptance_task = None
    app.state.reconciler_replacement_task = None
    app.state.resource_refresh_task = None
    if config["correctness_harness_enabled"]:
        if len(config["correctness_harness_secret"]) < 32:
            raise RuntimeError(
                "correctness harness requires a Secret-backed token of at least 32 characters"
            )
        if not config["multinode_e2e_enabled"] or not config["affinity_enabled"]:
            raise RuntimeError(
                "correctness harness requires multinode_e2e and affinity"
            )
        from prism_serve.gateway.correctness_harness import FaultInjectionGate

        app.state.correctness_fault_gate = FaultInjectionGate(
            timeout_s=float(config["correctness_fault_gate_timeout_s"])
        )
    if config["control_plane_replica_count"] != 1:
        raise RuntimeError(
            "process-local control-plane state requires one active gateway"
        )
    app.state.replacement_store = None
    if config["multinode_e2e_enabled"]:
        from prism_serve.scheduler.replacement_store import ReplacementDecisionStore

        if not config["replacement_store_path"]:
            raise RuntimeError("multinode_e2e requires a replacement store path")
        app.state.replacement_store = ReplacementDecisionStore(
            config["replacement_store_path"],
            max_records_per_run=config["replacement_store_max_records_per_run"],
            seal_retention=config["replacement_store_seal_retention"],
        )
        if app.state.replacement_store.last_error is not None:
            raise RuntimeError(
                "replacement store failed integrity preflight: "
                f"{app.state.replacement_store.last_error}"
            )
    app.state.http_infer_client = None
    app.state.worker_registry = None
    gateway_bootstrap_deadline = None
    if config["multinode_e2e_enabled"]:
        gateway_bootstrap_deadline = (
            asyncio.get_running_loop().time() + GATEWAY_BOOTSTRAP_TIMEOUT_S
        )
        await _run_gateway_bootstrap_stage(
            lambda: _bootstrap_week12_http_control(
                app, config, deadline=gateway_bootstrap_deadline
            ),
            stage="worker world bootstrap",
            deadline=gateway_bootstrap_deadline,
            retry_interval_s=_bootstrap_retry_interval_s(config),
        )

    # Start metrics before components that report through it.
    try:
        app.state.metrics = MetricsCollector(config)
    except Exception:
        logger.warning("MetricsCollector init failed; using NullMetrics")
        app.state.metrics = NullMetrics()
    metrics_task = asyncio.create_task(app.state.metrics.tick_loop())
    if app.state.http_infer_client is not None:
        app.state.http_infer_client.set_metrics(app.state.metrics)

    infer_client = getattr(app.state, "infer_client", None) or _make_stub_infer_client()
    app.state.governor = TransferGovernor(
        config, infer_client, app.state.metrics,
        worker_registry=app.state.worker_registry,
    )

    if hasattr(app.state.metrics, "set_governor"):
        app.state.metrics.set_governor(app.state.governor)
    if hasattr(app.state.metrics, "set_infer_client"):
        app.state.metrics.set_infer_client(infer_client)

    # Connect NATS before accepting traffic; production fails closed.
    app.state.queue = NATSQueue(config)
    try:
        if gateway_bootstrap_deadline is None:
            await app.state.queue.connect()
        else:
            await _run_gateway_bootstrap_stage(
                app.state.queue.connect,
                stage="NATS connect",
                deadline=gateway_bootstrap_deadline,
                retry_interval_s=_bootstrap_retry_interval_s(config),
            )
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

    operation_allocator = None
    network_control = None
    if config["multinode_e2e_enabled"]:
        from prism_serve.router.http_rpc import EndpointSequenceAllocator
        from prism_serve.router.network_rpc import (
            NetworkControlRPC,
        )
        operation_allocator = EndpointSequenceAllocator(
            config["topology_generation"], app.state.queue.owner_id
        )
        retry_interval_s = _bootstrap_retry_interval_s(config)
        app.state.owner_takeover_audit = await _bootstrap_replacement_owner(
            app.state.http_infer_client,
            ("p0", "p1", "d0", "d1"),
            app.state.queue.owner_id,
            max_audit_entries=4 * (
                config["active_operation_cap"] + config["terminal_snapshot_cap"]
            ),
            deadline=gateway_bootstrap_deadline,
            retry_interval_s=retry_interval_s,
        )


        await _run_gateway_bootstrap_stage(
            lambda: _pull_replacement_resource_reports(
                app,
                app.state.worker_registry,
                excluded_operation_ids=set(),
                excluded_resource_ids=set(),
                incomplete_is_not_ready=True,
            ),
            stage="post-owner resource reports",
            deadline=gateway_bootstrap_deadline,
            retry_interval_s=retry_interval_s,
        )
        members = app.state.worker_registry.members
        network_control = NetworkControlRPC(
            app.state.http_infer_client,
            operation_allocator,
            {name: identity.instance_epoch for name, identity in members.items()},
            query_interval_s=config["operation_query_interval_ms"] / 1000.0,
            operation_timeout_s=config["kv_transfer_timeout_s"],
            block_size=config["prefix_block_size"],
            block_bytes=config["prefix_block_bytes"],
            active_operation_cap=config["active_operation_cap"],
            terminal_snapshot_cap=config["terminal_snapshot_cap"],
            metrics=app.state.metrics,
        )
        network_control.set_correctness_fault_gate(
            app.state.correctness_fault_gate
        )
        app.state.governor.infer_client = network_control
        infer_client = network_control
        if hasattr(app.state.metrics, "set_infer_client"):
            app.state.metrics.set_infer_client(network_control)
    app.state.operation_allocator = operation_allocator
    app.state.network_control = network_control

    # ── 4. schedule_loop ────────────────────────────────────────────────
    app.state.tracker   = RequestTracker(app.state.metrics)
    from prism_serve.gateway.output import GatewayOutputBuffer
    app.state.output_buffer = GatewayOutputBuffer(
        active_operation_cap=config["active_operation_cap"],
        terminal_snapshot_cap=config["terminal_snapshot_cap"],
    )
    app.state.scheduler = PDScheduler(
        config, worker_registry=app.state.worker_registry
    )
    if config["multinode_e2e_enabled"]:
        for instance_id, identity in app.state.worker_registry.members.items():
            report = app.state.worker_registry.resource_signal(instance_id).report or {}
            app.state.scheduler.register_instance(
                instance_id=instance_id,
                role=identity.role,
                max_slots=int(
                    report.get("max_slots", 1 if identity.role == "decode" else 0)
                ),
                instance_epoch=identity.instance_epoch,
                active_request_ids=list(report.get("active_request_ids", ())),
            )
        from prism_serve.scheduler.resource_release import ResourceReleaseEvaluator
        app.state.resource_release_evaluator = ResourceReleaseEvaluator(
            app.state.scheduler, app.state.http_infer_client.finalize_release,
            app.state.metrics,
            active_operation_cap=config["active_operation_cap"],
            terminal_snapshot_cap=config["terminal_snapshot_cap"],
            replacement_store=app.state.replacement_store,
        )
        app.state.resource_release_evaluator.set_correctness_fault_gate(
            app.state.correctness_fault_gate
        )
        network_control.set_release_evaluator(
            app.state.resource_release_evaluator
        )
        from prism_serve.gateway.topology_admin import TopologyAcceptanceLedger
        app.state.topology_admin = TopologyAcceptanceLedger(
            app.state.worker_registry,
            terminal_snapshot_cap=config["terminal_snapshot_cap"],
        )
    if hasattr(app.state.metrics, "set_scheduler"):
        app.state.metrics.set_scheduler(app.state.scheduler)
    affinity_coordinator = None
    reconciler_task = None
    prefix_poll_interval_s = (
        _prefix_poll_interval_s(config)
        if config["affinity_enabled"]
        else None
    )
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
        if tokenizer_adapter is None and config["tokenizer_model"]:
            from transformers import AutoTokenizer
            from prism_serve.router.tokenizer import TokenizerAdapter, TokenizerIdentity

            encoder = AutoTokenizer.from_pretrained(
                config["tokenizer_model"],
                revision=config["tokenizer_revision"],
                use_fast=True,
            )
            tokenizer_adapter = TokenizerAdapter(encoder, TokenizerIdentity(
                model_id=config["tokenizer_model"],
                tokenizer_revision=config["tokenizer_revision"],
                chat_template_version=config["chat_template_version"],
                block_size=config["prefix_block_size"],
                hash_version="xxh64-chain-v1",
                kv_compatibility_id=config["kv_compatibility_id"],
            ))
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
            router, infer_client, app.state.queue, config, app.state.metrics,
            operation_allocator,
        )
        app.state.affinity_coordinator = affinity_coordinator
        app.state.prefix_reconciler = PrefixReconciler(
            app.state.prefix_index, infer_client, app.state.queue.owner_id,
            config["scheduler_generation"], app.state.metrics,
        )
        if config["multinode_e2e_enabled"]:
            expected_epochs = {
                name: identity.instance_epoch
                for name, identity in app.state.worker_registry.members.items()
            }
            app.state.prefix_world_publication = (
                await app.state.prefix_reconciler.rebuild_world(expected_epochs)
            )
        reconciler_task = asyncio.create_task(app.state.prefix_reconciler.run(
            lambda: (
                app.state.worker_registry.members.keys()
                if app.state.worker_registry is not None
                else app.state.scheduler.decode_instance_epochs().keys()
            ),
            prefix_poll_interval_s,
        ))
        reconciler_task.add_done_callback(
            lambda task: _on_control_plane_task_done(
                app,
                task,
                component="prefix_reconciler",
                invalidate_prefix=True,
            )
        )
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
            app.state.output_buffer,
            operation_allocator,
        )
    )
    app.state.loop_task = loop_task
    loop_task.add_done_callback(lambda task: _on_schedule_loop_done(app, task))
    resource_refresh_task = None
    if config["multinode_e2e_enabled"]:
        resource_refresh_task = asyncio.create_task(
            _refresh_worker_resources(app, config)
        )
        resource_refresh_task.add_done_callback(
            lambda task: _on_control_plane_task_done(
                app, task, component="resource_refresh"
            )
        )
        app.state.resource_refresh_task = resource_refresh_task


    # Give newly created supervisors one turn so immediate task exits cannot
    # slip through the first readiness publication.
    await asyncio.sleep(0)
    app.state.accepting = (
        _background_control_plane_tasks_healthy(app)
        and (
            not config["multinode_e2e_enabled"]
            or (
                app.state.worker_registry.world_fresh()
                and _replacement_store_allows_admission(app)
                and _topology_acceptance_allows_admission(app)
                and _prefix_world_allows_admission(app)
            )
        )
    )
    logger.info("prism-serve gateway ready (version %s)", __version__)

    yield

    logger.info("prism-serve shutting down")

    app.state.accepting = False

    if resource_refresh_task is not None:
        resource_refresh_task.cancel()
        await asyncio.gather(resource_refresh_task, return_exceptions=True)

    if affinity_coordinator is not None:
        await affinity_coordinator.shutdown()
    active_reconciler_task = getattr(app.state, "reconciler_task", None)
    if active_reconciler_task is not None:
        active_reconciler_task.cancel()
        try:
            await active_reconciler_task
        except asyncio.CancelledError:
            pass

    # Keep scheduling while existing requests and transfer ledgers drain.
    drained = await _wait_for_control_plane_drain(
        app.state.tracker,
        app.state.governor,
        timeout_s=config["shutdown_drain_timeout_s"],
    )
    active_loop_task = getattr(app.state, "loop_task", loop_task)
    active_loop_task.cancel()
    try:
        await asyncio.wait_for(
            asyncio.shield(active_loop_task),
            timeout=LOOP_SHUTDOWN_TIMEOUT_S,
        )
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
            join_timeout_s=NETWORK_CLEANUP_SHUTDOWN_TIMEOUT_S,
        )


    await _drain_governor(
        app.state.governor,
        timeout_s=GOVERNOR_SHUTDOWN_TIMEOUT_S,
    )


    await _close_runtime_io(
        app.state.queue,
        metrics_task,
        app.state.metrics,
        app.state.http_infer_client,
        timeout_s=RUNTIME_IO_CLOSE_SHUTDOWN_TIMEOUT_S,
    )
    logger.info("prism-serve shutdown complete")


app = FastAPI(
    title="prism-serve",
    version=__version__,
    summary="Kubernetes control plane for disaggregated LLM serving",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz(request: Request) -> JSONResponse:
    queue = getattr(request.app.state, "queue", None)
    nats_permanently_closed = (
        queue is not None
        and bool(getattr(queue, "is_permanently_closed", False))
    )
    if (
        getattr(request.app.state, "control_plane_failed", False)
        or nats_permanently_closed
        or not _background_control_plane_tasks_healthy(
            request.app,
            allow_reconciler_transition=True,
        )
    ):
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
    registry = getattr(request.app.state, "worker_registry", None)
    topology_healthy = registry is None or registry.world_fresh()
    prefix_healthy = _prefix_world_allows_admission(request.app)
    background_healthy = _background_control_plane_tasks_healthy(request.app)
    if (
        accepting and loop_healthy and queue_healthy
        and topology_healthy and prefix_healthy and background_healthy
    ):
        return JSONResponse({"status": "ready"})
    return JSONResponse({"status": "not_ready"}, status_code=503)


@app.get("/metrics")
def metrics() -> Response:
    """Expose metrics in the Prometheus text format."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _request_config(request: Request) -> dict:
    value = getattr(request.app.state, "runtime_config", None)
    return value if isinstance(value, dict) else _build_config()


def _correctness_auth_error(request: Request) -> JSONResponse | None:
    from prism_serve.gateway.correctness_harness import AUTH_HEADER, authorize

    config = _request_config(request)
    decision = authorize(
        enabled=bool(config.get("correctness_harness_enabled")),
        configured_secret=str(config.get("correctness_harness_secret") or ""),
        supplied_secret=str(request.headers.get(AUTH_HEADER) or ""),
    )
    if decision == "disabled":
        return JSONResponse({"error": "not_found"}, status_code=404)
    if decision != "ok":
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return None


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    registry = getattr(request.app.state, "worker_registry", None)
    queue = getattr(request.app.state, "queue", None)
    if (
        not getattr(request.app.state, "accepting", False)
        or queue is None
        or not queue.is_connected
        or (registry is not None and not registry.world_fresh())
        or not _prefix_world_allows_admission(request.app)
        or not _background_control_plane_tasks_healthy(request.app)
    ):
        return JSONResponse(
            {"error": "service_unavailable", "detail": "gateway not ready"},
            status_code=503,
        )
    try:
        body = await request.json()
        model = str(body["model"])
        correctness_route = None
        if "week12_correctness" in body:
            error = _correctness_auth_error(request)
            if error is not None:
                return error
            from prism_serve.gateway.correctness_harness import parse_route

            correctness_route = parse_route(body["week12_correctness"])
            config = _request_config(request)
            if model != config["model_id"]:
                raise ValueError("correctness model does not match runtime")
            tokenizer = getattr(request.app.state, "tokenizer_adapter", None)
            if tokenizer is None:
                raise ValueError("correctness checks require the frozen tokenizer")
            token_ids = list(body["input_token_ids"])
            if not token_ids or not all(
                type(token) is int and 0 <= token < 2**64 for token in token_ids
            ):
                raise ValueError("input_token_ids must be unsigned 64-bit integers")
            from prism_serve.router.fingerprint import PromptFingerprint

            fingerprint = PromptFingerprint.create(
                namespace=tokenizer.namespace,
                kv_compatibility_id=tokenizer.identity.kv_compatibility_id,
                request_context_digest="text-only",
                token_ids=token_ids,
                block_size=tokenizer.identity.block_size,
            )
        else:
            messages = body["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
            text = "\n".join(str(message["content"]) for message in messages)
            tokenizer = getattr(request.app.state, "tokenizer_adapter", None)
            if tokenizer is not None:
                token_ids = list(tokenizer.encoder.encode(text))
                fingerprint = tokenizer.fingerprint_request(text)
            else:
                token_ids = list(body["input_token_ids"])
                if not token_ids or not all(isinstance(token, int) for token in token_ids):
                    raise ValueError("input_token_ids are required without tokenizer")
                fingerprint = None
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse(
            {"error": "invalid_request", "detail": str(exc)}, status_code=422
        )
    tracker = getattr(request.app.state, "tracker", None)
    output_buffer = getattr(request.app.state, "output_buffer", None)
    if tracker is None or output_buffer is None:
        return JSONResponse({"error": "service_unavailable"}, status_code=503)
    from prism_serve.scheduler.sequence_state import RequestInfo

    req_id = str(body.get("request_id") or f"chatcmpl-{uuid.uuid4().hex}")
    sampling = {
        "temperature": float(body.get("temperature", 0.0)),
        "max_tokens": int(body.get("max_tokens", 32)),
    }
    if correctness_route is not None:
        if body.get("stream", False):
            return JSONResponse(
                {"error": "invalid_request", "detail": "correctness harness is non-streaming"},
                status_code=422,
            )
        sampling["ignore_eos"] = body.get("ignore_eos") is True
        from prism_serve.gateway.correctness_harness import validate_fixture

        try:
            validate_fixture(
                route=correctness_route, token_ids=token_ids, sampling=sampling
            )
        except ValueError as exc:
            return JSONResponse(
                {"error": "invalid_request", "detail": str(exc)}, status_code=422
            )
        control = getattr(request.app.state, "network_control", None)
        if control is None or not hasattr(control, "require_correctness_evidence"):
            return JSONResponse(
                {"error": "service_unavailable", "detail": "correctness control unavailable"},
                status_code=503,
            )
        try:
            control.require_correctness_evidence(req_id)
        except RuntimeError as exc:
            return JSONResponse(
                {"error": "service_unavailable", "detail": str(exc)}, status_code=503
            )
    try:
        tracker.add(RequestInfo(
            req_id=req_id,
            fingerprint=fingerprint,
            sampling_params=sampling,
            token_ids=token_ids,
            model=model,
            correctness_path=(correctness_route.path if correctness_route else ""),
            correctness_source_instance=(
                correctness_route.source_instance if correctness_route else ""
            ),
            correctness_target_instance=(
                correctness_route.target_instance if correctness_route else ""
            ),
            correctness_cached_prefix_tokens=(
                correctness_route.cached_prefix_tokens if correctness_route else 0
            ),
        ))
    except AssertionError as exc:
        if correctness_route is not None:
            control.cancel_correctness_evidence(req_id)
        return JSONResponse({"error": "conflict", "detail": str(exc)}, status_code=409)
    from prism_serve.gateway.output import GatewayOutputCapacity
    try:
        output_buffer.ensure(req_id)
    except GatewayOutputCapacity as exc:
        tracker.remove(req_id)
        if correctness_route is not None:
            control.cancel_correctness_evidence(req_id)
        return JSONResponse(
            {"error": "service_unavailable", "detail": str(exc)},
            status_code=503,
        )

    async def wait_with_query(cursor: int):
        while True:
            try:
                return await asyncio.wait_for(
                    output_buffer.wait_next(req_id, cursor),
                    timeout=_build_config()["operation_query_interval_ms"] / 1000.0,
                )
            except asyncio.TimeoutError:
                current = tracker.get(req_id)
                if current is None:
                    await output_buffer.fail(
                        req_id, "request_terminated_without_output"
                    )
                    output_buffer.mark_resource_free(req_id)
                    return output_buffer.snapshot(req_id)
                control = getattr(request.app.state, "network_control", None)
                from prism_serve.gateway.output import output_query_identity
                identity = output_query_identity(current)
                if control is None or identity is None or identity[2] != req_id:
                    continue
                instance_id, instance_epoch, _, operation_id = identity
                from prism_serve.gateway.output import repair_output_gap
                try:
                    await repair_output_gap(
                        control,
                        instance_id=instance_id,
                        instance_epoch=instance_epoch,
                        req_id=req_id,
                        operation_id=operation_id,
                        cursor=cursor,
                        output_buffer=output_buffer,
                        metrics=request.app.state.metrics,
                        still_current=(
                            lambda current=current, identity=identity: (
                                tracker.get(req_id) is current
                                and output_query_identity(current) == identity
                            )
                        ),
                    )
                except Exception:
                    # Query failure does not invent output or terminate the stream.
                    pass

    async def stream_events():
        cursor = 0
        while True:
            tokens, terminal, error = await wait_with_query(cursor)
            for token in tokens:
                cursor += 1
                event = {
                    "id": req_id, "object": "chat.completion.chunk", "model": model,
                    "choices": [{"index": 0, "delta": {"token_id": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
            if error or terminal:
                final = {
                    "id": req_id, "object": "chat.completion.chunk", "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error" if error else "stop"}],
                }
                yield f"data: {json.dumps(final, separators=(',', ':'))}\n\n"
                yield "data: [DONE]\n\n"
                return

    if body.get("stream", False):
        return StreamingResponse(stream_events(), media_type="text/event-stream")
    cursor = 0
    tokens = []
    try:
        while True:
            new_tokens, terminal, error = await wait_with_query(cursor)
            tokens.extend(new_tokens)
            cursor = len(tokens)
            if terminal or error:
                break
    finally:
        if correctness_route is not None:
            control.cancel_correctness_evidence(req_id)
    return JSONResponse({
        "id": req_id, "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "token_ids": tokens},
                     "finish_reason": "error" if error else "stop"}],
    })


async def _correctness_resource_snapshot(app: FastAPI) -> dict[str, object]:
    registry = getattr(app.state, "worker_registry", None)
    client = getattr(app.state, "http_infer_client", None)
    scheduler = getattr(app.state, "scheduler", None)
    governor = getattr(app.state, "governor", None)
    tracker = getattr(app.state, "tracker", None)
    if any(value is None for value in (registry, client, scheduler, governor, tracker)) \
            or not registry.world_fresh():
        raise RuntimeError("fixed 2P2D worker world is not fresh")
    members = registry.members
    reports: dict[str, dict[str, object]] = {}
    quarantine_operation_ids: set[str] = set()
    owner = app.state.queue.owner_id
    for instance_id in ("p0", "p1", "d0", "d1"):
        report = await client.get_resources(instance_id)
        identity = members[instance_id]
        buckets = report.get("block_buckets")
        resources = report.get("resources")
        expected_buckets = {
            "free", "pending", "sequence", "evictable", "quarantined"
        }
        valid_bucket_values = (
            isinstance(buckets, dict)
            and set(buckets) == expected_buckets
            and all(type(value) is int and value >= 0 for value in buckets.values())
        )
        if (
            report.get("instance_epoch") != identity.instance_epoch
            or report.get("complete") is not True
            or not isinstance(resources, dict)
            or not valid_bucket_values
            or report.get("block_conservation_valid") is not True
            or sum(buckets.values())
            != int(report.get("num_gpu_blocks", -1))
        ):
            raise RuntimeError(f"incomplete resource authority from {instance_id}")
        if int(buckets.get("quarantined", 0)) > 0:
            operations = await client.list_operations(instance_id, owner)
            instance_quarantine = {
                str(value["endpoint_ref"]["operation_id"])
                for value in operations.get("operations", ())
                if value.get("state") == "UNKNOWN" and value.get("resources_held") is True
            }
            if not instance_quarantine:
                raise RuntimeError("quarantined blocks lack operation identity")
            quarantine_operation_ids.update(instance_quarantine)
        reports[instance_id] = report

    quantities = [value["resources"] for value in reports.values()]
    lease_counts = scheduler.decode_slot_lease_counts()
    resources = {
        "slots": sum(int(value) for value in lease_counts.values()),
        "source_retain": sum(int(value.get("SOURCE_RETAIN", 0)) for value in quantities),
        "source_pins": sum(int(value.get("SOURCE_PIN", 0)) for value in quantities),
        "target_pending": sum(int(value.get("TARGET_PENDING", 0)) for value in quantities),
        "sequence_blocks": sum(
            int(value.get("TARGET_SEQUENCE", 0))
            + int(value.get("SOURCE_BLOCKS", 0))
            for value in quantities
        ),
        "pair_bytes": sum(governor.pair_bytes_inflight_snapshot().values()),
        "quarantine_operation_ids": sorted(quarantine_operation_ids),
    }
    return {
        "resources": resources,
        "active_requests": len(tracker),
        "worker_reports": reports,
        "decode_slot_lease_counts": lease_counts,
        "pair_bytes_by_pair": governor.pair_bytes_inflight_snapshot(),
    }


@app.get("/internal/week12/correctness/world")
async def correctness_world(request: Request):
    error = _correctness_auth_error(request)
    if error is not None:
        return error
    registry = getattr(request.app.state, "worker_registry", None)
    if registry is None or not registry.world_fresh():
        return JSONResponse({"error": "world_not_fresh"}, status_code=503)
    config = _request_config(request)
    from prism_serve.gateway.topology_admin import worker_identity_wire

    identities = []
    for identity in registry.members.values():
        identities.append(worker_identity_wire(identity))
    return {
        "topology_generation": registry.expected_topology_generation,
        "model_profile": dict(config["expected_model_profile"] or {}),
        "worker_identities": sorted(identities, key=lambda value: value["global_rank"]),
        "pair_capabilities": [
            asdict(value)
            for value in sorted(
                registry.capabilities.values(), key=lambda value: value.pair_id
            )
        ],
    }


@app.get("/internal/week12/correctness/takeover")
async def correctness_takeover(request: Request):
    error = _correctness_auth_error(request)
    if error is not None:
        return error
    audit = getattr(request.app.state, "owner_takeover_audit", None)
    reconciler = getattr(request.app.state, "prefix_reconciler", None)
    publication = (
        getattr(reconciler, "world_publication", None)
        or getattr(request.app.state, "prefix_world_publication", None)
    )
    if audit is None or publication is None:
        return JSONResponse({"error": "takeover_not_ready"}, status_code=503)
    return {
        "owner_takeover": asdict(audit),
        "prefix_world_publication": asdict(publication),
        "admission_ready": _prefix_world_allows_admission(request.app),
    }


@app.get("/internal/week12/correctness/resources")
async def correctness_resources(request: Request):
    error = _correctness_auth_error(request)
    if error is not None:
        return error
    try:
        return await _correctness_resource_snapshot(request.app)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return JSONResponse(
            {"error": "resource_authority_unavailable", "detail": str(exc)},
            status_code=503,
        )


@app.post("/internal/week12/correctness/faults")
async def correctness_arm_fault(request: Request):
    error = _correctness_auth_error(request)
    if error is not None:
        return error
    gate = getattr(request.app.state, "correctness_fault_gate", None)
    if gate is None:
        return JSONResponse({"error": "fault_gate_not_ready"}, status_code=503)
    body = await request.json()
    if not isinstance(body, dict) or set(body) != {"fault_kind"}:
        return JSONResponse({"error": "invalid_fault_request"}, status_code=400)
    try:
        return await gate.arm(str(body["fault_kind"]))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)


@app.get("/internal/week12/correctness/faults/{fault_run_id}")
async def correctness_fault_status(request: Request, fault_run_id: str):
    error = _correctness_auth_error(request)
    if error is not None:
        return error
    gate = getattr(request.app.state, "correctness_fault_gate", None)
    value = None if gate is None else await gate.snapshot()
    if value is None or value.get("fault_run_id") != fault_run_id:
        return JSONResponse({"error": "fault_not_found"}, status_code=404)
    return value


@app.post("/internal/week12/correctness/faults/{fault_run_id}/release")
async def correctness_release_fault(request: Request, fault_run_id: str):
    error = _correctness_auth_error(request)
    if error is not None:
        return error
    gate = getattr(request.app.state, "correctness_fault_gate", None)
    if gate is None:
        return JSONResponse({"error": "fault_gate_not_ready"}, status_code=503)
    try:
        return await gate.release(fault_run_id)
    except KeyError:
        return JSONResponse({"error": "fault_not_found"}, status_code=404)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)


@app.get("/internal/week12/correctness/faults/{fault_run_id}/endpoints")
async def correctness_fault_endpoint_evidence(
    request: Request, fault_run_id: str
):
    """Query the exact runtime refs captured at the armed checkpoint."""
    error = _correctness_auth_error(request)
    if error is not None:
        return error
    gate = getattr(request.app.state, "correctness_fault_gate", None)
    client = getattr(request.app.state, "http_infer_client", None)
    value = None if gate is None else await gate.snapshot()
    if value is None or value.get("fault_run_id") != fault_run_id:
        return JSONResponse({"error": "fault_not_found"}, status_code=404)
    if value.get("fault_kind") not in CORRECTNESS_ENDPOINT_AUTHORITY_FAULTS:
        return JSONResponse(
            {"error": "endpoint_registry_is_not_recovery_authority"},
            status_code=409,
        )
    details = value.get("details")
    if client is None or not isinstance(details, dict):
        return JSONResponse({"error": "endpoint_authority_not_ready"}, status_code=503)
    from dataclasses import asdict
    from prism_serve.router.http_rpc import EndpointOperationRef

    raw_refs: list[tuple[str, dict[str, object]]] = []
    single_ref = details.get("endpoint_ref")
    if isinstance(single_ref, dict):
        role = str(details.get("route_role") or "")
        if role not in {"source", "target"}:
            return JSONResponse(
                {"error": "endpoint_authority_not_ready"}, status_code=503
            )
        raw_refs.append((role, single_ref))
    else:
        for role in ("source", "target"):
            raw_ref = details.get(f"{role}_endpoint_ref")
            if not isinstance(raw_ref, dict):
                return JSONResponse(
                    {"error": "endpoint_authority_not_ready"}, status_code=503
                )
            raw_refs.append((role, raw_ref))
    results = []
    try:
        for role, raw_ref in raw_refs:
            ref = EndpointOperationRef(**raw_ref)
            snapshot = await client.operation_ref_status(
                ref.target_instance, ref
            )
            results.append({
                "endpoint_role": role,
                "instance_id": ref.target_instance,
                "endpoint_ref": asdict(ref),
                "action": "QUERY",
                "snapshot": snapshot,
            })
        resources = await _correctness_resource_snapshot(request.app)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return JSONResponse(
            {"error": "endpoint_authority_unavailable", "detail": str(exc)},
            status_code=503,
        )
    return {
        "fault_run_id": fault_run_id,
        "request_id": details.get("request_id"),
        "endpoint_proofs": results,
        "resources": resources,
        "events": value.get("events", []),
    }


@app.get("/internal/week12/correctness/replacements/{restart_run_id}")
async def correctness_replacement_evidence(
    request: Request, restart_run_id: str
):
    error = _correctness_auth_error(request)
    if error is not None:
        return error
    store = getattr(request.app.state, "replacement_store", None)
    if store is None:
        return JSONResponse({"error": "replacement_store_not_ready"}, status_code=503)
    seal = next((
        value for value in store.seals()
        if value.restart_run_id == restart_run_id
    ), None)
    if seal is None:
        return JSONResponse({"error": "replacement_run_not_found"}, status_code=404)
    return {
        "summary": store.state_summary(),
        "seal": {**asdict(seal), "seal_digest": seal.seal_digest},
    }


@app.get("/internal/week12/correctness/requests/{req_id}")
async def correctness_request_evidence(request: Request, req_id: str):
    error = _correctness_auth_error(request)
    if error is not None:
        return error
    control = getattr(request.app.state, "network_control", None)
    registry = getattr(request.app.state, "worker_registry", None)
    if control is None or registry is None:
        return JSONResponse({"error": "not_ready"}, status_code=503)
    observed = control.request_evidence(req_id)
    if observed is None:
        return JSONResponse({"error": "evidence_not_found"}, status_code=404)
    route = observed.get("route")
    src_blocks = observed.get("src_block_ids")
    dst_blocks = observed.get("dst_block_ids")
    if not isinstance(route, dict) or not isinstance(src_blocks, list) \
            or not isinstance(dst_blocks, list) or len(src_blocks) != len(dst_blocks):
        return JSONResponse({"error": "evidence_incomplete"}, status_code=503)
    source = str(route.get("source"))
    target = str(route.get("target"))
    mapping = []
    transport: dict[str, object]
    if source == target:
        if src_blocks or dst_blocks or int(observed.get("completed_bytes", -1)) != 0:
            return JSONResponse({"error": "local_evidence_contradiction"}, status_code=503)
        transport = {"selected_mode": "NO_TRANSFER", "completed_bytes": 0}
    else:
        capability = next((
            value for value in registry.capabilities.values()
            if set(value.pair_id.split("--")) == {source, target}
        ), None)
        if capability is None:
            return JSONResponse({"error": "pair_capability_missing"}, status_code=503)
        pair_members = capability.pair_id.split("--")
        members = registry.members
        mapping = [
            {
                "source_instance": source,
                "target_instance": target,
                "src_block": int(src),
                "dst_block": int(dst),
                "pair_id": capability.pair_id,
                "source_global_rank": members[source].global_rank,
                "target_global_rank": members[target].global_rank,
                "source_group_rank": pair_members.index(source),
                "target_group_rank": pair_members.index(target),
            }
            for src, dst in zip(src_blocks, dst_blocks)
        ]
        transport = {
            "selected_mode": capability.transport,
            "pair_id": capability.pair_id,
            "capability_generation": capability.probe_generation,
            "capability_evidence_path": capability.evidence_path,
            "completed_bytes": int(observed.get("completed_bytes", 0)),
            "work_terminal": observed.get("work_terminal") is True,
            "cuda_terminal": observed.get("cuda_terminal") is True,
        }
    return {**observed, "mapping": mapping, "transport": transport}


@app.post("/internal/register_instance")
async def register_instance(request: Request) -> JSONResponse:
    """Register a prefill or decode infer instance."""
    body = await request.json()
    if getattr(request.app.state, "worker_registry", None) is not None:
        return JSONResponse(
            {
                "error": "worker_registry_is_authority",
                "detail": "legacy registration is disabled for the fixed 2P2D worker world",
            },
            status_code=409,
        )
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


@app.get("/admin/topology")
async def topology_status(request: Request):
    ledger = getattr(request.app.state, "topology_admin", None)
    if ledger is None:
        return JSONResponse({"error": "topology authority unavailable"}, status_code=503)
    try:
        reports = await _pull_replacement_resource_reports(
            request.app,
            ledger.registry,
            excluded_operation_ids=set(),
            excluded_resource_ids=set(),
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return JSONResponse(
            {"error": "live resource authority unavailable", "detail": str(exc)},
            status_code=503,
        )
    from prism_serve.gateway.topology_admin import worker_identity_wire

    return {
        "state": ledger.registry.state.value,
        "topology_generation": ledger.registry.expected_topology_generation,
        "members": sorted(ledger.registry.members),
        "identities": [
            worker_identity_wire(identity)
            for identity in ledger.registry.members.values()
        ],
        "resource_reports": reports,
        "accepted_restart_runs": sorted(ledger.records),
    }


@app.get("/admin/topology/evidence")
async def topology_evidence(request: Request, generation: str):
    """Probe the replacement endpoints without publishing their world."""
    client = getattr(request.app.state, "http_infer_client", None)
    if client is None:
        return JSONResponse({"error": "worker client unavailable"}, status_code=503)
    instances = ("p0", "p1", "d0", "d1")
    try:
        identities, capability_sets, reports = await asyncio.gather(
            asyncio.gather(*(client.get_identity(name) for name in instances)),
            asyncio.gather(*(client.get_capabilities(name) for name in instances)),
            asyncio.gather(*(client.get_resources(name) for name in instances)),
        )
        identity_by_name = {
            str(value["instance_id"]): value for value in identities
        }
        if set(identity_by_name) != set(instances) or any(
            value.get("topology_generation") != generation
            for value in identities
        ):
            raise ValueError("replacement identities are not one fresh 2P2D world")
        pair_attestations = {}
        for instance, value in zip(instances, capability_sets):
            if value.get("ready") is not True:
                raise ValueError("replacement pair probe is not ready")
            for pair in value.get("pairs", ()):
                pair_id = str(pair["pair_id"])
                if instance not in set(pair_id.split("--")):
                    raise ValueError("non-member replacement pair attestation")
                pair_attestations.setdefault(pair_id, {})[instance] = pair
        try:
            pairs = _collapse_pair_attestations(pair_attestations)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        report_by_name = dict(zip(instances, reports))
        if any(
            report.get("complete") is not True
            or report.get("instance_epoch")
            != f"{identity_by_name[name]['pod_uid']}:{identity_by_name[name]['process_generation']}"
            for name, report in report_by_name.items()
        ):
            raise ValueError("replacement resource report is incomplete or stale")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return JSONResponse({"ready": False, "error": str(exc)}, status_code=503)
    return {
        "ready": True,
        "identities": list(identities),
        "pair_capabilities": [pairs[name] for name in sorted(pairs)],
        "resource_reports": report_by_name,
    }


@app.post("/admin/topology/accept")
async def topology_accept(request: Request):
    ledger = getattr(request.app.state, "topology_admin", None)
    if ledger is None:
        return JSONResponse({"error": "topology authority unavailable"}, status_code=503)
    try:
        record = await _accept_replacement_topology(
            request.app, await request.json(), _build_config()
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        payload = {"error": str(exc)}
        code = getattr(exc, "code", None)
        if isinstance(code, str):
            payload["code"] = code
        seal = getattr(exc, "seal", None)
        seal_digest = getattr(seal, "seal_digest", None)
        if isinstance(seal_digest, str):
            payload["seal_digest"] = seal_digest
        return JSONResponse(
            payload,
            status_code=int(getattr(exc, "http_status", 409)),
        )
    request.app.state.metrics.increment(
        "pd_world_restart_total",
        labels={"outcome": "accepted", "reason": "four_reports_and_probes"},
    )
    return {
        "restart_run_id": record.restart_run_id,
        "accepted": record.accepted,
        "new_topology_generation": record.new_topology_generation,
        "decision_digest": record.decision_digest,
    }


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


async def _accept_replacement_topology(
    app: FastAPI, body: dict[str, object], config: dict
):
    acceptance_task = asyncio.current_task()
    if acceptance_task is None:
        raise RuntimeError("topology acceptance requires an asyncio task")
    active_task = getattr(app.state, "topology_acceptance_task", None)
    if (
        active_task is not None
        and active_task is not acceptance_task
        and not active_task.done()
    ):
        raise RuntimeError(
            "another topology acceptance is already running"
        )
    app.state.topology_acceptance_task = acceptance_task
    try:
        return await _accept_replacement_topology_impl(app, body, config)
    finally:
        _finish_expected_reconciler_replacement(app, acceptance_task)
        if (
            getattr(app.state, "topology_acceptance_task", None)
            is acceptance_task
        ):
            app.state.topology_acceptance_task = None


async def _accept_replacement_topology_impl(
    app: FastAPI, body: dict[str, object], config: dict
):
    """Stage every generation-bound consumer, then publish one snapshot.

    Readiness stays false throughout.  A failed stage leaves every active
    object reference unchanged; whole-world replacement records are written by
    the old release authority before an old quarantined slot is released.
    """
    from prism_serve.router.http_rpc import EndpointSequenceAllocator
    from prism_serve.router.network_rpc import (
        NetworkControlRPC,
        activate_replacement_owner,
    )
    from prism_serve.scheduler.resource_release import (
        ReplacementEvidence,
        ResourceReleaseEvaluator,
    )
    from prism_serve.scheduler.scheduler import PDScheduler
    from prism_serve.scheduler.sequence_state import RequestTracker
    from prism_serve.gateway.output import GatewayOutputBuffer

    prefix_poll_interval_s = (
        _prefix_poll_interval_s(config)
        if getattr(app.state, "affinity_coordinator", None) is not None
        else None
    )
    ledger = app.state.topology_admin
    store = getattr(app.state, "replacement_store", None)
    if store is not None and store.transition_closed:
        decision_digest = ledger._digest(body)
        requested_generation = str(
            body.get("new_topology_generation") or ""
        )
        completed = store.exact_completed_run(
            restart_run_id=str(body.get("restart_run_id") or ""),
            old_topology_generation=str(
                body.get("old_topology_generation") or ""
            ),
            new_topology_generation=requested_generation,
            decision_digest=decision_digest,
        )
        if completed is not None:
            if (
                ledger.registry.expected_topology_generation
                != requested_generation
            ):
                from prism_serve.scheduler.replacement_store import (
                    RetiredReplacementRun,
                )

                raise RetiredReplacementRun(completed)
            _begin_topology_acceptance(
                app, requested_generation
            )
            return await _recover_sealed_replacement(app, body)
        if (
            not ledger.records
            and app.state.worker_registry is ledger.registry
            and ledger.registry.expected_topology_generation
            == str(body.get("new_topology_generation") or "")
        ):
            _begin_topology_acceptance(
                app, str(body.get("new_topology_generation") or "")
            )
            return await _recover_cold_replacement(app, body)
    if (
        store is not None
        and not store.transition_closed
        and app.state.worker_registry is ledger.registry
        and ledger.registry.expected_topology_generation
        == str(body.get("new_topology_generation") or "")
    ):
        _begin_topology_acceptance(
            app, str(body.get("new_topology_generation") or "")
        )
        return await _recover_unsealed_replacement(app, body)
    existing_record = ledger.records.get(
        str(body.get("restart_run_id") or "")
    )
    if (
        store is not None
        and (
            str(body.get("old_topology_generation") or "")
            != ledger.registry.expected_topology_generation
            or (
                existing_record is not None
                and existing_record.new_topology_generation
                != ledger.registry.expected_topology_generation
            )
        )
    ):
        if (
            existing_record is not None
            and existing_record.decision_digest != ledger._digest(body)
        ):
            raise ValueError(
                "restart run id reused with different evidence"
            )
        from prism_serve.scheduler.replacement_store import (
            UnknownReplacementRun,
        )

        raise UnknownReplacementRun(
            "UNKNOWN_REPLACEMENT_RUN: completion seal is no longer retained"
        )
    record, candidate_registry, replay = ledger.stage(body)
    _begin_topology_acceptance(app, record.new_topology_generation)
    if (
        replay
        and app.state.worker_registry is candidate_registry
        and _replacement_store_allows_admission(app)
    ):
        _finalize_topology_acceptance(
            app, record.new_topology_generation
        )
        return record

    old_scheduler = app.state.scheduler
    old_evaluator = app.state.resource_release_evaluator
    old_network = app.state.network_control
    old_allocator = app.state.operation_allocator
    old_tracker = getattr(app.state, "tracker", None)
    old_output_buffer = getattr(app.state, "output_buffer", None)
    old_loop = getattr(app.state, "loop_task", None)
    coordinator = getattr(app.state, "affinity_coordinator", None)
    reconciler_task = getattr(app.state, "reconciler_task", None)
    owner_generation = (
        old_allocator.owner_generation
        if old_allocator is not None
        else app.state.queue.owner_id
    )

    # Stop every old-world mutator before taking the replacement snapshot.
    # Admission was closed above, so no new request can enter this boundary.
    if old_loop is not None:
        old_loop.cancel()
        await asyncio.gather(old_loop, return_exceptions=True)
    if old_network is not None and hasattr(old_network, "quiesce"):
        await old_network.quiesce()
    if coordinator is not None:
        await coordinator.shutdown()
    if old_tracker is not None:
        await old_tracker.quiesce_cleanup_tasks()
    if reconciler_task is not None:
        _begin_expected_reconciler_replacement(app, reconciler_task)
        reconciler_task.cancel()
        await asyncio.gather(reconciler_task, return_exceptions=True)

    frozen_requests = (
        tuple(old_tracker.all_requests()) if old_tracker is not None else ()
    )
    frozen_leases = old_scheduler.replacement_decode_leases()

    excluded_operations = sorted(str(value) for value in body.get("old_operation_ids", ()))
    frozen_request_operations = set()
    for req in frozen_requests:
        frozen_request_operations.add(req.req_id)
        for value in (req.active_operation_id, req.transfer_operation_id):
            if value:
                frozen_request_operations.add(value)
        for name in (
            "dispatch_operation_ref", "suffix_operation_ref",
            "target_request_ref", "target_request_commit_ref",
            "transfer_source_ref", "transfer_target_ref",
        ):
            operation_ref = getattr(req, name, None)
            operation_id = getattr(operation_ref, "operation_id", "")
            if operation_id:
                frozen_request_operations.add(operation_id)
    missing = (
        {lease.operation_id for lease in frozen_leases}
        | frozen_request_operations
    ) - set(excluded_operations)
    if missing:
        raise ValueError(
            f"replacement evidence omits frozen operations: {sorted(missing)!r}"
        )
    exclusion_digest = _replacement_exclusion_digest(body)
    # New workers must accept the same active gateway owner before publication.
    app.state.owner_takeover_audit = await activate_replacement_owner(
        app.state.http_infer_client,
        tuple(sorted(candidate_registry.members)),
        owner_generation,
        max_audit_entries=4 * (
            config.get("active_operation_cap", 512)
            + config.get("terminal_snapshot_cap", 4096)
        ),
    )
    live_reports = await _pull_replacement_resource_reports(
        app,
        candidate_registry,
        excluded_operation_ids=set(excluded_operations),
        excluded_resource_ids={
            str(value) for value in body.get("old_resource_ids", ())
        },
    )


    for req in frozen_requests:
        if old_output_buffer is not None:
            await old_output_buffer.fail(req.req_id, "topology_replaced")
        old_tracker.remove(req.req_id)
    # Build every candidate consumer only after the replacement workers have
    # confirmed the active owner and published their current-epoch reports.
    candidate_scheduler = PDScheduler(
        config, worker_registry=candidate_registry
    )
    for instance_id, identity in candidate_registry.members.items():
        report = candidate_registry.resource_signal(instance_id).report or {}
        candidate_scheduler.register_instance(
            instance_id=instance_id,
            role=identity.role,
            max_slots=int(report.get("max_slots", 1 if identity.role == "decode" else 0)),
            instance_epoch=identity.instance_epoch,
            active_request_ids=list(report.get("active_request_ids", ())),
        )
    candidate_allocator = EndpointSequenceAllocator(
        record.new_topology_generation, owner_generation
    )
    candidate_network = NetworkControlRPC(
        app.state.http_infer_client,
        candidate_allocator,
        {
            name: identity.instance_epoch
            for name, identity in candidate_registry.members.items()
        },
        query_interval_s=config["operation_query_interval_ms"] / 1000.0,
        operation_timeout_s=config["kv_transfer_timeout_s"],
        block_size=config.get("prefix_block_size", 256),
        block_bytes=config.get("prefix_block_bytes", 0),
        active_operation_cap=config.get("active_operation_cap", 512),
        terminal_snapshot_cap=config.get("terminal_snapshot_cap", 4096),
        metrics=app.state.metrics,
    )
    candidate_evaluator = ResourceReleaseEvaluator(
        candidate_scheduler,
        app.state.http_infer_client.finalize_release,
        app.state.metrics,
        active_operation_cap=config.get("active_operation_cap", 512),
        terminal_snapshot_cap=config.get("terminal_snapshot_cap", 4096),
        replacement_store=getattr(app.state, "replacement_store", None),
    )
    candidate_evaluator.set_correctness_fault_gate(
        getattr(app.state, "correctness_fault_gate", None)
    )
    candidate_network.set_release_evaluator(candidate_evaluator)
    candidate_network.set_correctness_fault_gate(
        getattr(app.state, "correctness_fault_gate", None)
    )
    candidate_tracker = RequestTracker(app.state.metrics)
    candidate_output_buffer = GatewayOutputBuffer(
        active_operation_cap=config.get("active_operation_cap", 512),
        terminal_snapshot_cap=config.get("terminal_snapshot_cap", 4096),
    )
    from prism_serve.scheduler.transfer_governor import TransferGovernor
    candidate_governor = TransferGovernor(
        config, candidate_network, app.state.metrics,
        worker_registry=candidate_registry,
    )
    candidate_governor.set_expected_epochs(
        candidate_scheduler.decode_instance_epochs()
    )

    replacement_evidence = ReplacementEvidence(
        restart_run_id=record.restart_run_id,
        old_topology_generation=record.old_topology_generation,
        new_topology_generation=record.new_topology_generation,
        old_termination_proof_digests=record.termination_proof_digests,
        fresh_resource_report_digests=tuple(
            ledger._digest(live_reports[name]) for name in sorted(live_reports)
        ),
        excluded_old_operation_digest=exclusion_digest,
        accepted=record.accepted,
        decision_digest=record.decision_digest,
    )
    for lease in frozen_leases:
        old_scheduler.quarantine_decode_slot(lease.operation_id)
    entries = tuple(
        (
            f"replacement:{record.restart_run_id}:{lease.operation_id}",
            lease.operation_id,
            lease.lease_id,
            ("DECODE_SLOT",),
        )
        for lease in frozen_leases
    )
    fault_gate = getattr(app.state, "correctness_fault_gate", None)
    if fault_gate is not None:
        fault_gate.record_event(
            "release_predicates_satisfied",
            {
                "restart_run_id": record.restart_run_id,
                "operation_ids": [lease.operation_id for lease in frozen_leases],
                "old_termination_proof_digests": list(
                    replacement_evidence.old_termination_proof_digests
                ),
                "fresh_resource_report_digests": list(
                    replacement_evidence.fresh_resource_report_digests
                ),
            },
        )
    # A filesystem error can be reported after os.replace but before the
    # directory fsync returns. Therefore every persist exception is ambiguous:
    # leases stay QUARANTINED and admission stays closed; nothing is restored.
    persisted_records = old_evaluator.persist_whole_world_replaced_batch(
        entries, evidence=replacement_evidence
    )
    if fault_gate is not None:
        fault_gate.record_event(
            "replacement_record_durable",
            {
                "restart_run_id": record.restart_run_id,
                "operation_ids": [value.operation_id for value in persisted_records],
                "record_count": len(persisted_records),
            },
        )

    # The in-memory topology publication is replayable while the durable run is
    # still unsealed. Slot release remains strictly after the durable records.
    ledger.commit(record, candidate_registry)
    old_evaluator.release_persisted_replacement_batch(persisted_records)
    if fault_gate is not None:
        fault_gate.record_event(
            "slot_released",
            {
                "restart_run_id": record.restart_run_id,
                "operation_ids": [value.operation_id for value in persisted_records],
            },
        )
    candidate_prefix_reconciler = None
    candidate_prefix_publication = None
    if coordinator is not None:
        from prism_serve.router.reconciler import PrefixReconciler

        candidate_prefix_reconciler = PrefixReconciler(
            app.state.prefix_index,
            candidate_network,
            owner_generation,
            record.new_topology_generation,
            app.state.metrics,
        )
        candidate_prefix_publication = (
            await candidate_prefix_reconciler.rebuild_world({
                name: identity.instance_epoch
                for name, identity in candidate_registry.members.items()
            })
        )

    # No fallible construction follows this publication block.
    app.state.worker_registry = candidate_registry
    app.state.operation_allocator = candidate_allocator
    app.state.network_control = candidate_network
    app.state.scheduler = candidate_scheduler
    app.state.resource_release_evaluator = candidate_evaluator
    app.state.tracker = candidate_tracker
    app.state.output_buffer = candidate_output_buffer
    app.state.governor = candidate_governor
    if candidate_prefix_reconciler is not None:
        app.state.prefix_reconciler = candidate_prefix_reconciler
        app.state.prefix_world_publication = candidate_prefix_publication
    if coordinator is not None:
        coordinator.rpc = candidate_network
        coordinator.operation_allocator = candidate_allocator
        coordinator._contexts.clear()
    if hasattr(app.state.metrics, "set_scheduler"):
        app.state.metrics.set_scheduler(candidate_scheduler)
    if hasattr(app.state.metrics, "set_infer_client"):
        app.state.metrics.set_infer_client(candidate_network)
    if hasattr(app.state.metrics, "set_governor"):
        app.state.metrics.set_governor(candidate_governor)

    if candidate_prefix_reconciler is not None:
        app.state.reconciler_task = asyncio.create_task(
            candidate_prefix_reconciler.run(
                lambda: app.state.worker_registry.members.keys(),
                prefix_poll_interval_s,
            )
        )
        app.state.reconciler_task.add_done_callback(
            lambda task: _on_control_plane_task_done(
                app,
                task,
                component="prefix_reconciler",
                invalidate_prefix=True,
            )
        )


    _finish_expected_reconciler_replacement(app)

    if old_loop is not None:
        from prism_serve.scheduler.main_loop import schedule_loop
        app.state.loop_task = asyncio.create_task(schedule_loop(
            candidate_scheduler,
            candidate_governor,
            candidate_tracker,
            app.state.queue,
            app.state.metrics,
            config,
            coordinator,
            candidate_output_buffer,
            candidate_allocator,
        ))
        app.state.loop_task.add_done_callback(
            lambda task: _on_schedule_loop_done(app, task)
        )


    final_reports = await _pull_replacement_resource_reports(
        app,
        candidate_registry,
        excluded_operation_ids=set(excluded_operations),
        excluded_resource_ids={
            str(value) for value in body.get("old_resource_ids", ())
        },
    )
    final_replacement_evidence = ReplacementEvidence(
        restart_run_id=replacement_evidence.restart_run_id,
        old_topology_generation=replacement_evidence.old_topology_generation,
        new_topology_generation=replacement_evidence.new_topology_generation,
        old_termination_proof_digests=(
            replacement_evidence.old_termination_proof_digests
        ),
        fresh_resource_report_digests=tuple(
            ledger._digest(final_reports[name])
            for name in sorted(final_reports)
        ),
        excluded_old_operation_digest=(
            replacement_evidence.excluded_old_operation_digest
        ),
        accepted=replacement_evidence.accepted,
        decision_digest=replacement_evidence.decision_digest,
    )


    old_evaluator.seal_whole_world_replacement(final_replacement_evidence)



    await _pull_replacement_resource_reports(
        app,
        candidate_registry,
        excluded_operation_ids=set(excluded_operations),
        excluded_resource_ids={
            str(value) for value in body.get("old_resource_ids", ())
        },
    )
    _finalize_topology_acceptance(app, record.new_topology_generation)
    return record


def _replacement_exclusion_digest(body: dict[str, object]) -> str:
    exclusions = {
        "operation_ids": sorted(
            str(value) for value in body.get("old_operation_ids", ())
        ),
        "resource_ids": sorted(
            str(value) for value in body.get("old_resource_ids", ())
        ),
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(
            exclusions, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


async def _get_resource_report_with_received_at(
    client: object,
    instance_id: str,
    registry: object,
) -> tuple[object, float]:
    """Return one resource response with its actual local completion time."""
    method = getattr(client, "get_resources")
    report = await method(instance_id)
    capture = getattr(registry, "capture_resource_report_received_at")
    return report, capture()


async def _pull_replacement_resource_reports(
    app: FastAPI,
    registry,
    *,
    excluded_operation_ids: set[str],
    excluded_resource_ids: set[str],
    incomplete_is_not_ready: bool = False,
) -> dict[str, dict[str, object]]:
    """Pull one fresh, exhaustive report from every replacement worker."""
    instances = sorted(registry.members)
    report_samples = await asyncio.gather(
        *(
            _get_resource_report_with_received_at(
                app.state.http_infer_client, instance_id, registry,
            )
            for instance_id in instances
        )
    )
    by_instance: dict[str, dict[str, object]] = {}
    for instance_id, (report, received_at) in zip(instances, report_samples):
        if not isinstance(report, dict):
            raise ValueError(
                f"replacement resource report for {instance_id} is not an object"
            )
        if incomplete_is_not_ready and report.get("complete") is False:
            raise _GatewayBootstrapNotReady(
                f"replacement resource report for {instance_id} is incomplete"
            )
        if not registry.update_resource_report(
            instance_id, report, received_at=received_at,
        ):
            raise ValueError(
                f"replacement resource report for {instance_id} is invalid"
            )
        operation_ids = report.get("operation_ids")
        resource_ids = report.get("resource_ids")
        if not isinstance(operation_ids, list) or not isinstance(resource_ids, list):
            raise ValueError(
                "replacement resource report must exhaustively list operations "
                "and resources"
            )
        if {str(value) for value in operation_ids} & excluded_operation_ids:
            raise ValueError(
                f"replacement worker {instance_id} still reports an old operation"
            )
        if {str(value) for value in resource_ids} & excluded_resource_ids:
            raise ValueError(
                f"replacement worker {instance_id} still reports an old resource"
            )
        by_instance[instance_id] = dict(report)
    if not registry.world_fresh():
        error = "replacement worker world is not fresh after report pull"
        if incomplete_is_not_ready:
            raise _GatewayBootstrapNotReady(error)
        raise ValueError(error)
    return by_instance


async def _recover_cold_replacement(
    app: FastAPI, body: dict[str, object]
):
    """Accept after Gateway crashed before creating any local run state.

    The new Gateway has no old tracker or quarantine leases to release.  It may
    only persist an empty completion segment after revalidating the exact new
    world, then seal that segment as the replay authority.
    """
    from prism_serve.gateway.topology_admin import RestartRunRecord
    from prism_serve.scheduler.resource_release import ReplacementEvidence

    ledger = app.state.topology_admin
    registry = ledger.registry
    store = app.state.replacement_store
    run_id = str(body["restart_run_id"])
    old_generation = str(body["old_topology_generation"])
    new_generation = str(body["new_topology_generation"])
    if not run_id or not old_generation or not new_generation:
        raise ValueError("cold replacement recovery requires run and generations")
    if old_generation == new_generation:
        raise ValueError("cold replacement recovery requires a fresh generation")
    if registry.expected_topology_generation != new_generation:
        raise ValueError("cold replacement recovery registry generation mismatch")
    store.validate_cold_recovery_base(
        old_topology_generation=old_generation
    )
    if app.state.scheduler.replacement_decode_leases():
        raise ValueError("cold replacement recovery cannot own old local leases")
    tracker = getattr(app.state, "tracker", None)
    if tracker is not None and tuple(tracker.all_requests()):
        raise ValueError("cold replacement recovery cannot own old requests")

    terminations = ledger.validate_physical_termination_records(body)
    body_identities = _canonical_worker_identity_map(body.get("identities"))
    from prism_serve.gateway.topology_admin import worker_identity_wire

    live_identities = {
        name: worker_identity_wire(identity)
        for name, identity in registry.members.items()
    }
    if body_identities != live_identities:
        raise ValueError("cold replacement identities do not match live world")
    body_capabilities = {
        str(value["pair_id"]): dict(value)
        for value in body.get("pair_capabilities", ())
    }
    live_capabilities = {
        pair_id: asdict(capability)
        for pair_id, capability in registry.capabilities.items()
    }
    if body_capabilities != live_capabilities:
        raise ValueError("cold replacement pair probes do not match live world")

    operation_values = body.get("old_operation_ids")
    resource_values = body.get("old_resource_ids")
    if not isinstance(operation_values, list) or not isinstance(
        resource_values, list
    ):
        raise ValueError("cold replacement old id snapshots must be lists")
    excluded_operations = {str(value) for value in operation_values}
    excluded_resources = {str(value) for value in resource_values}
    if len(excluded_operations) != len(operation_values) or len(
        excluded_resources
    ) != len(resource_values):
        raise ValueError("cold replacement old id snapshots must be unique")
    reports = dict(body.get("resource_reports", {}))
    if set(reports) != set(registry.members):
        raise ValueError("cold replacement requires four resource reports")
    for instance, report in reports.items():
        if not registry.update_resource_report(instance, report):
            raise ValueError("cold replacement resource report is invalid")
        operation_ids = report.get("operation_ids")
        resource_ids = report.get("resource_ids")
        if not isinstance(operation_ids, list) or not isinstance(
            resource_ids, list
        ):
            raise ValueError("cold replacement report ids must be exhaustive lists")
        if {str(value) for value in operation_ids} & excluded_operations or {
            str(value) for value in resource_ids
        } & excluded_resources:
            raise ValueError("cold replacement report still contains old resources")
        if report.get("excluded_operation_ids") != sorted(excluded_operations):
            raise ValueError("cold replacement report omits old operations")
        if report.get("excluded_resource_ids") != sorted(excluded_resources):
            raise ValueError("cold replacement report omits old resources")

    live_reports = await _pull_replacement_resource_reports(
        app,
        registry,
        excluded_operation_ids=excluded_operations,
        excluded_resource_ids=excluded_resources,
    )
    decision_digest = ledger._digest(body)
    record = RestartRunRecord(
        restart_run_id=run_id,
        old_topology_generation=old_generation,
        new_topology_generation=new_generation,
        termination_proof_digests=tuple(
            ledger._digest(value) for value in terminations
        ),
        fresh_resource_report_digests=tuple(
            ledger._digest(reports[name]) for name in sorted(reports)
        ),
        pair_probe_digests=tuple(
            ledger._digest(body_capabilities[name])
            for name in sorted(body_capabilities)
        ),
        decision_digest=decision_digest,
    )
    replacement_evidence = ReplacementEvidence(
        restart_run_id=run_id,
        old_topology_generation=old_generation,
        new_topology_generation=new_generation,
        old_termination_proof_digests=record.termination_proof_digests,
        fresh_resource_report_digests=tuple(
            ledger._digest(live_reports[name]) for name in sorted(live_reports)
        ),
        excluded_old_operation_digest=_replacement_exclusion_digest(body),
        accepted=True,
        decision_digest=decision_digest,
    )
    store.persist_records(
        (),
        restart_run_id=run_id,
        old_topology_generation=old_generation,
        decision_digest=decision_digest,
    )
    app.state.resource_release_evaluator.seal_whole_world_replacement(
        replacement_evidence
    )
    ledger.recover_committed_run(record)
    _finalize_topology_acceptance(app, new_generation)
    return record


async def _recover_sealed_replacement(
    app: FastAPI, body: dict[str, object]
):
    """Replay the exact completion receipt after response loss + Gateway crash.

    The retained seal proves that this exact decision already completed.  This
    path only revalidates the current new world and rebuilds the process-local
    ledger; it must never persist, release, finalize, or create another seal.
    """
    from prism_serve.gateway.topology_admin import RestartRunRecord

    ledger = app.state.topology_admin
    registry = ledger.registry
    store = app.state.replacement_store
    run_id = str(body["restart_run_id"])
    old_generation = str(body["old_topology_generation"])
    new_generation = str(body["new_topology_generation"])
    decision_digest = ledger._digest(body)
    seal = store.exact_completed_run(
        restart_run_id=run_id,
        old_topology_generation=old_generation,
        new_topology_generation=new_generation,
        decision_digest=decision_digest,
    )
    if seal is None:
        raise ValueError("sealed replacement completion is no longer retained")
    if getattr(app.state, "worker_registry", None) is not registry:
        raise ValueError(
            "sealed replacement replay requires the installed candidate runtime"
        )
    if registry.expected_topology_generation != new_generation:
        raise ValueError("sealed replacement replay generation mismatch")

    body_identities = _canonical_worker_identity_map(body.get("identities"))
    from prism_serve.gateway.topology_admin import worker_identity_wire

    live_identities = {
        name: worker_identity_wire(identity)
        for name, identity in registry.members.items()
    }
    if body_identities != live_identities:
        raise ValueError("sealed replacement identities do not match live world")
    body_capabilities = {
        str(value["pair_id"]): dict(value)
        for value in body.get("pair_capabilities", ())
    }
    live_capabilities = {
        pair_id: asdict(capability)
        for pair_id, capability in registry.capabilities.items()
    }
    if body_capabilities != live_capabilities:
        raise ValueError("sealed replacement pair probes do not match live world")

    excluded_operations = {
        str(value) for value in body.get("old_operation_ids", ())
    }
    excluded_resources = {
        str(value) for value in body.get("old_resource_ids", ())
    }
    await _pull_replacement_resource_reports(
        app,
        registry,
        excluded_operation_ids=excluded_operations,
        excluded_resource_ids=excluded_resources,
    )
    terminations = list(body.get("termination_records", ()))
    reports = dict(body.get("resource_reports", {}))
    record = RestartRunRecord(
        restart_run_id=run_id,
        old_topology_generation=old_generation,
        new_topology_generation=new_generation,
        termination_proof_digests=tuple(
            ledger._digest(value) for value in terminations
        ),
        fresh_resource_report_digests=tuple(
            ledger._digest(reports[name]) for name in sorted(reports)
        ),
        pair_probe_digests=tuple(
            ledger._digest(body_capabilities[name])
            for name in sorted(body_capabilities)
        ),
        decision_digest=decision_digest,
    )
    ledger.recover_committed_run(record)
    _finalize_topology_acceptance(app, new_generation)
    return record


async def _recover_unsealed_replacement(
    app: FastAPI, body: dict[str, object]
):
    """Seal a durable replacement run after the accepting Gateway crashed."""
    from prism_serve.gateway.topology_admin import RestartRunRecord
    from prism_serve.scheduler.resource_release import ReplacementEvidence

    ledger = app.state.topology_admin
    registry = ledger.registry
    store = app.state.replacement_store
    decision_digest = ledger._digest(body)
    old_generation = str(body["old_topology_generation"])
    new_generation = str(body["new_topology_generation"])
    run_id = str(body["restart_run_id"])
    store.validate_active_run(
        restart_run_id=run_id,
        old_topology_generation=old_generation,
        decision_digest=decision_digest,
    )
    if registry.expected_topology_generation != new_generation:
        raise ValueError("replacement recovery registry generation mismatch")

    body_identities = _canonical_worker_identity_map(body.get("identities"))
    from prism_serve.gateway.topology_admin import worker_identity_wire

    live_identities = {
        name: worker_identity_wire(identity)
        for name, identity in registry.members.items()
    }
    if body_identities != live_identities:
        raise ValueError("replacement recovery identities do not match live world")
    body_capabilities = {
        str(value["pair_id"]): dict(value)
        for value in body.get("pair_capabilities", ())
    }
    live_capabilities = {
        pair_id: asdict(capability)
        for pair_id, capability in registry.capabilities.items()
    }
    if body_capabilities != live_capabilities:
        raise ValueError("replacement recovery pair probes do not match live world")

    terminations = ledger.validate_physical_termination_records(body)

    excluded_operations = {
        str(value) for value in body.get("old_operation_ids", ())
    }
    excluded_resources = {
        str(value) for value in body.get("old_resource_ids", ())
    }
    live_reports = await _pull_replacement_resource_reports(
        app,
        registry,
        excluded_operation_ids=excluded_operations,
        excluded_resource_ids=excluded_resources,
    )
    body_reports = dict(body.get("resource_reports", {}))
    if set(body_reports) != set(registry.members):
        raise ValueError("replacement recovery body requires four resource reports")
    record = RestartRunRecord(
        restart_run_id=run_id,
        old_topology_generation=old_generation,
        new_topology_generation=new_generation,
        termination_proof_digests=tuple(
            ledger._digest(value) for value in terminations
        ),
        fresh_resource_report_digests=tuple(
            ledger._digest(body_reports[name]) for name in sorted(body_reports)
        ),
        pair_probe_digests=tuple(
            ledger._digest(body_capabilities[name])
            for name in sorted(body_capabilities)
        ),
        decision_digest=decision_digest,
    )
    replacement_evidence = ReplacementEvidence(
        restart_run_id=run_id,
        old_topology_generation=old_generation,
        new_topology_generation=new_generation,
        old_termination_proof_digests=record.termination_proof_digests,
        fresh_resource_report_digests=tuple(
            ledger._digest(live_reports[name]) for name in sorted(live_reports)
        ),
        excluded_old_operation_digest=_replacement_exclusion_digest(body),
        accepted=True,
        decision_digest=decision_digest,
    )
    app.state.resource_release_evaluator.seal_whole_world_replacement(
        replacement_evidence
    )
    ledger.recover_committed_run(record)
    _finalize_topology_acceptance(app, new_generation)
    return record


async def _bootstrap_week12_http_control(
    app: FastAPI,
    config: dict,
    *,
    deadline: float | None = None,
) -> None:
    topology_path = config["worker_topology_path"]
    if not topology_path:
        raise RuntimeError("multinode_e2e requires topology path")
    topology = json.loads(Path(topology_path).read_text(encoding="utf-8"))
    generation = str(topology.get("topology_generation") or "")
    if not generation:
        raise RuntimeError("worker topology artifact requires generation")



    accepted_generation = topology.get("accepted_topology_generation")
    if accepted_generation is None:
        accepted_generation = config.get("topology_generation")
    accepted_generation = str(accepted_generation or "")
    app.state.topology_acceptance_required = (
        accepted_generation != generation
    )
    app.state.pending_topology_generation = (
        generation if app.state.topology_acceptance_required else None
    )
    endpoints = topology.get("endpoints")
    if not isinstance(endpoints, dict) or set(endpoints) != {"p0", "p1", "d0", "d1"}:
        raise RuntimeError("worker topology must contain exactly p0,p1,d0,d1")
    from prism_serve.router.http_rpc import HttpInferClient
    from prism_serve.router.worker_registry import (
        PairCapability,
        WorkerIdentity,
        WorkerRegistry,
    )

    retry_interval_s = _bootstrap_retry_interval_s(config)
    if deadline is None:
        deadline = (
            asyncio.get_running_loop().time() + GATEWAY_BOOTSTRAP_TIMEOUT_S
        )
    client = HttpInferClient(endpoints, timeout_s=config["infer_rpc_timeout_s"])
    registry = WorkerRegistry(
        expected_topology_generation=generation,
        expected_kv_compatibility_id=(
            config.get("kv_compatibility_id") or None
        ),
        expected_model_profile=config.get("expected_model_profile"),
        require_gpudirect_rdma=config["require_gpudirect_rdma"],
        resource_report_stale_after_s=config["resource_report_stale_after_s"],
    )

    async def pull_worker_evidence(instance_id: str):
        identity_value, capability_value, report_sample = await asyncio.gather(
            client.get_identity(instance_id),
            client.get_capabilities(instance_id),
            _get_resource_report_with_received_at(
                client, instance_id, registry,
            ),
        )
        report, received_at = report_sample
        identity_fields = (
            "instance_id", "role", "topology_generation", "pod_uid",
            "process_generation", "rpc_endpoint", "global_rank",
            "topology_digest", "kv_compatibility_id",
        )
        if not isinstance(identity_value, dict) or not all(
            field in identity_value for field in identity_fields
        ):
            raise ValueError(
                f"worker identity payload is malformed: {instance_id}"
            )
        string_fields = tuple(
            field for field in identity_fields if field != "global_rank"
        )
        if any(
            not isinstance(identity_value[field], str)
            or not identity_value[field]
            for field in string_fields
        ) or isinstance(identity_value["global_rank"], bool) \
                or not isinstance(identity_value["global_rank"], int):
            raise ValueError(
                f"worker identity fields are malformed: {instance_id}"
            )
        identity = WorkerIdentity(**{
            field: identity_value[field] for field in identity_fields
        })
        if identity.instance_id != instance_id:
            raise RuntimeError(
                f"worker endpoint {instance_id} returned duplicate/foreign "
                f"identity {identity.instance_id}"
            )

        if not isinstance(capability_value, dict) or "ready" not in capability_value:
            raise ValueError(
                f"worker capability payload is malformed: {instance_id}"
            )
        if capability_value["ready"] is False:
            raise _GatewayBootstrapNotReady(
                f"worker capability is not ready: {instance_id}"
            )
        if capability_value["ready"] is not True or not isinstance(
            capability_value.get("pairs"), (list, tuple)
        ):
            raise ValueError(
                f"worker capability payload is malformed: {instance_id}"
            )
        pair_fields = (
            "pair_id", "source_epoch", "target_epoch", "transport",
            "probe_generation", "probe_passed", "evidence_path",
        )
        capabilities = []
        seen_pairs: set[str] = set()
        for pair_value in capability_value["pairs"]:
            if not isinstance(pair_value, dict) or not all(
                field in pair_value for field in pair_fields
            ):
                raise ValueError(
                    f"worker pair attestation is malformed: {instance_id}"
                )
            string_pair_fields = tuple(
                field for field in pair_fields if field != "probe_passed"
            )
            if any(
                not isinstance(pair_value[field], str)
                or not pair_value[field]
                for field in string_pair_fields
            ) or not isinstance(pair_value["probe_passed"], bool):
                raise ValueError(
                    f"worker pair attestation fields are malformed: {instance_id}"
                )
            capability = PairCapability(**{
                field: pair_value[field] for field in pair_fields
            })
            if capability.pair_id in seen_pairs:
                raise RuntimeError(
                    f"worker returned duplicate pair attestation: "
                    f"{instance_id}/{capability.pair_id}"
                )
            seen_pairs.add(capability.pair_id)
            if instance_id not in set(capability.pair_id.split("--")):
                raise RuntimeError("non-member worker attested pair capability")
            capabilities.append(capability)

        if not isinstance(report, dict) or "complete" not in report:
            raise ValueError(
                f"worker resource report is malformed: {instance_id}"
            )
        if report["complete"] is False:
            raise _GatewayBootstrapNotReady(
                f"worker resource report is incomplete: {instance_id}"
            )
        if report["complete"] is not True:
            raise ValueError(
                f"worker resource report complete flag is malformed: {instance_id}"
            )
        return identity, capabilities, report, received_at

    try:
        identities = []
        capability_attestations = {}
        reports = {}
        for instance_id in WEEK12_WORKER_IDS:
            identity, capabilities, report, received_at = (
                await _run_gateway_bootstrap_stage(
                    lambda instance_id=instance_id: pull_worker_evidence(
                        instance_id
                    ),
                    stage=f"worker evidence {instance_id}",
                    deadline=deadline,
                    retry_interval_s=retry_interval_s,
                )
            )
            identities.append(identity)
            for capability in capabilities:
                capability_attestations.setdefault(
                    capability.pair_id, {}
                )[instance_id] = capability
            reports[instance_id] = (report, received_at)
        capability_by_pair = _collapse_pair_attestations(
            capability_attestations
        )
        if not registry.install_world(identities, list(capability_by_pair.values())):
            raise RuntimeError("worker identities/capabilities failed closed")
        if not all(
            registry.update_resource_report(
                instance_id,
                reports[instance_id][0],
                received_at=reports[instance_id][1],
            )
            for instance_id in WEEK12_WORKER_IDS
        ):
            raise RuntimeError("worker resource reports are incomplete")
        if not registry.world_fresh():
            raise _GatewayBootstrapNotReady(
                "worker resource reports are stale before bootstrap publication"
            )
    except BaseException:
        await client.close()
        raise
    # Commit the mounted desired generation only after one complete bootstrap
    # round succeeds.  A whole-round retry must keep comparing against the
    # original accepted generation instead of inheriting a failed attempt's
    # partially mutated config.
    config["topology_generation"] = generation
    app.state.http_infer_client = client
    app.state.worker_registry = registry


def _collapse_pair_attestations(attestations):
    from prism_serve.router.worker_registry import EXPECTED_PAIRS

    if set(attestations) != EXPECTED_PAIRS:
        raise RuntimeError("worker world requires exact five pair attestations")
    collapsed = {}
    for pair_id, by_worker in attestations.items():
        endpoints = set(pair_id.split("--"))
        if set(by_worker) != endpoints:
            raise RuntimeError("pair capability requires two endpoint attestations")
        values = list(by_worker.values())
        if values[0] != values[1]:
            raise RuntimeError("pair endpoint capability attestations disagree")
        collapsed[pair_id] = values[0]
    return collapsed


async def _refresh_worker_world_once(app: FastAPI) -> None:
    """Refresh identity, capability, and resource evidence as one fenced cycle."""
    from prism_serve.router.worker_registry import PairCapability, WorkerIdentity

    instances = ("p0", "p1", "d0", "d1")
    identity_fields = (
        "instance_id", "role", "topology_generation", "pod_uid",
        "process_generation", "rpc_endpoint", "global_rank",
        "topology_digest", "kv_compatibility_id",
    )
    capability_fields = (
        "pair_id", "source_epoch", "target_epoch", "transport",
        "probe_generation", "probe_passed", "evidence_path",
    )
    registry = app.state.worker_registry
    expected_capabilities = registry.capabilities
    observed_attestations: dict[str, dict[str, PairCapability]] = {}
    refresh_complete = True
    semantic_drift = False
    for instance in instances:
        try:
            identity_value, capability_value, report_sample = await asyncio.gather(
                app.state.http_infer_client.get_identity(instance),
                app.state.http_infer_client.get_capabilities(instance),
                _get_resource_report_with_received_at(
                    app.state.http_infer_client, instance, registry,
                ),
            )
            report, received_at = report_sample
        except asyncio.CancelledError:
            raise
        except Exception:
            registry.resource_report_failed(instance)
            # A transport exception is not process-death authority.  Preserve
            # the last validated identity/capability/report and let its
            # monotonic age become stale naturally.
            refresh_complete = False
            continue



        if not isinstance(identity_value, dict) or not all(
            field in identity_value for field in identity_fields
        ):
            refresh_complete = False
            continue
        try:
            identity = WorkerIdentity(**{
                field: identity_value[field] for field in identity_fields
            })
        except (TypeError, ValueError):
            refresh_complete = False
            continue
        if not registry.observe_identity(identity):
            semantic_drift = True
            app.state.metrics.increment(
                "worker_epoch_change_total", labels={"instance": instance}
            )
            continue

        if (
            not isinstance(capability_value, dict)
            or capability_value.get("ready") is not True
            or not isinstance(capability_value.get("pairs"), (list, tuple))
        ):
            refresh_complete = False
            continue
        worker_pairs: dict[str, PairCapability] = {}
        capability_malformed = False
        for pair_value in capability_value["pairs"]:
            if not isinstance(pair_value, dict) or not all(
                field in pair_value for field in capability_fields
            ):
                capability_malformed = True
                break
            try:
                pair = PairCapability(**{
                    field: pair_value[field] for field in capability_fields
                })
                endpoints = set(pair.pair_id.split("--"))
            except (AttributeError, TypeError, ValueError):
                capability_malformed = True
                break
            if len(endpoints) != 2 or instance not in endpoints:
                capability_malformed = True
                break
            worker_pairs[pair.pair_id] = pair
        expected_pair_ids = {
            pair_id for pair_id in expected_capabilities
            if instance in set(pair_id.split("--"))
        }
        if capability_malformed or set(worker_pairs) != expected_pair_ids:
            refresh_complete = False
            continue

        if not isinstance(report, dict) or not registry.update_resource_report(
            instance, report, received_at=received_at,
        ):
            refresh_complete = False
            app.state.metrics.increment(
                "control_message_error_total",
                labels={"operation": "resource_report"},
            )
            continue
        for pair_id, pair in worker_pairs.items():
            observed_attestations.setdefault(pair_id, {})[instance] = pair

    observed_capabilities = dict(expected_capabilities)
    if not semantic_drift and refresh_complete:
        try:
            observed_capabilities = _collapse_pair_attestations(
                observed_attestations
            )
        except RuntimeError:
            semantic_drift = True
            observed_capabilities = {}
    if semantic_drift and registry.state.value != "FAILED":
        registry.observe_capabilities([])
    elif not semantic_drift and refresh_complete and not registry.observe_capabilities(
        list(observed_capabilities.values())
    ):
        semantic_drift = True
    if semantic_drift:
        app.state.accepting = False

    app.state.accepting = (
        not semantic_drift
        and registry.world_fresh()
        and _replacement_store_allows_admission(app)
        and _topology_acceptance_allows_admission(app)
        and _prefix_world_allows_admission(app)
        and _background_control_plane_tasks_healthy(app)
    )

    for pair_id, expected in expected_capabilities.items():
        observed = observed_capabilities.get(pair_id)
        for transport in sorted({
            "NCCL_GDR", "NCCL_SOCKET", "CUDA_IPC", expected.transport,
            *(value.transport for value in observed_capabilities.values()),
        }):
            app.state.metrics.gauge(
                "pair_capability_ready", 0,
                labels={"pair": pair_id, "transport": transport},
            )
        app.state.metrics.gauge(
            "pair_capability_ready",
            1 if observed == expected and not semantic_drift else 0,
            labels={"pair": pair_id, "transport": expected.transport},
        )
    for instance in instances:
        signal = registry.resource_signal(instance)
        if signal.age_s is not None:
            app.state.metrics.gauge(
                "resource_report_age_seconds", signal.age_s,
                labels={"instance": instance},
            )
        accepted_report = signal.report if isinstance(signal.report, dict) else {}
        counts = accepted_report.get("resources", {})
        if not isinstance(counts, dict):
            counts = {}
        for resource_kind in (
            "SOURCE_BLOCKS", "SOURCE_PIN", "SOURCE_RETAIN", "TARGET_PENDING",
            "TARGET_SEQUENCE", "TRANSFER_BYTES",
        ):
            app.state.metrics.gauge(
                "cleanup_resources_held", float(counts.get(resource_kind, 0)),
                labels={"instance": instance, "resource_kind": resource_kind},
            )

    leases = app.state.scheduler.quarantined_decode_leases()
    quarantined_ids = {lease.operation_id for lease in leases}
    operations, bytes_by_pair = app.state.governor.quarantined_transfer_totals(
        quarantined_ids
    )
    app.state.metrics.gauge(
        "transfer_quarantined_operations", operations,
        labels={"reason": "decode_slot"},
    )
    for pair_id in expected_capabilities:
        app.state.metrics.gauge(
            "transfer_quarantined_bytes", bytes_by_pair.get(pair_id, 0),
            labels={"pair": pair_id},
        )

    if hasattr(app.state.http_infer_client, "owner_status"):
        expected_owner = app.state.queue.owner_id
        for instance in instances:
            orphan_count = 0
            try:
                status = await app.state.http_infer_client.owner_status(instance)
                active_owner = status.get("active_owner")
                if active_owner not in {None, expected_owner}:
                    report = await app.state.http_infer_client.list_operations(
                        instance, str(active_owner)
                    )
                    orphan_count = len(report.get("operations", ()))
            except Exception:
                pass
            app.state.metrics.gauge(
                "orphan_operations", orphan_count,
                labels={"instance": instance},
            )
    for state in ("STARTING", "READY", "FAILED", "STOPPED"):
        app.state.metrics.gauge(
            "pd_topology_state", int(state == registry.state.value),
            labels={"state": state},
        )


async def _refresh_worker_resources(app: FastAPI, config: dict) -> None:
    """Keep readiness tied to fresh, complete, epoch-fenced reports."""
    interval = min(0.5, config["resource_report_stale_after_s"] / 2)
    while True:
        await _refresh_worker_world_once(app)
        await asyncio.sleep(interval)


def _replacement_store_allows_admission(app: FastAPI) -> bool:
    store = getattr(app.state, "replacement_store", None)
    return bool(
        store is None
        or (store.ready and store.transition_closed)
    )


def _topology_acceptance_allows_admission(app: FastAPI) -> bool:
    return not bool(
        getattr(app.state, "topology_acceptance_required", False)
    )


def _begin_expected_reconciler_replacement(
    app: FastAPI,
    reconciler_task: asyncio.Task,
) -> None:
    if reconciler_task.done():
        raise RuntimeError(
            "prefix reconciler stopped before topology replacement"
        )
    acceptance_task = asyncio.current_task()
    if acceptance_task is None:
        raise RuntimeError("reconciler replacement requires an asyncio task")
    if (
        getattr(app.state, "topology_acceptance_task", None)
        is not acceptance_task
    ):
        raise RuntimeError(
            "reconciler replacement requires the active topology acceptance"
        )
    active_task = getattr(
        app.state, "reconciler_replacement_task", None
    )
    if (
        active_task is not None
        and active_task is not acceptance_task
        and not active_task.done()
    ):
        raise RuntimeError(
            "another topology acceptance is replacing the prefix reconciler"
        )
    app.state.reconciler_replacement_task = acceptance_task


def _finish_expected_reconciler_replacement(
    app: FastAPI,
    acceptance_task: asyncio.Task | None = None,
) -> None:
    if acceptance_task is None:
        acceptance_task = asyncio.current_task()
    if (
        acceptance_task is not None
        and getattr(
            app.state, "reconciler_replacement_task", None
        ) is acceptance_task
    ):
        app.state.reconciler_replacement_task = None


def _expected_reconciler_replacement_active(app: FastAPI) -> bool:
    acceptance_task = getattr(
        app.state, "reconciler_replacement_task", None
    )
    registered_task = getattr(
        app.state, "topology_acceptance_task", None
    )
    return bool(
        getattr(app.state, "topology_acceptance_required", False)
        and acceptance_task is not None
        and acceptance_task is registered_task
        and not acceptance_task.done()
    )


def _background_control_plane_tasks_healthy(
    app: FastAPI,
    *,
    allow_reconciler_transition: bool = False,
) -> bool:
    if getattr(app.state, "control_plane_failed", False):
        return False
    config = getattr(app.state, "runtime_config", {})
    if not (
        isinstance(config, dict)
        and config.get("multinode_e2e_enabled")
    ):
        return True
    resource_task = getattr(app.state, "resource_refresh_task", None)
    if resource_task is None or resource_task.done():
        return False
    if config.get("affinity_enabled"):
        reconciler_task = getattr(app.state, "reconciler_task", None)
        if reconciler_task is None or (
            reconciler_task.done()
            and not (
                allow_reconciler_transition
                and _expected_reconciler_replacement_active(app)
            )
        ):
            return False
    return True


def _prefix_world_allows_admission(app: FastAPI) -> bool:
    config = getattr(app.state, "runtime_config", {})
    if not (
        isinstance(config, dict)
        and config.get("multinode_e2e_enabled")
        and config.get("affinity_enabled")
    ):
        return True
    registry = getattr(app.state, "worker_registry", None)
    reconciler = getattr(app.state, "prefix_reconciler", None)
    reconciler_task = getattr(app.state, "reconciler_task", None)
    publication = (
        getattr(reconciler, "world_publication", None)
        or getattr(app.state, "prefix_world_publication", None)
    )
    if (
        registry is None
        or reconciler is None
        or reconciler_task is None
        or reconciler_task.done()
        or publication is None
    ):
        return False
    expected_epochs = {
        name: identity.instance_epoch
        for name, identity in registry.members.items()
    }
    return bool(
        set(expected_epochs) == {"p0", "p1", "d0", "d1"}
        and publication.matches(expected_epochs)
        and reconciler.world_ready(expected_epochs)
    )


def _begin_topology_acceptance(app: FastAPI, generation: str) -> None:
    """Fence admission while a validated replacement transaction is pending."""
    if not generation:
        raise ValueError("pending topology generation must be non-empty")
    if getattr(app.state, "topology_acceptance_required", False):
        pending = getattr(app.state, "pending_topology_generation", None)
        if pending != generation:
            raise ValueError(
                "another topology generation is already pending acceptance"
            )
    app.state.topology_acceptance_required = True
    app.state.pending_topology_generation = generation
    app.state.accepting = False


def _complete_topology_acceptance(app: FastAPI, generation: str) -> None:
    if getattr(app.state, "topology_acceptance_required", False):
        pending = getattr(app.state, "pending_topology_generation", None)
        if pending != generation:
            raise ValueError("accepted generation does not match pending topology")
    app.state.topology_acceptance_required = False
    app.state.pending_topology_generation = None


def _finalize_topology_acceptance(app: FastAPI, generation: str) -> None:
    """Commit READY only when every post-replacement admission gate is closed."""
    if not getattr(app.state, "topology_acceptance_required", False):
        raise ValueError("topology acceptance is not pending")
    if getattr(app.state, "pending_topology_generation", None) != generation:
        raise ValueError("pending topology generation does not match candidate")
    registry = getattr(app.state, "worker_registry", None)
    if (
        registry is None
        or registry.expected_topology_generation != generation
    ):
        raise ValueError("topology acceptance candidate generation mismatch")
    ledger = getattr(app.state, "topology_admin", None)
    if ledger is not None and ledger.registry is not registry:
        raise ValueError(
            "topology acceptance candidate is not the committed ledger registry"
        )
    if not registry.world_fresh():
        raise ValueError(
            "replacement worker world is not fresh before topology acceptance"
        )
    if not _replacement_store_allows_admission(app):
        raise RuntimeError(
            "replacement store is not sealed before topology acceptance"
        )
    if not _prefix_world_allows_admission(app):
        raise RuntimeError(
            "prefix world is not ready before topology acceptance"
        )
    loop_task = getattr(app.state, "loop_task", None)
    if getattr(app.state, "control_plane_failed", False) or (
        loop_task is not None and loop_task.done()
    ):
        raise RuntimeError(
            "control plane failed before topology acceptance"
        )
    if not _background_control_plane_tasks_healthy(app):
        raise RuntimeError(
            "background control plane task failed before topology acceptance"
        )
    _complete_topology_acceptance(app, generation)
    app.state.accepting = True


def _build_config() -> dict:
    expected_model_profile = None
    if settings.multinode_e2e_enabled:
        expected_model_profile = {
            "profile_id": settings.model_profile_id,
            "schema_version": 1,
            "model_id": settings.model_id,
            "model_revision": settings.model_revision,
            "tokenizer_revision": settings.tokenizer_revision,
            "config_sha256": settings.model_config_sha256,
            "dtype": settings.runtime_dtype,
            "kv_layout": settings.kv_layout,
            "tokens_per_block": settings.prefix_block_size,
            "kv_block_bytes": settings.kv_block_bytes,
            "tensor_parallel_size": settings.tensor_parallel_size,
            "num_hidden_layers": settings.model_num_hidden_layers,
            "num_key_value_heads": settings.model_num_key_value_heads,
            "head_dim": settings.model_head_dim,
            "rope_theta": settings.model_rope_theta,
            "kv_compatibility_id": settings.kv_compatibility_id,
        }
    return {
        "nats_url":                 settings.nats_url,
        "nats_connect_timeout_s":   settings.nats_connect_timeout_s,
        "nats_max_reconnect_attempts": settings.nats_max_reconnect_attempts,
        "nats_required":            settings.nats_required,
        "scheduler_id":             settings.gateway_pod_uid,
        "scheduler_generation":     settings.gateway_process_generation,
        "control_plane_replica_count": settings.control_plane_replica_count,
        "multinode_e2e_enabled":      settings.multinode_e2e_enabled,
        "worker_topology_path":       settings.worker_topology_path,
        "topology_generation":        settings.topology_generation,
        "infer_rpc_timeout_s":        settings.infer_rpc_timeout_s,
        "operation_query_interval_ms": settings.operation_query_interval_ms,
        "active_operation_cap":       settings.active_operation_cap,
        "operation_reorder_window":   settings.operation_reorder_window,
        "terminal_snapshot_cap":      settings.terminal_snapshot_cap,
        "replacement_store_path":    settings.replacement_store_path,
        "replacement_store_max_records_per_run": (
            settings.replacement_store_max_records_per_run
        ),
        "replacement_store_seal_retention": (
            settings.replacement_store_seal_retention
        ),
        "correctness_harness_enabled": settings.correctness_harness_enabled,
        "correctness_harness_secret": settings.correctness_harness_secret,
        "correctness_fault_gate_timeout_s": (
            settings.correctness_fault_gate_timeout_s
        ),
        "resource_report_stale_after_s": settings.resource_report_stale_after_s,
        "transfer_abort_timeout_s":   settings.transfer_abort_timeout_s,
        "nccl_watchdog_timeout_s":    settings.nccl_watchdog_timeout_s,
        "require_gpudirect_rdma":     settings.require_gpudirect_rdma,
        "allowed_fallback_transport": settings.allowed_fallback_transport,
        "model_id":                   settings.model_id,
        "model_revision":             settings.model_revision,
        "model_config_sha256":        settings.model_config_sha256,
        "runtime_dtype":              settings.runtime_dtype,
        "tensor_parallel_size":       settings.tensor_parallel_size,
        "kv_block_bytes":             settings.kv_block_bytes,
        "expected_model_profile":      expected_model_profile,
        "tokenizer_model":             settings.tokenizer_model,
        "tokenizer_revision":          settings.tokenizer_revision,
        "chat_template_version":       settings.chat_template_version,
        "kv_compatibility_id":         settings.kv_compatibility_id,
        "prefix_block_size":           settings.prefix_block_size,
        "HIGH_WATERMARK":             settings.high_watermark,
        "LOW_WATERMARK":              settings.low_watermark,
        "MAX_BYTES_INFLIGHT":         settings.max_bytes_inflight,
        "max_bytes_inflight_per_pair": settings.max_bytes_inflight_per_pair,
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

        def record_correctness_fault_event(self, event, details):
            logger.debug("stub correctness event=%s details=%s", event, details)

        async def wait_nats_command_fault_authority(self, endpoint_ref, fault_kind):
            raise RuntimeError(
                "stub infer client cannot observe NATS command fault authority"
            )

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


def _on_control_plane_task_done(
    app: FastAPI,
    task: asyncio.Task,
    *,
    component: str,
    invalidate_prefix: bool = False,
) -> None:
    """Fail liveness/readiness when a long-running control task stops."""
    if task.cancelled():
        return
    app.state.accepting = False
    app.state.control_plane_failed = True
    if invalidate_prefix:
        app.state.prefix_world_publication = None
        reconciler = getattr(app.state, "prefix_reconciler", None)
        if reconciler is not None:
            reconciler.world_publication = None
    error = task.exception()
    if error is not None:
        logger.error(
            "%s stopped; liveness and readiness disabled",
            component,
            exc_info=(type(error), error, error.__traceback__),
        )
    else:
        logger.error(
            "%s returned unexpectedly; liveness and readiness disabled",
            component,
        )


def _on_schedule_loop_done(app: FastAPI, task: asyncio.Task) -> None:
    """Fail readiness when the scheduler exits outside normal shutdown."""
    _on_control_plane_task_done(
        app, task, component="schedule_loop"
    )


async def _drain_governor(governor, timeout_s: float) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if governor.is_drained():
            break
        await asyncio.sleep(0.1)


async def _wait_for_control_plane_drain(tracker, governor, timeout_s: float) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if len(tracker) == 0 and governor.is_drained():
            return True
        await asyncio.sleep(0.1)
    return False


async def _close_runtime_io(
    queue,
    metrics_task: asyncio.Task,
    metrics,
    http_infer_client,
    *,
    timeout_s: float,
) -> bool:
    """Bound the final NATS drain and local client shutdown as one phase."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    deadline_reached = asyncio.Event()

    async def close_all() -> None:
        await queue.close()
        if deadline_reached.is_set():
            return
        metrics_task.cancel()
        await asyncio.gather(metrics_task, return_exceptions=True)
        if deadline_reached.is_set():
            return
        await metrics.flush()
        if deadline_reached.is_set():
            return
        if http_infer_client is not None:
            await http_infer_client.close()

    def consume_late_result(completed: asyncio.Task) -> None:
        try:
            completed.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug(
                "late runtime I/O close task failed after shutdown deadline",
                exc_info=True,
            )

    task = asyncio.create_task(close_all())
    done, pending = await asyncio.wait((task,), timeout=timeout_s)
    if pending:
        deadline_reached.set()
        logger.warning(
            "Gateway runtime I/O close deadline reached; requesting cancellation "
            "without waiting past the process-exit fence"
        )
        task.cancel()
        metrics_task.cancel()
        task.add_done_callback(consume_late_result)
        metrics_task.add_done_callback(consume_late_result)
        return False
    await task
    return True


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


async def _abort_remaining_requests(
    tracker,
    scheduler,
    governor,
    owner_id: str,
    timeout_s: float,
    transfer_timeout_s: float | None = None,
    join_timeout_s: float | None = None,
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
    if not tasks:
        return
    if join_timeout_s is None:
        for task in tasks:
            await asyncio.shield(task)
        return
    if join_timeout_s <= 0:
        raise ValueError("join_timeout_s must be positive")

    done, pending = await asyncio.wait(tasks, timeout=join_timeout_s)
    if pending:
        logger.warning(
            "Gateway shutdown cleanup deadline reached; cancelling %d local "
            "cleanup task(s) while remote resources remain quarantined",
            len(pending),
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    for task in tasks:
        if task in done:
            await task


def main() -> None:
    """Console-script entry point: ``prism-serve``."""
    import uvicorn

    process_identity_path = os.environ.get(
        "PRISM_SERVE_PROCESS_IDENTITY_PATH", ""
    )
    if process_identity_path:
        from prism_serve.process_identity import publish_process_identity

        publish_process_identity(
            process_identity_path,
            component="gateway",
            instance_id="gateway",
            pod_uid=settings.gateway_pod_uid,
            process_generation=settings.gateway_process_generation,
        )
    uvicorn.run(
        "prism_serve.gateway.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        timeout_graceful_shutdown=UVICORN_GRACEFUL_SHUTDOWN_TIMEOUT_S,
    )


if __name__ == "__main__":
    main()

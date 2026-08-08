"""Tick-based control loop for disaggregated prefill and decode."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from prism_serve.scheduler.resource_release import ResourceReleaseEvaluator

if TYPE_CHECKING:
    from prism_serve.scheduler.scheduler import PDScheduler
    from prism_serve.scheduler.transfer_governor import TransferGovernor
    from prism_serve.scheduler.sequence_state import RequestTracker, SeqState
    from prism_serve.scheduler.queue import NATSQueue

logger = logging.getLogger(__name__)

TICK_INTERVAL_S: float = 0.010

NATS_COMMAND_FAULTS = {
    "nats_drop": 0,
    "nats_duplicate": 2,
    "nats_publish_unknown": 1,
}


class InjectedNATSCommandFault(RuntimeError):
    """Deterministic one-shot command publish fault for the correctness gate."""


async def _publish_command_with_fault(
    queue,
    subject: str,
    command: dict[str, object],
    directive: dict[str, object] | None,
    record_event,
    observe_worker_authority,
) -> None:
    fault_kind = str((directive or {}).get("fault_kind") or "")
    if fault_kind not in NATS_COMMAND_FAULTS:
        await queue.publish(subject, command)
        return

    endpoint_ref = command.get("endpoint_ref")
    if not isinstance(endpoint_ref, dict):
        raise RuntimeError("NATS command fault requires an exact endpoint ref")
    publish_count = 0
    for _ in range(NATS_COMMAND_FAULTS[fault_kind]):
        await queue.publish(subject, command)
        publish_count += 1
    observation = {"delivery_count": 0, "execution_count": 0}
    if fault_kind != "nats_drop":
        observation = await observe_worker_authority(endpoint_ref, fault_kind)
    delivery_count = observation.get("delivery_count")
    execution_count = observation.get("execution_count")
    expected = {
        "nats_drop": (0, 0),
        "nats_duplicate": (2, 1),
        "nats_publish_unknown": (1, 1),
    }[fault_kind]
    if (delivery_count, execution_count) != expected:
        raise RuntimeError(
            f"{fault_kind} worker authority mismatch: "
            f"delivery={delivery_count!r} execution={execution_count!r}"
        )
    record_event(
        "fault_injected",
        {
            "fault_kind": fault_kind,
            "operation_id": str(endpoint_ref.get("operation_id") or ""),
            "endpoint_ref": dict(endpoint_ref),
            "subject": subject,
            "publish_count": publish_count,
            "delivery_count": delivery_count,
            "execution_count": execution_count,
        },
    )
    raise InjectedNATSCommandFault(
        f"injected {fault_kind} after {publish_count} publish call(s)"
    )


def _normal_prefill_payload(req, queue) -> dict[str, object]:
    """Build the immutable NATS payload covered by the endpoint ref digest."""
    return {
        "instance_id": req.prefill_instance,
        "instance_epoch": req.prefill_instance_epoch,
        "decode_instance_epoch": req.decode_instance_epoch,
        "req_id": req.req_id,
        "owner_id": queue.owner_id,
        "command_id": req.command_id,
        # Retries resend this exact envelope/ref; attempt count is local-only.
        "dispatch_attempt": 1,
        "prefill_done_subject": queue.reply_subject("prefill_done"),
        "recompute_done_subject": queue.reply_subject("recompute_done"),
        "first_token_subject": queue.reply_subject("first_token"),
        "decode_done_subject": queue.reply_subject("decode_done"),
        "decode_progress_subject": queue.reply_subject("decode_progress"),
        "reply_subject": queue.reply_subject("prefill_done"),
        "token_ids": list(req.token_ids),
        "sampling_params": dict(req.sampling_params),
        "held_resource_kinds": ["SOURCE_BLOCKS"],
    }


def _uses_week12_network_control(governor) -> bool:
    return getattr(
        getattr(governor, "infer_client", None),
        "week12_network_control",
        False,
    ) is True


async def schedule_loop(
    scheduler: "PDScheduler",
    governor: "TransferGovernor",
    tracker: "RequestTracker",
    queue: "NATSQueue",
    metrics,
    config: dict,
    affinity_coordinator=None,
    output_buffer=None,
    operation_allocator=None,
) -> None:
    """Run scheduling ticks until cancelled."""
    from prism_serve.scheduler.sequence_state import SeqState, TransferTask
    from prism_serve.scheduler.transfer_governor import TransferDispatchError

    kv_transfer_timeout_s: float = config.get("kv_transfer_timeout_s", 30.0)
    prefill_timeout_s: float = config.get("prefill_timeout_s", 30.0)
    recompute_timeout_s: float = config.get("recompute_timeout_s", 30.0)
    decode_timeout_s: float = config.get("decode_timeout_s", 300.0)
    suffix_prefill_timeout_s: float = config.get("suffix_prefill_timeout_s", 30.0)
    max_dispatch_attempts: int = config.get("max_dispatch_attempts", 3)
    abort_timeout_s: float = config.get("abort_request_timeout_s", 5.0)
    tick_interval_s = config.get("schedule_loop_tick_ms", 10) / 1000.0
    output_query_interval_s = config.get(
        "operation_query_interval_ms", 50
    ) / 1000.0

    while True:
        tick_start = time.monotonic()

        # ── Phase 1: WAITING → PREFILLING ─────────────────────────────


        waiting = (
            list(tracker.iter_by_state(SeqState.WAITING))
            if scheduler.admission_ready()
            else ()
        )
        for req in waiting:
            if (
                config.get("affinity_enabled", False)
                and affinity_coordinator is not None
                and req.fingerprint is not None
                and req.correctness_path not in {"cold", "seed_d0", "seed_d1"}
                and await affinity_coordinator.try_start(req, scheduler, tracker)
            ):
                continue
            p = scheduler.pick_prefill_instance(
                req.req_id,
                required_instance=(req.correctness_source_instance or None),
            )
            if p is None:
                break
            d = scheduler.pick_decode_instance(
                req.req_id,
                req.kv_size_bytes,
                required_instance=(req.correctness_target_instance or None),
            )
            if d is None:
                # Release P load before trying another request.
                scheduler.on_prefill_done(p)
                continue
            p_epoch = scheduler.instance_epoch(p)
            d_epoch = scheduler.instance_epoch(d)
            target_request_ref = None
            target_blocks: tuple[int, ...] = ()
            network_control = _uses_week12_network_control(governor)
            slot_lease = None
            if network_control:
                slot_lease = scheduler.adopt_picked_decode_slot(
                    d, req.req_id, req.req_id
                )
                try:
                    target_request_ref, target_blocks = await governor.infer_client.prepare_normal_request(
                        d, d_epoch, req.req_id, req.token_ids, req.sampling_params,
                        {
                            "first_token_subject": queue.reply_subject("first_token"),
                            "decode_progress_subject": queue.reply_subject("decode_progress"),
                            "decode_done_subject": queue.reply_subject("decode_done"),
                        },
                        prefix_identity=(
                            {
                                "namespace": req.fingerprint.namespace,
                                "kv_compatibility_id": req.fingerprint.kv_compatibility_id,
                                "request_context_digest": req.fingerprint.request_context_digest,
                            }
                            if req.fingerprint is not None else None
                        ),
                    )
                except Exception as exc:
                    from prism_serve.router.http_rpc import AmbiguousRPCError
                    scheduler.on_prefill_done(p, p_epoch)
                    metrics.increment("infer_rpc_ambiguous_total", labels={
                        "reason": "target_prepare_failed",
                    })
                    logger.warning("target prepare failed req=%s: %s", req.req_id, exc)
                    if isinstance(exc, AmbiguousRPCError):
                        target_request_ref = exc.endpoint_ref
                        tracker.transition(
                            req.req_id, SeqState.PREFILLING,
                            prefill_instance=p, decode_instance=d,
                            prefill_instance_epoch=p_epoch,
                            decode_instance_epoch=d_epoch,
                            command_id=f"{queue.owner_id}:{req.req_id}",
                            publish_outcome="NOT_STARTED",
                            active_operation_id=req.req_id,
                            decode_slot_lease_id=slot_lease.lease_id,
                            target_request_ref=target_request_ref,
                        )
                        await _join_canonical_cleanup(
                            req, tracker, scheduler, governor, queue.owner_id,
                            abort_timeout_s, abort_timeout_s,
                        )
                    else:
                        scheduler.release_decode_slot(req.req_id)
                    continue
            command_id = f"{queue.owner_id}:{req.req_id}"
            tracker.transition(
                req.req_id, SeqState.PREFILLING,
                prefill_instance=p,
                decode_instance=d,
                prefill_instance_epoch=p_epoch,
                decode_instance_epoch=d_epoch,
                command_id=command_id,
                publish_outcome="NOT_STARTED",
                active_operation_id=req.req_id if network_control else "",
                decode_slot_lease_id=(slot_lease.lease_id if slot_lease else ""),
                target_request_ref=target_request_ref,
                target_block_table=list(target_blocks),
            )
            publish_task = None
            fault_directive = None
            try:
                payload = _normal_prefill_payload(req, queue)
                command = payload
                if operation_allocator is not None:
                    from dataclasses import asdict
                    endpoint_ref = operation_allocator.allocate(
                        target_instance=p,
                        target_worker_epoch=p_epoch,
                        operation_id=req.req_id,
                        payload=payload,
                    )
                    req.dispatch_operation_ref = endpoint_ref
                    command = {"schema_version": 1,
                               "endpoint_ref": asdict(endpoint_ref),
                               "payload": payload}
                    if req.correctness_path:
                        fault_directive = await governor.infer_client.correctness_fault_checkpoint(
                            "before_nats_dispatch",
                            {
                                "request_id": req.req_id,
                                "path": req.correctness_path,
                                "source_instance": p,
                                "target_instance": d,
                                "source_endpoint_ref": asdict(endpoint_ref),
                                "target_endpoint_ref": (
                                    asdict(target_request_ref)
                                    if target_request_ref is not None else None
                                ),
                            },
                        )
                publish_task = asyncio.create_task(
                    _publish_command_with_fault(
                        queue,
                        queue.dispatch_subject(p),
                        command,
                        fault_directive,
                        governor.infer_client.record_correctness_fault_event,
                        governor.infer_client.wait_nats_command_fault_authority,
                    )
                )
                await publish_task
                tracker.set_publish_outcome(req.req_id, "ACKED")
            except asyncio.CancelledError:
                tracker.set_publish_outcome(req.req_id, "UNKNOWN")
                cleanup = _get_or_create_canonical_cleanup(
                    req, tracker, scheduler, governor, queue.owner_id,
                    abort_timeout_s, abort_timeout_s,
                )
                await asyncio.shield(cleanup)
                raise
            except Exception as exc:
                tracker.set_publish_outcome(
                    req.req_id, "NOT_STARTED" if publish_task is None else "UNKNOWN"
                )
                metrics.increment("control_message_error_total", labels={
                    "operation": "dispatch_prefill",
                })
                logger.warning("dispatch publish failed req=%s: %s", req.req_id, exc)
                cleanup = _get_or_create_canonical_cleanup(
                    req, tracker, scheduler, governor, queue.owner_id,
                    abort_timeout_s, abort_timeout_s,
                )
                await asyncio.shield(cleanup)
                continue

        for msg in await queue.poll("suffix_prefill_done"):
            req_id = msg.get("req_id")
            req = tracker.get(req_id) if req_id is not None else None
            if (
                req is None
                or req.state != SeqState.PREFIX_PREFILLING
                or msg.get("operation_id") != req.active_operation_id
                or msg.get("instance_epoch") != req.decode_instance_epoch
                or not scheduler.epoch_matches(
                    req.decode_instance, req.decode_instance_epoch
                )
            ):
                continue
            tracker.transition(req_id, SeqState.DECODING)

        if affinity_coordinator is not None:
            for req in tracker.get_stuck_suffix_prefills(suffix_prefill_timeout_s):
                outcome = await affinity_coordinator.cleanup_suffix(
                    req, scheduler, tracker
                )
                if outcome is None:
                    # Exact-ref UNKNOWN is not an ACK.  The coordinator owns a
                    # reconciliation task; tracker and quarantined slot remain.
                    continue
                if outcome.action == "ABORTED" and tracker.get(req.req_id) is req:
                    tracker.transition(req.req_id, SeqState.ABORTED)
                    tracker.remove(req.req_id)

        for req in tracker.get_stuck_prefills(prefill_timeout_s):
            if req.dispatch_attempts >= max_dispatch_attempts:
                await _join_canonical_cleanup(
                    req, tracker, scheduler, governor, queue.owner_id,
                    abort_timeout_s, abort_timeout_s,
                )
                metrics.increment(
                    "prefill_dispatch_abort_total", labels={"reason": "timeout"}
                )
                continue
            publish_task = None
            try:
                payload = _normal_prefill_payload(req, queue)
                command = payload
                if req.dispatch_operation_ref is not None:
                    from dataclasses import asdict
                    command = {
                        "schema_version": 1,
                        "endpoint_ref": asdict(req.dispatch_operation_ref),
                        "payload": payload,
                    }
                publish_task = asyncio.create_task(
                    queue.publish(queue.dispatch_subject(req.prefill_instance), command)
                )
                await publish_task
                tracker.set_publish_outcome(req.req_id, "ACKED")
            except asyncio.CancelledError:
                tracker.set_publish_outcome(req.req_id, "UNKNOWN")
                cleanup = _get_or_create_canonical_cleanup(
                    req, tracker, scheduler, governor, queue.owner_id,
                    abort_timeout_s, abort_timeout_s,
                )
                await asyncio.shield(cleanup)
                raise
            except Exception as exc:
                tracker.set_publish_outcome(
                    req.req_id, "NOT_STARTED" if publish_task is None else "UNKNOWN"
                )
                metrics.increment("control_message_error_total", labels={
                    "operation": "retry_prefill",
                })
                logger.warning("retry publish failed req=%s: %s", req.req_id, exc)
                await _join_canonical_cleanup(
                    req, tracker, scheduler, governor, queue.owner_id,
                    abort_timeout_s, abort_timeout_s,
                )
                continue
            req.prefill_start = time.monotonic()
            req.dispatch_attempts += 1
            metrics.increment("prefill_dispatch_retry_total")

        # Submit completed prefills for KV transfer.
        for msg in await queue.poll("prefill_done"):
            req_id = msg.get("req_id")
            if req_id is None or req_id not in tracker:
                continue
            req = tracker.get(req_id)
            if (
                req is None
                or req.state != SeqState.PREFILLING
                or msg.get("instance_epoch") != req.prefill_instance_epoch
                or not scheduler.epoch_matches(
                    req.prefill_instance, req.prefill_instance_epoch
                )
            ):
                continue

            kv_size   = msg.get("kv_size_bytes", 0)
            blk_table = msg.get("block_table", [])
            first_token = msg.get("first_token")
            if _uses_week12_network_control(governor) and (
                isinstance(first_token, bool) or not isinstance(first_token, int)
            ):
                metrics.increment(
                    "control_message_error_total",
                    labels={"operation": "prefill_done_first_token"},
                )
                await _join_canonical_cleanup(
                    req, tracker, scheduler, governor, queue.owner_id,
                    abort_timeout_s, abort_timeout_s,
                )
                continue

            tracker.transition(
                req_id, SeqState.KV_PENDING,
                kv_size_bytes=kv_size,
                block_table=blk_table,
                transfer_operation_id=(
                    req.active_operation_id or uuid.uuid4().hex
                ),
            )
            scheduler.on_prefill_done(
                req.prefill_instance, req.prefill_instance_epoch
            )

            try:
                transfer_task = TransferTask(
                    req_id=req_id,
                    operation_id=req.transfer_operation_id,
                    src=req.prefill_instance,
                    dst=req.decode_instance,
                    kv_size=kv_size,
                    src_epoch=req.prefill_instance_epoch,
                    dst_epoch=req.decode_instance_epoch,
                    src_block_ids=tuple(blk_table),
                    dst_block_ids=tuple(req.target_block_table),
                    target_request_ref=req.target_request_ref,
                    token_ids=tuple(req.token_ids),
                    sampling_params=dict(req.sampling_params),
                    first_token=first_token,
                    correctness_path=req.correctness_path,
                )
                transfer_task.on_complete = (
                    lambda task=transfer_task: _on_kv_done(
                        task.req_id, task.operation_id, tracker, scheduler,
                        metrics, output_buffer, task.target_request_commit_ref,
                    )
                )
                governor.submit(transfer_task)
            except TransferDispatchError as exc:
                metrics.increment(
                    "kv_transfer_dispatch_error_total",
                    labels={"dst": req.decode_instance},
                )
                logger.warning("KV transfer dispatch failed req=%s: %s", req_id, exc)
                decision = governor.on_transfer_failure(
                    req_id, req.decode_instance, "dispatch_error"
                )
                if decision == "recompute":
                    if not scheduler.epoch_matches(
                        req.decode_instance, req.decode_instance_epoch
                    ):
                        await _join_canonical_cleanup(
                            req, tracker, scheduler, governor, queue.owner_id,
                            abort_timeout_s, abort_timeout_s,
                        )
                        continue
                    if not await _trigger_recompute_epoch_fenced(
                        req, scheduler, governor
                    ):
                        governor.finish_request(req.req_id)
                        await _join_canonical_cleanup(
                            req, tracker, scheduler, governor, queue.owner_id,
                            abort_timeout_s, abort_timeout_s,
                        )
                        continue
                    tracker.transition(req_id, SeqState.RECOMPUTING)
                else:
                    await _join_canonical_cleanup(
                        req, tracker, scheduler, governor, queue.owner_id,
                        abort_timeout_s, abort_timeout_s,
                    )

        # Handle timeouts before deferred work can be flushed in this tick.
        for req in tracker.get_stuck_requests(kv_transfer_timeout_s):
            operation_id = req.transfer_operation_id
            if classify_transfer_epochs(
                req, scheduler
            ) != TransferEpochResult.BOTH_CURRENT:
                await _join_canonical_cleanup(
                    req, tracker, scheduler, governor, queue.owner_id,
                    abort_timeout_s, abort_timeout_s,
                )
                continue
            if not _matches_pending_operation(
                req.req_id, operation_id, tracker
            ):
                continue
            transfer_state = governor.task_state(req.req_id, operation_id)
            if transfer_state == "inflight":
                stopped = await _abort_remote_transfer(
                    governor.infer_client,
                    req.prefill_instance,
                    req.decode_instance,
                    queue.owner_id,
                    req.req_id,
                    operation_id,
                    abort_timeout_s,
                )
                if not _owns_pending_transfer(
                    req.req_id, operation_id, tracker, governor
                ):
                    # Completion or cancellation won the race while abort awaited.
                    continue
                if classify_transfer_epochs(
                    req, scheduler
                ) != TransferEpochResult.BOTH_CURRENT:
                    await _join_canonical_cleanup(
                        req, tracker, scheduler, governor, queue.owner_id,
                        abort_timeout_s, abort_timeout_s,
                    )
                    continue
                if not stopped:
                    await _join_canonical_cleanup(
                        req, tracker, scheduler, governor, queue.owner_id,
                        abort_timeout_s, abort_timeout_s,
                    )
                    metrics.increment(
                        "kv_transfer_abort_total",
                        labels={"reason": "transfer_abort_unconfirmed"},
                    )
                    continue
            elif transfer_state == "none":
                # No local ownership and no completion: remote handoff is unknown.
                await _join_canonical_cleanup(
                    req, tracker, scheduler, governor, queue.owner_id,
                    abort_timeout_s, abort_timeout_s,
                )
                continue
            governor.cancel(req.req_id, operation_id)
            decision = governor.on_transfer_failure(
                req.req_id, req.decode_instance, "timeout"
            )
            if decision == "recompute":
                if not scheduler.epoch_matches(
                    req.decode_instance, req.decode_instance_epoch
                ):
                    await _join_canonical_cleanup(
                        req, tracker, scheduler, governor, queue.owner_id,
                        abort_timeout_s, abort_timeout_s,
                    )
                    continue
                if not await _trigger_recompute_epoch_fenced(
                    req, scheduler, governor
                ):
                    governor.finish_request(req.req_id)
                    await _join_canonical_cleanup(
                        req, tracker, scheduler, governor, queue.owner_id,
                        abort_timeout_s, abort_timeout_s,
                    )
                    continue
                tracker.transition(req.req_id, SeqState.RECOMPUTING)
                logger.info("recompute triggered req=%s dst=%s",
                            req.req_id, req.decode_instance)
            else:  # abort
                await _join_canonical_cleanup(
                    req, tracker, scheduler, governor, queue.owner_id,
                    abort_timeout_s, abort_timeout_s,
                )
                logger.warning("request aborted after recompute limit req=%s", req.req_id)

        # A successful local recompute enters decode on the original D instance.
        for msg in await queue.poll("recompute_done"):
            req_id = msg.get("req_id")
            req = tracker.get(req_id) if req_id is not None else None
            if (
                req is None
                or req.state != SeqState.RECOMPUTING
                or msg.get("instance_epoch") != req.decode_instance_epoch
                or not scheduler.epoch_matches(
                    req.decode_instance, req.decode_instance_epoch
                )
            ):
                continue
            tracker.transition(req_id, SeqState.DECODING)

        for req in tracker.get_stuck_recomputes(recompute_timeout_s):
            if not scheduler.epoch_matches(
                req.decode_instance, req.decode_instance_epoch
            ):
                await _join_canonical_cleanup(
                    req, tracker, scheduler, governor, queue.owner_id,
                    abort_timeout_s, abort_timeout_s,
                )
                continue
            decision = governor.on_transfer_failure(
                req.req_id, req.decode_instance, "recompute_timeout"
            )
            if decision == "recompute":
                if not scheduler.epoch_matches(
                    req.decode_instance, req.decode_instance_epoch
                ):
                    await _join_canonical_cleanup(
                        req, tracker, scheduler, governor, queue.owner_id,
                        abort_timeout_s, abort_timeout_s,
                    )
                    continue
                if not await _trigger_recompute_epoch_fenced(
                    req, scheduler, governor
                ):
                    governor.finish_request(req.req_id)
                    await _join_canonical_cleanup(
                        req, tracker, scheduler, governor, queue.owner_id,
                        abort_timeout_s, abort_timeout_s,
                    )
                    continue
                req.recompute_start = time.monotonic()
            else:
                await _join_canonical_cleanup(
                    req, tracker, scheduler, governor, queue.owner_id,
                    abort_timeout_s, abort_timeout_s,
                )

        # Flush deferred transfers after timeout handling.
        governor.tick()

        # Process decode progress and completion.
        for msg in await queue.poll("first_token"):
            req_id = msg.get("req_id")
            req = tracker.get(req_id) if req_id is not None else None
            if (
                req is not None
                and (
                    not _uses_week12_network_control(governor)
                    or msg.get("operation_id") == req.active_operation_id
                )
                and msg.get("instance_epoch") == req.decode_instance_epoch
                and scheduler.epoch_matches(
                    req.decode_instance, req.decode_instance_epoch
                )
            ):
                if output_buffer is not None and msg.get("token_ids") is not None:
                    try:
                        await output_buffer.apply_cumulative(
                            req_id, list(msg.get("token_ids", ())),
                            int(msg.get("output_seq_no", -1)),
                        )
                    except (TypeError, ValueError):
                        pass
                if req.state == SeqState.DECODING:
                    tracker.record_first_token(req_id)

        for msg in await queue.poll("decode_progress"):
            req_id = msg.get("req_id")
            req = tracker.get(req_id) if req_id is not None else None
            if (
                output_buffer is None
                or req is None
                or (
                    _uses_week12_network_control(governor)
                    and msg.get("operation_id") != req.active_operation_id
                )
                or msg.get("instance_epoch") != req.decode_instance_epoch
                or not scheduler.epoch_matches(
                    req.decode_instance, req.decode_instance_epoch
                )
            ):
                continue
            try:
                await output_buffer.apply_cumulative(
                    req_id,
                    list(msg.get("token_ids", ())),
                    int(msg.get("output_seq_no", -1)),
                )
            except (TypeError, ValueError):
                metrics.increment("control_message_error_total", labels={
                    "operation": "decode_progress",
                })
                if _uses_week12_network_control(governor):
                    from prism_serve.gateway.output import (
                        output_query_identity,
                        repair_output_gap,
                    )
                    identity = output_query_identity(req)
                    if identity is None:
                        continue
                    (
                        instance_id, instance_epoch,
                        query_req_id, operation_id,
                    ) = identity
                    try:
                        cursor = len(output_buffer.snapshot(req_id)[0])
                        await repair_output_gap(
                            governor.infer_client,
                            instance_id=instance_id,
                            instance_epoch=instance_epoch,
                            req_id=query_req_id,
                            operation_id=operation_id,
                            cursor=cursor,
                            output_buffer=output_buffer,
                            metrics=metrics,
                            still_current=(
                                lambda req=req, identity=identity: (
                                    tracker.get(req.req_id) is req
                                    and output_query_identity(req) == identity
                                )
                            ),
                        )
                    except Exception:
                        metrics.increment("control_message_error_total", labels={
                            "operation": "output_gap_query",
                        })

        for msg in await queue.poll("decode_done"):
            req_id = msg.get("req_id")
            if req_id is None or req_id not in tracker:
                continue
            req = tracker.get(req_id)
            if (
                req is None
                or req.state not in {SeqState.KV_PENDING, SeqState.DECODING}
                or (
                    _uses_week12_network_control(governor)
                    and msg.get("operation_id") != req.active_operation_id
                )
                or msg.get("instance_epoch") != req.decode_instance_epoch
                or not scheduler.epoch_matches(
                    req.decode_instance, req.decode_instance_epoch
                )
            ):
                continue
            if output_buffer is not None:
                token_ids = list(msg.get("token_ids", ()))
                try:
                    await output_buffer.apply_cumulative(
                        req_id, token_ids, int(msg.get("output_seq_no", len(token_ids))),
                        terminal=True,
                    )
                except (TypeError, ValueError):
                    await output_buffer.apply_cumulative(
                        req_id, [], 0, terminal=True, error="invalid decode_done cursor"
                    )
            if req.state == SeqState.DECODING:
                tracker.transition(req_id, SeqState.FINISHED)
            if not _uses_week12_network_control(governor):
                scheduler.on_decode_finished(
                    req.decode_instance, req.decode_instance_epoch,
                    req.active_operation_id or None,
                )
                governor.finish_request(req_id)
                tracker.remove(req_id)
                if output_buffer is not None:
                    output_buffer.mark_resource_free(req_id)

        if _uses_week12_network_control(governor):
            if output_buffer is not None:
                now = time.monotonic()
                for req in list(tracker.iter_by_state(SeqState.DECODING)):
                    tokens, terminal, error = output_buffer.snapshot(req.req_id)
                    if tokens:
                        tracker.record_first_token(req.req_id)
                    if terminal or error:
                        tracker.transition(req.req_id, SeqState.FINISHED)
                        continue
                    if now - req.last_output_query_at < output_query_interval_s:
                        continue
                    req.last_output_query_at = now
                    try:
                        await _reconcile_authoritative_output(
                            req, tracker, governor.infer_client, output_buffer
                        )
                    except Exception:
                        metrics.increment(
                            "control_message_error_total",
                            labels={"operation": "output_authoritative_query"},
                        )
            for req in list(tracker.iter_by_state(SeqState.FINISHED)):
                if await governor.infer_client.cleanup_request(
                    scheduler, req, abort=False
                ):
                    if affinity_coordinator is not None:
                        affinity_coordinator.terminal_cleanup_complete(req)
                    governor.finish_request(req.req_id)
                    tracker.remove(req.req_id)
                    if output_buffer is not None:
                        output_buffer.mark_resource_free(req.req_id)
            # UNKNOWN cleanup keeps the request ledger and quarantined slot.  A
            # later exact-ref query may close it without inventing a new op.
            for req in list(tracker.iter_by_state(SeqState.ABORTED)):
                if await governor.infer_client.cleanup_request(
                    scheduler, req, abort=True
                ):
                    if affinity_coordinator is not None:
                        affinity_coordinator.terminal_cleanup_complete(req)
                    governor.finish_request(req.req_id)
                    tracker.remove(req.req_id)
                    if output_buffer is not None:
                        await output_buffer.fail(req.req_id, "request_aborted")
                        output_buffer.mark_resource_free(req.req_id)

        for req in tracker.get_stuck_decodes(decode_timeout_s):
            await _join_canonical_cleanup(
                req, tracker, scheduler, governor, queue.owner_id,
                abort_timeout_s, abort_timeout_s,
            )
            metrics.increment(
                "decode_abort_total", labels={"reason": "timeout"}
            )

        # Publish the current scheduler snapshot.
        metrics.gauge("active_requests", len(tracker))
        metrics.gauge(
            "waiting_requests",
            sum(1 for _ in tracker.iter_by_state(SeqState.WAITING)),
        )
        metrics.gauge(
            "kv_pending_requests",
            sum(1 for _ in tracker.iter_by_state(SeqState.KV_PENDING)),
        )

        elapsed = time.monotonic() - tick_start
        await asyncio.sleep(max(0.0, tick_interval_s - elapsed))


async def _abort_and_release(
    req,
    scheduler: "PDScheduler",
    governor: "TransferGovernor",
    owner_id: str,
    timeout_s: float,
    *,
    include_prefill: bool,
) -> None:
    """Release local capacity only after remote abort is acknowledged."""
    instances = []
    if include_prefill and req.prefill_instance:
        instances.append(("prefill", req.prefill_instance))
    if req.decode_instance:
        instances.append(("decode", req.decode_instance))

    for role, instance_id in instances:
        assigned_epoch = (
            req.prefill_instance_epoch
            if role == "prefill" else req.decode_instance_epoch
        )
        if not scheduler.epoch_matches(instance_id, assigned_epoch):
            continue
        acknowledged = await _abort_remote_request(
            governor.infer_client, instance_id, owner_id, req.req_id, timeout_s
        )
        if not acknowledged:
            # Remote capacity remains unsafe until an explicit reconciliation.
            scheduler.quarantine_instance(instance_id)
            continue
        if role == "prefill":
            scheduler.on_prefill_done(instance_id, assigned_epoch)
        else:
            scheduler.on_decode_finished(instance_id, assigned_epoch)
    governor.finish_request(req.req_id)


class RemoteRequestState(Enum):
    NOT_OWNED = auto()
    NEVER_DISPATCHED = auto()
    ABORT_ACK = auto()
    UNCERTAIN = auto()
    ALREADY_COMPLETED = auto()


class TransferCleanupState(Enum):
    NONE = auto()
    DEFERRED_CANCELLED = auto()
    FENCED_ACK = auto()
    UNCERTAIN = auto()


class TransferEpochResult(Enum):
    BOTH_CURRENT = auto()
    SOURCE_CHANGED_TARGET_CURRENT = auto()
    TARGET_CHANGED = auto()


def classify_transfer_epochs(req, scheduler: "PDScheduler") -> TransferEpochResult:
    """Target precedence prevents a both-changed race from mutating new D state."""
    if not scheduler.epoch_matches(
        req.decode_instance, req.decode_instance_epoch
    ):
        return TransferEpochResult.TARGET_CHANGED
    if not scheduler.epoch_matches(
        req.prefill_instance, req.prefill_instance_epoch
    ):
        return TransferEpochResult.SOURCE_CHANGED_TARGET_CURRENT
    return TransferEpochResult.BOTH_CURRENT


def quarantine_source_changed_transfer(
    req, scheduler: "PDScheduler", governor: "TransferGovernor", owner_id: str
) -> None:
    scheduler.quarantine_instance(
        req.decode_instance,
        uncertain_transfer=(
            owner_id, req.req_id, req.transfer_operation_id,
            req.decode_instance_epoch,
        ),
    )
    governor.cancel(req.req_id, req.transfer_operation_id)


@dataclass(slots=True, frozen=True)
class CleanupProof:
    request_state: "SeqState"
    p_remote_request_state: RemoteRequestState
    d_remote_request_state: RemoteRequestState
    transfer_state: TransferCleanupState
    transfer_mode: str
    publish_outcome: str
    request_owns_p_load: bool
    request_owns_original_d_slot: bool
    d_epoch_changed: bool = False


def p_load_releasable(proof: CleanupProof) -> bool:
    return (
        not proof.request_owns_p_load
        or proof.p_remote_request_state in {
            RemoteRequestState.NOT_OWNED,
            RemoteRequestState.ALREADY_COMPLETED,
        }
        or proof.publish_outcome == "NOT_STARTED"
        and proof.p_remote_request_state == RemoteRequestState.NEVER_DISPATCHED
        or proof.publish_outcome == "ACKED"
        and proof.p_remote_request_state == RemoteRequestState.ABORT_ACK
    )


def validate_cleanup_proof(proof: CleanupProof) -> None:
    from prism_serve.scheduler.sequence_state import SeqState

    state = proof.request_state
    valid = False
    if state == SeqState.WAITING:
        valid = (
            not proof.request_owns_p_load
            and not proof.request_owns_original_d_slot
            and proof.p_remote_request_state == RemoteRequestState.NOT_OWNED
            and proof.d_remote_request_state == RemoteRequestState.NEVER_DISPATCHED
            and proof.transfer_state == TransferCleanupState.NONE
            and proof.transfer_mode == "none"
        )
    elif state == SeqState.PREFILLING:
        p_valid = (
            proof.publish_outcome == "NOT_STARTED"
            and proof.p_remote_request_state == RemoteRequestState.NEVER_DISPATCHED
            or proof.publish_outcome == "ACKED"
            and proof.p_remote_request_state in {
                RemoteRequestState.ABORT_ACK, RemoteRequestState.UNCERTAIN,
            }
            or proof.publish_outcome == "UNKNOWN"
            and proof.p_remote_request_state in {
                RemoteRequestState.NEVER_DISPATCHED,
                RemoteRequestState.ABORT_ACK,
                RemoteRequestState.UNCERTAIN,
            }
        )
        valid = (
            proof.request_owns_p_load and proof.request_owns_original_d_slot
            and p_valid
            and proof.d_remote_request_state == RemoteRequestState.NEVER_DISPATCHED
            and proof.transfer_state == TransferCleanupState.NONE
            and proof.transfer_mode == "none"
        )
    elif state == SeqState.KV_PENDING:
        valid = (
            not proof.request_owns_p_load and proof.request_owns_original_d_slot
            and proof.p_remote_request_state == RemoteRequestState.ALREADY_COMPLETED
            and proof.d_remote_request_state in {
                RemoteRequestState.ABORT_ACK, RemoteRequestState.UNCERTAIN,
            }
            and (
                proof.transfer_mode == "deferred"
                and proof.transfer_state == TransferCleanupState.DEFERRED_CANCELLED
                or proof.transfer_mode == "inflight"
                and proof.transfer_state in {
                    TransferCleanupState.FENCED_ACK,
                    TransferCleanupState.UNCERTAIN,
                }
            )
        )
    elif state == SeqState.DECODING:
        valid = (
            not proof.request_owns_p_load and proof.request_owns_original_d_slot
            and proof.p_remote_request_state == RemoteRequestState.ALREADY_COMPLETED
            and proof.d_remote_request_state in {
                RemoteRequestState.ABORT_ACK, RemoteRequestState.UNCERTAIN,
            }
            and proof.transfer_state == TransferCleanupState.NONE
            and proof.transfer_mode == "none"
        )
    elif state == SeqState.RECOMPUTING:
        valid = (
            not proof.request_owns_p_load and proof.request_owns_original_d_slot
            and proof.p_remote_request_state == RemoteRequestState.ALREADY_COMPLETED
            and proof.d_remote_request_state in {
                RemoteRequestState.ABORT_ACK, RemoteRequestState.UNCERTAIN,
            }
            and proof.transfer_state == TransferCleanupState.FENCED_ACK
            and proof.transfer_mode == "none"
        )
    if not valid:
        raise ValueError("invalid cleanup proof ownership combination")


def d_slot_releasable(proof: CleanupProof) -> bool:
    from prism_serve.scheduler.sequence_state import SeqState

    if not proof.request_owns_original_d_slot or proof.d_epoch_changed:
        return False
    state = proof.request_state
    return (
        state == SeqState.PREFILLING
        and proof.d_remote_request_state == RemoteRequestState.NEVER_DISPATCHED
        and proof.transfer_state == TransferCleanupState.NONE
        or state == SeqState.KV_PENDING
        and proof.d_remote_request_state == RemoteRequestState.ABORT_ACK
        and proof.transfer_state
        in {TransferCleanupState.DEFERRED_CANCELLED, TransferCleanupState.FENCED_ACK}
        or state == SeqState.DECODING
        and proof.d_remote_request_state == RemoteRequestState.ABORT_ACK
        and proof.transfer_state == TransferCleanupState.NONE
        or state == SeqState.RECOMPUTING
        and proof.d_remote_request_state == RemoteRequestState.ABORT_ACK
        and proof.transfer_state == TransferCleanupState.FENCED_ACK
    )


def _get_or_create_canonical_cleanup(
    req,
    tracker: "RequestTracker",
    scheduler: "PDScheduler",
    governor: "TransferGovernor",
    owner_id: str,
    request_timeout_s: float,
    transfer_timeout_s: float,
) -> asyncio.Task:
    return tracker.get_or_create_cleanup_task(
        req.req_id,
        lambda: _canonical_cleanup(
            req, tracker, scheduler, governor, owner_id,
            request_timeout_s, transfer_timeout_s,
        ),
    )


async def _canonical_cleanup(
    req,
    tracker: "RequestTracker",
    scheduler: "PDScheduler",
    governor: "TransferGovernor",
    owner_id: str,
    request_timeout_s: float,
    transfer_timeout_s: float,
) -> None:
    """Own all terminal resource mutations for one request."""
    from prism_serve.scheduler.sequence_state import SeqState

    if tracker.get(req.req_id) is not req:
        return
    if req.state == SeqState.PREFILLING and req.publish_outcome in {
        "UNKNOWN", "NOT_STARTED",
    }:
        tracker.metrics.increment(
            "operation_cancelled_before_arrival_total",
            labels={
                "endpoint": "dispatch.prefill",
                "publish_outcome": req.publish_outcome,
            },
        )
    if _uses_week12_network_control(governor):
        if req.prefill_instance:
            scheduler.on_prefill_done(
                req.prefill_instance, req.prefill_instance_epoch
            )
        released = await governor.infer_client.cleanup_request(
            scheduler, req, abort=True
        )
        if released:
            # UNKNOWN remote state still owns the transfer ledger and both
            # pair/destination byte credits. Releasing those credits before
            # exact-ref terminal/finalize proof could admit a second writer.
            governor.finish_request(req.req_id)
        if tracker.get(req.req_id) is req:
            if req.state not in {SeqState.FINISHED, SeqState.ABORTED}:
                tracker.transition(req.req_id, SeqState.ABORTED)
            if released:
                tracker.remove(req.req_id)
        current = asyncio.current_task()
        if current is not None:
            tracker.clear_cleanup_task(req.req_id, current)
        return
    if req.state == SeqState.PREFILLING and req.publish_outcome == "UNKNOWN":
        p_epoch_matches = scheduler.epoch_matches(
            req.prefill_instance, req.prefill_instance_epoch
        )
        if p_epoch_matches:
            scheduler.quarantine_instance(
                req.prefill_instance,
                uncertain_dispatch=(
                    owner_id, req.req_id, req.command_id,
                    req.prefill_instance_epoch,
                ),
            )
        scheduler.on_decode_finished(
            req.decode_instance, req.decode_instance_epoch
        )
        # Quarantine is already authoritative; abort is bounded best effort only.
        if p_epoch_matches:
            await _abort_remote_request(
                governor.infer_client, req.prefill_instance, owner_id,
                req.req_id, request_timeout_s,
            )
        governor.finish_request(req.req_id)
    elif req.state == SeqState.PREFILLING and req.publish_outcome == "NOT_STARTED":
        scheduler.on_prefill_done(
            req.prefill_instance, req.prefill_instance_epoch
        )
        scheduler.on_decode_finished(
            req.decode_instance, req.decode_instance_epoch
        )
        governor.finish_request(req.req_id)
    else:
        await _fence_and_abort_request(
            req, scheduler, governor, owner_id,
            request_timeout_s, transfer_timeout_s,
        )
    if tracker.get(req.req_id) is req:
        if req.state not in {SeqState.FINISHED, SeqState.ABORTED}:
            tracker.transition(req.req_id, SeqState.ABORTED)
        tracker.remove(req.req_id)
    current = asyncio.current_task()
    if current is not None:
        tracker.clear_cleanup_task(req.req_id, current)


async def _join_canonical_cleanup(
    req,
    tracker: "RequestTracker",
    scheduler: "PDScheduler",
    governor: "TransferGovernor",
    owner_id: str,
    request_timeout_s: float,
    transfer_timeout_s: float,
) -> None:
    task = _get_or_create_canonical_cleanup(
        req, tracker, scheduler, governor, owner_id,
        request_timeout_s, transfer_timeout_s,
    )
    await asyncio.shield(task)


async def _fence_and_abort_request(
    req,
    scheduler: "PDScheduler",
    governor: "TransferGovernor",
    owner_id: str,
    request_timeout_s: float,
    transfer_timeout_s: float,
) -> CleanupProof:
    """Build independent P, D, and transfer cleanup evidence."""
    from prism_serve.scheduler.sequence_state import SeqState

    owns_p = req.state == SeqState.PREFILLING and bool(req.prefill_instance)
    owns_d = bool(req.decode_instance) and req.state != SeqState.WAITING
    p_epoch_changed = owns_p and not scheduler.epoch_matches(
        req.prefill_instance, req.prefill_instance_epoch
    )
    target_epoch_changed = owns_d and not scheduler.epoch_matches(
        req.decode_instance, req.decode_instance_epoch
    )
    source_changed_target_current = False
    p_state = (
        RemoteRequestState.NEVER_DISPATCHED
        if owns_p
        else RemoteRequestState.NOT_OWNED
        if req.state == SeqState.WAITING
        else RemoteRequestState.ALREADY_COMPLETED
    )
    d_state = RemoteRequestState.NEVER_DISPATCHED
    transfer_state = TransferCleanupState.NONE

    task_state = governor.task_state(req.req_id, req.transfer_operation_id or None)
    if target_epoch_changed and req.state == SeqState.KV_PENDING:
        # Reconciliation into the current epoch already proved old operations absent.
        transfer_state = TransferCleanupState.FENCED_ACK
    elif req.state == SeqState.KV_PENDING and task_state == "deferred":
        governor.cancel(req.req_id, req.transfer_operation_id)
        transfer_state = TransferCleanupState.DEFERRED_CANCELLED
    elif req.state == SeqState.KV_PENDING and task_state == "inflight":
        epoch_result = classify_transfer_epochs(req, scheduler)
        if epoch_result == TransferEpochResult.BOTH_CURRENT:
            fenced = await _abort_remote_transfer(
                governor.infer_client,
                req.prefill_instance,
                req.decode_instance,
                owner_id,
                req.req_id,
                req.transfer_operation_id,
                transfer_timeout_s,
            )
            epoch_result = classify_transfer_epochs(req, scheduler)
            if epoch_result == TransferEpochResult.BOTH_CURRENT:
                transfer_state = (
                    TransferCleanupState.FENCED_ACK
                    if fenced else TransferCleanupState.UNCERTAIN
                )
            elif epoch_result == TransferEpochResult.TARGET_CHANGED:
                target_epoch_changed = True
                governor.cancel(req.req_id, req.transfer_operation_id)
            else:
                source_changed_target_current = True
                transfer_state = TransferCleanupState.UNCERTAIN
                quarantine_source_changed_transfer(
                    req, scheduler, governor, owner_id
                )
        elif epoch_result == TransferEpochResult.TARGET_CHANGED:
            target_epoch_changed = True
            governor.cancel(req.req_id, req.transfer_operation_id)
        else:
            source_changed_target_current = True
            transfer_state = TransferCleanupState.UNCERTAIN
            quarantine_source_changed_transfer(req, scheduler, governor, owner_id)
    elif req.state == SeqState.RECOMPUTING:
        transfer_state = TransferCleanupState.FENCED_ACK
    elif req.state == SeqState.KV_PENDING and task_state == "none":
        task_state = "inflight"
        transfer_state = TransferCleanupState.UNCERTAIN

    if owns_p and not p_epoch_changed:
        acknowledged = await _abort_remote_request(
            governor.infer_client, req.prefill_instance, owner_id,
            req.req_id, request_timeout_s,
        )
        if scheduler.epoch_matches(
            req.prefill_instance, req.prefill_instance_epoch
        ):
            p_state = (
                RemoteRequestState.ABORT_ACK
                if acknowledged else RemoteRequestState.UNCERTAIN
            )
        else:
            p_epoch_changed = True
    if (
        not target_epoch_changed
        and not source_changed_target_current
        and req.state in {SeqState.KV_PENDING, SeqState.RECOMPUTING, SeqState.DECODING}
    ):
        acknowledged = await _abort_remote_request(
            governor.infer_client, req.decode_instance, owner_id,
            req.req_id, request_timeout_s,
        )
        if scheduler.epoch_matches(
            req.decode_instance, req.decode_instance_epoch
        ):
            d_state = (
                RemoteRequestState.ABORT_ACK
                if acknowledged else RemoteRequestState.UNCERTAIN
            )
        else:
            target_epoch_changed = True
            governor.cancel(req.req_id, req.transfer_operation_id or None)

    proof = CleanupProof(
        request_state=req.state,
        p_remote_request_state=p_state,
        d_remote_request_state=d_state,
        transfer_state=transfer_state,
        transfer_mode=(
            task_state if req.state == SeqState.KV_PENDING else "none"
        ),
        publish_outcome=req.publish_outcome,
        request_owns_p_load=owns_p,
        request_owns_original_d_slot=owns_d,
        d_epoch_changed=target_epoch_changed,
    )
    if (
        not p_epoch_changed
        and not target_epoch_changed
        and not source_changed_target_current
    ):
        validate_cleanup_proof(proof)
    if owns_p and not p_epoch_changed:
        if p_load_releasable(proof):
            scheduler.on_prefill_done(
                req.prefill_instance, req.prefill_instance_epoch
            )
        else:
            scheduler.quarantine_instance(req.prefill_instance)
    if owns_d and not target_epoch_changed and not source_changed_target_current:
        if d_slot_releasable(proof):
            scheduler.on_decode_finished(
                req.decode_instance, req.decode_instance_epoch
            )
        else:
            uncertain = None
            if transfer_state == TransferCleanupState.UNCERTAIN:
                uncertain = (
                    owner_id,
                    req.req_id,
                    req.transfer_operation_id,
                    scheduler.instance_epoch(req.decode_instance),
                )
            scheduler.quarantine_instance(
                req.decode_instance, uncertain_transfer=uncertain
            )

    # UNKNOWN remote transfer ownership moves to quarantine before ledger deletion.
    governor.finish_request(req.req_id)
    return proof


async def _abort_remote_request(
    infer_client,
    instance_id: str,
    owner_id: str,
    req_id: str,
    timeout_s: float,
) -> bool:
    """Call the idempotent abort RPC and normalize sync and async clients."""
    try:
        method = infer_client.abort_request

        async def invoke():
            kwargs = {
                "instance_id": instance_id,
                "owner_id": owner_id,
                "req_id": req_id,
            }
            if inspect.iscoroutinefunction(method):
                return await method(**kwargs)
            result = await asyncio.to_thread(method, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        result = await asyncio.wait_for(invoke(), timeout=timeout_s)
        if isinstance(result, dict):
            return bool(result.get("success"))
        return bool(result)
    except Exception as exc:
        logger.warning(
            "remote abort failed req=%s instance=%s: %s",
            req_id, instance_id, exc,
        )
        return False


async def _abort_remote_transfer(
    infer_client,
    src_instance: str,
    dst_instance: str,
    owner_id: str,
    req_id: str,
    operation_id: str,
    timeout_s: float,
) -> bool:
    """Confirm that a handed-off transfer can no longer write target KV."""
    method = getattr(infer_client, "abort_transfer", None)
    if method is None:
        return False
    try:
        kwargs = {
            "src_instance": src_instance,
            "dst_instance": dst_instance,
            "owner_id": owner_id,
            "req_id": req_id,
            "operation_id": operation_id,
        }

        async def invoke():
            if inspect.iscoroutinefunction(method):
                return await method(**kwargs)
            result = await asyncio.to_thread(method, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        result = await asyncio.wait_for(invoke(), timeout=timeout_s)
        if isinstance(result, dict):
            return bool(result.get("success"))
        return bool(result)
    except Exception as exc:
        logger.warning(
            "remote transfer abort failed req=%s operation=%s: %s",
            req_id, operation_id, exc,
        )
        return False


def _on_kv_done(
    req_id: str,
    operation_id: str,
    tracker: "RequestTracker",
    scheduler: "PDScheduler",
    metrics,
    output_buffer=None,
    target_request_commit_ref=None,
) -> None:
    """Called by TransferGovernor._on_complete when KV reaches the D instance."""
    from prism_serve.scheduler.sequence_state import SeqState
    req = tracker.get(req_id)
    if req is None:
        return
    if (
        req.state != SeqState.KV_PENDING
        or req.transfer_operation_id != operation_id
        or not scheduler.epoch_matches(
            req.decode_instance, req.decode_instance_epoch
        )
    ):
        # Recompute or abort made this callback stale.
        return
    if req.active_operation_id:
        scheduler.commit_decode_slot(req.active_operation_id)
    tracker.transition(
        req_id, SeqState.DECODING,
        target_request_commit_ref=target_request_commit_ref,
    )
    if output_buffer is not None:
        tokens, terminal, error = output_buffer.snapshot(req_id)
        if tokens:
            tracker.record_first_token(req_id)
        if terminal or error:
            tracker.transition(req_id, SeqState.FINISHED)


async def _reconcile_authoritative_output(
    req, tracker: "RequestTracker", infer_client, output_buffer
) -> None:
    """Advance tracker only from the exact request operation and worker epoch."""
    from prism_serve.gateway.output import output_query_identity
    from prism_serve.scheduler.sequence_state import SeqState
    identity = output_query_identity(req)
    if identity is None:
        raise ValueError("output query is not ready")
    instance_id, instance_epoch, req_id, operation_id = identity
    tokens, _terminal, _error = output_buffer.snapshot(req_id)
    value = await infer_client.request_output(
        instance_id, req_id, len(tokens)
    )
    still_current = lambda: (
        tracker.get(req_id) is req
        and output_query_identity(req) == identity
    )
    if (
        not still_current()
        or value.get("req_id") != req_id
        or value.get("instance_epoch") != instance_epoch
        or value.get("operation_id") != operation_id
    ):
        raise ValueError("output query identity changed")
    await output_buffer.apply_cumulative(
        req_id, list(value.get("token_ids", ())),
        int(value.get("output_seq_no", -1)),
        terminal=bool(value.get("terminal", False)),
        error=value.get("error"),
        still_current=still_current,
    )
    current_tokens, current_terminal, current_error = output_buffer.snapshot(
        req_id
    )
    if current_tokens:
        tracker.record_first_token(req_id)
    if current_terminal or current_error:
        tracker.transition(req_id, SeqState.FINISHED)


async def _trigger_recompute_epoch_fenced(
    req,
    scheduler: "PDScheduler",
    governor: "TransferGovernor",
) -> bool:
    """Commit recompute only if the decode epoch is stable across the action."""
    if not scheduler.epoch_matches(
        req.decode_instance, req.decode_instance_epoch
    ):
        return False
    result = governor.trigger_recompute(req.req_id, req.decode_instance)
    if inspect.isawaitable(result):
        result = await result
    if result is False:
        return False
    return scheduler.epoch_matches(
        req.decode_instance, req.decode_instance_epoch
    )


def _owns_pending_transfer(
    req_id: str,
    operation_id: str,
    tracker: "RequestTracker",
    governor: "TransferGovernor",
) -> bool:
    """Check the operation-scoped precondition shared by completion and timeout."""
    return (
        _matches_pending_operation(req_id, operation_id, tracker)
        and governor.owns(req_id, operation_id)
    )


def _matches_pending_operation(
    req_id: str,
    operation_id: str,
    tracker: "RequestTracker",
) -> bool:
    """Return whether this operation may still win completion or timeout."""
    from prism_serve.scheduler.sequence_state import SeqState

    req = tracker.get(req_id)
    return bool(
        req is not None
        and req.state == SeqState.KV_PENDING
        and req.transfer_operation_id == operation_id
    )

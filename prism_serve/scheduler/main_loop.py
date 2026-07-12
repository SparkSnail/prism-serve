"""Tick-based control loop for disaggregated prefill and decode."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism_serve.scheduler.scheduler import PDScheduler
    from prism_serve.scheduler.transfer_governor import TransferGovernor
    from prism_serve.scheduler.sequence_state import RequestTracker, SeqState
    from prism_serve.scheduler.queue import NATSQueue

logger = logging.getLogger(__name__)

TICK_INTERVAL_S: float = 0.010


async def schedule_loop(
    scheduler: "PDScheduler",
    governor: "TransferGovernor",
    tracker: "RequestTracker",
    queue: "NATSQueue",
    metrics,
    config: dict,
) -> None:
    """Run scheduling ticks until cancelled."""
    from prism_serve.scheduler.sequence_state import SeqState, TransferTask
    from prism_serve.scheduler.transfer_governor import TransferDispatchError

    kv_transfer_timeout_s: float = config.get("kv_transfer_timeout_s", 30.0)
    prefill_timeout_s: float = config.get("prefill_timeout_s", 30.0)
    recompute_timeout_s: float = config.get("recompute_timeout_s", 30.0)
    decode_timeout_s: float = config.get("decode_timeout_s", 300.0)
    max_dispatch_attempts: int = config.get("max_dispatch_attempts", 3)
    abort_timeout_s: float = config.get("abort_request_timeout_s", 5.0)
    tick_interval_s = config.get("schedule_loop_tick_ms", 10) / 1000.0

    while True:
        tick_start = time.monotonic()

        # Dispatch waiting requests to prefill and decode instances.
        # Stop as soon as there are no available P instances (all remaining
        # WAITING requests will be retried next tick).
        for req in list(tracker.iter_by_state(SeqState.WAITING)):
            p = scheduler.pick_prefill_instance(req.req_id)
            if p is None:
                # No P instance available; leave all WAITING for next tick.
                break
            d = scheduler.pick_decode_instance(req.req_id, req.kv_size_bytes)
            if d is None:
                # No D slot / all D instances congested; return P load counter.
                scheduler.on_prefill_done(p)
                continue  # try other WAITING requests that might fit elsewhere
            try:
                await queue.publish(queue.dispatch_subject(p), {
                    "instance_id": p,
                    "req_id": req.req_id,
                    "owner_id": queue.owner_id,
                    "command_id": f"{queue.owner_id}:{req.req_id}",
                    "dispatch_attempt": 1,
                    "prefill_done_subject": queue.reply_subject("prefill_done"),
                    "recompute_done_subject": queue.reply_subject("recompute_done"),
                    "first_token_subject": queue.reply_subject("first_token"),
                    "decode_done_subject": queue.reply_subject("decode_done"),
                })
            except Exception as exc:
                scheduler.on_prefill_done(p)
                scheduler.on_decode_finished(d)
                metrics.increment(
                    "control_message_error_total",
                    labels={"operation": "dispatch_prefill"},
                )
                logger.warning("dispatch publish failed req=%s: %s", req.req_id, exc)
                continue
            tracker.transition(
                req.req_id, SeqState.PREFILLING,
                prefill_instance=p,
                decode_instance=d,
            )

        for req in tracker.get_stuck_prefills(prefill_timeout_s):
            if req.dispatch_attempts >= max_dispatch_attempts:
                await _abort_and_release(
                    req, scheduler, governor, queue.owner_id,
                    abort_timeout_s, include_prefill=True,
                )
                tracker.transition(req.req_id, SeqState.ABORTED)
                tracker.remove(req.req_id)
                metrics.increment(
                    "prefill_dispatch_abort_total", labels={"reason": "timeout"}
                )
                continue
            try:
                await queue.publish(queue.dispatch_subject(req.prefill_instance), {
                    "instance_id": req.prefill_instance,
                    "req_id": req.req_id,
                    "owner_id": queue.owner_id,
                    "command_id": f"{queue.owner_id}:{req.req_id}",
                    "dispatch_attempt": req.dispatch_attempts + 1,
                    "prefill_done_subject": queue.reply_subject("prefill_done"),
                    "recompute_done_subject": queue.reply_subject("recompute_done"),
                    "first_token_subject": queue.reply_subject("first_token"),
                    "decode_done_subject": queue.reply_subject("decode_done"),
                })
            except Exception as exc:
                metrics.increment(
                    "control_message_error_total",
                    labels={"operation": "retry_prefill"},
                )
                logger.warning("retry publish failed req=%s: %s", req.req_id, exc)
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
            if req is None or req.state != SeqState.PREFILLING:
                continue

            kv_size   = msg.get("kv_size_bytes", 0)
            blk_table = msg.get("block_table", [])

            tracker.transition(
                req_id, SeqState.KV_PENDING,
                kv_size_bytes=kv_size,
                block_table=blk_table,
                transfer_operation_id=uuid.uuid4().hex,
            )
            scheduler.on_prefill_done(req.prefill_instance)

            try:
                governor.submit(TransferTask(
                    req_id=req_id,
                    operation_id=req.transfer_operation_id,
                    src=req.prefill_instance,
                    dst=req.decode_instance,
                    kv_size=kv_size,
                    on_complete=lambda rid=req_id, op=req.transfer_operation_id: _on_kv_done(
                        rid, op, tracker, scheduler, metrics
                    ),
                ))
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
                    governor.trigger_recompute(req_id, req.decode_instance)
                    tracker.transition(req_id, SeqState.RECOMPUTING)
                else:
                    await _abort_and_release(
                        req, scheduler, governor, queue.owner_id,
                        abort_timeout_s, include_prefill=False,
                    )
                    tracker.transition(req_id, SeqState.ABORTED)
                    tracker.remove(req_id)

        # Recover or abort timed-out work.
        # Process before the deferred queue flush so timed-out tasks are not
        # re-flushed from the deferred queue in the same tick.
        for req in tracker.get_stuck_requests(kv_transfer_timeout_s):
            operation_id = req.transfer_operation_id
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
                    # Completion or cancellation won while abort was pending.
                    continue
                if not stopped:
                    governor.cancel(req.req_id, operation_id)
                    scheduler.quarantine_instance(req.decode_instance)
                    governor.finish_request(req.req_id)
                    tracker.transition(req.req_id, SeqState.ABORTED)
                    tracker.remove(req.req_id)
                    metrics.increment(
                        "kv_transfer_abort_total",
                        labels={"reason": "transfer_abort_unconfirmed"},
                    )
                    continue
            elif transfer_state == "none":
                scheduler.quarantine_instance(req.decode_instance)
                governor.finish_request(req.req_id)
                tracker.transition(req.req_id, SeqState.ABORTED)
                tracker.remove(req.req_id)
                continue
            governor.cancel(req.req_id, operation_id)
            decision = governor.on_transfer_failure(
                req.req_id, req.decode_instance, "timeout"
            )
            if decision == "recompute":
                governor.trigger_recompute(req.req_id, req.decode_instance)
                tracker.transition(req.req_id, SeqState.RECOMPUTING)
                logger.info(
                    "recompute triggered for req=%s dst=%s",
                    req.req_id, req.decode_instance,
                )
            else:  # abort
                await _abort_and_release(
                    req, scheduler, governor, queue.owner_id,
                    abort_timeout_s, include_prefill=False,
                )
                tracker.transition(req.req_id, SeqState.ABORTED)
                tracker.remove(req.req_id)
                logger.warning(
                    "request aborted (max_recompute exhausted) req=%s",
                    req.req_id,
                )

        for msg in await queue.poll("recompute_done"):
            req_id = msg.get("req_id")
            req = tracker.get(req_id) if req_id is not None else None
            if req is None or req.state != SeqState.RECOMPUTING:
                continue
            tracker.transition(req_id, SeqState.DECODING)

        for req in tracker.get_stuck_recomputes(recompute_timeout_s):
            decision = governor.on_transfer_failure(
                req.req_id, req.decode_instance, "recompute_timeout"
            )
            if decision == "recompute":
                governor.trigger_recompute(req.req_id, req.decode_instance)
                req.recompute_start = time.monotonic()
            else:
                await _abort_and_release(
                    req, scheduler, governor, queue.owner_id,
                    abort_timeout_s, include_prefill=False,
                )
                tracker.transition(req.req_id, SeqState.ABORTED)
                tracker.remove(req.req_id)

        # Flush deferred transfers after timeout handling.
        governor.tick()

        # Process decode progress and completion.
        for msg in await queue.poll("first_token"):
            req_id = msg.get("req_id")
            if req_id is not None:
                tracker.record_first_token(req_id)

        for msg in await queue.poll("decode_done"):
            req_id = msg.get("req_id")
            if req_id is None or req_id not in tracker:
                continue
            req = tracker.get(req_id)
            if req is None or req.state != SeqState.DECODING:
                continue
            tracker.transition(req_id, SeqState.FINISHED)
            scheduler.on_decode_finished(req.decode_instance)
            governor.finish_request(req_id)
            tracker.remove(req_id)

        for req in tracker.get_stuck_decodes(decode_timeout_s):
            await _abort_and_release(
                req, scheduler, governor, queue.owner_id,
                abort_timeout_s, include_prefill=False,
            )
            tracker.transition(req.req_id, SeqState.ABORTED)
            tracker.remove(req.req_id)
            metrics.increment("decode_abort_total", labels={"reason": "timeout"})

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
        acknowledged = await _abort_remote_request(
            governor.infer_client, instance_id, owner_id, req.req_id, timeout_s
        )
        if not acknowledged:
            # Remote capacity remains unsafe until an explicit reconciliation.
            scheduler.quarantine_instance(instance_id)
            continue
        if role == "prefill":
            scheduler.on_prefill_done(instance_id)
        else:
            scheduler.on_decode_finished(instance_id)
    governor.finish_request(req.req_id)


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
) -> None:
    """Called by TransferGovernor._on_complete when KV reaches the D instance."""
    from prism_serve.scheduler.sequence_state import SeqState
    req = tracker.get(req_id)
    if req is None:
        return
    if (
        req.state != SeqState.KV_PENDING
        or req.transfer_operation_id != operation_id
    ):
        # Request was already recomputed or aborted; ignore stale callback.
        return
    tracker.transition(req_id, SeqState.DECODING)
    metrics.increment("kv_transfer_success_total")


def _matches_pending_operation(
    req_id: str,
    operation_id: str,
    tracker: "RequestTracker",
) -> bool:
    """Return whether the tracker still binds the pending operation."""
    from prism_serve.scheduler.sequence_state import SeqState

    req = tracker.get(req_id)
    return bool(
        req is not None
        and req.state == SeqState.KV_PENDING
        and req.transfer_operation_id == operation_id
    )


def _owns_pending_transfer(
    req_id: str,
    operation_id: str,
    tracker: "RequestTracker",
    governor: "TransferGovernor",
) -> bool:
    """Return whether tracker and governor still own the same operation."""
    return (
        _matches_pending_operation(req_id, operation_id, tracker)
        and governor.owns(req_id, operation_id)
    )

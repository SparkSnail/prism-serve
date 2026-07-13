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
    affinity_coordinator=None,
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

    while True:
        tick_start = time.monotonic()

        # Dispatch waiting requests to prefill and decode instances.
        # Stop as soon as there are no available P instances (all remaining
        # WAITING requests will be retried next tick).
        for req in list(tracker.iter_by_state(SeqState.WAITING)):
            if (
                config.get("affinity_enabled", False)
                and affinity_coordinator is not None
                and req.fingerprint is not None
                and await affinity_coordinator.try_start(req, scheduler, tracker)
            ):
                continue
            p = scheduler.pick_prefill_instance(req.req_id)
            if p is None:
                # No P instance available; leave all WAITING for next tick.
                break
            d = scheduler.pick_decode_instance(req.req_id, req.kv_size_bytes)
            if d is None:
                # Release P load before trying another request.
                scheduler.on_prefill_done(p)
                continue
            p_epoch = scheduler.instance_epoch(p)
            d_epoch = scheduler.instance_epoch(d)
            command_id = f"{queue.owner_id}:{req.req_id}"
            tracker.transition(
                req.req_id, SeqState.PREFILLING,
                prefill_instance=p,
                decode_instance=d,
                prefill_instance_epoch=p_epoch,
                decode_instance_epoch=d_epoch,
                command_id=command_id,
                publish_outcome="NOT_STARTED",
            )
            publish_task = None
            try:
                publish_task = asyncio.create_task(queue.publish(queue.dispatch_subject(p), {
                    "instance_id": p,
                    "instance_epoch": p_epoch,
                    "decode_instance_epoch": d_epoch,
                    "req_id": req.req_id,
                    "owner_id": queue.owner_id,
                    "command_id": command_id,
                    "dispatch_attempt": 1,
                    "prefill_done_subject": queue.reply_subject("prefill_done"),
                    "recompute_done_subject": queue.reply_subject("recompute_done"),
                    "first_token_subject": queue.reply_subject("first_token"),
                    "decode_done_subject": queue.reply_subject("decode_done"),
                }))
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
                stopped = await affinity_coordinator.abort_suffix(req)
                if stopped:
                    released = scheduler.on_decode_finished(
                        req.decode_instance, req.decode_instance_epoch,
                        req.active_operation_id,
                    )
                    if not released:
                        scheduler.release_quarantined_decode_slot(
                            req.active_operation_id
                        )
                else:
                    scheduler.quarantine_decode_slot(req.active_operation_id)
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
                publish_task = asyncio.create_task(queue.publish(queue.dispatch_subject(req.prefill_instance), {
                    "instance_id": req.prefill_instance,
                    "instance_epoch": req.prefill_instance_epoch,
                    "decode_instance_epoch": req.decode_instance_epoch,
                    "req_id": req.req_id,
                    "owner_id": queue.owner_id,
                    "command_id": f"{queue.owner_id}:{req.req_id}",
                    "dispatch_attempt": req.dispatch_attempts + 1,
                    "prefill_done_subject": queue.reply_subject("prefill_done"),
                    "recompute_done_subject": queue.reply_subject("recompute_done"),
                    "first_token_subject": queue.reply_subject("first_token"),
                    "decode_done_subject": queue.reply_subject("decode_done"),
                }))
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

            tracker.transition(
                req_id, SeqState.KV_PENDING,
                kv_size_bytes=kv_size,
                block_table=blk_table,
                transfer_operation_id=uuid.uuid4().hex,
            )
            scheduler.on_prefill_done(
                req.prefill_instance, req.prefill_instance_epoch
            )

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
                and msg.get("instance_epoch") == req.decode_instance_epoch
                and scheduler.epoch_matches(
                    req.decode_instance, req.decode_instance_epoch
                )
            ):
                tracker.record_first_token(req_id)

        for msg in await queue.poll("decode_done"):
            req_id = msg.get("req_id")
            if req_id is None or req_id not in tracker:
                continue
            req = tracker.get(req_id)
            if (
                req is None
                or req.state != SeqState.DECODING
                or msg.get("instance_epoch") != req.decode_instance_epoch
                or not scheduler.epoch_matches(
                    req.decode_instance, req.decode_instance_epoch
                )
            ):
                continue
            tracker.transition(req_id, SeqState.FINISHED)
            scheduler.on_decode_finished(
                req.decode_instance, req.decode_instance_epoch,
                req.active_operation_id or None,
            )
            governor.finish_request(req_id)
            tracker.remove(req_id)

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
    tracker.transition(req_id, SeqState.DECODING)


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
        await result
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

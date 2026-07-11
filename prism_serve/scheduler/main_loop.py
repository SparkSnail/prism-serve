"""Main scheduling loop for prism-serve.

A 10 ms tick-based coroutine that drives all serve-side PD orchestration.
Phase ordering is designed so that within a single tick each request
advances at most one state.

Phase sequence:
  1  WAITING     → PREFILLING  (assign P+D, dispatch via NATS)
  2  PREFILLING  → KV_PENDING  (poll prefill_done, submit to governor)
  3  stuck check → recompute / abort  (KV_PENDING timeout)
  4  deferred flush (governor.tick)
  5  DECODING    → FINISHED    (poll decode_done, release slots)
  6  metrics snapshot

Borrowing:
  - 10 ms tick interval ← Ray Serve reconcile loop (empirical optimum:
    < 1 ms spins CPU; > 100 ms delays recompute and slot release)
  - Phase ordering ← Flink StreamTask (process new → advance in-flight
    → clean terminal → emit metrics)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism_serve.scheduler.scheduler import PDScheduler
    from prism_serve.scheduler.transfer_governor import TransferGovernor
    from prism_serve.scheduler.sequence_state import RequestTracker, SeqState
    from prism_serve.scheduler.queue import NATSQueue

logger = logging.getLogger(__name__)

TICK_INTERVAL_S: float = 0.010  # 10 ms — Ray Serve reconcile loop interval


async def schedule_loop(
    scheduler: "PDScheduler",
    governor: "TransferGovernor",
    tracker: "RequestTracker",
    queue: "NATSQueue",
    metrics,
    config: dict,
) -> None:
    """Main scheduling coroutine.  Runs until cancelled.

    Args:
        scheduler: PD instance selection (PDScheduler)
        governor:  KV transfer flow control (TransferGovernor)
        tracker:   per-request state machine (RequestTracker)
        queue:     NATS publish/subscribe wrapper (NATSQueue)
        metrics:   MetricsCollector (or NullMetrics in tests)
        config:    runtime config dict
    """
    from prism_serve.scheduler.sequence_state import SeqState, TransferTask

    kv_transfer_timeout_s: float = config.get("kv_transfer_timeout_s", 30.0)
    tick_interval_s = config.get("schedule_loop_tick_ms", 10) / 1000.0

    while True:
        tick_start = time.monotonic()

        # ── Phase 1: WAITING → PREFILLING ─────────────────────────────
        # Assign P+D instances and dispatch prefill via NATS.
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
            # Dispatch prefill instruction via NATS.
            await queue.publish("dispatch_prefill", {
                "instance_id": p,
                "req_id":      req.req_id,
                # token_ids are carried in the request object (set by gateway)
            })
            tracker.transition(
                req.req_id, SeqState.PREFILLING,
                prefill_instance=p,
                decode_instance=d,
            )

        # ── Phase 2: PREFILLING → KV_PENDING ──────────────────────────
        # Poll prefill_done notifications from P instances (via NATS).
        # For each, release P load counter and submit KV transfer to governor.
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
            )
            scheduler.on_prefill_done(req.prefill_instance)

            # Submit to governor — may dispatch immediately or defer.
            governor.submit(TransferTask(
                req_id=req_id,
                src=req.prefill_instance,
                dst=req.decode_instance,
                kv_size=kv_size,
                on_complete=lambda rid=req_id: _on_kv_done(
                    rid, tracker, scheduler, metrics
                ),
            ))

        # ── Phase 3: stuck detection → recompute / abort ──────────────
        # Scan KV_PENDING requests that have exceeded the transfer timeout.
        # Process before Phase 4 so already-timed-out tasks are not
        # re-flushed from the deferred queue in the same tick.
        for req in tracker.get_stuck_requests(kv_transfer_timeout_s):
            governor.cancel(req.req_id)
            decision = governor.on_transfer_failure(
                req.req_id, req.decode_instance, "timeout"
            )
            if decision == "recompute":
                governor.trigger_recompute(req.req_id, req.decode_instance)
                tracker.transition(req.req_id, SeqState.WAITING)
                # Release the D slot we pre-reserved — the request will
                # re-claim one when it enters PREFILLING again next tick.
                scheduler.on_decode_finished(req.decode_instance)
                logger.info(
                    "recompute triggered for req=%s dst=%s",
                    req.req_id, req.decode_instance,
                )
            else:  # abort
                tracker.transition(req.req_id, SeqState.ABORTED)
                scheduler.on_decode_finished(req.decode_instance)
                tracker.remove(req.req_id)
                logger.warning(
                    "request aborted (max_recompute exhausted) req=%s",
                    req.req_id,
                )

        # ── Phase 4: flush deferred transfer queue ────────────────────
        # Let governor attempt to dispatch tasks that were deferred due to
        # high-watermark or byte-cap congestion.
        governor.tick()

        # ── Phase 5: DECODING → FINISHED ──────────────────────────────
        # Poll decode_done notifications from D instances.
        for msg in await queue.poll("decode_done"):
            req_id = msg.get("req_id")
            if req_id is None or req_id not in tracker:
                continue
            req = tracker.get(req_id)
            if req is None or req.state != SeqState.DECODING:
                continue
            tracker.transition(req_id, SeqState.FINISHED)
            scheduler.on_decode_finished(req.decode_instance)
            tracker.remove(req_id)

        # ── Phase 6: metrics snapshot ─────────────────────────────────
        metrics.gauge("active_requests", len(tracker))
        metrics.gauge(
            "waiting_requests",
            sum(1 for _ in tracker.iter_by_state(SeqState.WAITING)),
        )
        metrics.gauge(
            "kv_pending_requests",
            sum(1 for _ in tracker.iter_by_state(SeqState.KV_PENDING)),
        )

        # ── sleep until next tick ─────────────────────────────────────
        elapsed = time.monotonic() - tick_start
        await asyncio.sleep(max(0.0, tick_interval_s - elapsed))


# ---------------------------------------------------------------------------
# KV transfer complete callback
# ---------------------------------------------------------------------------

def _on_kv_done(
    req_id: str,
    tracker: "RequestTracker",
    scheduler: "PDScheduler",
    metrics,
) -> None:
    """Called by TransferGovernor._on_complete when KV reaches the D instance."""
    from prism_serve.scheduler.sequence_state import SeqState
    req = tracker.get(req_id)
    if req is None:
        return
    if req.state != SeqState.KV_PENDING:
        # Request was already recomputed or aborted; ignore stale callback.
        return
    tracker.transition(req_id, SeqState.DECODING)
    metrics.increment("kv_transfer_success_total")

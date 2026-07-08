"""Integration tests for schedule_loop (scheduler/main_loop.py).

No GPU, no NATS server required.  All infer-side behaviour is driven by
injecting messages directly into NATSQueue._put_mock() and by controlling
TransferGovernor's infer_client stub.

Each test runs schedule_loop for a bounded number of ticks (via
asyncio.wait_for) and asserts on the final state of RequestTracker,
PDScheduler slot counters, and governor inflight bytes.

Scenarios covered:
  happy_path_single       — 1 request: WAITING → PREFILLING → KV_PENDING
                            → DECODING → FINISHED, slots restored
  happy_path_four         — 4 concurrent requests (E2E §11 scenario),
                            all finish, no ABORTED
  kv_transfer_deferred    — D-instance at HIGH_WATERMARK; KV dispatch deferred
                            until watermark drops, then flushed
  recompute_fallback      — KV transfer times out; request falls back to
                            WAITING, recompute_count incremented
  abort_after_max_recompute — recompute budget exhausted; request ABORTED
  no_prefill_instance     — no P registered; requests stay WAITING
  no_decode_slot          — D fully booked; requests stay WAITING until slot
                            becomes available
  phase_ordering          — within one tick a request advances at most one
                            state (WAITING stays WAITING after Phase-1 dispatch
                            until prefill_done arrives in a later tick)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from prism_serve.metrics.collector import NullMetrics
from prism_serve.scheduler.main_loop import schedule_loop
from prism_serve.scheduler.queue import NATSQueue
from prism_serve.scheduler.scheduler import PDScheduler
from prism_serve.scheduler.sequence_state import RequestInfo, RequestTracker, SeqState
from prism_serve.scheduler.transfer_governor import TransferGovernor


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "HIGH_WATERMARK":         0.85,
    "LOW_WATERMARK":          0.70,
    "MAX_BYTES_INFLIGHT":     512 * 1024 ** 2,  # 512 MB — generous for tests
    "kv_transfer_timeout_s":  0.05,             # 50 ms — fast timeout for tests
    "max_recompute_attempts": 2,
    "schedule_loop_tick_ms":  10,
}

KV_SIZE_1BLOCK = 112 * 1024 ** 2   # 112 MB


def make_components(config: dict | None = None):
    """Return (scheduler, governor, tracker, queue, metrics, infer_client)."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    metrics = NullMetrics()
    infer_client = MagicMock()
    # transfer() calls on_complete immediately by default (instant KV transfer)
    infer_client.transfer.side_effect = (
        lambda src, dst, req_id, on_complete=None: on_complete() if on_complete else None
    )
    infer_client.reset_to_waiting = MagicMock()

    governor = TransferGovernor(cfg, infer_client, metrics)
    scheduler = PDScheduler(cfg)
    tracker = RequestTracker(metrics)
    queue = NATSQueue(cfg, use_mock=True)
    return scheduler, governor, tracker, queue, metrics, infer_client


async def run_ticks(
    scheduler, governor, tracker, queue, metrics, config,
    *,
    n_ticks: int = 20,
    tick_s: float = 0.001,   # 1 ms per tick in tests (faster than 10 ms)
) -> None:
    """Run schedule_loop for exactly n_ticks, then cancel."""
    ticks_done = 0
    orig_sleep = asyncio.sleep

    async def fast_sleep(delay):
        nonlocal ticks_done
        ticks_done += 1
        if ticks_done >= n_ticks:
            raise asyncio.CancelledError
        await orig_sleep(0)   # yield but don't actually wait

    import prism_serve.scheduler.main_loop as ml
    ml_sleep_orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0  # disable real sleep; we control ticks via fast_sleep

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )

    # Patch asyncio.sleep inside the running loop to count ticks
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(asyncio, "sleep", fast_sleep)
        try:
            await asyncio.wait_for(loop_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        finally:
            loop_task.cancel()
            try:
                await loop_task
            except (asyncio.CancelledError, Exception):
                pass

    ml.TICK_INTERVAL_S = ml_sleep_orig


async def drive_loop(
    scheduler, governor, tracker, queue, metrics, config,
    *,
    max_ticks: int = 50,
    stop_when=None,      # callable(tracker) -> bool; stop when True
) -> None:
    """Run ticks until stop_when() is satisfied or max_ticks reached."""
    import prism_serve.scheduler.main_loop as ml

    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    tick = 0
    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )

    while tick < max_ticks:
        await asyncio.sleep(0)   # yield to let loop_task run one iteration
        tick += 1
        if stop_when and stop_when(tracker):
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig


# ---------------------------------------------------------------------------
# Test 1: Happy path — single request end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_single():
    """WAITING → PREFILLING → KV_PENDING → DECODING → FINISHED."""
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    req = RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK)
    tracker.add(req)

    # After Phase 1 the request should be PREFILLING.
    # We inject prefill_done after 2 ticks to simulate P-infer.
    injected = False

    async def _inject_prefill_done_once(tracker_ref):
        nonlocal injected
        if not injected and tracker_ref.get("R1") and \
                tracker_ref.get("R1").state == SeqState.PREFILLING:
            await queue._put_mock("prefill_done", {
                "req_id": "R1",
                "kv_size_bytes": KV_SIZE_1BLOCK,
                "block_table": [3],
            })
            injected = True

    # Patch: after KV transfer completes (infer_client.transfer called
    # on_complete immediately), inject decode_done.
    decode_injected = False
    orig_transfer = infer_client.transfer.side_effect

    def transfer_and_decode(src, dst, req_id, on_complete=None):
        nonlocal decode_injected
        if on_complete:
            on_complete()
        # Schedule decode_done injection for next tick
        decode_injected = True

    infer_client.transfer.side_effect = transfer_and_decode

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )

    for _ in range(60):
        await asyncio.sleep(0)
        await _inject_prefill_done_once(tracker)
        # Once decode_injected, feed decode_done on the next yield
        if decode_injected and tracker.get("R1") and \
                tracker.get("R1").state == SeqState.DECODING:
            await queue._put_mock("decode_done", {"req_id": "R1"})
            decode_injected = False
        if "R1" not in tracker:
            break   # FINISHED + removed

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    # Request must be gone (removed on FINISHED)
    assert "R1" not in tracker
    # Slot must be returned
    assert scheduler._decode_free_slots["d-0"] == 10
    # P load must be 0
    assert scheduler._prefill_load["p-0"] == 0


# ---------------------------------------------------------------------------
# Test 2: Happy path — 4 concurrent requests (E2E §11 scenario)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_four_concurrent():
    """4 concurrent requests, 2P+2D, all finish with no ABORTED."""
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("p-1", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=127)
    scheduler.register_instance("d-1", "decode", max_slots=127)

    req_sizes = {
        "R1": 2 * KV_SIZE_1BLOCK,   # 512-token, 2 blocks
        "R2": 1 * KV_SIZE_1BLOCK,   # 256-token, 1 block
        "R3": 4 * KV_SIZE_1BLOCK,   # 1024-token, 4 blocks
        "R4": 2 * KV_SIZE_1BLOCK,
    }
    for rid, sz in req_sizes.items():
        tracker.add(RequestInfo(req_id=rid, kv_size_bytes=sz))

    prefill_injected: set[str] = set()
    decode_injected: set[str] = set()

    def transfer_side_effect(src, dst, req_id, on_complete=None):
        if on_complete:
            on_complete()
        decode_injected.add(req_id)

    infer_client.transfer.side_effect = transfer_side_effect

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )

    for _ in range(200):
        await asyncio.sleep(0)

        # Inject prefill_done for any PREFILLING request not yet injected
        for rid, sz in req_sizes.items():
            r = tracker.get(rid)
            if r and r.state == SeqState.PREFILLING and rid not in prefill_injected:
                await queue._put_mock("prefill_done", {
                    "req_id": rid,
                    "kv_size_bytes": sz,
                    "block_table": [0],
                })
                prefill_injected.add(rid)

        # Inject decode_done for any DECODING request not yet injected
        for rid in list(decode_injected):
            r = tracker.get(rid)
            if r and r.state == SeqState.DECODING:
                await queue._put_mock("decode_done", {"req_id": rid})
                decode_injected.discard(rid)

        # Stop when all requests are gone
        if len(tracker) == 0:
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    # All requests finished
    assert len(tracker) == 0
    # All slots restored
    assert scheduler._decode_free_slots["d-0"] + \
           scheduler._decode_free_slots["d-1"] == 254
    # P load cleared
    assert scheduler._prefill_load["p-0"] == 0
    assert scheduler._prefill_load["p-1"] == 0


# ---------------------------------------------------------------------------
# Test 3: KV transfer deferred by high watermark, then flushed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kv_transfer_deferred_then_flushed():
    """D-instance at HIGH_WATERMARK: transfer deferred; drops to LOW → flushed."""
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    # Set D congested before prefill_done arrives
    governor._kv_usage["d-0"] = 0.90

    # infer_client.transfer should NOT be called while congested
    infer_client.transfer.side_effect = (
        lambda src, dst, req_id, on_complete=None: on_complete() if on_complete else None
    )

    req = RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK)
    tracker.add(req)

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )

    prefill_injected = False
    for tick in range(100):
        await asyncio.sleep(0)

        r = tracker.get("R1")
        if r and r.state == SeqState.PREFILLING and not prefill_injected:
            await queue._put_mock("prefill_done", {
                "req_id": "R1",
                "kv_size_bytes": KV_SIZE_1BLOCK,
                "block_table": [3],
            })
            prefill_injected = True

        # After prefill_done injected: request should be KV_PENDING,
        # deferred because of high watermark
        if prefill_injected and tick == 20:
            r2 = tracker.get("R1")
            assert r2 is not None and r2.state == SeqState.KV_PENDING, (
                f"expected KV_PENDING, got {r2.state if r2 else 'gone'}"
            )
            assert governor.deferred_depth("d-0") == 1
            assert infer_client.transfer.call_count == 0

            # Simulate watermark drop → governor should flush
            governor.update_kv_usage("d-0", 0.60)

        if tick == 30:
            # transfer should have been called by now
            assert infer_client.transfer.call_count == 1
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    assert infer_client.transfer.call_count == 1
    assert governor.deferred_depth("d-0") == 0


# ---------------------------------------------------------------------------
# Test 4: recompute fallback on KV transfer timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recompute_fallback_on_timeout():
    """KV_PENDING times out → request returns to WAITING, recompute_count=1."""
    # Very short timeout so the test doesn't take seconds
    config = {**DEFAULT_CONFIG, "kv_transfer_timeout_s": 0.01}  # 10 ms

    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    # transfer() hangs — never calls on_complete
    infer_client.transfer.side_effect = lambda src, dst, req_id, on_complete=None: None

    req = RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK)
    tracker.add(req)

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )

    prefill_injected = False
    waited = False
    for _ in range(300):
        await asyncio.sleep(0)

        r = tracker.get("R1")
        if r and r.state == SeqState.PREFILLING and not prefill_injected:
            await queue._put_mock("prefill_done", {
                "req_id": "R1",
                "kv_size_bytes": KV_SIZE_1BLOCK,
                "block_table": [3],
            })
            prefill_injected = True

        # After prefill_done: wait for kv_sent_at to be set, then sleep past
        # the timeout threshold so Phase 3 detects stuck
        if prefill_injected and not waited:
            r2 = tracker.get("R1")
            if r2 and r2.state == SeqState.KV_PENDING and r2.kv_sent_at > 0:
                # Backdate kv_sent_at to force timeout
                r2.kv_sent_at = time.monotonic() - 1.0  # 1 s ago > 10 ms timeout
                waited = True

        # Once request returns to WAITING, we're done
        if r and r.state == SeqState.WAITING and prefill_injected and waited:
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    r = tracker.get("R1")
    assert r is not None, "request should still exist (in WAITING for recompute)"
    assert r.state == SeqState.WAITING
    assert governor._recompute_counts["R1"] == 1
    infer_client.reset_to_waiting.assert_called_once_with("d-0", "R1")
    # Slot should be returned after recompute so it can be re-allocated
    assert scheduler._decode_free_slots["d-0"] == 10


# ---------------------------------------------------------------------------
# Test 5: abort after max_recompute exhausted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_abort_after_max_recompute():
    """After max_recompute_attempts timeouts, request is ABORTED."""
    config = {**DEFAULT_CONFIG,
              "kv_transfer_timeout_s": 0.01,
              "max_recompute_attempts": 1}

    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    # transfer always hangs
    infer_client.transfer.side_effect = lambda src, dst, req_id, on_complete=None: None

    req = RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK)
    tracker.add(req)

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )

    prefill_injected = False
    recompute_count = 0

    for _ in range(500):
        await asyncio.sleep(0)

        r = tracker.get("R1")

        # Inject prefill_done whenever we see PREFILLING
        # (covers both initial run and after recompute → re-schedule)
        if r and r.state == SeqState.PREFILLING:
            await queue._put_mock("prefill_done", {
                "req_id": "R1",
                "kv_size_bytes": KV_SIZE_1BLOCK,
                "block_table": [3],
            })
            prefill_injected = True

        # Backdate kv_sent_at whenever we see KV_PENDING
        if r and r.state == SeqState.KV_PENDING and r.kv_sent_at > 0:
            # Only backdate once per KV_PENDING entry
            if r.kv_sent_at > time.monotonic() - 0.5:
                r.kv_sent_at = time.monotonic() - 1.0

        # Track recompute transitions: WAITING after prefill_injected
        if r and r.state == SeqState.WAITING and prefill_injected:
            recompute_count += 1

        # Request removed from tracker means ABORTED (removed in main_loop)
        if "R1" not in tracker and prefill_injected:
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    # Request must be gone (aborted and removed)
    assert "R1" not in tracker
    # Slot must be returned
    assert scheduler._decode_free_slots["d-0"] == 10


# ---------------------------------------------------------------------------
# Test 6: no P instance → requests stay WAITING
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_prefill_instance_stays_waiting():
    """Without any registered P instance, all requests remain WAITING."""
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    # No prefill instance registered
    scheduler.register_instance("d-0", "decode", max_slots=10)

    for i in range(3):
        tracker.add(RequestInfo(req_id=f"R{i}", kv_size_bytes=KV_SIZE_1BLOCK))

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )

    for _ in range(20):
        await asyncio.sleep(0)

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    assert len(tracker) == 3
    for i in range(3):
        r = tracker.get(f"R{i}")
        assert r is not None and r.state == SeqState.WAITING


# ---------------------------------------------------------------------------
# Test 7: D slots full → requests stay WAITING until slot freed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_waiting_until_decode_slot_freed():
    """D fully booked: R1 stays WAITING until R0's slot is released."""
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)   # only 1 slot

    # Book the slot by hand (simulating an existing sequence)
    scheduler._decode_free_slots["d-0"] = 0

    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )

    # Run 20 ticks — R1 must stay WAITING
    for _ in range(20):
        await asyncio.sleep(0)

    r = tracker.get("R1")
    assert r is not None and r.state == SeqState.WAITING

    # Free the slot (simulating previous sequence finishing)
    scheduler.on_decode_finished("d-0")
    assert scheduler._decode_free_slots["d-0"] == 1

    # Run more ticks — R1 should now advance to PREFILLING
    for _ in range(20):
        await asyncio.sleep(0)
        r2 = tracker.get("R1")
        if r2 and r2.state == SeqState.PREFILLING:
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    r = tracker.get("R1")
    assert r is not None and r.state == SeqState.PREFILLING


# ---------------------------------------------------------------------------
# Test 8: phase ordering — within one tick a request advances at most one state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_phase_ordering_single_advance_per_tick():
    """After Phase-1 dispatch, the request is PREFILLING — not yet KV_PENDING.

    prefill_done is only injected after the first tick completes, so the
    request cannot skip PREFILLING→KV_PENDING within the same tick it was
    dispatched.
    """
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    req = RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK)
    tracker.add(req)

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )

    # Yield exactly once — gives loop_task one full tick
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    r = tracker.get("R1")
    # After first tick: Phase-1 dispatched → PREFILLING
    # Phase-2 ran but inbox is empty → still PREFILLING (not KV_PENDING)
    assert r is not None and r.state == SeqState.PREFILLING, (
        f"expected PREFILLING after first tick, got {r.state if r else 'gone'}"
    )

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig


# ---------------------------------------------------------------------------
# Test 9: kv_usage update via governor propagates correctly
# ---------------------------------------------------------------------------

def test_governor_kv_usage_update_unblocks_deferred():
    """Synchronous check: update_kv_usage below LOW_WATERMARK flushes deferred."""
    from prism_serve.scheduler.sequence_state import TransferTask
    infer_client = MagicMock()
    metrics = NullMetrics()
    config = {**DEFAULT_CONFIG, "MAX_BYTES_INFLIGHT": 512 * 1024 ** 2}
    governor = TransferGovernor(config, infer_client, metrics)

    governor._kv_usage["d-0"] = 0.90   # congested
    task = TransferTask(req_id="R1", src="p-0", dst="d-0", kv_size=KV_SIZE_1BLOCK)
    governor.submit(task)

    assert governor.deferred_depth("d-0") == 1
    infer_client.transfer.assert_not_called()

    # Watermark recovery
    governor.update_kv_usage("d-0", 0.65)

    assert governor.deferred_depth("d-0") == 0
    infer_client.transfer.assert_called_once()


# ---------------------------------------------------------------------------
# Test 10: metrics gauges called on each tick
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_metrics_gauges_updated_each_tick():
    """Phase 6 must call metrics.gauge for active/waiting/kv_pending."""
    scheduler, governor, tracker, queue, _, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    metrics = MagicMock()

    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )

    for _ in range(5):
        await asyncio.sleep(0)

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    gauge_names = [call.args[0] for call in metrics.gauge.call_args_list]
    assert "active_requests"     in gauge_names
    assert "waiting_requests"    in gauge_names
    assert "kv_pending_requests" in gauge_names

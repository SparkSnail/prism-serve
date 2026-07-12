"""Schedule-loop integration tests with mock transport and infer clients."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from prism_serve.metrics.collector import NullMetrics
from prism_serve.scheduler.main_loop import _abort_remote_request, schedule_loop
from prism_serve.scheduler.queue import NATSQueue
from prism_serve.scheduler.scheduler import PDScheduler
from prism_serve.scheduler.sequence_state import RequestInfo, RequestTracker, SeqState
from prism_serve.scheduler.transfer_governor import TransferGovernor


DEFAULT_CONFIG = {
    "HIGH_WATERMARK":         0.85,
    "LOW_WATERMARK":          0.70,
    "MAX_BYTES_INFLIGHT":     512 * 1024 ** 2,  # 512 MB — generous for tests
    "kv_transfer_timeout_s":  0.05,             # 50 ms — fast timeout for tests
    "prefill_timeout_s":      30.0,
    "max_dispatch_attempts":  3,
    "recompute_timeout_s":    30.0,
    "decode_timeout_s":       300.0,
    "abort_request_timeout_s": 0.1,
    "max_recompute_attempts": 2,
    "schedule_loop_tick_ms":  0,
}

KV_SIZE_1BLOCK = 28 * 1024 ** 2   # TP=1 reference block: 28 MiB


def make_components(config: dict | None = None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    metrics = NullMetrics()
    infer_client = MagicMock()
    infer_client.transfer.side_effect = (
        lambda src, dst, req_id, operation_id, on_complete=None:
        on_complete() if on_complete else None
    )
    infer_client.reset_to_waiting = MagicMock()
    infer_client.abort_request = AsyncMock(return_value={"success": True})
    infer_client.abort_transfer = AsyncMock(return_value={"success": True})

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
    ticks_done = 0
    orig_sleep = asyncio.sleep

    async def fast_sleep(delay):
        nonlocal ticks_done
        ticks_done += 1
        if ticks_done >= n_ticks:
            raise asyncio.CancelledError
        await orig_sleep(0)

    import prism_serve.scheduler.main_loop as ml
    ml_sleep_orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )

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


@pytest.mark.asyncio
async def test_schedule_loop_uses_configured_tick_interval(monkeypatch):
    config = {**DEFAULT_CONFIG, "schedule_loop_tick_ms": 123}
    scheduler, governor, tracker, queue, metrics, _ = make_components(config)
    delays = []

    async def capture_sleep(delay):
        delays.append(delay)
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", capture_sleep)
    with pytest.raises(asyncio.CancelledError):
        await schedule_loop(scheduler, governor, tracker, queue, metrics, config)

    assert delays
    assert 0.100 < delays[0] <= 0.123


async def drive_loop(
    scheduler, governor, tracker, queue, metrics, config,
    *,
    max_ticks: int = 50,
    stop_when=None,      # callable(tracker) -> bool; stop when True
) -> None:
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


@pytest.mark.asyncio
async def test_happy_path_single():
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    req = RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK)
    tracker.add(req)

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

    decode_injected = False
    orig_transfer = infer_client.transfer.side_effect

    def transfer_and_decode(src, dst, req_id, operation_id, on_complete=None):
        nonlocal decode_injected
        if on_complete:
            on_complete()
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
        if decode_injected and tracker.get("R1") and \
                tracker.get("R1").state == SeqState.DECODING:
            await queue._put_mock("decode_done", {"req_id": "R1"})
            decode_injected = False
        if "R1" not in tracker:
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    assert "R1" not in tracker
    assert scheduler._decode_free_slots["d-0"] == 10
    assert scheduler._prefill_load["p-0"] == 0


@pytest.mark.asyncio
async def test_happy_path_four_concurrent():
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

    def transfer_side_effect(src, dst, req_id, operation_id, on_complete=None):
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

        for rid, sz in req_sizes.items():
            r = tracker.get(rid)
            if r and r.state == SeqState.PREFILLING and rid not in prefill_injected:
                await queue._put_mock("prefill_done", {
                    "req_id": rid,
                    "kv_size_bytes": sz,
                    "block_table": [0],
                })
                prefill_injected.add(rid)

        for rid in list(decode_injected):
            r = tracker.get(rid)
            if r and r.state == SeqState.DECODING:
                await queue._put_mock("decode_done", {"req_id": rid})
                decode_injected.discard(rid)

        if len(tracker) == 0:
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    assert len(tracker) == 0
    assert scheduler._decode_free_slots["d-0"] + \
           scheduler._decode_free_slots["d-1"] == 254
    assert scheduler._prefill_load["p-0"] == 0
    assert scheduler._prefill_load["p-1"] == 0


@pytest.mark.asyncio
async def test_kv_transfer_deferred_then_flushed():
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    governor._kv_usage["d-0"] = 0.90

    infer_client.transfer.side_effect = (
        lambda src, dst, req_id, operation_id, on_complete=None:
        on_complete() if on_complete else None
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

        if prefill_injected and tick == 20:
            r2 = tracker.get("R1")
            assert r2 is not None and r2.state == SeqState.KV_PENDING, (
                f"expected KV_PENDING, got {r2.state if r2 else 'gone'}"
            )
            assert governor.deferred_depth("d-0") == 1
            assert infer_client.transfer.call_count == 0

            governor.update_kv_usage("d-0", 0.60)

        if tick == 30:
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


@pytest.mark.asyncio
async def test_recompute_fallback_on_timeout():
    """Keep recompute on the decode instance selected before transfer."""
    config = {**DEFAULT_CONFIG, "kv_transfer_timeout_s": 0.01}  # 10 ms

    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    infer_client.transfer.side_effect = (
        lambda src, dst, req_id, operation_id, on_complete=None: None
    )

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

        if prefill_injected and not waited:
            r2 = tracker.get("R1")
            if r2 and r2.state == SeqState.KV_PENDING and r2.kv_sent_at > 0:
                r2.kv_sent_at = time.monotonic() - 1.0
                waited = True

        if r and r.state == SeqState.RECOMPUTING and prefill_injected and waited:
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    r = tracker.get("R1")
    assert r is not None, "request should remain tracked during recompute"
    assert r.state == SeqState.RECOMPUTING
    assert governor._recompute_counts["R1"] == 1
    infer_client.reset_to_waiting.assert_called_once_with("d-0", "R1")
    infer_client.abort_transfer.assert_awaited_once()
    assert scheduler._decode_free_slots["d-0"] == 9
    assert governor.deferred_depth("d-0") == 0


@pytest.mark.asyncio
async def test_transfer_abort_failure_quarantines_before_recompute():
    config = {**DEFAULT_CONFIG, "kv_transfer_timeout_s": 0.01}
    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    infer_client.transfer.side_effect = lambda **_kwargs: None
    infer_client.abort_transfer = AsyncMock(return_value={"success": False})
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    try:
        for _ in range(200):
            await asyncio.sleep(0)
            request = tracker.get("R1")
            if request and request.state == SeqState.PREFILLING:
                await queue._put_mock("prefill_done", {
                    "req_id": "R1",
                    "kv_size_bytes": KV_SIZE_1BLOCK,
                    "block_table": [3],
                })
            if request and request.state == SeqState.KV_PENDING:
                request.kv_sent_at = time.monotonic() - 1.0
            if "R1" not in tracker:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "R1" not in tracker
    assert scheduler.quarantine_record("d-0") is not None
    assert "d-0" not in scheduler.decode_free_slots()
    infer_client.reset_to_waiting.assert_not_called()


@pytest.mark.asyncio
async def test_late_completion_during_transfer_abort_await_wins_once():
    config = {**DEFAULT_CONFIG, "kv_transfer_timeout_s": 0.01}
    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    infer_client.transfer.side_effect = lambda **_kwargs: None
    abort_started = asyncio.Event()
    allow_abort_reply = asyncio.Event()

    async def abort_transfer(**_kwargs):
        abort_started.set()
        await allow_abort_reply.wait()
        return {"success": True}

    infer_client.abort_transfer.side_effect = abort_transfer
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    prefill_injected = False
    try:
        for _ in range(200):
            await asyncio.sleep(0)
            request = tracker.get("R1")
            if request and request.state == SeqState.PREFILLING and not prefill_injected:
                await queue._put_mock("prefill_done", {
                    "req_id": "R1",
                    "kv_size_bytes": KV_SIZE_1BLOCK,
                    "block_table": [3],
                })
                prefill_injected = True
            if request and request.state == SeqState.KV_PENDING:
                request.kv_sent_at = time.monotonic() - 1.0
            if abort_started.is_set():
                break

        await asyncio.wait_for(abort_started.wait(), timeout=1.0)
        completion = infer_client.transfer.call_args.kwargs["on_complete"]
        completion()
        assert tracker.get("R1").state == SeqState.DECODING

        allow_abort_reply.set()
        for _ in range(20):
            await asyncio.sleep(0)
        assert not task.done()
        assert tracker.get("R1").state == SeqState.DECODING
        assert governor._recompute_counts.get("R1", 0) == 0
        infer_client.reset_to_waiting.assert_not_called()

        await queue._put_mock("decode_done", {"req_id": "R1"})
        for _ in range(100):
            await asyncio.sleep(0)
            if "R1" not in tracker:
                break
    finally:
        allow_abort_reply.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "R1" not in tracker
    assert scheduler.decode_free_slots() == {"d-0": 1}
    assert governor.is_drained()


@pytest.mark.asyncio
async def test_transfer_ownership_loss_quarantines_and_terminates():
    config = {**DEFAULT_CONFIG, "kv_transfer_timeout_s": 0.01}
    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    infer_client.transfer.side_effect = lambda **_kwargs: None
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    prefill_injected = False
    try:
        for _ in range(200):
            await asyncio.sleep(0)
            request = tracker.get("R1")
            if request and request.state == SeqState.PREFILLING and not prefill_injected:
                await queue._put_mock("prefill_done", {
                    "req_id": "R1",
                    "kv_size_bytes": KV_SIZE_1BLOCK,
                    "block_table": [3],
                })
                prefill_injected = True
            if request and request.state == SeqState.KV_PENDING:
                operation_id = request.transfer_operation_id
                assert governor.cancel("R1", operation_id)
                request.kv_sent_at = time.monotonic() - 1.0
            if "R1" not in tracker:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "R1" not in tracker
    assert scheduler.quarantine_record("d-0") is not None
    assert "d-0" not in scheduler.decode_free_slots()
    assert governor.is_drained()
    infer_client.abort_transfer.assert_not_called()
    infer_client.reset_to_waiting.assert_not_called()


@pytest.mark.asyncio
async def test_transfer_dispatch_error_does_not_stop_schedule_loop():
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    infer_client.transfer.side_effect = ConnectionError("transport unavailable")
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )
    try:
        for _ in range(100):
            await asyncio.sleep(0)
            request = tracker.get("R1")
            if request and request.state == SeqState.PREFILLING:
                await queue._put_mock("prefill_done", {
                    "req_id": "R1",
                    "kv_size_bytes": KV_SIZE_1BLOCK,
                    "block_table": [3],
                })
            if request and request.state == SeqState.RECOMPUTING:
                break
        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    request = tracker.get("R1")
    assert request is not None and request.state == SeqState.RECOMPUTING
    assert governor.is_drained()
    infer_client.reset_to_waiting.assert_called_once_with("d-0", "R1")


@pytest.mark.asyncio
async def test_abort_after_max_recompute():
    config = {**DEFAULT_CONFIG,
              "kv_transfer_timeout_s": 0.01,
              "recompute_timeout_s": 0.01,
              "max_recompute_attempts": 1}

    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    infer_client.transfer.side_effect = (
        lambda src, dst, req_id, operation_id, on_complete=None: None
    )

    req = RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK)
    tracker.add(req)

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )

    prefill_injected = False
    for _ in range(500):
        await asyncio.sleep(0)

        r = tracker.get("R1")

        if r and r.state == SeqState.PREFILLING:
            await queue._put_mock("prefill_done", {
                "req_id": "R1",
                "kv_size_bytes": KV_SIZE_1BLOCK,
                "block_table": [3],
            })
            prefill_injected = True

        if r and r.state == SeqState.KV_PENDING and r.kv_sent_at > 0:
            if r.kv_sent_at > time.monotonic() - 0.5:
                r.kv_sent_at = time.monotonic() - 1.0

        if r and r.state == SeqState.RECOMPUTING:
            r.recompute_start = time.monotonic() - 1.0

        if "R1" not in tracker and prefill_injected:
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    assert "R1" not in tracker
    assert scheduler._decode_free_slots["d-0"] == 10


@pytest.mark.asyncio
async def test_no_prefill_instance_stays_waiting():
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
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

@pytest.mark.asyncio
async def test_recompute_done_finishes_and_clears_retry_state():
    config = {**DEFAULT_CONFIG, "kv_transfer_timeout_s": 0.01}
    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    infer_client.transfer.side_effect = lambda **kwargs: None
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    try:
        for _ in range(100):
            await asyncio.sleep(0)
            req = tracker.get("R1")
            if req and req.state == SeqState.PREFILLING:
                await queue._put_mock("prefill_done", {
                    "req_id": "R1",
                    "kv_size_bytes": KV_SIZE_1BLOCK,
                    "block_table": [3],
                })
            if req and req.state == SeqState.KV_PENDING:
                req.kv_sent_at = time.monotonic() - 1.0
            if req and req.state == SeqState.RECOMPUTING:
                break

        await queue._put_mock("recompute_done", {"req_id": "R1"})
        for _ in range(100):
            await asyncio.sleep(0)
            if tracker.get("R1").state == SeqState.DECODING:
                break
        req = tracker.get("R1")
        assert req.prefill_instance == "p-0"
        assert req.decode_instance == "d-0"
        assert governor._recompute_counts["R1"] == 1

        await queue._put_mock("decode_done", {"req_id": "R1"})
        for _ in range(100):
            await asyncio.sleep(0)
            if "R1" not in tracker:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "R1" not in tracker
    assert "R1" not in governor._recompute_counts
    assert scheduler.decode_free_slots() == {"d-0": 1}


@pytest.mark.asyncio
async def test_prefill_timeout_retries_same_assignment_then_aborts():
    config = {
        **DEFAULT_CONFIG,
        "prefill_timeout_s": 0.01,
        "max_dispatch_attempts": 2,
    }
    scheduler, governor, tracker, queue, metrics, _ = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))
    published = []

    async def capture_publish(subject, data):
        published.append((subject, data))

    queue.publish = capture_publish
    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    try:
        for _ in range(100):
            await asyncio.sleep(0)
            req = tracker.get("R1")
            if req is None:
                break
            if req.state == SeqState.PREFILLING:
                req.prefill_start = time.monotonic() - 1.0
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "R1" not in tracker
    assert [subject for subject, _ in published] == [
        "dispatch_prefill.p-0", "dispatch_prefill.p-0",
    ]
    command_ids = {data["command_id"] for _, data in published}
    assert len(command_ids) == 1
    assert next(iter(command_ids)).endswith(":R1")
    assert all(data["instance_id"] == "p-0" for _, data in published)
    assert all(data["first_token_subject"].startswith("first_token.")
               for _, data in published)
    assert all(data["recompute_done_subject"].startswith("recompute_done.")
               for _, data in published)
    assert scheduler.prefill_queue_depths() == {"p-0": 0}
    assert scheduler.decode_free_slots() == {"d-0": 1}


@pytest.mark.asyncio
async def test_waiting_until_decode_slot_freed():
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)

    scheduler._decode_free_slots["d-0"] = 0

    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )

    for _ in range(20):
        await asyncio.sleep(0)

    r = tracker.get("R1")
    assert r is not None and r.state == SeqState.WAITING

    scheduler.on_decode_finished("d-0")
    assert scheduler._decode_free_slots["d-0"] == 1

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


@pytest.mark.asyncio
async def test_phase_ordering_single_advance_per_tick():
    """A request cannot leave PREFILLING in its dispatch tick."""
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

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    r = tracker.get("R1")
    assert r is not None and r.state == SeqState.PREFILLING, (
        f"expected PREFILLING after first tick, got {r.state if r else 'gone'}"
    )

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig


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


@pytest.mark.asyncio
async def test_sync_abort_client_is_bounded_by_timeout():
    """A blocking synchronous RPC must not stall the event loop."""
    import time as wall_time

    client = MagicMock()

    def blocking_abort(**_kwargs):
        wall_time.sleep(0.1)
        return {"success": True}

    client.abort_request.side_effect = blocking_abort
    started = wall_time.monotonic()
    acknowledged = await _abort_remote_request(
        client, "d-0", "owner-uid", "R1", timeout_s=0.01
    )

    assert acknowledged is False
    assert wall_time.monotonic() - started < 0.08


@pytest.mark.asyncio
async def test_metrics_gauges_updated_each_tick():
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


@pytest.mark.asyncio
async def test_decode_timeout_releases_reserved_slot():
    config = {**DEFAULT_CONFIG, "decode_timeout_s": 0.01}
    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    try:
        for _ in range(100):
            await asyncio.sleep(0)
            req = tracker.get("R1")
            if req and req.state == SeqState.PREFILLING:
                await queue._put_mock("prefill_done", {
                    "req_id": "R1",
                    "kv_size_bytes": KV_SIZE_1BLOCK,
                    "block_table": [3],
                })
            if req and req.state == SeqState.DECODING:
                req.decode_start = time.monotonic() - 1.0
            if "R1" not in tracker:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "R1" not in tracker
    assert scheduler.decode_free_slots() == {"d-0": 1}
    infer_client.abort_request.assert_called_once_with(
        instance_id="d-0", owner_id=queue.owner_id, req_id="R1"
    )


@pytest.mark.asyncio
async def test_dispatch_publish_failure_keeps_loop_alive_and_rolls_back():
    scheduler, governor, tracker, queue, metrics, _ = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))
    calls = 0

    async def flaky_publish(_subject, _data):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("NATS reconnecting")

    queue.publish = flaky_publish
    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )
    try:
        for _ in range(100):
            await asyncio.sleep(0)
            request = tracker.get("R1")
            if request and request.state == SeqState.PREFILLING:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert calls >= 2
    assert tracker.get("R1").state == SeqState.PREFILLING
    assert scheduler.prefill_queue_depths() == {"p-0": 1}
    assert scheduler.decode_free_slots() == {"d-0": 0}


@pytest.mark.asyncio
async def test_decode_abort_failure_quarantines_instance_capacity():
    config = {**DEFAULT_CONFIG, "decode_timeout_s": 0.01}
    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("d-0", "decode", max_slots=1)
    scheduler._decode_free_slots["d-0"] = 0
    infer_client.abort_request.return_value = {"success": False}
    request = RequestInfo(
        req_id="R1", state=SeqState.DECODING, decode_instance="d-0"
    )
    request.decode_start = time.monotonic() - 1.0
    tracker.add(request)

    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    try:
        for _ in range(100):
            await asyncio.sleep(0)
            if "R1" not in tracker:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "R1" not in tracker
    assert "d-0" not in scheduler.decode_free_slots()

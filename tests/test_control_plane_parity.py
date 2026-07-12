"""Compare schedule-loop state and resource traces with independent expectations.

Token parity requires the real infer RPC, which is not available yet. These tests
cover the observable control-plane contract without pretending to validate tokens.
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


CONFIG = {
    "HIGH_WATERMARK": 0.85,
    "LOW_WATERMARK": 0.70,
    "MAX_BYTES_INFLIGHT": 512 * 1024**2,
    "kv_transfer_timeout_s": 0.01,
    "max_recompute_attempts": 2,
    "schedule_loop_tick_ms": 0,
}
KV_SIZE = 112 * 1024**2


def _components():
    metrics = NullMetrics()
    client = MagicMock()
    client.abort_transfer = AsyncMock(return_value={"success": True})
    governor = TransferGovernor(CONFIG, client, metrics)
    scheduler = PDScheduler(CONFIG)
    tracker = RequestTracker(metrics)
    queue = NATSQueue(CONFIG, use_mock=True)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE))
    return scheduler, governor, tracker, queue, metrics, client


async def _wait_for_state(
    tracker: RequestTracker,
    expected: SeqState,
    max_yields: int = 100,
) -> None:
    for _ in range(max_yields):
        request = tracker.get("R1")
        if request is not None and request.state == expected:
            return
        await asyncio.sleep(0)
    request = tracker.get("R1")
    actual = request.state.name if request is not None else "REMOVED"
    pytest.fail(f"expected {expected.name}, got {actual}")


@pytest.mark.asyncio
async def test_happy_path_trace_matches_reference():
    """Completion events produce the expected request-state trace."""
    scheduler, governor, tracker, queue, metrics, client = _components()
    client.transfer.side_effect = (
        lambda src, dst, req_id, operation_id, on_complete=None: on_complete()
        if on_complete else None
    )
    observed = [tracker.get("R1").state.name]
    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, CONFIG)
    )
    try:
        await _wait_for_state(tracker, SeqState.PREFILLING)
        observed.append(tracker.get("R1").state.name)

        await queue._put_mock("prefill_done", {
            "req_id": "R1",
            "kv_size_bytes": KV_SIZE,
            "block_table": [3],
        })
        await _wait_for_state(tracker, SeqState.DECODING)
        observed.append(tracker.get("R1").state.name)

        await queue._put_mock("decode_done", {"req_id": "R1"})
        for _ in range(100):
            if "R1" not in tracker:
                break
            await asyncio.sleep(0)
        observed.append("REMOVED" if "R1" not in tracker else "PRESENT")
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    reference = ["WAITING", "PREFILLING", "DECODING", "REMOVED"]
    assert observed == reference
    assert scheduler.prefill_queue_depths() == {"p-0": 0}
    assert scheduler.decode_free_slots() == {"d-0": 1}
    assert governor.all_inflight_zero()


@pytest.mark.asyncio
async def test_timeout_trace_drops_stale_transfer():
    """Timeout recomputes on the assigned D and drops the stale transfer."""
    scheduler, governor, tracker, queue, metrics, client = _components()
    governor.update_kv_usage("d-0", 0.90)
    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, CONFIG)
    )
    try:
        await _wait_for_state(tracker, SeqState.PREFILLING)
        await queue._put_mock("prefill_done", {
            "req_id": "R1",
            "kv_size_bytes": KV_SIZE,
            "block_table": [3],
        })
        await _wait_for_state(tracker, SeqState.KV_PENDING)
        request = tracker.get("R1")
        request.kv_sent_at = time.monotonic() - 1.0
        await _wait_for_state(tracker, SeqState.RECOMPUTING)

        governor.update_kv_usage("d-0", 0.0)
        governor.tick()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert tracker.get("R1").state == SeqState.RECOMPUTING
    assert governor.deferred_depth("d-0") == 0
    assert governor.all_inflight_zero()
    assert scheduler.decode_free_slots() == {"d-0": 0}
    client.transfer.assert_not_called()
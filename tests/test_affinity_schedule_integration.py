from __future__ import annotations

import asyncio
import time

import pytest

from prism_serve.metrics.collector import NullMetrics
from prism_serve.scheduler.main_loop import schedule_loop
from prism_serve.scheduler.scheduler import PDScheduler
from prism_serve.scheduler.sequence_state import RequestInfo, RequestTracker, SeqState


class EmptyQueue:
    owner_id = "gateway"

    async def poll(self, _subject):
        return []


class SuffixQueue(EmptyQueue):
    def __init__(self, message):
        self.message = message

    async def poll(self, subject):
        if subject == "suffix_prefill_done" and self.message is not None:
            message, self.message = self.message, None
            return [message]
        return []


class IdleGovernor:
    def tick(self):
        pass


class SuffixAbortCoordinator:
    def __init__(self, stopped):
        self.stopped = stopped

    async def abort_suffix(self, req):
        return self.stopped


class Coordinator:
    def __init__(self):
        self.calls = 0

    async def try_start(self, req, scheduler, tracker):
        self.calls += 1
        lease = scheduler.reserve_decode_slot("d0", req.req_id, "op1")
        tracker.transition(
            req.req_id, SeqState.AFFINITY_LOADING,
            active_operation_id="op1", decode_instance="d0",
            decode_slot_lease_id=lease.lease_id,
        )
        return True


@pytest.mark.asyncio
async def test_affinity_seam_runs_before_cold_prefill() -> None:
    metrics = NullMetrics()
    tracker = RequestTracker(metrics)
    tracker.add(RequestInfo("r1", fingerprint=object()))
    scheduler = PDScheduler({})
    scheduler.register_instance("p0", "prefill", instance_epoch="pe")
    scheduler.register_instance("d0", "decode", max_slots=1, instance_epoch="de")
    coordinator = Coordinator()
    task = asyncio.create_task(schedule_loop(
        scheduler, IdleGovernor(), tracker, EmptyQueue(), metrics,
        {"affinity_enabled": True, "schedule_loop_tick_ms": 1}, coordinator,
    ))
    for _ in range(20):
        if tracker.get("r1").state == SeqState.AFFINITY_LOADING:
            break
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert coordinator.calls == 1
    assert tracker.get("r1").state == SeqState.AFFINITY_LOADING
    assert scheduler.prefill_queue_depths()["p0"] == 0


@pytest.mark.asyncio
async def test_suffix_completion_requires_operation_and_epoch_match() -> None:
    metrics = NullMetrics()
    tracker = RequestTracker(metrics)
    request = RequestInfo(
        "r1", state=SeqState.PREFIX_PREFILLING,
        active_operation_id="op", decode_instance="d0", decode_instance_epoch="de",
    )
    tracker.add(request)
    scheduler = PDScheduler({})
    scheduler.register_instance("d0", "decode", max_slots=1, instance_epoch="de")
    queue = SuffixQueue({
        "req_id": "r1", "operation_id": "op", "instance_epoch": "de",
    })
    task = asyncio.create_task(schedule_loop(
        scheduler, IdleGovernor(), tracker, queue, metrics,
        {"schedule_loop_tick_ms": 1},
    ))
    for _ in range(20):
        if tracker.get("r1").state == SeqState.DECODING:
            break
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tracker.get("r1").state == SeqState.DECODING


@pytest.mark.asyncio
@pytest.mark.parametrize("stopped, expected_free, expected_state", [
    (True, 1, "RELEASED"),
    (False, 0, "QUARANTINED"),
])
async def test_suffix_timeout_releases_only_after_abort_ack(
    stopped, expected_free, expected_state
) -> None:
    metrics = NullMetrics()
    tracker = RequestTracker(metrics)
    scheduler = PDScheduler({})
    scheduler.register_instance("d0", "decode", max_slots=1, instance_epoch="de")
    scheduler.reserve_decode_slot("d0", "r1", "op")
    scheduler.commit_decode_slot("op")
    request = RequestInfo(
        "r1", state=SeqState.PREFIX_PREFILLING,
        active_operation_id="op", decode_instance="d0", decode_instance_epoch="de",
        suffix_prefill_started_at=time.monotonic() - 10,
    )
    tracker.add(request)
    task = asyncio.create_task(schedule_loop(
        scheduler, IdleGovernor(), tracker, EmptyQueue(), metrics,
        {"schedule_loop_tick_ms": 1, "suffix_prefill_timeout_s": 0.01},
        SuffixAbortCoordinator(stopped),
    ))
    for _ in range(20):
        if tracker.get("r1") is None:
            break
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert scheduler.decode_free_slots()["d0"] == expected_free
    assert scheduler.decode_slot_lease("op").state == expected_state

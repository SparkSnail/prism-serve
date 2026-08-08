from __future__ import annotations

import asyncio
import time

import pytest

from prism_serve.metrics.collector import NullMetrics
from prism_serve.router.coordinator import AffinityCoordinator
from prism_serve.router.fingerprint import PromptFingerprint
from prism_serve.router.http_rpc import EndpointSequenceAllocator
from prism_serve.router.http_rpc import AmbiguousRPCError
from prism_serve.router.protocol import (
    CachedPrefixDecision,
    MappedTransferStatus,
    PreparedPrefix,
    PrefixOperationStatus,
    ResolvedPrefix,
)
from prism_serve.scheduler.scheduler import PDScheduler
from prism_serve.scheduler.sequence_state import RequestInfo, RequestTracker, SeqState


class StaticRouter:
    def __init__(self, decisions):
        self.decisions = decisions

    def iter_decisions(self, *args, **kwargs):
        return list(self.decisions)


class Queue:
    owner_id = "gateway"

    def __init__(self):
        self.messages = []

    def dispatch_subject(self, instance):
        return f"dispatch.{instance}"

    def reply_subject(self, kind):
        return f"reply.{kind}"

    async def publish(self, subject, payload):
        self.messages.append((subject, payload))


class RPC:
    def __init__(self):
        self.unpinned = []
        self.aborted = []

    async def resolve_prefix(self, source, epoch, operation_id, expected, **identity):
        return ResolvedPrefix(operation_id, epoch, (1, 2))

    async def prepare_prefix(self, target, epoch, operation_id, req_id, **kwargs):
        return PreparedPrefix(operation_id, kwargs["mode"], (8, 9))

    async def transfer_cached_prefix(self, plan):
        return MappedTransferStatus.COMPLETED

    async def commit_cached_prefix(self, target, operation_id, plan):
        pass

    async def unpin_prefix(self, source, operation_id):
        self.unpinned.append((source, operation_id))

    async def abort_mapped_prefix(self, source, target, operation_id):
        return MappedTransferStatus.COMPLETED

    async def get_prefix_operation(self, target, operation_id):
        return PrefixOperationStatus.COMMITTED

    async def abort_cached_prefix(self, target, operation_id):
        self.aborted.append((target, operation_id))


def _request():
    fingerprint = PromptFingerprint(
        "ns", "compat", "text", tuple(range(10)), (11, 22), 4
    )
    return RequestInfo("r", fingerprint=fingerprint, sampling_params={})


def _scheduler(slots=1):
    scheduler = PDScheduler({})
    scheduler.register_instance("dst", "decode", max_slots=slots, instance_epoch="de")
    return scheduler


@pytest.mark.asyncio
async def test_coordinator_commits_slot_and_dispatches_only_suffix():
    queue = Queue()
    decision = CachedPrefixDecision("src", "se", "dst", 2, 8, 64, 1.0)
    coordinator = AffinityCoordinator(
        StaticRouter([decision]), RPC(), queue,
        {"prefill_ms_per_token": 0.1, "locality_wait_ms": 20},
    )
    scheduler = _scheduler()
    tracker = RequestTracker(NullMetrics())
    tracker.add(_request())
    assert await coordinator.try_start(tracker.get("r"), scheduler, tracker)
    await asyncio.gather(*coordinator._tasks.values())
    assert tracker.get("r").state == SeqState.PREFIX_PREFILLING
    assert scheduler.decode_slot_lease(tracker.get("r").active_operation_id).state == "ACTIVE"
    assert queue.messages[0][1]["remaining_token_ids"] == [8, 9]


@pytest.mark.asyncio
async def test_suffix_dispatch_uses_versioned_endpoint_envelope():
    queue = Queue()
    decision = CachedPrefixDecision("src", "se", "dst", 2, 8, 64, 1.0)
    allocator = EndpointSequenceAllocator("world-a", "gateway-a")
    coordinator = AffinityCoordinator(
        StaticRouter([decision]), RPC(), queue,
        {"prefill_ms_per_token": 0.1, "locality_wait_ms": 20},
        operation_allocator=allocator,
    )
    scheduler = _scheduler()
    tracker = RequestTracker(NullMetrics())
    tracker.add(_request())
    assert await coordinator.try_start(tracker.get("r"), scheduler, tracker)
    await asyncio.gather(*coordinator._tasks.values())

    command = queue.messages[0][1]
    assert command["schema_version"] == 1
    assert command["endpoint_ref"]["target_instance"] == "dst"
    assert command["payload"]["remaining_token_ids"] == [8, 9]
    assert tracker.get("r").suffix_operation_ref.operation_seq == 1


@pytest.mark.asyncio
async def test_no_match_and_overload_do_not_add_affinity_delay():
    tracker = RequestTracker(NullMetrics())
    tracker.add(_request())
    no_match = AffinityCoordinator(StaticRouter([]), RPC(), Queue(), {})
    assert not await no_match.try_start(tracker.get("r"), _scheduler(), tracker)

    overloaded = PDScheduler({})
    coordinator = AffinityCoordinator(
        StaticRouter([CachedPrefixDecision("src", "se", "dst", 1, 4, 4, 1)]),
        RPC(), Queue(), {"max_affinity_wait_ms": 100},
    )
    assert not await coordinator.try_start(tracker.get("r"), overloaded, tracker)


class RacyScheduler(PDScheduler):
    def reserve_decode_slot(self, instance_id, req_id, operation_id):
        return None


@pytest.mark.asyncio
async def test_reservation_race_wait_is_bounded():
    scheduler = RacyScheduler({})
    scheduler.register_instance("dst", "decode", max_slots=1, instance_epoch="de")
    decision = CachedPrefixDecision("src", "se", "dst", 1, 4, 4, 1)
    coordinator = AffinityCoordinator(
        StaticRouter([decision]), RPC(), Queue(),
        {"locality_wait_ms": 20, "max_affinity_wait_ms": 100},
    )
    tracker = RequestTracker(NullMetrics())
    req = _request()
    tracker.add(req)
    assert await coordinator.try_start(req, scheduler, tracker)
    req.arrived_at = time.monotonic() - 1
    assert not await coordinator.try_start(req, scheduler, tracker)


class HungResolveRPC(RPC):
    async def resolve_prefix(self, *args, **kwargs):
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_load_deadline_releases_reservation_and_unpins_source():
    rpc = HungResolveRPC()
    decision = CachedPrefixDecision("src", "se", "dst", 2, 8, 64, 1.0)
    coordinator = AffinityCoordinator(
        StaticRouter([decision]), rpc, Queue(), {"prefix_load_timeout_s": 0.01}
    )
    scheduler = _scheduler()
    tracker = RequestTracker(NullMetrics())
    tracker.add(_request())
    assert await coordinator.try_start(tracker.get("r"), scheduler, tracker)
    operation_id = tracker.get("r").active_operation_id
    await asyncio.gather(*coordinator._tasks.values())
    assert scheduler.decode_slot_lease(operation_id).state == "RELEASED"
    assert rpc.unpinned == [("src", operation_id)]
    assert tracker.get("r").state == SeqState.WAITING


class PrepareResponseLossRPC(RPC):
    async def prepare_prefix(self, *args, **kwargs):
        raise ConnectionError("response lost after prepare")

    async def get_prefix_operation(self, target, operation_id):
        return PrefixOperationStatus.PREPARED


@pytest.mark.asyncio
async def test_prepare_response_loss_aborts_target_and_releases_pin_and_slot():
    rpc = PrepareResponseLossRPC()
    decision = CachedPrefixDecision("src", "se", "dst", 2, 8, 64, 1.0)
    coordinator = AffinityCoordinator(StaticRouter([decision]), rpc, Queue(), {})
    scheduler = _scheduler()
    tracker = RequestTracker(NullMetrics())
    tracker.add(_request())
    assert await coordinator.try_start(tracker.get("r"), scheduler, tracker)
    operation_id = tracker.get("r").active_operation_id
    await asyncio.gather(*coordinator._tasks.values())
    assert rpc.aborted == [("dst", operation_id)]
    assert rpc.unpinned == [("src", operation_id)]
    assert scheduler.decode_slot_lease(operation_id).state == "RELEASED"


class FlakyCleanupRPC(HungResolveRPC):
    def __init__(self):
        super().__init__()
        self.unpin_attempts = 0

    async def unpin_prefix(self, source, operation_id):
        self.unpin_attempts += 1
        if self.unpin_attempts == 1:
            raise ConnectionError("unpin response lost")
        await super().unpin_prefix(source, operation_id)


@pytest.mark.asyncio
async def test_cleanup_supervisor_quarantines_then_retries_to_proven_release():
    rpc = FlakyCleanupRPC()
    decision = CachedPrefixDecision("src", "se", "dst", 2, 8, 64, 1.0)
    coordinator = AffinityCoordinator(
        StaticRouter([decision]), rpc, Queue(),
        {"prefix_load_timeout_s": 0.01, "prefix_operation_watchdog_s": 0.2},
    )
    scheduler = _scheduler()
    tracker = RequestTracker(NullMetrics())
    tracker.add(_request())
    assert await coordinator.try_start(tracker.get("r"), scheduler, tracker)
    operation_id = tracker.get("r").active_operation_id
    await asyncio.gather(*coordinator._tasks.values())
    assert rpc.unpin_attempts == 2
    assert scheduler.decode_slot_lease(operation_id).state == "RELEASED"
    assert tracker.get("r").state == SeqState.WAITING


class BrokenCleanupRPC(HungResolveRPC):
    async def unpin_prefix(self, source, operation_id):
        raise ConnectionError("still unavailable")


@pytest.mark.asyncio
async def test_cleanup_watchdog_leaves_unproven_slot_quarantined():
    decision = CachedPrefixDecision("src", "se", "dst", 2, 8, 64, 1.0)
    coordinator = AffinityCoordinator(
        StaticRouter([decision]), BrokenCleanupRPC(), Queue(),
        {"prefix_load_timeout_s": 0.005, "prefix_operation_watchdog_s": 0.03,
         "prefix_reconcile_interval_s": 0.005},
    )
    scheduler = _scheduler()
    tracker = RequestTracker(NullMetrics())
    tracker.add(_request())
    assert await coordinator.try_start(tracker.get("r"), scheduler, tracker)
    operation_id = tracker.get("r").active_operation_id
    await asyncio.gather(*coordinator._tasks.values())
    assert scheduler.decode_slot_lease(operation_id).state == "QUARANTINED"
    assert tracker.get("r").state == SeqState.AFFINITY_LOADING
    assert operation_id in coordinator._contexts
    coordinator.rpc = RPC()
    await asyncio.wait_for(coordinator._recovery_tasks[operation_id], 0.2)


class RecoveringCleanupRPC(HungResolveRPC):
    def __init__(self):
        super().__init__()
        self.available = False

    async def unpin_prefix(self, source, operation_id):
        if not self.available:
            raise ConnectionError("temporarily unavailable")
        await super().unpin_prefix(source, operation_id)


@pytest.mark.asyncio
async def test_uncertain_cleanup_keeps_reconciling_after_initial_watchdog():
    rpc = RecoveringCleanupRPC()
    decision = CachedPrefixDecision("src", "se", "dst", 2, 8, 64, 1.0)
    coordinator = AffinityCoordinator(
        StaticRouter([decision]), rpc, Queue(),
        {"prefix_load_timeout_s": 0.005, "prefix_operation_watchdog_s": 0.02,
         "prefix_reconcile_interval_s": 0.005},
    )
    scheduler = _scheduler()
    tracker = RequestTracker(NullMetrics())
    tracker.add(_request())
    assert await coordinator.try_start(tracker.get("r"), scheduler, tracker)
    operation_id = tracker.get("r").active_operation_id
    await asyncio.gather(*coordinator._tasks.values())
    assert operation_id in coordinator._recovery_tasks
    rpc.available = True
    await asyncio.wait_for(coordinator._recovery_tasks[operation_id], 0.2)
    assert scheduler.decode_slot_lease(operation_id).state == "RELEASED"
    assert tracker.get("r").state == SeqState.WAITING
    assert operation_id not in coordinator._contexts


class FailingQueue(Queue):
    async def publish(self, subject, payload):
        raise ConnectionError("publish failed")


@pytest.mark.asyncio
async def test_suffix_publish_failure_keeps_active_lease_for_watchdog_abort():
    decision = CachedPrefixDecision("src", "se", "dst", 2, 8, 64, 1.0)
    coordinator = AffinityCoordinator(StaticRouter([decision]), RPC(), FailingQueue(), {})
    scheduler = _scheduler()
    tracker = RequestTracker(NullMetrics())
    tracker.add(_request())
    assert await coordinator.try_start(tracker.get("r"), scheduler, tracker)
    operation_id = tracker.get("r").active_operation_id
    await asyncio.gather(*coordinator._tasks.values())
    assert tracker.get("r").state == SeqState.PREFIX_PREFILLING
    assert scheduler.decode_slot_lease(operation_id).state == "ACTIVE"


class CommittedSourceReleaseRPC(RPC):
    def __init__(self, first_error):
        super().__init__()
        self.first_error = first_error
        self.unpin_attempts = 0
        self.source_released = asyncio.Event()

    async def unpin_prefix(self, source, operation_id):
        self.unpin_attempts += 1
        if self.unpin_attempts == 1:
            raise self.first_error
        await super().unpin_prefix(source, operation_id)
        self.source_released.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_error",
    [
        ConnectionError("pre-send connection failure"),
        AmbiguousRPCError(None, "post-send response lost"),
    ],
    ids=["ordinary-pre-send", "ambiguous-post-send"],
)
async def test_committed_target_dispatches_suffix_while_source_release_retries(
    first_error,
):
    rpc = CommittedSourceReleaseRPC(first_error)
    queue = Queue()
    decision = CachedPrefixDecision("src", "se", "dst", 2, 8, 64, 1.0)
    coordinator = AffinityCoordinator(
        StaticRouter([decision]),
        rpc,
        queue,
        {"prefix_reconcile_interval_s": 0.001},
    )
    scheduler = _scheduler()
    tracker = RequestTracker(NullMetrics())
    tracker.add(_request())

    assert await coordinator.try_start(tracker.get("r"), scheduler, tracker)
    operation_id = tracker.get("r").active_operation_id
    await asyncio.gather(*coordinator._tasks.values())
    await asyncio.wait_for(rpc.source_released.wait(), 0.2)

    request = tracker.get("r")
    assert request.state == SeqState.PREFIX_PREFILLING
    assert scheduler.decode_slot_lease(operation_id).state == "ACTIVE"
    assert len(queue.messages) == 1
    assert queue.messages[0][1]["operation_id"] == operation_id
    assert rpc.unpin_attempts == 2
    assert rpc.unpinned == [("src", operation_id)]
    assert coordinator._contexts[operation_id].source_pinned is False
    await coordinator.shutdown()


class Week12RPC(RPC):
    week12_network_control = True


@pytest.mark.asyncio
async def test_week12_keeps_cross_source_pin_until_request_cleanup_proof():
    rpc = Week12RPC()
    queue = Queue()
    decision = CachedPrefixDecision("src", "se", "dst", 2, 8, 64, 1.0)
    coordinator = AffinityCoordinator(
        StaticRouter([decision]), rpc, queue,
        {"prefill_ms_per_token": 0.1, "locality_wait_ms": 20},
    )
    scheduler = _scheduler()
    tracker = RequestTracker(NullMetrics())
    tracker.add(_request())

    assert await coordinator.try_start(tracker.get("r"), scheduler, tracker)
    operation_id = tracker.get("r").active_operation_id
    await asyncio.gather(*coordinator._tasks.values())

    request = tracker.get("r")
    assert request.state == SeqState.PREFIX_PREFILLING
    assert rpc.unpinned == []
    assert coordinator._contexts[operation_id].source_pinned is True

    # In production this hook runs only after NetworkControlRPC's generic
    # request cleanup returns its stored release proof.
    coordinator.terminal_cleanup_complete(request)
    assert operation_id not in coordinator._contexts
    assert rpc.unpinned == []

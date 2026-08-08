from __future__ import annotations

import time

import pytest

from prism_serve.router.prefix_index import CacheLocation, FullReportRequired, PrefixEvent, PrefixIndex
from prism_serve.router.reconciler import PrefixReconciler, PrefixReport


class FakeRPC:
    def __init__(self):
        now = time.monotonic()
        self.location = CacheLocation("d0", "e1", "ns", "c", "text", 11, 0, 7, 4, now)
        self.events = [PrefixEvent("hash_added", self.location, 1)]
        self.acks = []
        self.fail_peek = False
        self.report_epoch = "e1"

    async def full_report_and_register(self, instance_id, consumer_id, generation):
        return PrefixReport("d0", self.report_epoch, 0, ())

    async def peek_events(self, instance_id, consumer_id, generation, after_seq, limit):
        if self.fail_peek:
            self.fail_peek = False
            raise FullReportRequired("gap")
        return [event for event in self.events if event.seq_no > after_seq]

    async def ack_events(self, instance_id, consumer_id, generation, up_to_seq):
        self.acks.append(up_to_seq)


@pytest.mark.asyncio
async def test_apply_happens_before_ack_and_replay_is_idempotent() -> None:
    rpc = FakeRPC()
    index = PrefixIndex()
    reconciler = PrefixReconciler(index, rpc, "gateway", "gen")
    assert await reconciler.rebuild("d0") == 0
    assert await reconciler.sync_once("d0") == 1
    assert rpc.acks == [1]
    # Simulate ACK response loss: reset remote-facing cursor and replay event.
    reconciler._owners["d0"] = ("e1", 0)
    assert await reconciler.sync_once("d0") == 1
    assert rpc.acks == [1, 1]
    index.assert_consistent()


@pytest.mark.asyncio
async def test_gap_disables_old_hits_rotates_generation_and_rebuilds() -> None:
    rpc = FakeRPC()
    index = PrefixIndex()
    reconciler = PrefixReconciler(index, rpc, "gateway", "old-gen")
    await reconciler.rebuild("d0")
    rpc.fail_peek = True
    rpc.report_epoch = "e2"
    assert await reconciler.sync_once("d0") == 0
    assert reconciler.generation != "old-gen"
    assert reconciler._owners["d0"] == ("e2", 0)


@pytest.mark.asyncio
async def test_epoch_owner_mismatch_fails_closed_to_full_report() -> None:
    rpc = FakeRPC()
    index = PrefixIndex()
    reconciler = PrefixReconciler(index, rpc, "gateway", "gen")
    await reconciler.rebuild("d0")
    rpc.location = CacheLocation(
        "d0", "new-epoch", "ns", "c", "text", 11, 0, 7, 4, time.monotonic()
    )
    rpc.events = [PrefixEvent("hash_added", rpc.location, 1)]
    rpc.report_epoch = "new-epoch"
    assert await reconciler.sync_once("d0") == 0
    assert reconciler._owners["d0"] == ("new-epoch", 0)


class WorldRPC(FakeRPC):
    def __init__(self):
        super().__init__()
        self.epochs = {
            "p0": "p0-e1", "p1": "p1-e1", "d0": "d0-e1", "d1": "d1-e1"
        }

    async def full_report_and_register(self, instance_id, consumer_id, generation):
        epoch = self.epochs[instance_id]
        location = CacheLocation(
            instance_id, epoch, "ns", "c", "text", 11, 0, 7, 4,
            time.monotonic(),
        )
        return PrefixReport(instance_id, epoch, 3, (location,))


@pytest.mark.asyncio
async def test_world_rebuild_publishes_four_epoch_matched_reports_together() -> None:
    rpc = WorldRPC()
    index = PrefixIndex()
    reconciler = PrefixReconciler(index, rpc, "gateway", "gen")

    publication = await reconciler.rebuild_world(rpc.epochs)

    assert publication.matches(rpc.epochs)
    assert reconciler.world_ready(rpc.epochs)
    assert len(publication.reports) == 4
    assert all(value.content_digest.startswith("sha256:") for value in publication.reports)
    index.assert_consistent()


@pytest.mark.asyncio
async def test_world_rebuild_epoch_mismatch_keeps_prior_publication() -> None:
    rpc = WorldRPC()
    index = PrefixIndex()
    reconciler = PrefixReconciler(index, rpc, "gateway", "gen")
    first = await reconciler.rebuild_world(rpc.epochs)
    prior_owners = dict(reconciler._owners)
    expected = dict(rpc.epochs)
    rpc.epochs["d1"] = "foreign-epoch"

    with pytest.raises(ValueError, match="epoch"):
        await reconciler.rebuild_world(expected)

    assert reconciler._owners == prior_owners
    assert first.matches(expected)
    assert reconciler.world_ready(expected)
    index.assert_consistent()


@pytest.mark.asyncio
async def test_world_event_gap_rotates_and_rebuilds_all_reports_before_ready() -> None:
    rpc = WorldRPC()
    index = PrefixIndex()
    reconciler = PrefixReconciler(index, rpc, "gateway", "gen")
    await reconciler.rebuild_world(rpc.epochs)
    rpc.fail_peek = True

    assert await reconciler.sync_once("p0") == 3

    assert reconciler.generation != "gen"
    assert reconciler.world_ready(rpc.epochs)
    assert reconciler.world_publication is not None
    assert reconciler.world_publication.generation == reconciler.generation
    assert len(reconciler.world_publication.reports) == 4

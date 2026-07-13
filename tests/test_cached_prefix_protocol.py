from __future__ import annotations

import pytest

from prism_serve.router.fingerprint import PromptFingerprint
from prism_serve.router.loader import cleanup_mapped_operation, load_cached_prefix
from prism_serve.router.protocol import (
    CachedPrefixDecision,
    CachedPrefixPlan,
    MappedTransferStatus,
    PreparedPrefix,
    PrefixOperationStatus,
    ResolvedPrefix,
)
from prism_serve.scheduler.scheduler import PDScheduler


class FakeRPC:
    def __init__(self, transfer, target):
        self.transfer = transfer
        self.target = target
        self.calls = []

    async def abort_mapped_prefix(self, source, target, operation_id):
        self.calls.append(("abort_mapped", operation_id))
        return self.transfer

    async def get_prefix_operation(self, target, operation_id):
        self.calls.append(("query", operation_id))
        return self.target

    async def abort_cached_prefix(self, target, operation_id):
        self.calls.append(("abort_target", operation_id))

    async def unpin_prefix(self, source, operation_id):
        self.calls.append(("unpin", operation_id))


class LoadRPC(FakeRPC):
    def __init__(self):
        super().__init__(MappedTransferStatus.COMPLETED, PrefixOperationStatus.COMMITTED)

    async def resolve_prefix(self, source, source_epoch, operation_id, expected_blocks, **identity):
        self.calls.append(("resolve", operation_id))
        return ResolvedPrefix(operation_id, source_epoch, (1, 2))

    async def prepare_prefix(self, target, target_epoch, operation_id, req_id, **kwargs):
        self.calls.append(("prepare", operation_id))
        return PreparedPrefix(operation_id, kwargs["mode"], (8, 9))

    async def transfer_cached_prefix(self, plan):
        self.calls.append(("transfer", plan.operation_id))
        return MappedTransferStatus.COMPLETED

    async def commit_cached_prefix(self, target, operation_id, plan):
        self.calls.append(("commit", operation_id))


def _case():
    scheduler = PDScheduler({})
    scheduler.register_instance("d0", "decode", max_slots=1, instance_epoch="de")
    scheduler.reserve_decode_slot("d0", "r1", "op1")
    plan = CachedPrefixPlan("op1", "r1", "s0", "d0", "se", "de", (1,), (2,), 4)
    return scheduler, plan


@pytest.mark.asyncio
async def test_fenced_prepared_aborts_then_releases() -> None:
    scheduler, plan = _case()
    rpc = FakeRPC(MappedTransferStatus.FENCED, PrefixOperationStatus.PREPARED)
    outcome = await cleanup_mapped_operation(rpc, scheduler, plan)
    assert outcome.action == "ABORTED"
    assert rpc.calls == [
        ("abort_mapped", "op1"), ("query", "op1"),
        ("abort_target", "op1"), ("unpin", "op1"),
    ]
    assert scheduler.decode_free_slots()["d0"] == 1


@pytest.mark.asyncio
async def test_completed_committed_never_rolls_back() -> None:
    scheduler, plan = _case()
    rpc = FakeRPC(MappedTransferStatus.COMPLETED, PrefixOperationStatus.COMMITTED)
    outcome = await cleanup_mapped_operation(rpc, scheduler, plan)
    assert outcome.action == "COMMITTED"
    assert ("abort_target", "op1") not in rpc.calls
    assert scheduler.decode_slot_lease("op1").state == "ACTIVE"


@pytest.mark.asyncio
async def test_unknown_quarantines_without_release_or_unpin() -> None:
    scheduler, plan = _case()
    rpc = FakeRPC(MappedTransferStatus.UNKNOWN, PrefixOperationStatus.PREPARED)
    outcome = await cleanup_mapped_operation(rpc, scheduler, plan)
    assert outcome.action == "QUARANTINED"
    assert rpc.calls == [("abort_mapped", "op1")]
    assert scheduler.decode_free_slots()["d0"] == 0


@pytest.mark.asyncio
async def test_completed_aborted_duplicate_cleanup_is_idempotent() -> None:
    scheduler, plan = _case()
    rpc = FakeRPC(MappedTransferStatus.COMPLETED, PrefixOperationStatus.ABORTED)
    first = await cleanup_mapped_operation(rpc, scheduler, plan)
    second = await cleanup_mapped_operation(rpc, scheduler, plan)
    assert first.action == second.action == "ABORTED"
    assert scheduler.decode_free_slots()["d0"] == 1


@pytest.mark.asyncio
async def test_load_runs_transaction_and_preserves_suffix() -> None:
    rpc = LoadRPC()
    fingerprint = PromptFingerprint(
        "ns", "compat", "text", tuple(range(10)), (11, 22), 4
    )
    decision = CachedPrefixDecision("s0", "se", "d0", 2, 8, 16, 4.0)
    plan = await load_cached_prefix(
        rpc, req_id="r1", operation_id="op1", fingerprint=fingerprint,
        sampling_params={"temperature": 0}, decision=decision, target_epoch="de",
    )
    assert plan.token_ids[plan.cached_prefix_tokens:] == (8, 9)
    assert rpc.calls == [
        ("resolve", "op1"), ("prepare", "op1"), ("transfer", "op1"),
        ("commit", "op1"), ("unpin", "op1"),
    ]


@pytest.mark.asyncio
async def test_local_cleanup_does_not_require_mapped_abort() -> None:
    scheduler, base = _case()
    plan = CachedPrefixPlan(
        base.operation_id, base.req_id, "d0", "d0", "de", "de",
        (1,), (), 4, mode="local_reuse",
    )
    rpc = FakeRPC(MappedTransferStatus.UNKNOWN, PrefixOperationStatus.ABORTED)
    outcome = await cleanup_mapped_operation(rpc, scheduler, plan)
    assert outcome.action == "ABORTED"
    assert ("abort_mapped", "op1") not in rpc.calls
    assert scheduler.decode_free_slots()["d0"] == 1

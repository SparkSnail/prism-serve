"""Operation-scoped cached-prefix loading and compensation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from prism_serve.router.protocol import (
    CachedPrefixPlan,
    CachedPrefixDecision,
    CleanupOutcome,
    ExpectedPrefixBlock,
    MappedTransferStatus,
    PreparedPrefix,
    PrefixOperationStatus,
    ResolvedPrefix,
)
from prism_serve.router.fingerprint import PromptFingerprint
from prism_serve.scheduler.scheduler import PDScheduler


class PrefixCacheRPC(Protocol):
    async def resolve_prefix(
        self, source: str, source_epoch: str, operation_id: str,
        expected_blocks: list[ExpectedPrefixBlock], *, namespace: str,
        kv_compatibility_id: str, request_context_digest: str,
    ) -> ResolvedPrefix | None: ...

    async def prepare_prefix(
        self, target: str, target_epoch: str, operation_id: str, req_id: str,
        *, mode: str, block_count: int, token_ids: tuple[int, ...],
        sampling_params: dict,
    ) -> PreparedPrefix | None: ...

    async def transfer_cached_prefix(self, plan: CachedPrefixPlan) -> MappedTransferStatus: ...

    async def commit_cached_prefix(
        self, target: str, operation_id: str, plan: CachedPrefixPlan
    ) -> None: ...

    async def abort_mapped_prefix(
        self, source: str, target: str, operation_id: str
    ) -> MappedTransferStatus: ...

    async def get_prefix_operation(
        self, target: str, operation_id: str
    ) -> PrefixOperationStatus: ...

    async def abort_cached_prefix(self, target: str, operation_id: str) -> None: ...

    async def unpin_prefix(self, source: str, operation_id: str) -> None: ...

    async def abort_suffix_prefill(
        self, target: str, operation_id: str
    ) -> bool: ...


class PrefixLoadError(RuntimeError):
    def __init__(self, message: str, plan: CachedPrefixPlan | None = None):
        super().__init__(message)
        self.plan = plan


@dataclass(slots=True)
class PrefixLoadContext:
    operation_id: str
    req_id: str
    source_instance: str
    target_instance: str
    stage: str = "STARTED"
    plan: CachedPrefixPlan | None = None
    source_pinned: bool = False
    target_pending: bool = False
    on_change: Callable[[], None] | None = None

    def changed(self) -> None:
        if self.on_change is not None:
            self.on_change()


async def load_cached_prefix(
    rpc: PrefixCacheRPC,
    *,
    req_id: str,
    operation_id: str,
    fingerprint: PromptFingerprint,
    sampling_params: dict,
    decision: CachedPrefixDecision,
    target_epoch: str,
    context: PrefixLoadContext | None = None,
) -> CachedPrefixPlan:
    """Run resolve -> prepare -> mapped copy/local reuse -> commit -> unpin."""
    block_count = decision.cached_prefix_blocks
    expected = [
        ExpectedPrefixBlock(
            index,
            fingerprint.chain_hashes[index],
            fingerprint.token_ids[
                index * fingerprint.block_size:(index + 1) * fingerprint.block_size
            ],
        )
        for index in range(block_count)
    ]
    if context is None:
        context = PrefixLoadContext(
            operation_id, req_id, decision.source_instance, decision.decode_instance
        )
    context.stage = "RESOLVING"
    resolved = await rpc.resolve_prefix(
        decision.source_instance, decision.source_epoch, operation_id, expected,
        namespace=fingerprint.namespace,
        kv_compatibility_id=fingerprint.kv_compatibility_id,
        request_context_digest=fingerprint.request_context_digest,
    )
    if resolved is None or len(resolved.src_block_ids) != block_count:
        raise PrefixLoadError("source prefix became stale")
    context.source_pinned = True
    context.changed()
    context.stage = "RESOLVED"
    mode = (
        "local_reuse"
        if decision.source_instance == decision.decode_instance
        else "remote_transfer"
    )
    plan = CachedPrefixPlan(
        operation_id=operation_id,
        req_id=req_id,
        source_instance=decision.source_instance,
        target_instance=decision.decode_instance,
        source_epoch=decision.source_epoch,
        target_epoch=target_epoch,
        src_block_ids=resolved.src_block_ids,
        dst_block_ids=(),
        cached_prefix_tokens=decision.cached_prefix_tokens,
        namespace=fingerprint.namespace,
        kv_compatibility_id=fingerprint.kv_compatibility_id,
        request_context_digest=fingerprint.request_context_digest,
        token_ids=fingerprint.token_ids,
        sampling_params=sampling_params,
        mode=mode,
    )
    context.plan = plan
    context.stage = "PREPARING"
    context.target_pending = True
    context.changed()
    prepared = await rpc.prepare_prefix(
        decision.decode_instance, target_epoch, operation_id, req_id,
        mode=mode, block_count=block_count,
        token_ids=fingerprint.token_ids, sampling_params=sampling_params,
    )
    if prepared is None:
        raise PrefixLoadError("target prefix prepare failed")
    plan = CachedPrefixPlan(
        operation_id=plan.operation_id,
        req_id=plan.req_id,
        source_instance=plan.source_instance,
        target_instance=plan.target_instance,
        source_epoch=plan.source_epoch,
        target_epoch=plan.target_epoch,
        src_block_ids=plan.src_block_ids,
        dst_block_ids=prepared.dst_block_ids,
        cached_prefix_tokens=plan.cached_prefix_tokens,
        namespace=plan.namespace,
        kv_compatibility_id=plan.kv_compatibility_id,
        request_context_digest=plan.request_context_digest,
        token_ids=plan.token_ids,
        sampling_params=plan.sampling_params,
        mode=mode,
    )
    context.plan = plan
    context.stage = "PREPARED"
    if mode == "remote_transfer":
        context.stage = "TRANSFERRING"
        transfer = await rpc.transfer_cached_prefix(plan)
        if transfer != MappedTransferStatus.COMPLETED:
            raise PrefixLoadError(f"mapped transfer did not complete: {transfer}", plan)
    context.stage = "COMMITTING"
    await rpc.commit_cached_prefix(decision.decode_instance, operation_id, plan)
    context.stage = "COMMITTED"
    context.target_pending = False
    context.changed()
    await rpc.unpin_prefix(decision.source_instance, operation_id)
    context.source_pinned = False
    context.changed()
    return plan


async def cleanup_load_context(
    rpc: PrefixCacheRPC,
    scheduler: PDScheduler,
    context: PrefixLoadContext,
) -> CleanupOutcome:
    """Compensate every load stage, including response-loss boundaries."""
    if context.stage in {"STARTED", "RESOLVING", "RESOLVED"}:
        await rpc.unpin_prefix(context.source_instance, context.operation_id)
        _release_reserved_or_quarantined(scheduler, context.operation_id)
        return CleanupOutcome("ABORTED", PrefixOperationStatus.ABORTED)

    plan = context.plan
    assert plan is not None
    if context.stage in {"PREPARING", "PREPARED"}:
        target = await rpc.get_prefix_operation(
            context.target_instance, context.operation_id
        )
        if target == PrefixOperationStatus.COMMITTED:
            scheduler.commit_decode_slot(context.operation_id)
            await rpc.unpin_prefix(context.source_instance, context.operation_id)
            context.source_pinned = False
            context.target_pending = False
            context.changed()
            return CleanupOutcome("COMMITTED", target)
        if target == PrefixOperationStatus.UNKNOWN:
            scheduler.quarantine_decode_slot(context.operation_id)
            return CleanupOutcome("QUARANTINED", target)
        if target == PrefixOperationStatus.PREPARED:
            await rpc.abort_cached_prefix(context.target_instance, context.operation_id)
        context.target_pending = False
        context.changed()
        await rpc.unpin_prefix(context.source_instance, context.operation_id)
        context.source_pinned = False
        context.changed()
        _release_reserved_or_quarantined(scheduler, context.operation_id)
        return CleanupOutcome("ABORTED", PrefixOperationStatus.ABORTED)

    outcome = await cleanup_mapped_operation(rpc, scheduler, plan)
    if outcome.action != "QUARANTINED":
        context.source_pinned = False
        context.target_pending = False
        context.changed()
    return outcome


def _release_reserved_or_quarantined(
    scheduler: PDScheduler, operation_id: str
) -> None:
    lease = scheduler.decode_slot_lease(operation_id)
    if lease is not None and lease.state == "QUARANTINED":
        scheduler.release_quarantined_decode_slot(operation_id)
    else:
        scheduler.release_decode_slot(operation_id)


async def cleanup_mapped_operation(
    rpc: PrefixCacheRPC,
    scheduler: PDScheduler,
    plan: CachedPrefixPlan,
) -> CleanupOutcome:
    """Apply the reviewed partial-cleanup matrix for one old operation."""
    if plan.mode == "local_reuse":
        target = await rpc.get_prefix_operation(plan.target_instance, plan.operation_id)
        if target == PrefixOperationStatus.COMMITTED:
            scheduler.commit_decode_slot(plan.operation_id)
            await rpc.unpin_prefix(plan.source_instance, plan.operation_id)
            return CleanupOutcome("COMMITTED", target)
        if target == PrefixOperationStatus.PREPARED:
            await rpc.abort_cached_prefix(plan.target_instance, plan.operation_id)
        elif target == PrefixOperationStatus.UNKNOWN:
            scheduler.quarantine_decode_slot(plan.operation_id)
            return CleanupOutcome("QUARANTINED", target)
        await rpc.unpin_prefix(plan.source_instance, plan.operation_id)
        _release_reserved_or_quarantined(scheduler, plan.operation_id)
        return CleanupOutcome("ABORTED", PrefixOperationStatus.ABORTED)

    transfer = await rpc.abort_mapped_prefix(
        plan.source_instance, plan.target_instance, plan.operation_id
    )
    if transfer == MappedTransferStatus.UNKNOWN:
        scheduler.quarantine_decode_slot(plan.operation_id)
        return CleanupOutcome("QUARANTINED", PrefixOperationStatus.UNKNOWN)

    target = await rpc.get_prefix_operation(plan.target_instance, plan.operation_id)
    if target == PrefixOperationStatus.COMMITTED:
        scheduler.commit_decode_slot(plan.operation_id)
        await rpc.unpin_prefix(plan.source_instance, plan.operation_id)
        return CleanupOutcome("COMMITTED", target)
    if target == PrefixOperationStatus.UNKNOWN:
        scheduler.quarantine_decode_slot(plan.operation_id)
        return CleanupOutcome("QUARANTINED", target)
    if target == PrefixOperationStatus.PREPARED:
        if transfer not in {MappedTransferStatus.FENCED, MappedTransferStatus.COMPLETED}:
            scheduler.quarantine_decode_slot(plan.operation_id)
            return CleanupOutcome("QUARANTINED", PrefixOperationStatus.UNKNOWN)
        await rpc.abort_cached_prefix(plan.target_instance, plan.operation_id)

    # ABORTED is the duplicate cleanup path; target blocks are already safe.
    await rpc.unpin_prefix(plan.source_instance, plan.operation_id)
    _release_reserved_or_quarantined(scheduler, plan.operation_id)
    return CleanupOutcome("ABORTED", PrefixOperationStatus.ABORTED)

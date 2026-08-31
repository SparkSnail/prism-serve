"""In-process PrefixCacheRPC adapter for CPU integration tests."""

from __future__ import annotations

import time

from prism_serve.router.prefix_index import CacheLocation, PrefixEvent
from prism_serve.router.protocol import (
    CachedPrefixPlan,
    ExpectedPrefixBlock,
    MappedTransferStatus,
    PreparedPrefix,
    PrefixOperationStatus,
    ResolvedPrefix,
)
from prism_serve.router.reconciler import PrefixReport


class InProcessPrefixCacheRPC:
    """Execute the cross-layer contract against injected infer service objects."""

    def __init__(self, services: dict[str, object]):
        self.services = services

    def _service(self, instance_id: str, epoch: str | None = None):
        service = self.services[instance_id]
        if epoch is not None and service.instance_epoch != epoch:
            raise ValueError("infer instance epoch changed")
        return service

    async def resolve_prefix(
        self, source, source_epoch, operation_id, expected_blocks, **identity
    ):
        service = self._service(source, source_epoch)
        resolved = service.resolve_prefix(
            operation_id,
            [(item.chain_hash, list(item.token_ids)) for item in expected_blocks],
            **identity,
        )
        return None if resolved is None else ResolvedPrefix(
            operation_id, source_epoch, tuple(resolved)
        )

    async def prepare_prefix(
        self, target, target_epoch, operation_id, req_id, **kwargs
    ):
        try:
            from prism_infer.sampling_params import SamplingParams
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "InProcessPrefixCacheRPC requires the optional prism-infer "
                "adapter; install prism-serve[infer]"
            ) from exc

        service = self._service(target, target_epoch)
        params = kwargs["sampling_params"]
        sampling = SamplingParams(**params) if isinstance(params, dict) else params
        operation = service.prepare(
            operation_id, req_id, mode=kwargs["mode"],
            block_count=kwargs["block_count"], token_ids=list(kwargs["token_ids"]),
            sampling_params=sampling,
        )
        return None if operation is None else PreparedPrefix(
            operation_id, operation.mode, tuple(operation.dst_block_ids)
        )

    async def transfer_cached_prefix(self, plan: CachedPrefixPlan):
        try:
            from prism_infer.engine.kv_transfer import MappedPrefixTransferReq
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "InProcessPrefixCacheRPC requires the optional prism-infer "
                "adapter; install prism-serve[infer]"
            ) from exc

        source = self._service(plan.source_instance, plan.source_epoch)
        target = self._service(plan.target_instance, plan.target_epoch)
        request = MappedPrefixTransferReq(
            plan.operation_id, plan.req_id,
            plan.source_instance, plan.source_epoch,
            plan.target_instance, plan.target_epoch,
            plan.src_block_ids, plan.dst_block_ids,
            plan.namespace, plan.kv_compatibility_id, plan.request_context_digest,
        )
        return MappedTransferStatus(target.transfer_from(source, request).value)

    async def commit_cached_prefix(self, target, operation_id, plan):
        service = self._service(target, plan.target_epoch)
        service.commit(
            operation_id, namespace=plan.namespace,
            kv_compatibility_id=plan.kv_compatibility_id,
            request_context_digest=plan.request_context_digest,
            cached_prefix_tokens=plan.cached_prefix_tokens,
        )

    async def abort_mapped_prefix(self, source, target, operation_id):
        service = self._service(target)
        status = service.transfers.status(operation_id)
        if status.value == "COMPLETED":
            return MappedTransferStatus.COMPLETED
        result = service.transfers.abort_result(
            operation_id, source_fenced=True, target_fenced=True
        )
        return MappedTransferStatus(result.value)

    async def get_prefix_operation(self, target, operation_id):
        return PrefixOperationStatus(self._service(target).status(operation_id).value)

    async def abort_cached_prefix(self, target, operation_id):
        self._service(target).abort(operation_id)

    async def unpin_prefix(self, source, operation_id):
        self._service(source).unpin(operation_id)

    async def abort_suffix_prefill(self, target, operation_id):
        return self._service(target).abort_sequence(operation_id)

    async def get_prefix_resource_counts(self, instance):
        return self._service(instance).resource_counts()

    async def full_report_and_register(self, instance_id, consumer_id, generation):
        report = self._service(instance_id).block_manager.full_report_and_register(
            consumer_id, generation
        )
        now = time.monotonic()
        return PrefixReport(
            report.instance_id, report.instance_epoch, report.snapshot_seq_no,
            tuple(self._location(event, now) for event in report.locations),
        )

    async def peek_events(
        self, instance_id, consumer_id, generation, after_seq, limit
    ):
        events = self._service(instance_id).block_manager.peek_events(
            consumer_id, generation, after_seq, limit
        )
        now = time.monotonic()
        return [PrefixEvent(event.kind, self._location(event, now), event.seq_no) for event in events]

    async def ack_events(self, instance_id, consumer_id, generation, up_to_seq):
        self._service(instance_id).block_manager.ack_events(
            consumer_id, generation, up_to_seq
        )

    @staticmethod
    def _location(event, received_at):
        return CacheLocation(
            event.instance_id, event.instance_epoch, event.namespace,
            event.kv_compatibility_id, event.request_context_digest,
            event.chain_hash, event.block_index, event.block_id,
            event.prefix_tokens, received_at,
        )

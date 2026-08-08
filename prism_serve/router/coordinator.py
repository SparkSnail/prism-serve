"""Integrate non-blocking affinity loads with the schedule loop."""

from __future__ import annotations

import asyncio
import logging
import uuid

from prism_serve.router.loader import (
    PrefixCacheRPC,
    PrefixLoadContext,
    PrefixLoadError,
    cleanup_load_context,
    load_cached_prefix,
)
from prism_serve.router.router import AffinityRouter
from prism_serve.scheduler.sequence_state import SeqState

logger = logging.getLogger(__name__)


class AffinityCoordinator:
    def __init__(
        self, router: AffinityRouter, rpc: PrefixCacheRPC, queue, config: dict,
        metrics=None, operation_allocator=None,
    ):
        self.router = router
        self.rpc = rpc
        self.queue = queue
        self.config = config
        self.metrics = metrics
        self.operation_allocator = operation_allocator
        self._tasks: dict[str, asyncio.Task] = {}
        self._contexts: dict[str, PrefixLoadContext] = {}
        self._recovery_tasks: dict[str, asyncio.Task] = {}
        self._source_release_tasks: dict[str, asyncio.Task] = {}

    def _record_leases(self, scheduler) -> None:
        if self.metrics is None:
            return
        counts = scheduler.decode_slot_lease_counts()
        for state, value in counts.items():
            self.metrics.gauge("decode_slot_leases", value, labels={"state": state})
        self.metrics.gauge("decode_slot_quarantined", counts["QUARANTINED"])

    async def _cleanup(self, scheduler, context: PrefixLoadContext):
        deadline = asyncio.get_running_loop().time() + self.config.get(
            "prefix_operation_watchdog_s", 30.0
        )
        last_error = None
        while asyncio.get_running_loop().time() < deadline:
            try:
                cleanup = (
                    self.rpc.cleanup_prefix_context(scheduler, context)
                    if getattr(self.rpc, "week12_network_control", False) is True
                    else cleanup_load_context(self.rpc, scheduler, context)
                )
                outcome = await asyncio.wait_for(
                    cleanup,
                    timeout=self.config.get("prefix_load_timeout_s", 5.0),
                )
                self._record_leases(scheduler)
                return outcome
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                lease = scheduler.decode_slot_lease(context.operation_id)
                if lease is not None and lease.state not in {"RELEASED", "QUARANTINED"}:
                    scheduler.quarantine_decode_slot(context.operation_id)
                self._record_leases(scheduler)
                await asyncio.sleep(min(0.05, max(0, deadline - asyncio.get_running_loop().time())))
        logger.error(
            "affinity cleanup watchdog expired operation=%s: %r",
            context.operation_id, last_error,
        )
        return None

    def _retain_for_recovery(self, scheduler, tracker, context, req_id) -> None:
        if context.operation_id in self._recovery_tasks:
            return
        task = asyncio.create_task(
            self._recover_until_proven(scheduler, tracker, context, req_id)
        )
        self._recovery_tasks[context.operation_id] = task
        task.add_done_callback(
            lambda done, op=context.operation_id: self._recovery_tasks.pop(op, None)
        )

    def _release_source_in_background(self, context: PrefixLoadContext) -> None:
        if not context.source_pinned \
                or context.operation_id in self._source_release_tasks:
            return
        task = asyncio.create_task(self._release_source_until_proven(context))
        self._source_release_tasks[context.operation_id] = task
        task.add_done_callback(
            lambda done, op=context.operation_id: self._source_release_tasks.pop(
                op, None
            )
        )

    async def _release_source_until_proven(
        self, context: PrefixLoadContext
    ) -> None:
        while context.source_pinned:
            try:
                await asyncio.wait_for(
                    self.rpc.unpin_prefix(
                        context.source_instance, context.operation_id
                    ),
                    timeout=self.config.get("prefix_load_timeout_s", 5.0),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "source pin release remains ambiguous operation=%s",
                    context.operation_id,
                    exc_info=True,
                )
                await asyncio.sleep(
                    self.config.get("prefix_reconcile_interval_s", 1.0)
                )
                continue
            context.source_pinned = False
            context.changed()
            return

    def _release_source_after_commit(self, context: PrefixLoadContext) -> None:

        # cleanup plan until the request itself is terminal.  Its
        # NetworkControlRPC/ResourceReleaseEvaluator is the sole release
        # authority.  The background path remains only for the legacy RPC
        # surface, which has no request-level generic finalizer.
        if getattr(self.rpc, "week12_network_control", False) is True:
            return
        self._release_source_in_background(context)

    def terminal_cleanup_complete(self, req) -> None:
        """Forget a committed context only after request-level release proof."""
        context = self._contexts.pop(req.active_operation_id, None)
        if context is None:
            return
        context.source_pinned = False
        context.target_pending = False
        context.changed()

    async def _recover_until_proven(self, scheduler, tracker, context, req_id) -> None:
        while True:
            outcome = await self._cleanup(scheduler, context)
            if outcome is None or outcome.action == "QUARANTINED":
                await asyncio.sleep(self.config.get("prefix_reconcile_interval_s", 1.0))
                continue
            current = tracker.get(req_id)
            if current is not None and current.active_operation_id == context.operation_id:
                if outcome.action == "ABORTED" and current.state == SeqState.AFFINITY_LOADING:
                    tracker.transition(req_id, SeqState.WAITING, active_operation_id="")
                elif outcome.action == "ABORTED" and current.state == SeqState.PREFIX_PREFILLING:
                    tracker.transition(req_id, SeqState.ABORTED)
                    tracker.remove(req_id)
                elif outcome.action == "COMMITTED" and current.state == SeqState.AFFINITY_LOADING:
                    tracker.transition(req_id, SeqState.PREFIX_PREFILLING)
            self._contexts.pop(context.operation_id, None)
            self._record_leases(scheduler)
            return

    async def try_start(self, req, scheduler, tracker) -> bool:
        fingerprint = req.fingerprint
        free = scheduler.decode_free_slots()
        candidates = [instance for instance, slots in free.items() if slots > 0]
        if not candidates:
            return False
        per_token_ms = self.config.get("prefill_ms_per_token", 0.05)
        decisions = self.router.iter_decisions(
            fingerprint,
            candidates,
            scheduler.epoch_matches,
            len(fingerprint.token_ids) * per_token_ms,
            per_token_ms,
            required_source=(req.correctness_source_instance or None),
            required_target=(req.correctness_target_instance or None),
            required_cached_prefix_blocks=(
                req.correctness_cached_prefix_tokens // fingerprint.block_size
                if req.correctness_cached_prefix_tokens else None
            ),
        )
        for decision in decisions:
            operation_id = uuid.uuid4().hex
            lease = scheduler.reserve_decode_slot(
                decision.decode_instance, req.req_id, operation_id
            )
            if lease is None:
                continue
            self._record_leases(scheduler)
            tracker.transition(
                req.req_id, SeqState.AFFINITY_LOADING,
                active_operation_id=operation_id,
                decode_slot_lease_id=lease.lease_id,
                cache_source=decision.source_instance,
                decode_instance=decision.decode_instance,
                decode_instance_epoch=scheduler.instance_epoch(decision.decode_instance),
                cached_prefix_tokens=decision.cached_prefix_tokens,
            )
            task = asyncio.create_task(
                self._run(req.req_id, operation_id, decision, scheduler, tracker)
            )
            self._tasks[operation_id] = task
            context = PrefixLoadContext(
                operation_id, req.req_id, decision.source_instance,
                decision.decode_instance,
            )
            context.on_change = lambda: self._record_leases(scheduler)
            self._contexts[operation_id] = context
            task.add_done_callback(
                lambda done, op=operation_id: self._tasks.pop(op, None)
            )
            return True

        if req.correctness_path in {"same_instance", "cross_instance"}:
            # The operator seam must never silently fall back to cold.  It waits
            # for the exact observed cache location until the client deadline.
            return True
        return bool(decisions) and self._should_wait(req)

    def _should_wait(self, req) -> bool:
        elapsed_ms = (asyncio.get_running_loop().time() - req.arrived_at) * 1000
        limit = min(
            self.config.get("locality_wait_ms", 20),
            self.config.get("max_affinity_wait_ms", 100),
        )
        return elapsed_ms < limit

    async def _run(self, req_id, operation_id, decision, scheduler, tracker) -> None:
        req = tracker.get(req_id)
        if req is None:
            return
        plan = None
        context = self._contexts[operation_id]
        try:
            plan = await asyncio.wait_for(
                load_cached_prefix(
                    self.rpc,
                    req_id=req_id,
                    operation_id=operation_id,
                    fingerprint=req.fingerprint,
                    sampling_params=req.sampling_params,
                    decision=decision,
                    target_epoch=req.decode_instance_epoch,
                    context=context,
                    defer_source_release=True,
                ),
                timeout=self.config.get("prefix_load_timeout_s", 5.0),
            )
            current = tracker.get(req_id)
            if current is None or current.state != SeqState.AFFINITY_LOADING \
                    or current.active_operation_id != operation_id:
                await self._cleanup(scheduler, context)
                return
            scheduler.commit_decode_slot(operation_id)
            self._record_leases(scheduler)
            tracker.transition(req_id, SeqState.PREFIX_PREFILLING)

            # cross-instance source pin remains held through decode and joins
            # the request-terminal generic finalize plan.
            self._release_source_after_commit(context)
            if self.metrics is not None:
                self.metrics.increment(
                    "cached_prefix_tokens_total", plan.cached_prefix_tokens,
                )
                self.metrics.observe(
                    "cached_prefix_transfer_bytes", decision.transfer_bytes,
                )
                self.metrics.increment(
                    "suffix_prefill_tokens_total",
                    len(plan.token_ids) - plan.cached_prefix_tokens,
                )
            suffix_subject = (
                self.queue.dispatch_suffix_subject(decision.decode_instance)
                if hasattr(self.queue, "dispatch_suffix_subject")
                else self.queue.dispatch_subject(decision.decode_instance)
            )
            payload = {
                "kind": "dispatch_suffix_prefill",
                "req_id": req_id,
                "owner_id": self.queue.owner_id,
                "operation_id": operation_id,
                "instance_epoch": req.decode_instance_epoch,
                "cached_prefix_tokens": plan.cached_prefix_tokens,
                "remaining_token_ids": list(plan.token_ids[plan.cached_prefix_tokens:]),
                "sampling_params": plan.sampling_params,
                "suffix_prefill_done_subject": self.queue.reply_subject("suffix_prefill_done"),
                "reply_subject": self.queue.reply_subject("suffix_prefill_done"),
                "first_token_subject": self.queue.reply_subject("first_token"),
                "decode_progress_subject": self.queue.reply_subject("decode_progress"),
                "decode_done_subject": self.queue.reply_subject("decode_done"),
            }
            command = payload
            endpoint_ref = None
            if self.operation_allocator is not None:
                from dataclasses import asdict
                endpoint_ref = self.operation_allocator.allocate(
                    target_instance=decision.decode_instance,
                    target_worker_epoch=req.decode_instance_epoch,
                    operation_id=operation_id,
                    payload=payload,
                )
                req.suffix_operation_ref = endpoint_ref
                if getattr(self.rpc, "week12_network_control", False) is True:
                    self.rpc.remember_external_ref(
                        "suffix", decision.decode_instance, operation_id, endpoint_ref
                    )
                command = {"schema_version": 1,
                           "endpoint_ref": asdict(endpoint_ref),
                           "payload": payload}
            if endpoint_ref is not None and getattr(
                self.rpc, "week12_network_control", False
            ) is True:
                self.rpc.mark_external_ref_attempted(endpoint_ref)
            try:
                await self.queue.publish(suffix_subject, command)
            except BaseException:
                if endpoint_ref is not None and getattr(
                    self.rpc, "week12_network_control", False
                ) is True:
                    self.rpc.mark_external_ref_ambiguous(endpoint_ref)
                raise
        except asyncio.CancelledError:
            await asyncio.shield(self._cleanup(scheduler, context))
            raise
        except (PrefixLoadError, asyncio.TimeoutError):
            outcome = await self._cleanup(scheduler, context)
            self._record_leases(scheduler)
            if self.metrics is not None:
                self.metrics.increment(
                    "affinity_fallback_total",
                    labels={"reason": "load_timeout_or_failure"},
                )
            current = tracker.get(req_id)
            if outcome is not None and outcome.action == "ABORTED" \
                    and current is not None and current.state == SeqState.AFFINITY_LOADING \
                    and current.active_operation_id == operation_id:
                tracker.transition(req_id, SeqState.WAITING, active_operation_id="")
            elif outcome is None:
                self._retain_for_recovery(scheduler, tracker, context, req_id)
        except Exception:
            logger.exception("affinity load failed req=%s operation=%s", req_id, operation_id)
            if context.stage == "COMMITTED":
                # The request was already promoted before suffix publication.

                # must not grant an independent early source-unpin authority.
                self._release_source_after_commit(context)
                self._record_leases(scheduler)
                return
            outcome = await self._cleanup(scheduler, context)
            if outcome is None:
                self._retain_for_recovery(scheduler, tracker, context, req_id)
            self._record_leases(scheduler)

    async def shutdown(self) -> None:
        tasks = (
            list(self._tasks.values())
            + list(self._recovery_tasks.values())
            + list(self._source_release_tasks.values())
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def abort_suffix(self, req) -> bool:
        try:
            return bool(await asyncio.wait_for(
                self.rpc.abort_suffix_prefill(
                    req.decode_instance, req.active_operation_id
                ),
                timeout=self.config.get("prefix_load_timeout_s", 5.0),
            ))
        except (asyncio.TimeoutError, ConnectionError):
            return False

    async def cleanup_suffix(self, req, scheduler, tracker):
        """Use the correlated-ref evaluator; UNKNOWN keeps lease/tracker quarantined."""
        context = self._contexts.get(req.active_operation_id)
        if context is None:
            return None
        outcome = await self._cleanup(scheduler, context)
        if outcome is None or outcome.action == "QUARANTINED":
            self._retain_for_recovery(
                scheduler, tracker, context, req.req_id
            )
            return None
        self._contexts.pop(context.operation_id, None)
        return outcome

"""Reconcile prefix directories with full reports and acknowledged events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import uuid
import asyncio
import logging

from prism_serve.router.prefix_index import (
    CacheLocation,
    FullReportRequired,
    PrefixEvent,
    PrefixIndex,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PrefixReport:
    instance_id: str
    instance_epoch: str
    snapshot_seq_no: int
    locations: tuple[CacheLocation, ...]


class PrefixDirectoryRPC(Protocol):
    async def full_report_and_register(
        self, instance_id: str, consumer_id: str, generation: str
    ) -> PrefixReport: ...

    async def peek_events(
        self, instance_id: str, consumer_id: str, generation: str,
        after_seq: int, limit: int,
    ) -> list[PrefixEvent]: ...

    async def ack_events(
        self, instance_id: str, consumer_id: str, generation: str, up_to_seq: int
    ) -> None: ...


class PrefixReconciler:
    def __init__(
        self,
        index: PrefixIndex,
        rpc: PrefixDirectoryRPC,
        consumer_id: str,
        generation: str,
        metrics=None,
    ) -> None:
        self.index = index
        self.rpc = rpc
        self.consumer_id = consumer_id
        self.generation = generation
        self.metrics = metrics
        self._owners: dict[str, tuple[str, int]] = {}
        self._resource_counts: dict[str, dict[str, int]] = {}

    async def rebuild(self, instance_id: str) -> int:
        report = await self.rpc.full_report_and_register(
            instance_id, self.consumer_id, self.generation
        )
        owner = (report.instance_id, report.instance_epoch)
        self.index.remove_instance(instance_id)
        self.index.install_full_report(owner, report.snapshot_seq_no, report.locations)
        self._owners[instance_id] = (report.instance_epoch, report.snapshot_seq_no)
        if self.metrics is not None:
            self.metrics.increment(
                "prefix_full_report_total", labels={"reason": "startup_or_rebuild"}
            )
        return report.snapshot_seq_no

    async def sync_once(self, instance_id: str, limit: int = 1024) -> int:
        if self.metrics is not None and hasattr(self.rpc, "get_prefix_resource_counts"):
            counts = await self.rpc.get_prefix_resource_counts(instance_id)
            self._resource_counts[instance_id] = counts
            self.metrics.gauge(
                "prefix_transfer_pins",
                sum(item["transfer_pins"] for item in self._resource_counts.values()),
            )
            self.metrics.gauge(
                "prefix_pending_allocations",
                sum(item["pending_allocations"] for item in self._resource_counts.values()),
            )
        if instance_id not in self._owners:
            return await self.rebuild(instance_id)
        epoch, cursor = self._owners[instance_id]
        try:
            events = await self.rpc.peek_events(
                instance_id, self.consumer_id, self.generation, cursor, limit
            )
            applied = self.index.apply_events((instance_id, epoch), events)
            # ACK only after every local mutation succeeded.
            await self.rpc.ack_events(
                instance_id, self.consumer_id, self.generation, applied
            )
            self._owners[instance_id] = (epoch, applied)
            return applied
        except (FullReportRequired, ValueError):
            # Never route from a directory whose cursor/epoch is uncertain.
            self.index.remove_instance(instance_id)
            self._owners.pop(instance_id, None)
            self.generation = uuid.uuid4().hex
            if self.metrics is not None:
                self.metrics.increment(
                    "prefix_event_gap_total", labels={"reason": "gap_or_epoch"}
                )
            return await self.rebuild(instance_id)

    async def run(self, instances_provider, interval_s: float) -> None:
        """Continuously reconcile current worker incarnations; failures stay fail-closed."""
        assert interval_s > 0, f"invalid prefix poll interval: {interval_s=}"
        while True:
            for instance_id in tuple(instances_provider()):
                try:
                    await self.sync_once(instance_id)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.index.remove_instance(instance_id)
                    self._owners.pop(instance_id, None)
                    logger.warning("prefix reconcile failed instance=%s: %s", instance_id, exc)
            await asyncio.sleep(interval_s)

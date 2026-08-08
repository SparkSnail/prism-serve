"""Reconcile prefix directories with full reports and acknowledged events."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Protocol
import uuid

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


@dataclass(slots=True, frozen=True)
class PrefixWorldReportEvidence:
    instance_id: str
    instance_epoch: str
    snapshot_seq_no: int
    location_count: int
    content_digest: str


@dataclass(slots=True, frozen=True)
class PrefixWorldPublication:
    consumer_id: str
    generation: str
    reports: tuple[PrefixWorldReportEvidence, ...]

    def matches(self, expected_epochs: dict[str, str]) -> bool:
        return {
            value.instance_id: value.instance_epoch for value in self.reports
        } == expected_epochs


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
        self._expected_epochs: dict[str, str] = {}
        self.world_publication: PrefixWorldPublication | None = None

    @staticmethod
    def _report_digest(report: PrefixReport) -> str:
        locations = [
            {
                "instance_id": value.instance_id,
                "instance_epoch": value.instance_epoch,
                "namespace": value.namespace,
                "kv_compatibility_id": value.kv_compatibility_id,
                "request_context_digest": value.request_context_digest,
                "chain_hash": value.chain_hash,
                "block_index": value.block_index,
                "block_id": value.block_id,
                "prefix_tokens": value.prefix_tokens,
            }
            for value in report.locations
        ]
        locations.sort(key=lambda value: json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ))
        payload = {
            "instance_id": report.instance_id,
            "instance_epoch": report.instance_epoch,
            "snapshot_seq_no": report.snapshot_seq_no,
            "locations": locations,
        }
        return "sha256:" + hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()

    def world_ready(self, expected_epochs: dict[str, str]) -> bool:
        owners_match = {
            instance_id: value[0] for instance_id, value in self._owners.items()
        } == expected_epochs
        return bool(
            owners_match
            and self.world_publication is not None
            and self.world_publication.matches(expected_epochs)
        )

    async def rebuild_world(
        self, expected_epochs: dict[str, str]
    ) -> PrefixWorldPublication:
        if not expected_epochs:
            raise ValueError("prefix full-report world cannot be empty")
        instances = tuple(sorted(expected_epochs))
        reports = await asyncio.gather(*(
            self.rpc.full_report_and_register(
                instance_id, self.consumer_id, self.generation
            )
            for instance_id in instances
        ))
        by_instance: dict[str, PrefixReport] = {}
        for requested, report in zip(instances, reports):
            if report.instance_id != requested:
                raise ValueError("prefix full report returned another instance")
            if report.instance_epoch != expected_epochs[requested]:
                raise ValueError("prefix full report epoch does not match current world")
            if report.instance_id in by_instance:
                raise ValueError("duplicate prefix full report instance")
            by_instance[report.instance_id] = report

        expected_owners = {
            (instance_id, epoch) for instance_id, epoch in expected_epochs.items()
        }
        self.index.install_world_full_reports(
            (
                (
                    (report.instance_id, report.instance_epoch),
                    report.snapshot_seq_no,
                    report.locations,
                )
                for report in by_instance.values()
            ),
            expected_owners=expected_owners,
        )
        self._owners = {
            report.instance_id: (report.instance_epoch, report.snapshot_seq_no)
            for report in by_instance.values()
        }
        self._expected_epochs = dict(expected_epochs)
        if self.metrics is not None:
            for _ in reports:
                self.metrics.increment(
                    "prefix_full_report_total",
                    labels={"reason": "startup_world"},
                )
        publication = PrefixWorldPublication(
            consumer_id=self.consumer_id,
            generation=self.generation,
            reports=tuple(
                PrefixWorldReportEvidence(
                    instance_id=report.instance_id,
                    instance_epoch=report.instance_epoch,
                    snapshot_seq_no=report.snapshot_seq_no,
                    location_count=len(report.locations),
                    content_digest=self._report_digest(report),
                )
                for report in (by_instance[name] for name in instances)
            ),
        )
        self.world_publication = publication
        return publication

    async def rebuild(self, instance_id: str) -> int:
        report = await self.rpc.full_report_and_register(
            instance_id, self.consumer_id, self.generation
        )
        if report.instance_id != instance_id:
            raise ValueError("prefix full report returned another instance")
        owner = (report.instance_id, report.instance_epoch)
        self.index.remove_instance(instance_id)
        self.index.install_full_report(owner, report.snapshot_seq_no, report.locations)
        self._owners[instance_id] = (report.instance_epoch, report.snapshot_seq_no)
        self._expected_epochs[instance_id] = report.instance_epoch
        if self.metrics is not None:
            self.metrics.increment(
                "prefix_full_report_total", labels={"reason": "startup_or_rebuild"}
            )
        return report.snapshot_seq_no

    async def sync_once(self, instance_id: str, limit: int = 1024) -> int:
        if self.world_publication is None and len(self._expected_epochs) > 1:
            publication = await self.rebuild_world(self._expected_epochs)
            return next(
                value.snapshot_seq_no
                for value in publication.reports
                if value.instance_id == instance_id
            )
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
            expected_epochs = dict(self._expected_epochs) or {
                name: value[0] for name, value in self._owners.items()
            }
            if len(expected_epochs) > 1:
                self.index.clear()
                self._owners.clear()
                self.world_publication = None
                self.generation = uuid.uuid4().hex
                if self.metrics is not None:
                    self.metrics.increment(
                        "prefix_event_gap_total", labels={"reason": "gap_or_epoch"}
                    )
                publication = await self.rebuild_world(expected_epochs)
                return next(
                    value.snapshot_seq_no
                    for value in publication.reports
                    if value.instance_id == instance_id
                )
            self.index.remove_instance(instance_id)
            self._owners.pop(instance_id, None)
            self.world_publication = None
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
                    if len(self._expected_epochs) > 1:
                        self.index.clear()
                        self._owners.clear()
                        self.world_publication = None
                    else:
                        self.index.remove_instance(instance_id)
                        self._owners.pop(instance_id, None)
                    logger.warning("prefix reconcile failed instance=%s: %s", instance_id, exc)
            await asyncio.sleep(interval_s)

"""Pair-wise topology cost model for affinity decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class LinkInfo:
    kind: str
    measured_gbps: float
    fixed_overhead_ms: float = 0.0


class TopologyMatrix:
    def __init__(self) -> None:
        self._links: dict[tuple[str, str], LinkInfo] = {}

    def set_link(self, source: str, target: str, link: LinkInfo) -> None:
        self._links[(source, target)] = link

    def transfer_cost_ms(self, source: str, target: str, size_bytes: int) -> float:
        assert size_bytes >= 0, f"negative transfer size: {size_bytes=}"
        if source == target:
            return 0.0
        link = self._links.get((source, target))
        if link is None or link.measured_gbps <= 0:
            return float("inf")
        bytes_per_ms = link.measured_gbps * 1_000_000_000 / 8 / 1000
        return link.fixed_overhead_ms + size_bytes / bytes_per_ms

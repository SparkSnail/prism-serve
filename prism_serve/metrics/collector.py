"""Metrics collector for prism-serve.

Provides a thin wrapper over prometheus_client that:
  - increments counters
  - sets gauges
  - records histogram observations
  - runs a periodic tick_loop to scrape kv_usage from infer instances

For unit tests, a NullMetrics stub is provided that accepts all calls
without recording anything (avoids importing prometheus_client in CI).

Borrowing:
  - Histogram for TTFT ← Prometheus best-practice (cross-instance aggregation
    via histogram_quantile requires Histogram, not Summary)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NullMetrics (test stub / fallback when prometheus_client not installed)
# ---------------------------------------------------------------------------

class NullMetrics:
    """No-op metrics backend used in unit tests."""

    def increment(self, name: str, *, labels: dict | None = None) -> None:
        pass

    def gauge(self, name: str, value: float, *, labels: dict | None = None) -> None:
        pass

    def observe(self, name: str, value: float, *, labels: dict | None = None) -> None:
        pass

    async def tick_loop(self) -> None:
        """Never-ending no-op coroutine (mirrors MetricsCollector.tick_loop)."""
        while True:
            await asyncio.sleep(3600)

    async def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# MetricsCollector (real implementation backed by prometheus_client)
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Prometheus-backed metrics collector.

    Metric types used:
      Counter   — kv_transfer_success_total, recompute_total, abort_total,
                  congestion_total
      Gauge     — active_requests, waiting_requests, kv_pending_requests,
                  deferred_queue_depth, decode_slot_utilization, stale_slots
      Histogram — request_ttft_ms (P50/P90/P99 via histogram_quantile)

    Why Histogram for TTFT (not Summary):
      Summary computes quantiles client-side and cannot be aggregated across
      serve replicas.  histogram_quantile() works across any number of
      instances:
        histogram_quantile(0.99, sum(rate(request_ttft_ms_bucket[5m])) by (le))
    """

    # Histogram bucket boundaries for TTFT (ms)
    TTFT_BUCKETS = (1, 5, 10, 50, 100, 500, 1000, 5000, 30000)

    def __init__(self, config: dict) -> None:
        self.config = config
        self._kv_usage_scrape_interval_s: float = config.get(
            "kv_usage_scrape_interval_s", 5.0
        )
        self._infer_client = None  # set later via set_infer_client()
        self._governor = None      # set later via set_governor()

        try:
            from prometheus_client import Counter, Gauge, Histogram
            self._available = True
        except ImportError:
            logger.warning(
                "prometheus_client not installed; metrics will be no-ops. "
                "Install with: pip install prometheus-client"
            )
            self._available = False
            return

        # --- Counters ---
        self._kv_success = Counter(
            "kv_transfer_success_total",
            "KV cache transfers completed successfully",
        )
        self._kv_recompute = Counter(
            "kv_transfer_recompute_total",
            "KV transfer recompute fallback count",
            ["reason", "attempt"],
        )
        self._kv_abort = Counter(
            "kv_transfer_abort_total",
            "KV transfer aborts after max_recompute_attempts",
            ["reason"],
        )
        self._kv_congestion = Counter(
            "kv_transfer_congestion_total",
            "Times dynamic watermark rejected a transfer (per dst)",
            ["dst"],
        )

        # --- Gauges ---
        self._active_reqs = Gauge(
            "active_requests",
            "Total active requests (all states)",
        )
        self._waiting_reqs = Gauge(
            "waiting_requests",
            "Requests in WAITING state (P-instance queue backlog)",
        )
        self._kv_pending_reqs = Gauge(
            "kv_pending_requests",
            "Requests in KV_PENDING state (KV transfer in-flight)",
        )
        self._deferred_depth = Gauge(
            "deferred_queue_depth",
            "Depth of TransferGovernor deferred queue per dst instance",
            ["dst"],
        )
        self._slot_util = Gauge(
            "decode_slot_utilization",
            "Fraction of slots in use per decode instance",
            ["instance_id"],
        )
        self._stale_slots = Gauge(
            "stale_slots_count",
            "Slots held beyond stale_timeout (leak indicator; should be 0)",
        )

        # --- Histograms ---
        self._ttft = Histogram(
            "request_ttft_ms",
            "Time to first token (ms)",
            ["state"],
            buckets=self.TTFT_BUCKETS,
        )

    # ------------------------------------------------------------------
    # Public API (called by scheduler components)
    # ------------------------------------------------------------------

    def increment(self, name: str, *, labels: dict | None = None) -> None:
        """Increment a counter by 1."""
        if not self._available:
            return
        counter = self._counter_map().get(name)
        if counter is None:
            return
        if labels:
            counter.labels(**labels).inc()
        else:
            counter.inc()

    def gauge(self, name: str, value: float, *, labels: dict | None = None) -> None:
        """Set a gauge to value."""
        if not self._available:
            return
        g = self._gauge_map().get(name)
        if g is None:
            return
        if labels:
            g.labels(**labels).set(value)
        else:
            g.set(value)

    def observe(self, name: str, value: float, *, labels: dict | None = None) -> None:
        """Record a histogram observation."""
        if not self._available:
            return
        h = self._histogram_map().get(name)
        if h is None:
            return
        if labels:
            h.labels(**labels).observe(value)
        else:
            h.observe(value)

    # ------------------------------------------------------------------
    # Metric name → object maps (lazy, built once)
    # ------------------------------------------------------------------

    def _counter_map(self):
        if not hasattr(self, "_cm"):
            self._cm = {
                "kv_transfer_success_total":    self._kv_success,
                "kv_transfer_recompute_total":  self._kv_recompute,
                "kv_transfer_abort_total":      self._kv_abort,
                "kv_transfer_congestion_total": self._kv_congestion,
            }
        return self._cm

    def _gauge_map(self):
        if not hasattr(self, "_gm"):
            self._gm = {
                "active_requests":        self._active_reqs,
                "waiting_requests":       self._waiting_reqs,
                "kv_pending_requests":    self._kv_pending_reqs,
                "deferred_queue_depth":   self._deferred_depth,
                "decode_slot_utilization": self._slot_util,
                "stale_slots_count":      self._stale_slots,
            }
        return self._gm

    def _histogram_map(self):
        if not hasattr(self, "_hm"):
            self._hm = {
                "request_ttft_ms": self._ttft,
            }
        return self._hm

    # ------------------------------------------------------------------
    # Periodic scrape loop (runs as an asyncio task)
    # ------------------------------------------------------------------

    def set_infer_client(self, infer_client) -> None:
        """Inject the interface-contract [03] client for kv_usage scraping."""
        self._infer_client = infer_client

    async def tick_loop(self) -> None:
        """Periodically scrape kv_usage from D instances and update governor.

        Called as an asyncio.Task from lifespan.  Interval is configurable
        via kv_usage_scrape_interval_s (default 5 s).
        """
        while True:
            await asyncio.sleep(self._kv_usage_scrape_interval_s)
            await self._scrape_kv_usage()

    async def _scrape_kv_usage(self) -> None:
        """Ask each D instance for its current KV usage ratio."""
        if self._infer_client is None:
            return
        try:
            usages: dict[str, float] = await self._infer_client.get_kv_usage_all()
            for instance_id, ratio in usages.items():
                # Propagate to governor (injected via set_governor)
                if self._governor is not None:
                    self._governor.update_kv_usage(instance_id, ratio)
                if self._available:
                    self._slot_util.labels(instance_id=instance_id).set(ratio)
        except Exception as exc:
            logger.warning("kv_usage scrape failed: %s", exc)

    def set_governor(self, governor) -> None:
        """Inject TransferGovernor so tick_loop can push kv_usage updates."""
        self._governor = governor

    async def flush(self) -> None:
        """Best-effort flush on shutdown (prometheus_client handles this)."""
        pass

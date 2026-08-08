"""Prometheus metrics and periodic decode-instance KV usage scraping."""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class NullMetrics:
    """No-op metrics backend used in unit tests."""

    def increment(
        self, name: str, amount: float = 1, *, labels: dict | None = None
    ) -> None:
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

    TTFT_BUCKETS = (1, 5, 10, 50, 100, 500, 1000, 5000, 30000)

    def __init__(self, config: dict) -> None:
        self.config = config
        self._kv_usage_scrape_interval_s: float = config.get(
            "kv_usage_scrape_interval_s", 5.0
        )
        self._infer_client = None  # set later via set_infer_client()
        self._governor = None      # set later via set_governor()
        self._scheduler = None

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
        self._kv_dispatch_error = Counter(
            "kv_transfer_dispatch_error_total",
            "Transfer RPC failures before remote handoff",
            ["dst"],
        )
        self._prefill_retry = Counter(
            "prefill_dispatch_retry_total",
            "Prefill dispatch retries after a stage deadline",
        )
        self._prefill_abort = Counter(
            "prefill_dispatch_abort_total",
            "Prefill dispatch aborts after retry exhaustion",
            ["reason"],
        )
        self._decode_abort = Counter(
            "decode_abort_total",
            "Decode aborts after a stage deadline",
            ["reason"],
        )
        self._control_message_error = Counter(
            "control_message_error_total",
            "Control-message operations that failed before completion",
            ["operation"],
        )
        self._prefix_event_gap = Counter(
            "prefix_event_gap_total", "Prefix event gap or overflow", ["reason"]
        )
        self._prefix_full_report = Counter(
            "prefix_full_report_total", "Prefix full reports", ["reason"]
        )
        self._prefix_stale = Counter(
            "prefix_directory_stale_total", "Stale directory entries rejected"
        )
        self._affinity_fallback = Counter(
            "affinity_fallback_total", "Affinity cold fallbacks", ["reason"]
        )
        self._cached_prefix_tokens = Counter(
            "cached_prefix_tokens_total", "Tokens reused from cached prefixes"
        )
        self._suffix_prefill_tokens = Counter(
            "suffix_prefill_tokens_total", "Suffix tokens prefetched after prefix reuse"
        )
        self._output_gap_repair = Counter(
            "output_gap_repair_total", "Cumulative output gaps repaired", ["source"]
        )
        self._infer_rpc_requests = Counter(
            "infer_rpc_requests_total", "Infer HTTP RPC calls", ["endpoint", "status"]
        )
        self._infer_rpc_ambiguous = Counter(
            "infer_rpc_ambiguous_total", "Ambiguous infer RPC outcomes", ["reason"]
        )
        self._worker_epoch_change = Counter(
            "worker_epoch_change_total", "Observed worker epoch changes", ["instance"]
        )
        self._pd_world_restart = Counter(
            "pd_world_restart_total", "Whole-world restart outcomes", ["outcome", "reason"]
        )
        self._operation_stale = Counter(
            "operation_stale_total", "Rejected stale operations", ["endpoint", "reason"]
        )
        self._operation_cancelled_before_arrival = Counter(
            "operation_cancelled_before_arrival_total",
            "Operations terminalized before command arrival", ["endpoint", "publish_outcome"]
        )
        self._cleanup_finalize = Counter(
            "cleanup_finalize_total", "Endpoint terminal finalize", ["endpoint", "status"]
        )
        self._cleanup_finalize_replay = Counter(
            "cleanup_finalize_replay_total", "Finalize snapshot replay", ["endpoint"]
        )
        self._cleanup_replacement_record = Counter(
            "cleanup_replacement_record_total", "Replacement release records", ["outcome"]
        )
        self._cleanup_replacement_replay = Counter(
            "cleanup_replacement_record_replay_total", "Replacement record replay", ["outcome"]
        )

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
        self._prefix_pins = Gauge("prefix_transfer_pins", "Active prefix transfer pins")
        self._prefix_pending = Gauge("prefix_pending_allocations", "Pending target prefix allocations")
        self._decode_slot_leases = Gauge("decode_slot_leases", "Decode slot leases", ["state"])
        self._decode_slot_quarantined = Gauge("decode_slot_quarantined", "Quarantined decode slots")
        self._pd_topology_state = Gauge("pd_topology_state", "PD topology state", ["state"])
        self._cleanup_resources_held = Gauge("cleanup_resources_held", "Held cleanup resources", ["instance", "resource_kind"])
        self._resource_report_age = Gauge("resource_report_age_seconds", "Gateway-local resource report age", ["instance"])
        self._pair_capability_ready = Gauge("pair_capability_ready", "Pair capability readiness", ["pair", "transport"])
        self._transfer_quarantined_bytes = Gauge("transfer_quarantined_bytes", "Quarantined transfer bytes", ["pair"])
        self._transfer_quarantined_operations = Gauge("transfer_quarantined_operations", "Quarantined transfer operations", ["reason"])
        self._orphan_operations = Gauge("orphan_operations", "Old-owner operations", ["instance"])

        self._ttft = Histogram(
            "request_ttft_ms",
            "Time to first token (ms)",
            ["state"],
            buckets=self.TTFT_BUCKETS,
        )
        self._cached_prefix_bytes = Histogram(
            "cached_prefix_transfer_bytes", "Mapped prefix transfer bytes"
        )
        self._gateway_first_token_stage = Histogram(
            "gateway_first_token_stage_latency_ms", "Gateway admission to first token", ["path"],
            buckets=self.TTFT_BUCKETS,
        )
        self._nccl_transfer_latency = Histogram(
            "nccl_transfer_latency_ms", "Measured NCCL transfer latency", ["pair", "path"]
        )
        self._nccl_transfer_bytes = Counter(
            "nccl_transfer_bytes", "Completed NCCL transfer bytes", ["pair", "path"]
        )

    def increment(
        self, name: str, amount: float = 1, *, labels: dict | None = None
    ) -> None:
        if not self._available:
            return
        counter = self._counter_map().get(name)
        if counter is None:
            return
        if labels:
            counter.labels(**labels).inc(amount)
        else:
            counter.inc(amount)

    def gauge(self, name: str, value: float, *, labels: dict | None = None) -> None:
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
        if not self._available:
            return
        h = self._histogram_map().get(name)
        if h is None:
            return
        if labels:
            h.labels(**labels).observe(value)
        else:
            h.observe(value)

    def _counter_map(self):
        if not hasattr(self, "_cm"):
            self._cm = {
                "kv_transfer_success_total":    self._kv_success,
                "kv_transfer_recompute_total":  self._kv_recompute,
                "kv_transfer_abort_total":      self._kv_abort,
                "kv_transfer_congestion_total": self._kv_congestion,
                "kv_transfer_dispatch_error_total": self._kv_dispatch_error,
                "prefill_dispatch_retry_total": self._prefill_retry,
                "prefill_dispatch_abort_total": self._prefill_abort,
                "decode_abort_total":           self._decode_abort,
                "control_message_error_total":  self._control_message_error,
            }
            optional = {
                "prefix_event_gap_total": "_prefix_event_gap",
                "prefix_full_report_total": "_prefix_full_report",
                "prefix_directory_stale_total": "_prefix_stale",
                "affinity_fallback_total": "_affinity_fallback",
                "cached_prefix_tokens_total": "_cached_prefix_tokens",
                "suffix_prefill_tokens_total": "_suffix_prefill_tokens",
                "output_gap_repair_total": "_output_gap_repair",
                "infer_rpc_requests_total": "_infer_rpc_requests",
                "infer_rpc_ambiguous_total": "_infer_rpc_ambiguous",
                "worker_epoch_change_total": "_worker_epoch_change",
                "pd_world_restart_total": "_pd_world_restart",
                "operation_stale_total": "_operation_stale",
                "operation_cancelled_before_arrival_total": "_operation_cancelled_before_arrival",
                "cleanup_finalize_total": "_cleanup_finalize",
                "cleanup_finalize_replay_total": "_cleanup_finalize_replay",
                "cleanup_replacement_record_total": "_cleanup_replacement_record",
                "cleanup_replacement_record_replay_total": "_cleanup_replacement_replay",
                "nccl_transfer_bytes": "_nccl_transfer_bytes",
            }
            self._cm.update({
                name: getattr(self, attr) for name, attr in optional.items()
                if hasattr(self, attr)
            })
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
            optional = {
                "prefix_transfer_pins": "_prefix_pins",
                "prefix_pending_allocations": "_prefix_pending",
                "decode_slot_leases": "_decode_slot_leases",
                "decode_slot_quarantined": "_decode_slot_quarantined",
                "pd_topology_state": "_pd_topology_state",
                "cleanup_resources_held": "_cleanup_resources_held",
                "resource_report_age_seconds": "_resource_report_age",
                "pair_capability_ready": "_pair_capability_ready",
                "transfer_quarantined_bytes": "_transfer_quarantined_bytes",
                "transfer_quarantined_operations": "_transfer_quarantined_operations",
                "orphan_operations": "_orphan_operations",
            }
            self._gm.update({
                name: getattr(self, attr) for name, attr in optional.items()
                if hasattr(self, attr)
            })
        return self._gm

    def _histogram_map(self):
        if not hasattr(self, "_hm"):
            self._hm = {
                "request_ttft_ms": self._ttft,
            }
            if hasattr(self, "_cached_prefix_bytes"):
                self._hm["cached_prefix_transfer_bytes"] = self._cached_prefix_bytes
            if hasattr(self, "_gateway_first_token_stage"):
                self._hm["gateway_first_token_stage_latency_ms"] = self._gateway_first_token_stage
            if hasattr(self, "_nccl_transfer_latency"):
                self._hm["nccl_transfer_latency_ms"] = self._nccl_transfer_latency
        return self._hm

    def set_infer_client(self, infer_client) -> None:
        """Inject the client used for KV usage scraping."""
        self._infer_client = infer_client

    async def tick_loop(self) -> None:
        """Periodically propagate decode KV usage to scheduler components."""
        while True:
            await asyncio.sleep(self._kv_usage_scrape_interval_s)
            await self._scrape_kv_usage()

    async def _scrape_kv_usage(self) -> None:
        """Refresh epoch-scoped KV usage without blocking the scheduler on failure."""
        if self._infer_client is None:
            return
        expected_epochs = (
            self._scheduler.decode_instance_epochs()
            if self._scheduler is not None else {}
        )
        if self._governor is not None:
            self._governor.set_expected_epochs(expected_epochs)
        try:
            usages: dict[str, dict] = await self._infer_client.get_kv_usage_all()
            if not isinstance(usages, dict):
                raise ValueError("kv_usage response must be a mapping")
            sampled_at = time.monotonic()
            from prism_serve.scheduler.scheduler import KVUsageSample

            for instance_id, expected_epoch in expected_epochs.items():
                payload = usages.get(instance_id)
                sample = None
                if isinstance(payload, dict):
                    ratio = payload.get("ratio")
                    epoch = payload.get("instance_epoch")
                    if isinstance(ratio, (int, float)) and epoch == expected_epoch:
                        sample = KVUsageSample(float(ratio), epoch, sampled_at)
                if self._governor is not None:
                    self._governor.update_kv_usage(instance_id, sample)
                if self._scheduler is not None:
                    self._scheduler.update_kv_usage(instance_id, sample)
                if self._available and sample is not None:
                    self._slot_util.labels(instance_id=instance_id).set(sample.ratio)
        except Exception as exc:
            logger.warning("kv_usage scrape failed: %s", exc)
            for instance_id in expected_epochs:
                if self._governor is not None:
                    self._governor.update_kv_usage(instance_id, None)
                if self._scheduler is not None:
                    self._scheduler.update_kv_usage(instance_id, None)

    def set_governor(self, governor) -> None:
        """Inject the transfer governor for KV usage updates."""
        self._governor = governor

    def set_scheduler(self, scheduler) -> None:
        """Share observed KV usage with decode instance selection."""
        self._scheduler = scheduler

    async def flush(self) -> None:
        """Best-effort flush on shutdown (prometheus_client handles this)."""
        pass

"""KV transfer flow controller for prism-serve.

Governs all P→D KV cache transfers with three layers of protection:
  1. Dynamic watermark (primary): based on real-time KV usage of the D
     instance — self-adaptive congestion control.
  2. Byte cap (secondary): hard per-dst in-flight byte limit to prevent
     single-burst saturation.
  3. Deferred queue: tasks that cannot be sent immediately are buffered
     and flushed when congestion clears.

Borrowing:
  - per-dst bytes_inflight + deferred queue ← Spark ShuffleBlockPusher
  - high/low watermark adaptive control    ← Celeborn CongestionController
  - backpressure to upstream queue         ← Flink credit-based flow control
  - softLimit/hardLimit two-stage fallback ← HDFS LeaseManager
  - recompute fallback policy              ← vLLM kv_load_failure_policy
"""

from __future__ import annotations

from collections import defaultdict, deque

from prism_serve.scheduler.sequence_state import TransferTask


class TransferGovernor:
    """Cluster-level KV transfer flow controller.

    Three-layer protection (outer → inner):
      1. Dynamic watermark (primary): D-instance KV usage ratio.
      2. Byte cap (secondary): fixed per-dst in-flight byte upper bound.
      3. Deferred queue: buffer on congestion, flush on low watermark.

    Public interface called by schedule_loop:
      submit(task)              — enqueue or dispatch a transfer task
      tick()                    — called every ~5 s to flush deferred queues
      update_kv_usage(id, ratio) — called by metrics collector
      on_transfer_failure(...)  — decide recompute vs abort
      trigger_recompute(...)    — RPC reset_to_waiting on D instance
    """

    # Class-level defaults (can be overridden via config)
    HIGH_WATERMARK:     float = 0.85   # pause sending above this KV usage
    LOW_WATERMARK:      float = 0.70   # resume sending below this KV usage
    MAX_BYTES_INFLIGHT: int   = 256 * 1024 * 1024  # 256 MB byte cap

    def __init__(self, config: dict, infer_client, metrics) -> None:
        """
        Args:
            config:       dict with optional overrides for watermarks, timeouts…
            infer_client: interface-contract [03] RPC client
                          must expose .transfer(src, dst, req_id, on_complete)
                                  and .reset_to_waiting(dst, req_id)
            metrics:      MetricsCollector (increment / gauge / observe)
        """
        self.config = config
        self.infer_client = infer_client
        self.metrics = metrics

        # Apply config overrides
        self.HIGH_WATERMARK     = config.get("HIGH_WATERMARK",     self.HIGH_WATERMARK)
        self.LOW_WATERMARK      = config.get("LOW_WATERMARK",      self.LOW_WATERMARK)
        self.MAX_BYTES_INFLIGHT = config.get("MAX_BYTES_INFLIGHT", self.MAX_BYTES_INFLIGHT)

        # Per-dst runtime state
        self._kv_usage:       dict[str, float]          = defaultdict(float)
        self._bytes_inflight: dict[str, int]            = defaultdict(int)
        # Per-dst FIFO deferred queues (preserve ordering within each dst)
        self._deferred:       dict[str, deque[TransferTask]] = defaultdict(deque)
        # Identity tracking invalidates callbacks from cancelled or superseded tasks.
        self._inflight_tasks: dict[str, TransferTask] = {}

        # Recompute guard (prevents infinite retry loops)
        self._recompute_counts: dict[str, int] = defaultdict(int)
        self._max_recompute: int = config.get("max_recompute_attempts", 2)

        self._transfer_timeout_s: float = config.get("kv_transfer_timeout_s", 30.0)

    # ------------------------------------------------------------------
    # KV usage update (called by metrics collector ~every 5 s)
    # ------------------------------------------------------------------

    def update_kv_usage(self, instance_id: str, ratio: float) -> None:
        """Update KV usage ratio for a D instance.

        Called by MetricsCollector.tick_loop on a ~5 s interval.
        Triggers deferred-queue flush if the instance has dropped below
        LOW_WATERMARK.
        """
        prev = self._kv_usage[instance_id]
        self._kv_usage[instance_id] = max(0.0, min(1.0, ratio))
        # If the instance just recovered from congestion, try to flush.
        if prev >= self.HIGH_WATERMARK and ratio < self.LOW_WATERMARK:
            self._flush_deferred(instance_id)

    # ------------------------------------------------------------------
    # Flow-control gate
    # ------------------------------------------------------------------

    def can_send(self, dst: str, size_bytes: int, priority: int = 1) -> bool:
        """Two-layer check: dynamic watermark (primary) + byte cap (secondary).

        Args:
            dst:        destination D instance ID
            size_bytes: KV bytes about to be sent
            priority:   1 = normal PD transfer; 0 = migration/replica
                        (low-priority traffic gets a stricter byte cap)

        Returns:
            True  → dispatch immediately
            False → congested, enqueue in deferred
        """
        # Primary: dynamic watermark
        kv_usage = self._kv_usage.get(dst, 0.0)
        if kv_usage >= self.HIGH_WATERMARK:
            self.metrics.increment(
                "kv_transfer_congestion_total", labels={"dst": dst}
            )
            return False

        # Secondary: byte cap (low-priority traffic gets 30 % of the cap)
        cap = self.MAX_BYTES_INFLIGHT * (1.0 if priority >= 1 else 0.3)
        if self._bytes_inflight[dst] + size_bytes > cap:
            return False

        return True

    # ------------------------------------------------------------------
    # Submit / dispatch
    # ------------------------------------------------------------------

    def submit(self, task: TransferTask) -> None:
        """Submit a KV transfer task.

        Either dispatches immediately or enqueues in the deferred queue.
        The caller does not need to check can_send.
        """
        dst = task.dst
        if self.can_send(dst, task.kv_size, task.priority):
            self._dispatch(task)
        else:
            self._deferred[dst].append(task)
            self.metrics.gauge(
                "deferred_queue_depth",
                len(self._deferred[dst]),
                labels={"dst": dst},
            )

    def _dispatch(self, task: TransferTask) -> None:
        """Actually send: update accounting, register identity, call transfer RPC."""
        self._bytes_inflight[task.dst] += task.kv_size
        self._inflight_tasks[task.req_id] = task
        # Interface contract [03]: instruct P instance to push KV to D.
        # KVBlockPusher on the infer side executes the real NCCL P2P transfer.
        self.infer_client.transfer(
            src=task.src,
            dst=task.dst,
            req_id=task.req_id,
            on_complete=lambda: self._on_complete(task),
        )

    def _on_complete(self, task: TransferTask) -> None:
        """Release accounting and fire the callback unless the task is stale."""
        if self._inflight_tasks.get(task.req_id) is not task:
            return
        del self._inflight_tasks[task.req_id]
        self._bytes_inflight[task.dst] = max(
            0, self._bytes_inflight[task.dst] - task.kv_size
        )
        if task.on_complete:
            task.on_complete()
        self._flush_deferred(task.dst)

    def cancel(self, req_id: str) -> bool:
        """Cancel deferred/in-flight accounting and invalidate late callbacks."""
        cancelled = False
        for dst, queue in self._deferred.items():
            kept = deque(task for task in queue if task.req_id != req_id)
            if len(kept) != len(queue):
                self._deferred[dst] = kept
                cancelled = True
                self.metrics.gauge(
                    "deferred_queue_depth", len(kept), labels={"dst": dst}
                )

        task = self._inflight_tasks.pop(req_id, None)
        if task is not None:
            self._bytes_inflight[task.dst] = max(
                0, self._bytes_inflight[task.dst] - task.kv_size
            )
            cancelled = True
        return cancelled

    # ------------------------------------------------------------------
    # Deferred queue management
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Periodic flush of all deferred queues.

        Called by schedule_loop Phase 4 (every 10 ms) and by an independent
        governor tick coroutine (every ~5 s) for low-watermark recovery.
        Mirrors Celeborn lowWatermark exit-congestion logic.
        """
        for dst in list(self._deferred.keys()):
            if self._deferred[dst]:
                self._flush_deferred(dst)

    def _flush_deferred(self, dst: str) -> None:
        """Greedily flush the deferred queue for dst while can_send is True.

        Mirrors Spark ShuffleBlockPusher.pushUpToMax() / deferred drain.
        """
        q = self._deferred[dst]
        while q:
            task = q[0]
            if not self.can_send(dst, task.kv_size, task.priority):
                break  # still congested, stop
            q.popleft()
            self._dispatch(task)

    # ------------------------------------------------------------------
    # Recompute fallback
    # ------------------------------------------------------------------

    def on_transfer_failure(
        self,
        req_id: str,
        dst: str,
        failure_reason: str,
    ) -> str:
        """Decide the response to a KV transfer failure.

        Two-stage protection (← HDFS LeaseManager softLimit/hardLimit):
          soft: recompute (give system a self-healing chance)
          hard: abort     (prevent infinite retry)

        Args:
            req_id:         request identifier
            dst:            D instance ID
            failure_reason: "timeout" | "network" | "dst_oom"

        Returns:
            "recompute" if retry budget remains, "abort" otherwise.
        """
        count = self._recompute_counts[req_id]
        if count >= self._max_recompute:
            self._recompute_counts.pop(req_id, None)
            self.metrics.increment(
                "kv_transfer_abort_total", labels={"reason": failure_reason}
            )
            return "abort"

        self._recompute_counts[req_id] = count + 1
        self.metrics.increment(
            "kv_transfer_recompute_total",
            labels={"reason": failure_reason, "attempt": str(count + 1)},
        )
        return "recompute"

    def trigger_recompute(self, req_id: str, dst: str) -> None:
        """Tell the D instance to roll back the sequence to WAITING.

        D-infer's scheduler will detect the WAITING state on its next step
        and re-run prefill locally (no re-routing required).

        Interface contract [03]: reset_to_waiting RPC.
        infer side: seq.status = SequenceStatus.WAITING
        """
        self.infer_client.reset_to_waiting(dst, req_id)
        # bytes_inflight for a failed transfer is no longer valid; do not
        # double-subtract (it was already cleared in _on_complete or was
        # never incremented if the task sat in deferred).

    # ------------------------------------------------------------------
    # Introspection helpers (used by schedule_loop + tests)
    # ------------------------------------------------------------------

    def bytes_inflight(self, dst: str) -> int:
        return self._bytes_inflight.get(dst, 0)

    def deferred_depth(self, dst: str) -> int:
        return len(self._deferred.get(dst, []))

    def all_inflight_zero(self) -> bool:
        """True when no in-flight bytes remain (used by lifespan drain)."""
        return all(v == 0 for v in self._bytes_inflight.values())

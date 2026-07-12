"""KV transfer flow control and retry bookkeeping."""

from __future__ import annotations

from collections import defaultdict, deque

from prism_serve.scheduler.sequence_state import TransferTask


class TransferDispatchError(RuntimeError):
    """Raised when transfer dispatch fails before remote ownership begins."""


class TransferGovernor:
    """Apply per-destination watermarks, byte caps, and retry limits."""

    # Class-level defaults (can be overridden via config)
    HIGH_WATERMARK:     float = 0.85   # pause sending above this KV usage
    LOW_WATERMARK:      float = 0.70   # resume sending below this KV usage
    MAX_BYTES_INFLIGHT: int   = 256 * 1024 * 1024  # 256 MB byte cap

    def __init__(self, config: dict, infer_client, metrics) -> None:
        self.config = config
        self.infer_client = infer_client
        self.metrics = metrics

        self.HIGH_WATERMARK     = config.get("HIGH_WATERMARK",     self.HIGH_WATERMARK)
        self.LOW_WATERMARK      = config.get("LOW_WATERMARK",      self.LOW_WATERMARK)
        self.MAX_BYTES_INFLIGHT = config.get("MAX_BYTES_INFLIGHT", self.MAX_BYTES_INFLIGHT)

        self._kv_usage:       dict[str, float]          = defaultdict(float)
        self._bytes_inflight: dict[str, int]            = defaultdict(int)
        self._deferred:       dict[str, deque[TransferTask]] = defaultdict(deque)
        # Identity tracking invalidates callbacks from cancelled or superseded tasks.
        self._inflight_tasks: dict[str, TransferTask] = {}

        # Bound retries so persistent transfer failures eventually abort.
        self._recompute_counts: dict[str, int] = defaultdict(int)
        self._max_recompute: int = config.get("max_recompute_attempts", 2)

        self._transfer_timeout_s: float = config.get("kv_transfer_timeout_s", 30.0)

    def update_kv_usage(self, instance_id: str, ratio: float) -> None:
        """Update KV usage and flush when congestion clears."""
        prev = self._kv_usage[instance_id]
        self._kv_usage[instance_id] = max(0.0, min(1.0, ratio))
        if prev >= self.HIGH_WATERMARK and ratio < self.LOW_WATERMARK:
            self._flush_deferred(instance_id)

    def can_send(self, dst: str, size_bytes: int, priority: int = 1) -> bool:
        """Apply the destination watermark and in-flight byte cap."""
        kv_usage = self._kv_usage.get(dst, 0.0)
        if kv_usage >= self.HIGH_WATERMARK:
            self.metrics.increment(
                "kv_transfer_congestion_total", labels={"dst": dst}
            )
            return False

        # The cap limits concurrency, not request size. An oversized transfer
        # may own an idle destination so it cannot block the FIFO forever.
        cap = self.MAX_BYTES_INFLIGHT * (1.0 if priority >= 1 else 0.3)
        if size_bytes > cap:
            return self._bytes_inflight[dst] == 0
        if self._bytes_inflight[dst] + size_bytes > cap:
            return False

        return True

    def submit(self, task: TransferTask) -> None:
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
        self._bytes_inflight[task.dst] += task.kv_size
        self._inflight_tasks[task.req_id] = task
        try:
            self.infer_client.transfer(
                src=task.src,
                dst=task.dst,
                req_id=task.req_id,
                operation_id=task.operation_id,
                on_complete=lambda: self._on_complete(task),
            )
        except Exception as exc:
            if self._inflight_tasks.get(task.req_id) is task:
                del self._inflight_tasks[task.req_id]
                self._bytes_inflight[task.dst] = max(
                    0, self._bytes_inflight[task.dst] - task.kv_size
                )
            raise TransferDispatchError(
                f"transfer dispatch failed for req={task.req_id!r}"
            ) from exc

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

    def task_state(self, req_id: str) -> str:
        """Return whether a transfer is deferred, in flight, or absent."""
        if req_id in self._inflight_tasks:
            return "inflight"
        if any(
            task.req_id == req_id
            for queue in self._deferred.values()
            for task in queue
        ):
            return "deferred"
        return "none"

    def tick(self) -> None:
        """Attempt to flush every deferred destination queue."""
        for dst in list(self._deferred.keys()):
            if self._deferred[dst]:
                self._flush_deferred(dst)

    def _flush_deferred(self, dst: str) -> None:
        """Flush a destination queue while preserving FIFO order."""
        q = self._deferred[dst]
        while q:
            task = q[0]
            if not self.can_send(dst, task.kv_size, task.priority):
                break  # still congested, stop
            q.popleft()
            self._dispatch(task)

    def on_transfer_failure(
        self,
        req_id: str,
        dst: str,
        failure_reason: str,
    ) -> str:
        """Return ``recompute`` within the retry budget, otherwise ``abort``."""
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
        """Tell the assigned D instance to recompute prefill locally."""
        self.infer_client.reset_to_waiting(dst, req_id)

    def finish_request(self, req_id: str) -> None:
        """Release transfer and retry bookkeeping for a terminal request."""
        self.cancel(req_id)
        self._recompute_counts.pop(req_id, None)

    def bytes_inflight(self, dst: str) -> int:
        return self._bytes_inflight.get(dst, 0)

    def deferred_depth(self, dst: str) -> int:
        return len(self._deferred.get(dst, []))

    def all_inflight_zero(self) -> bool:
        """True when no in-flight bytes remain (used by lifespan drain)."""
        return all(v == 0 for v in self._bytes_inflight.values())

    def is_drained(self) -> bool:
        """Return true when no transfer is active or deferred."""
        return self.all_inflight_zero() and all(
            not queue for queue in self._deferred.values()
        )

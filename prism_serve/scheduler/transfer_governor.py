"""KV transfer flow control and retry bookkeeping."""

from __future__ import annotations

from collections import defaultdict, deque
import time

from prism_serve.scheduler.scheduler import KVUsageSample
from prism_serve.scheduler.sequence_state import TransferTask


class TransferDispatchError(RuntimeError):
    """Raised when transfer dispatch fails before remote ownership begins."""


class TransferGovernor:
    """Control KV transfers with watermarks and per-pair byte caps."""


    HIGH_WATERMARK:     float = 0.85
    LOW_WATERMARK:      float = 0.70
    MAX_BYTES_INFLIGHT_PER_PAIR: int = 1024 * 1024 * 1024
    MAX_BYTES_INFLIGHT: int = MAX_BYTES_INFLIGHT_PER_PAIR

    def __init__(
        self, config: dict, infer_client, metrics, worker_registry=None
    ) -> None:
        self.config = config
        self.infer_client = infer_client
        self.metrics = metrics
        self.worker_registry = worker_registry

        self.HIGH_WATERMARK     = config.get("HIGH_WATERMARK",     self.HIGH_WATERMARK)
        self.LOW_WATERMARK      = config.get("LOW_WATERMARK",      self.LOW_WATERMARK)
        self.MAX_BYTES_INFLIGHT_PER_PAIR = config.get(
            "max_bytes_inflight_per_pair",
            config.get("MAX_BYTES_INFLIGHT", self.MAX_BYTES_INFLIGHT_PER_PAIR),
        )
        # Compatibility alias for callers that inspect the old attribute.
        self.MAX_BYTES_INFLIGHT = self.MAX_BYTES_INFLIGHT_PER_PAIR

        # Usage samples are valid only for the destination's current epoch.
        self._kv_usage:       dict[str, KVUsageSample]       = {}
        self._expected_epochs: dict[str, str]                 = {}
        self._kv_usage_stale_after_s: float = config.get(
            "kv_usage_stale_after_s", 10.0
        )


        self._pair_bytes_inflight: dict[str, int]            = defaultdict(int)
        self._bytes_inflight: dict[str, int]                 = defaultdict(int)

        self._deferred:       dict[str, deque[TransferTask]] = defaultdict(deque)
        # Identity tracking invalidates callbacks from cancelled or superseded tasks.
        self._inflight_tasks: dict[str, TransferTask] = {}

        # Bound retries so persistent transfer failures eventually abort.
        self._recompute_counts: dict[str, int] = defaultdict(int)
        self._max_recompute: int = config.get("max_recompute_attempts", 2)

        self._transfer_timeout_s: float = config.get("kv_transfer_timeout_s", 30.0)

    def update_kv_usage(
        self, instance_id: str, sample: KVUsageSample | None
    ) -> None:
        """Replace a usage sample and flush on a fresh high-to-low crossing."""
        prev = self._kv_usage.get(instance_id)
        if sample is None:
            self._kv_usage.pop(instance_id, None)
            return
        normalized = KVUsageSample(
            max(0.0, min(1.0, sample.ratio)),
            sample.instance_epoch,
            sample.sampled_at,
        )
        self._kv_usage[instance_id] = normalized
        if (
            prev is not None
            and prev.ratio >= self.HIGH_WATERMARK
            and normalized.ratio < self.LOW_WATERMARK
            and self._sample_is_fresh(instance_id)
        ):
            self._flush_deferred_for_dst(instance_id)

    def set_expected_epochs(self, epochs: dict[str, str]) -> None:
        self._expected_epochs = dict(epochs)
        for instance_id, sample in list(self._kv_usage.items()):
            if sample.instance_epoch != self._expected_epochs.get(instance_id):
                self._kv_usage.pop(instance_id, None)

    def _sample_is_fresh(self, instance_id: str, now: float | None = None) -> bool:
        sample = self._kv_usage.get(instance_id)
        return bool(
            sample is not None
            and sample.instance_epoch == self._expected_epochs.get(instance_id)
            and (time.monotonic() if now is None else now) - sample.sampled_at
            <= self._kv_usage_stale_after_s
        )

    @staticmethod
    def _pair_id(src: str, dst: str) -> str:
        return f"{src}--{dst}"

    def _flush_deferred_for_dst(self, dst: str) -> None:
        self._flush_deferred(dst)

    def can_send(
        self,
        dst: str,
        size_bytes: int,
        priority: int = 1,
        *,
        src: str | None = None,
    ) -> bool:
        if self.worker_registry is not None:
            if not self.worker_registry.can_govern(dst):
                return False
            if src is not None and not self.worker_registry.can_govern(src):
                return False


        if not self._sample_is_fresh(dst):
            return False
        if self._kv_usage[dst].ratio >= self.HIGH_WATERMARK:
            self.metrics.increment(
                "kv_transfer_congestion_total", labels={"dst": dst}
            )
            return False


        cap = self.MAX_BYTES_INFLIGHT_PER_PAIR * (1.0 if priority >= 1 else 0.3)
        pair_id = self._pair_id(src, dst) if src is not None else None
        current = (
            self._pair_bytes_inflight[pair_id]
            if pair_id is not None
            else self._bytes_inflight[dst]
        )


        if size_bytes > cap:
            return current == 0
        if current + size_bytes > cap:
            return False

        return True

    def submit(self, task: TransferTask) -> None:
        dst = task.dst
        if self.can_send(dst, task.kv_size, task.priority, src=task.src):
            self._dispatch(task)
        else:
            self._deferred[dst].append(task)
            self.metrics.gauge(
                "deferred_queue_depth",
                len(self._deferred[dst]),
                labels={"dst": dst},
            )

    def _dispatch(self, task: TransferTask) -> None:
        pair_id = self._pair_id(task.src, task.dst)
        self._pair_bytes_inflight[pair_id] += task.kv_size
        self._bytes_inflight[task.dst] += task.kv_size
        self._inflight_tasks[task.req_id] = task
        try:
            if getattr(self.infer_client, "week12_network_control", False) is True:
                self.infer_client.transfer_task(
                    task, lambda: self._on_complete(task)
                )
            else:
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
                self._pair_bytes_inflight[pair_id] = max(
                    0, self._pair_bytes_inflight[pair_id] - task.kv_size
                )
                self._bytes_inflight[task.dst] = max(
                    0, self._bytes_inflight[task.dst] - task.kv_size
                )
            raise TransferDispatchError(
                f"transfer dispatch failed for req={task.req_id!r}: {exc}"
            ) from exc

    def _on_complete(self, task: TransferTask) -> None:
        """Release accounting and fire the callback unless the task is stale."""
        if self._inflight_tasks.get(task.req_id) is not task:
            return
        del self._inflight_tasks[task.req_id]
        pair_id = self._pair_id(task.src, task.dst)
        self._pair_bytes_inflight[pair_id] = max(
            0, self._pair_bytes_inflight[pair_id] - task.kv_size
        )
        self._bytes_inflight[task.dst] = max(
            0, self._bytes_inflight[task.dst] - task.kv_size
        )
        if task.on_complete:
            task.on_complete()
        self._flush_deferred(task.dst)

    def owns(self, req_id: str, operation_id: str) -> bool:
        """Return whether the ledger owns this request and operation."""
        task = self._inflight_tasks.get(req_id)
        if task is not None:
            return task.operation_id == operation_id
        return any(
            task.req_id == req_id and task.operation_id == operation_id
            for queue in self._deferred.values()
            for task in queue
        )

    def cancel(self, req_id: str, operation_id: str | None = None) -> bool:
        """Cancel one operation, or all request work when no operation is given."""
        cancelled = False
        for dst, queue in self._deferred.items():
            kept = deque(
                task for task in queue
                if not (
                    task.req_id == req_id
                    and (operation_id is None or task.operation_id == operation_id)
                )
            )
            if len(kept) != len(queue):
                self._deferred[dst] = kept
                cancelled = True
                self.metrics.gauge(
                    "deferred_queue_depth", len(kept), labels={"dst": dst},
                )

        task = self._inflight_tasks.get(req_id)
        if task is not None and (
            operation_id is None or task.operation_id == operation_id
        ):
            del self._inflight_tasks[req_id]
            pair_id = self._pair_id(task.src, task.dst)
            self._pair_bytes_inflight[pair_id] = max(
                0, self._pair_bytes_inflight[pair_id] - task.kv_size
            )
            self._bytes_inflight[task.dst] = max(
                0, self._bytes_inflight[task.dst] - task.kv_size
            )
            cancelled = True
        return cancelled

    def task_state(self, req_id: str, operation_id: str | None = None) -> str:
        """Return whether a transfer is deferred, handed off, or absent."""
        task = self._inflight_tasks.get(req_id)
        if task is not None and (
            operation_id is None or task.operation_id == operation_id
        ):
            return "inflight"
        if any(
            task.req_id == req_id
            and (operation_id is None or task.operation_id == operation_id)
            for queue in self._deferred.values()
            for task in queue
        ):
            return "deferred"
        return "none"

    def quarantined_transfer_totals(
        self, operation_ids: set[str]
    ) -> tuple[int, dict[str, int]]:
        tasks = [
            *self._inflight_tasks.values(),
            *(task for queue in self._deferred.values() for task in queue),
        ]
        matching = {
            task.operation_id: task for task in tasks
            if task.operation_id in operation_ids
        }
        bytes_by_pair: dict[str, int] = {}
        for task in matching.values():
            pair = f"{task.src}--{task.dst}"
            bytes_by_pair[pair] = bytes_by_pair.get(pair, 0) + task.kv_size
        return len(matching), bytes_by_pair

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
            if not self.can_send(
                task.dst, task.kv_size, task.priority, src=task.src
            ):
                break
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

    def trigger_recompute(self, req_id: str, dst: str):
        return self.infer_client.reset_to_waiting(dst, req_id)

    def finish_request(self, req_id: str) -> None:
        """Release transfer and retry bookkeeping for a terminal request."""
        self.cancel(req_id)
        self._recompute_counts.pop(req_id, None)

    def bytes_inflight(self, dst: str) -> int:
        return self._bytes_inflight.get(dst, 0)

    def bytes_inflight_for_pair(self, src: str, dst: str) -> int:
        return self._pair_bytes_inflight.get(self._pair_id(src, dst), 0)

    def pair_bytes_inflight_snapshot(self) -> dict[str, int]:
        return {
            pair: value for pair, value in self._pair_bytes_inflight.items()
            if value > 0
        }

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

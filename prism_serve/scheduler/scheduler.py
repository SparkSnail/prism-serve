"""Cluster-level PD orchestration scheduler.

Decides which Prefill instance handles each request and which Decode instance
receives the resulting KV cache.  Does NOT handle KV transfer flow control
(see TransferGovernor) or per-request state tracking (see RequestTracker).

Borrowing:
  - pick_prefill_instance  ← shortest-queue strategy (standard load balancing)
  - pick_decode_instance   ← most-free-slots strategy (KV-pressure-aware)
  - decide_decode_instance_count ← Flink AdaptiveBatch (runtime parallelism)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# PDScheduler
# ---------------------------------------------------------------------------

class PDScheduler:
    """Cluster-level PD orchestration decision maker.

    Responsibilities:
      1. Maintain a slot-view of all P/D instances.
      2. For each incoming request: select the optimal P instance and
         pre-select a D instance.
      3. Dynamically adjust the recommended P:D ratio at runtime
         (← Flink AdaptiveBatch).

    Not responsible for: KV transfer flow control (TransferGovernor),
    per-request state tracking (RequestTracker).
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        # instance_id → queue depth (prefill instances)
        self._prefill_load: dict[str, int] = {}
        # instance_id → free slot count (decode instances)
        self._decode_free_slots: dict[str, int] = {}
        # instance_id → KV usage ratio 0.0–1.0 (updated by metrics collector)
        self._kv_usage: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_instance(
        self,
        instance_id: str,
        role: str,
        max_slots: int = 0,
    ) -> None:
        """Register a new instance when a K8S Pod comes up.

        Args:
            instance_id: unique instance ID, e.g. "prefill-0", "decode-1"
            role:        "prefill" | "decode"
            max_slots:   decode instances must supply this (> 0)
        """
        if role == "prefill":
            self._prefill_load[instance_id] = 0
        elif role == "decode":
            assert max_slots > 0, (
                f"decode instance must specify max_slots > 0, got {max_slots=}"
            )
            self._decode_free_slots[instance_id] = max_slots
            self._kv_usage[instance_id] = 0.0
        else:
            raise ValueError(f"unknown role {role!r}; expected 'prefill' or 'decode'")

    def deregister_instance(self, instance_id: str) -> None:
        """Remove an instance that is going offline."""
        self._prefill_load.pop(instance_id, None)
        self._decode_free_slots.pop(instance_id, None)
        self._kv_usage.pop(instance_id, None)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def pick_prefill_instance(self, req_id: str) -> str | None:
        """Select the P instance with the shortest queue (least loaded).

        Returns None if no prefill instance is registered.

        Why not round-robin: queue depth is proportional to actual GPU
        load for prefill.  A short-queue instance is more likely to start
        the new request promptly.
        """
        if not self._prefill_load:
            return None
        # O(n_prefill_instances); n is typically < 10 in a single cluster
        best = min(self._prefill_load, key=lambda k: self._prefill_load[k])
        self._prefill_load[best] += 1
        return best

    def pick_decode_instance(
        self,
        req_id: str,
        kv_size_bytes: int,
    ) -> str | None:
        """Pre-select the D instance with the most free slots.

        Filters out instances whose KV usage is at or above HIGH_WATERMARK
        (transmission flow control is already congested there).

        Returns None if all D instances are full or congested.

        Why not round-robin: KV size varies wildly (short prompt = 0.11 GB,
        long prompt = 3.5 GB).  Round-robin ignores KV pressure; most-free-slots
        naturally balances it.
        """
        high_wm: float = self.config.get("HIGH_WATERMARK", 0.85)
        candidates = [
            (iid, slots)
            for iid, slots in self._decode_free_slots.items()
            if slots > 0 and self._kv_usage.get(iid, 0.0) < high_wm
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda x: x[1])[0]
        self._decode_free_slots[best] -= 1
        return best

    # ------------------------------------------------------------------
    # Feedback from infer layer
    # ------------------------------------------------------------------

    def on_prefill_done(self, instance_id: str) -> None:
        """Decrement P instance queue depth when prefill completes."""
        if instance_id in self._prefill_load:
            self._prefill_load[instance_id] = max(
                0, self._prefill_load[instance_id] - 1
            )

    def on_decode_finished(self, instance_id: str) -> None:
        """Return a slot to the D instance when a sequence finishes."""
        if instance_id in self._decode_free_slots:
            self._decode_free_slots[instance_id] += 1

    def update_kv_usage(self, instance_id: str, ratio: float) -> None:
        """Update KV usage for a D instance (called by metrics collector)."""
        self._kv_usage[instance_id] = max(0.0, min(1.0, ratio))

    # ------------------------------------------------------------------
    # Adaptive decode instance count
    # ------------------------------------------------------------------

    def decide_decode_instance_count(self, active_kv_bytes: int) -> int:
        """Compute recommended number of decode instances at runtime.

        Formula: ceil(active_kv_bytes / kv_per_instance), clamped to
                 [min_decode_instances, max_decode_instances].

        Flink AdaptiveBatch analogy:
          Flink : parallelism = ceil(totalBytes / avgDataVolumePerTask)
          prism : decode_count = ceil(active_kv_bytes / kv_per_instance)

        Args:
            active_kv_bytes: total KV size across all active sequences (bytes)

        Returns:
            recommended decode instance count (for autoscaler)
        """
        kv_per = self.config.get("kv_per_instance_bytes", 56 * 1024 ** 3)
        n = math.ceil(active_kv_bytes / kv_per) if active_kv_bytes > 0 else 1
        return max(
            self.config.get("min_decode_instances", 1),
            min(n, self.config.get("max_decode_instances", 64)),
        )

    # ------------------------------------------------------------------
    # Introspection (for tests / monitoring)
    # ------------------------------------------------------------------

    def prefill_queue_depths(self) -> dict[str, int]:
        return dict(self._prefill_load)

    def decode_free_slots(self) -> dict[str, int]:
        return dict(self._decode_free_slots)

    def kv_usage(self) -> dict[str, float]:
        return dict(self._kv_usage)

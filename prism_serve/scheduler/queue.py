"""NATS publish/subscribe wrapper for prism-serve scheduler.

Decouples the schedule_loop from the NATS SDK so tests can substitute a
plain asyncio.Queue mock without a real NATS server.

Design decisions:
  1. inbox buffer: NATS callbacks are async push; the inbox converts "push"
     to "pull" so schedule_loop can poll synchronously in each phase.
  2. queue_group="serve": multiple serve replicas share the same group so
     each message is processed by exactly one instance (≡ Kafka Consumer
     Group).
  3. Wildcard subscription "kv_usage.*": infer instances publish per their
     own ID; serve subscribes once and handles all without restart.

Borrowing:
  - queue_group semantics ← Kafka Consumer Group
  - subject routing       ← NATS native feature (not Kafka-comparable)
"""

from __future__ import annotations

import asyncio
import json
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NATSQueue
# ---------------------------------------------------------------------------

class NATSQueue:
    """NATS publish/subscribe wrapper.

    Usage:
        q = NATSQueue(config)
        await q.connect()
        ...
        await q.publish("dispatch_prefill", {"instance_id": "p-0", ...})
        msgs = await q.poll("prefill_done")   # non-blocking drain
        ...
        await q.close()

    For unit tests, pass use_mock=True to bypass NATS entirely and use
    in-process asyncio.Queue objects.
    """

    def __init__(self, config: dict, *, use_mock: bool = False) -> None:
        self._url: str = config.get("nats_url", "nats://localhost:4222")
        self._use_mock = use_mock
        self._nc = None                                   # NATSClient | None
        # subject → asyncio.Queue (the inbox buffer)
        self._inbox: dict[str, asyncio.Queue] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to NATS and set up subscriptions.

        Subscriptions (queue_group="serve" for all → exactly-once delivery):
          "prefill_done"  — P instance signals prefill completed
          "decode_done"   — D instance signals sequence finished
          "kv_usage.*"    — D instances report KV usage (wildcard)
        """
        if self._use_mock:
            # Tests inject messages directly via _put_mock(); no NATS needed.
            return

        try:
            import nats
        except ImportError as exc:
            raise ImportError(
                "nats-py is required for NATS transport; "
                "install with: pip install nats-py"
            ) from exc

        self._nc = await nats.connect(self._url)
        # queue_group ensures exactly-one delivery across serve replicas
        await self._nc.subscribe(
            "prefill_done",
            queue="serve",
            cb=self._make_handler("prefill_done"),
        )
        await self._nc.subscribe(
            "decode_done",
            queue="serve",
            cb=self._make_handler("decode_done"),
        )
        # Wildcard: matches kv_usage.d-0, kv_usage.d-1, …
        # New infer instances publish without serve restart.
        await self._nc.subscribe(
            "kv_usage.*",
            cb=self._make_handler("kv_usage"),
        )
        logger.info("NATSQueue connected to %s", self._url)

    async def close(self) -> None:
        """Drain in-flight messages and close the NATS connection."""
        if self._nc is not None:
            try:
                await self._nc.drain()
                await self._nc.close()
            except Exception:
                pass
            self._nc = None

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish(self, subject: str, data: dict) -> None:
        """Fire-and-forget publish.

        With JetStream enabled (nats -js flag) the server persists the
        message; without it the message is at-most-once.
        """
        if self._use_mock:
            # In mock mode, publishing to "dispatch_prefill" etc. is
            # intentionally a no-op; tests drive infer behaviour directly.
            return
        if self._nc is None:
            logger.warning("publish called before connect(); dropping %s", subject)
            return
        payload = json.dumps(data).encode()
        await self._nc.publish(subject, payload)

    # ------------------------------------------------------------------
    # Poll (non-blocking drain)
    # ------------------------------------------------------------------

    async def poll(self, subject: str) -> list[dict]:
        """Drain all pending messages for subject from the inbox.

        Non-blocking: returns an empty list if no messages are waiting.
        Called by schedule_loop Phase 2 ("prefill_done") and Phase 5
        ("decode_done") on every tick.
        """
        q = self._inbox.get(subject)
        if not q:
            return []
        msgs: list[dict] = []
        while not q.empty():
            try:
                msgs.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break
        return msgs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_handler(self, subject: str):
        """Return a NATS message callback that pushes into the inbox queue."""
        async def handler(msg) -> None:
            try:
                data = json.loads(msg.data.decode())
            except Exception:
                logger.warning("malformed NATS message on subject %r", subject)
                return
            inbox = self._inbox.setdefault(subject, asyncio.Queue())
            await inbox.put(data)
        return handler

    # ------------------------------------------------------------------
    # Test helpers (mock mode)
    # ------------------------------------------------------------------

    async def _put_mock(self, subject: str, data: dict) -> None:
        """Directly inject a message into the inbox (for unit tests)."""
        inbox = self._inbox.setdefault(subject, asyncio.Queue())
        await inbox.put(data)

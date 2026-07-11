"""NATS transport for scheduler commands and completion events."""

from __future__ import annotations

import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)

class NATSQueue:
    """Route commands to infer instances and replies to their owning gateway."""

    def __init__(self, config: dict, *, use_mock: bool = False) -> None:
        self._url: str = config.get("nats_url", "nats://localhost:4222")
        self._connect_timeout_s: float = config.get("nats_connect_timeout_s", 2.0)
        self._max_reconnect_attempts: int = config.get(
            "nats_max_reconnect_attempts", 60
        )
        configured_owner = config.get("scheduler_id")
        self._owner_id = (
            os.getenv("HOSTNAME", "serve-local")
            if configured_owner is None
            else configured_owner
        )
        _validate_subject_token(self._owner_id, "scheduler_id")
        self._use_mock = use_mock
        self._nc = None
        self._inbox: dict[str, asyncio.Queue] = {}

    async def connect(self) -> None:
        """Connect and subscribe to replies owned by this gateway."""
        if self._use_mock:
            return

        try:
            import nats
        except ImportError as exc:
            raise ImportError(
                "nats-py is required for NATS transport; "
                "install with: pip install nats-py"
            ) from exc

        self._nc = await nats.connect(
            self._url,
            connect_timeout=self._connect_timeout_s,
            retry_on_failed_connect=False,
            max_reconnect_attempts=self._max_reconnect_attempts,
        )
        await self._nc.subscribe(
            self.reply_subject("prefill_done"),
            cb=self._make_handler("prefill_done"),
        )
        await self._nc.subscribe(
            self.reply_subject("decode_done"),
            cb=self._make_handler("decode_done"),
        )
        await self._nc.subscribe(
            self.reply_subject("recompute_done"),
            cb=self._make_handler("recompute_done"),
        )
        await self._nc.subscribe(
            self.reply_subject("first_token"),
            cb=self._make_handler("first_token"),
        )
        await self._nc.subscribe(
            "kv_usage.*",
            cb=self._make_handler("kv_usage"),
        )
        logger.info("NATSQueue connected to %s", self._url)

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def is_connected(self) -> bool:
        if self._use_mock:
            return True
        return bool(self._nc is not None and self._nc.is_connected)

    def dispatch_subject(self, instance_id: str) -> str:
        _validate_subject_token(instance_id, "instance_id")
        return f"dispatch_prefill.{instance_id}"

    def reply_subject(self, event: str) -> str:
        assert event in {
            "prefill_done", "decode_done", "recompute_done", "first_token",
        }, f"unsupported completion event: {event!r}"
        return f"{event}.{self._owner_id}"

    async def close(self) -> None:
        """Drain in-flight messages and close the NATS connection."""
        if self._nc is not None:
            try:
                await self._nc.drain()
                await self._nc.close()
            except Exception:
                pass
            self._nc = None

    async def publish(self, subject: str, data: dict) -> None:
        """Publish a Core NATS message with at-most-once delivery."""
        if self._use_mock:
            return
        if not self.is_connected:
            raise ConnectionError(f"NATS unavailable for publish: {subject}")
        await self._nc.publish(subject, json.dumps(data).encode())

    async def poll(self, subject: str) -> list[dict]:
        """Drain pending messages from one logical inbox without blocking."""
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

    async def _put_mock(self, subject: str, data: dict) -> None:
        """Directly inject a message into the inbox (for unit tests)."""
        inbox = self._inbox.setdefault(subject, asyncio.Queue())
        await inbox.put(data)


def _validate_subject_token(value: str, field_name: str) -> None:
    """Reject identifiers that would change NATS subject matching."""
    assert value and not any(char.isspace() for char in value), (
        f"{field_name} must be a non-empty NATS token without whitespace: {value!r}"
    )
    assert not any(char in value for char in ".*>"), (
        f"{field_name} must not contain '.', '*', or '>': {value!r}"
    )

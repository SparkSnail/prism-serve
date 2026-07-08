"""Unit tests for NATSQueue (scheduler/queue.py) in mock mode.

No real NATS server required — all tests use use_mock=True.
Covers: connect (no-op in mock), publish (no-op in mock), poll (drain inbox),
_put_mock (direct injection), _make_handler (JSON decode + put).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from prism_serve.scheduler.queue import NATSQueue


def make_queue() -> NATSQueue:
    return NATSQueue({"nats_url": "nats://localhost:4222"}, use_mock=True)


# ---------------------------------------------------------------------------
# connect / close (mock mode — should be silent no-ops)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connect_mock_is_noop():
    q = make_queue()
    await q.connect()          # must not raise
    assert q._nc is None       # no real connection


@pytest.mark.asyncio
async def test_close_mock_is_noop():
    q = make_queue()
    await q.connect()
    await q.close()            # must not raise
    assert q._nc is None


# ---------------------------------------------------------------------------
# publish (mock mode — fire-and-forget, no-op)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_publish_mock_does_not_raise():
    q = make_queue()
    await q.connect()
    await q.publish("dispatch_prefill", {"instance_id": "p-0", "req_id": "R1"})
    # Nothing to assert — just must not raise and not put into inbox
    assert q._inbox.get("dispatch_prefill") is None


# ---------------------------------------------------------------------------
# poll — empty inbox
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_poll_empty_returns_empty_list():
    q = make_queue()
    msgs = await q.poll("prefill_done")
    assert msgs == []


@pytest.mark.asyncio
async def test_poll_unknown_subject_returns_empty_list():
    q = make_queue()
    msgs = await q.poll("does_not_exist")
    assert msgs == []


# ---------------------------------------------------------------------------
# _put_mock + poll — core inbox mechanics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_mock_then_poll_single():
    q = make_queue()
    await q._put_mock("prefill_done", {"req_id": "R1", "kv_size_bytes": 112})
    msgs = await q.poll("prefill_done")
    assert msgs == [{"req_id": "R1", "kv_size_bytes": 112}]


@pytest.mark.asyncio
async def test_put_mock_then_poll_multiple():
    q = make_queue()
    for i in range(4):
        await q._put_mock("prefill_done", {"req_id": f"R{i}"})
    msgs = await q.poll("prefill_done")
    assert len(msgs) == 4
    assert [m["req_id"] for m in msgs] == ["R0", "R1", "R2", "R3"]


@pytest.mark.asyncio
async def test_poll_drains_inbox():
    """poll() removes messages from inbox; second poll returns empty."""
    q = make_queue()
    await q._put_mock("decode_done", {"req_id": "R1"})
    msgs1 = await q.poll("decode_done")
    msgs2 = await q.poll("decode_done")
    assert len(msgs1) == 1
    assert msgs2 == []


@pytest.mark.asyncio
async def test_poll_different_subjects_independent():
    """Messages on different subjects don't interfere."""
    q = make_queue()
    await q._put_mock("prefill_done", {"req_id": "R1"})
    await q._put_mock("decode_done",  {"req_id": "R2"})

    prefill_msgs = await q.poll("prefill_done")
    decode_msgs  = await q.poll("decode_done")

    assert [m["req_id"] for m in prefill_msgs] == ["R1"]
    assert [m["req_id"] for m in decode_msgs]  == ["R2"]


@pytest.mark.asyncio
async def test_poll_preserves_fifo_order():
    q = make_queue()
    payloads = [{"seq": i} for i in range(10)]
    for p in payloads:
        await q._put_mock("prefill_done", p)
    msgs = await q.poll("prefill_done")
    assert [m["seq"] for m in msgs] == list(range(10))


# ---------------------------------------------------------------------------
# _make_handler — JSON decode path (used by real NATS callbacks)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_make_handler_valid_json():
    """Handler correctly parses JSON and pushes into inbox."""
    q = make_queue()
    handler = q._make_handler("prefill_done")

    class FakeMsg:
        data = json.dumps({"req_id": "R42"}).encode()

    await handler(FakeMsg())
    msgs = await q.poll("prefill_done")
    assert msgs == [{"req_id": "R42"}]


@pytest.mark.asyncio
async def test_make_handler_malformed_json_is_dropped():
    """Handler silently drops malformed JSON without raising."""
    q = make_queue()
    handler = q._make_handler("prefill_done")

    class BadMsg:
        data = b"not valid json{{{"

    await handler(BadMsg())   # must not raise
    msgs = await q.poll("prefill_done")
    assert msgs == []


@pytest.mark.asyncio
async def test_make_handler_multiple_subjects_separate_inboxes():
    """Handlers for different subjects write to separate inbox queues."""
    q = make_queue()
    h_prefill = q._make_handler("prefill_done")
    h_decode  = q._make_handler("decode_done")

    class Msg:
        def __init__(self, d): self.data = json.dumps(d).encode()

    await h_prefill(Msg({"req_id": "P1"}))
    await h_decode(Msg({"req_id": "D1"}))

    assert (await q.poll("prefill_done")) == [{"req_id": "P1"}]
    assert (await q.poll("decode_done"))  == [{"req_id": "D1"}]

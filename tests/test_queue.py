"""NATS queue tests using the local mock transport."""

from __future__ import annotations

import asyncio
import json

import pytest

from prism_serve.scheduler.queue import NATSQueue


def make_queue() -> NATSQueue:
    return NATSQueue(
        {
            "nats_url": "nats://localhost:4222",
            "scheduler_id": "serve-0",
            "scheduler_generation": "boot-1",
        },
        use_mock=True,
    )


@pytest.mark.asyncio
async def test_connect_mock_is_noop():
    q = make_queue()
    await q.connect()
    assert q._nc is None


@pytest.mark.asyncio
async def test_close_mock_is_noop():
    q = make_queue()
    await q.connect()
    await q.close()
    assert q._nc is None


@pytest.mark.asyncio
async def test_publish_mock_does_not_raise():
    q = make_queue()
    await q.connect()
    await q.publish("dispatch_prefill", {"instance_id": "p-0", "req_id": "R1"})
    assert q._inbox.get("dispatch_prefill") is None


def test_subjects_are_scoped_to_target_and_owner():
    q = make_queue()
    assert q.dispatch_subject("p-0") == "dispatch_prefill.p-0"
    assert q.reply_subject("prefill_done") == "prefill_done.serve-0--boot-1"
    assert q.reply_subject("decode_done") == "decode_done.serve-0--boot-1"


@pytest.mark.parametrize("bad_id", ["", "serve.0", "serve *", "serve>"])
def test_subject_scope_rejects_wildcard_or_multitoken_ids(bad_id):
    with pytest.raises((AssertionError, ValueError)):
        NATSQueue({"scheduler_id": bad_id}, use_mock=True)


def test_process_restart_changes_owner_for_same_pod_uid():
    first = NATSQueue({
        "scheduler_id": "pod-uid", "scheduler_generation": "boot-1",
    }, use_mock=True)
    second = NATSQueue({
        "scheduler_id": "pod-uid", "scheduler_generation": "boot-2",
    }, use_mock=True)
    assert first.owner_id != second.owner_id


@pytest.mark.asyncio
async def test_publish_fails_when_real_connection_is_unavailable():
    q = NATSQueue({
        "scheduler_id": "serve-0", "scheduler_generation": "boot-1",
    })
    with pytest.raises(ConnectionError, match="NATS unavailable"):
        await q.publish("dispatch_prefill.p-0", {"req_id": "R1"})


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
    q = make_queue()
    await q._put_mock("decode_done", {"req_id": "R1"})
    msgs1 = await q.poll("decode_done")
    msgs2 = await q.poll("decode_done")
    assert len(msgs1) == 1
    assert msgs2 == []


@pytest.mark.asyncio
async def test_poll_different_subjects_independent():
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


@pytest.mark.asyncio
async def test_make_handler_valid_json():
    q = make_queue()
    handler = q._make_handler("prefill_done")

    class FakeMsg:
        data = json.dumps({"req_id": "R42"}).encode()

    await handler(FakeMsg())
    msgs = await q.poll("prefill_done")
    assert msgs == [{"req_id": "R42"}]


@pytest.mark.asyncio
async def test_make_handler_malformed_json_is_dropped():
    q = make_queue()
    handler = q._make_handler("prefill_done")

    class BadMsg:
        data = b"not valid json{{{"

    await handler(BadMsg())
    msgs = await q.poll("prefill_done")
    assert msgs == []


@pytest.mark.asyncio
async def test_make_handler_multiple_subjects_separate_inboxes():
    q = make_queue()
    h_prefill = q._make_handler("prefill_done")
    h_decode  = q._make_handler("decode_done")

    class Msg:
        def __init__(self, d): self.data = json.dumps(d).encode()

    await h_prefill(Msg({"req_id": "P1"}))
    await h_decode(Msg({"req_id": "D1"}))

    assert (await q.poll("prefill_done")) == [{"req_id": "P1"}]
    assert (await q.poll("decode_done"))  == [{"req_id": "D1"}]

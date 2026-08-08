from __future__ import annotations

import asyncio

import pytest

from prism_serve.gateway.output import (
    GatewayOutputBuffer, GatewayOutputCapacity, repair_output_gap,
)
from prism_serve.metrics.collector import NullMetrics
from prism_serve.scheduler.main_loop import (
    _on_kv_done, _reconcile_authoritative_output,
)
from prism_serve.scheduler.scheduler import PDScheduler
from prism_serve.scheduler.sequence_state import (
    RequestInfo, RequestTracker, SeqState,
)


@pytest.mark.asyncio
async def test_cumulative_output_cursor_deduplicates_and_finishes():
    output = GatewayOutputBuffer()
    assert await output.apply_cumulative("r1", [10, 11], 2) is True
    assert await output.apply_cumulative("r1", [10, 11], 2) is False
    assert await output.apply_cumulative("r1", [10, 11, 12], 3, terminal=True) is True
    assert output.snapshot("r1") == ([10, 11, 12], True, None)


@pytest.mark.asyncio
async def test_cumulative_output_rejects_changed_prefix():
    output = GatewayOutputBuffer()
    await output.apply_cumulative("r1", [10, 11], 2)
    with pytest.raises(ValueError, match="changed committed prefix"):
        await output.apply_cumulative("r1", [10, 99, 12], 3)


@pytest.mark.asyncio
async def test_output_gap_query_repairs_from_authoritative_cumulative_cursor():
    class Client:
        async def request_output(self, instance_id, req_id, after_seq):
            assert (instance_id, req_id, after_seq) == ("d0", "r1", 1)
            return {
                "req_id": "r1", "instance_epoch": "d0-epoch",
                "operation_id": "op-1",
                "output_seq_no": 3, "token_ids": [10, 11, 12],
                "terminal": False,
            }

    output = GatewayOutputBuffer()
    await output.apply_cumulative("r1", [10], 1)
    changed = await repair_output_gap(
        Client(), instance_id="d0", instance_epoch="d0-epoch", req_id="r1",
        operation_id="op-1", cursor=1, output_buffer=output,
    )
    assert changed is True
    assert output.snapshot("r1") == ([10, 11, 12], False, None)


@pytest.mark.asyncio
async def test_output_gap_query_rejects_another_request_operation():
    class Client:
        async def request_output(self, _instance_id, req_id, _after_seq):
            return {
                "req_id": req_id, "instance_epoch": "d0-epoch",
                "operation_id": "stale-op", "output_seq_no": 1,
                "token_ids": [10], "terminal": False,
            }

    with pytest.raises(ValueError, match="output query identity changed"):
        await repair_output_gap(
            Client(), instance_id="d0", instance_epoch="d0-epoch", req_id="r1",
            operation_id="op-1", cursor=0, output_buffer=GatewayOutputBuffer(),
        )


@pytest.mark.asyncio
async def test_output_gap_query_rejects_another_request_id():
    class Client:
        async def request_output(self, _instance_id, _req_id, _after_seq):
            return {
                "req_id": "r2", "instance_epoch": "d0-epoch",
                "operation_id": "op-1", "output_seq_no": 1,
                "token_ids": [10], "terminal": False,
            }

    with pytest.raises(ValueError, match="output query identity changed"):
        await repair_output_gap(
            Client(), instance_id="d0", instance_epoch="d0-epoch", req_id="r1",
            operation_id="op-1", cursor=0, output_buffer=GatewayOutputBuffer(),
        )


@pytest.mark.asyncio
async def test_output_gap_query_rejects_request_replaced_during_rpc():
    current = True

    class Client:
        async def request_output(self, _instance_id, req_id, _after_seq):
            nonlocal current
            current = False
            return {
                "req_id": req_id, "instance_epoch": "d0-epoch",
                "operation_id": "op-1", "output_seq_no": 1,
                "token_ids": [10], "terminal": False,
            }

    output = GatewayOutputBuffer()
    with pytest.raises(ValueError, match="output query request changed"):
        await repair_output_gap(
            Client(), instance_id="d0", instance_epoch="d0-epoch", req_id="r1",
            operation_id="op-1", cursor=0, output_buffer=output,
            still_current=lambda: current,
        )
    assert output.snapshot("r1") == ([], False, None)


@pytest.mark.asyncio
async def test_low_cap_evicts_only_resource_free_terminal_output():
    output = GatewayOutputBuffer(active_operation_cap=1, terminal_snapshot_cap=1)
    await output.apply_cumulative("held", [1], 1, terminal=True)
    with pytest.raises(GatewayOutputCapacity):
        output.ensure("blocked")

    output.mark_resource_free("held")
    await output.apply_cumulative("next", [2], 1, terminal=True)
    output.mark_resource_free("next")

    assert "held" not in output._states
    assert output.snapshot("next") == ([2], True, None)
    assert output.state_counts() == {
        "active_or_held": 0, "resource_free_terminal": 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_token", [7, 9])
async def test_done_before_commit_callback_converges_terminal_once(terminal_token):
    metrics = NullMetrics()
    scheduler = PDScheduler({})
    scheduler.register_instance("d0", "decode", 1, "d-epoch")
    lease = scheduler.reserve_decode_slot("d0", "r1", "op-1")
    tracker = RequestTracker(metrics)
    tracker.add(RequestInfo(
        req_id="r1", state=SeqState.KV_PENDING,
        decode_instance="d0", decode_instance_epoch="d-epoch",
        active_operation_id="op-1", transfer_operation_id="op-1",
    ))
    output = GatewayOutputBuffer()
    await output.apply_cumulative(
        "r1", [terminal_token], 1, terminal=True
    )
    commit_ref = object()

    _on_kv_done(
        "r1", "op-1", tracker, scheduler, metrics, output, commit_ref
    )

    req = tracker.get("r1")
    assert req.state == SeqState.FINISHED
    assert req.target_request_commit_ref is commit_ref
    assert req.first_token_at > 0
    assert lease.state == "ACTIVE"


@pytest.mark.asyncio
async def test_authoritative_output_query_advances_exact_operation_to_terminal():
    metrics = NullMetrics()
    tracker = RequestTracker(metrics)
    req = RequestInfo(
        req_id="r1", state=SeqState.DECODING,
        decode_instance="d0", decode_instance_epoch="d-epoch",
        active_operation_id="op-1",
    )
    tracker.add(req)
    output = GatewayOutputBuffer()

    class Client:
        async def request_output(self, instance, req_id, after_seq):
            return {
                "req_id": req_id, "instance_epoch": "d-epoch",
                "operation_id": "op-1", "token_ids": [7],
                "output_seq_no": 1, "terminal": True,
            }

    await _reconcile_authoritative_output(req, tracker, Client(), output)

    assert req.state == SeqState.FINISHED
    assert req.first_token_at > 0
    assert output.snapshot("r1") == ([7], True, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "instance_id", "instance_epoch", "operation_id"),
    [
        (SeqState.KV_PENDING, "d0", "d-epoch", "op-1"),
        (SeqState.DECODING, "", "d-epoch", "op-1"),
        (SeqState.DECODING, "d0", "", "op-1"),
        (SeqState.DECODING, "d0", "d-epoch", ""),
    ],
)
async def test_authoritative_output_query_requires_ready_complete_identity(
    state, instance_id, instance_epoch, operation_id,
):
    tracker = RequestTracker(NullMetrics())
    req = RequestInfo(
        req_id="r1", state=state,
        decode_instance=instance_id, decode_instance_epoch=instance_epoch,
        active_operation_id=operation_id,
    )
    tracker.add(req)

    class Client:
        def __init__(self):
            self.calls = 0

        async def request_output(self, _instance, _req_id, _after_seq):
            self.calls += 1
            raise AssertionError("query must not start")

    client = Client()
    with pytest.raises(ValueError, match="output query is not ready"):
        await _reconcile_authoritative_output(
            req, tracker, client, GatewayOutputBuffer()
        )
    assert client.calls == 0


@pytest.mark.asyncio
async def test_authoritative_output_query_rejects_replaced_request_after_rpc():
    tracker = RequestTracker(NullMetrics())
    req = RequestInfo(
        req_id="r1", state=SeqState.DECODING,
        decode_instance="d0", decode_instance_epoch="d-epoch",
        active_operation_id="op-1",
    )
    tracker.add(req)
    output = GatewayOutputBuffer()

    class Client:
        async def request_output(self, _instance, _req_id, _after_seq):
            tracker.remove("r1")
            tracker.add(RequestInfo(
                req_id="r1", state=SeqState.DECODING,
                decode_instance="d0", decode_instance_epoch="d-epoch",
                active_operation_id="op-1",
            ))
            return {
                "req_id": "r1", "instance_epoch": "d-epoch",
                "operation_id": "op-1", "token_ids": [7],
                "output_seq_no": 1, "terminal": True,
            }

    with pytest.raises(ValueError, match="output query identity changed"):
        await _reconcile_authoritative_output(req, tracker, Client(), output)
    assert output.snapshot("r1") == ([], False, None)


@pytest.mark.asyncio
async def test_authoritative_output_query_rechecks_identity_at_apply():
    tracker = RequestTracker(NullMetrics())
    req = RequestInfo(
        req_id="r1", state=SeqState.DECODING,
        decode_instance="d0", decode_instance_epoch="d-epoch",
        active_operation_id="op-1",
    )
    tracker.add(req)
    rpc_returned = asyncio.Event()
    output = GatewayOutputBuffer()
    state = output.ensure("r1")
    await state.condition.acquire()

    class Client:
        async def request_output(self, _instance, _req_id, _after_seq):
            rpc_returned.set()
            return {
                "req_id": "r1", "instance_epoch": "d-epoch",
                "operation_id": "op-1", "token_ids": [7],
                "output_seq_no": 1, "terminal": True,
            }

    task = asyncio.create_task(
        _reconcile_authoritative_output(req, tracker, Client(), output)
    )
    await rpc_returned.wait()
    await asyncio.sleep(0)
    tracker.remove("r1")
    tracker.add(RequestInfo(
        req_id="r1", state=SeqState.DECODING,
        decode_instance="d0", decode_instance_epoch="d-epoch",
        active_operation_id="op-1",
    ))
    state.condition.release()

    with pytest.raises(ValueError, match="output query request changed"):
        await task
    assert output.snapshot("r1") == ([], False, None)


@pytest.mark.asyncio
async def test_authoritative_output_query_rejects_another_request_id():
    tracker = RequestTracker(NullMetrics())
    req = RequestInfo(
        req_id="r1", state=SeqState.DECODING,
        decode_instance="d0", decode_instance_epoch="d-epoch",
        active_operation_id="op-1",
    )
    tracker.add(req)

    class Client:
        async def request_output(self, _instance, _req_id, _after_seq):
            return {
                "req_id": "r2", "instance_epoch": "d-epoch",
                "operation_id": "op-1", "token_ids": [7],
                "output_seq_no": 1, "terminal": True,
            }

    with pytest.raises(ValueError, match="output query identity changed"):
        await _reconcile_authoritative_output(
            req, tracker, Client(), GatewayOutputBuffer()
        )

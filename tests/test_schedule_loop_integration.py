"""Schedule-loop integration tests with mock transport and infer clients."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from prism_serve.metrics.collector import NullMetrics
from prism_serve.gateway.app import _abort_remaining_requests
from prism_serve.router.http_rpc import EndpointSequenceAllocator
from prism_serve.scheduler.main_loop import (
    InjectedNATSCommandFault,
    _publish_command_with_fault,
    schedule_loop,
)
from prism_serve.scheduler.main_loop import _abort_remote_request
from prism_serve.scheduler.main_loop import (
    CleanupProof,
    RemoteRequestState,
    TransferCleanupState,
    d_slot_releasable,
    p_load_releasable,
    validate_cleanup_proof,
    _canonical_cleanup,
    _trigger_recompute_epoch_fenced,
)
from prism_serve.scheduler.queue import NATSQueue
from prism_serve.scheduler.scheduler import KVUsageSample, PDScheduler
from prism_serve.scheduler.sequence_state import (
    RequestInfo, RequestTracker, SeqState, TransferTask,
)
from prism_serve.scheduler.transfer_governor import TransferGovernor


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fault_kind", "expected_publish_count"),
    [("nats_drop", 0), ("nats_duplicate", 2), ("nats_publish_unknown", 1)],
)
async def test_nats_command_fault_uses_exact_envelope_and_reports_actual_count(
    fault_kind: str,
    expected_publish_count: int,
) -> None:
    queue = MagicMock()
    queue.publish = AsyncMock()
    command = {
        "schema_version": 1,
        "endpoint_ref": {
            "owner_generation": "gateway-1",
            "operation_seq": 7,
            "target_instance": "p0",
            "target_worker_epoch": "p0-e1",
            "operation_id": "request-1",
            "payload_digest": "sha256:exact",
            "topology_generation": "world-1",
        },
        "payload": {"req_id": "request-1"},
    }
    events: list[tuple[str, dict[str, object]]] = []
    observe = AsyncMock(return_value={
        "delivery_count": expected_publish_count,
        "execution_count": 1 if expected_publish_count else 0,
    })

    with pytest.raises(InjectedNATSCommandFault):
        await _publish_command_with_fault(
            queue,
            "dispatch_prefill.p0",
            command,
            {"fault_kind": fault_kind},
            lambda name, details: events.append((name, details)),
            observe,
        )

    assert queue.publish.await_count == expected_publish_count
    assert all(call.args == ("dispatch_prefill.p0", command)
               for call in queue.publish.await_args_list)
    assert events == [("fault_injected", {
        "fault_kind": fault_kind,
        "operation_id": "request-1",
        "endpoint_ref": command["endpoint_ref"],
        "subject": "dispatch_prefill.p0",
        "publish_count": expected_publish_count,
        "delivery_count": expected_publish_count,
        "execution_count": 1 if expected_publish_count else 0,
    })]
    if fault_kind == "nats_drop":
        observe.assert_not_awaited()
    else:
        observe.assert_awaited_once_with(command["endpoint_ref"], fault_kind)


DEFAULT_CONFIG = {
    "HIGH_WATERMARK":         0.85,
    "LOW_WATERMARK":          0.70,
    "MAX_BYTES_INFLIGHT":     512 * 1024 ** 2,  # 512 MB — generous for tests
    "kv_transfer_timeout_s":  0.05,             # 50 ms — fast timeout for tests
    "prefill_timeout_s":      30.0,
    "max_dispatch_attempts":  3,
    "recompute_timeout_s":    30.0,
    "decode_timeout_s":       300.0,
    "abort_request_timeout_s": 0.1,
    "max_recompute_attempts": 2,
    "schedule_loop_tick_ms":  0,
}

KV_SIZE_1BLOCK = 28 * 1024 ** 2   # TP=1 illustrative block: 28 MiB


REACHABLE_CLEANUP_DOMAIN = [
    (SeqState.WAITING, RemoteRequestState.NOT_OWNED,
     RemoteRequestState.NEVER_DISPATCHED, TransferCleanupState.NONE,
     "none", "NOT_STARTED", True, False),
]
REACHABLE_CLEANUP_DOMAIN += [
    (SeqState.PREFILLING, p_state, RemoteRequestState.NEVER_DISPATCHED,
     TransferCleanupState.NONE, "none", outcome, p_safe, True)
    for outcome, p_states in [
        ("NOT_STARTED", [(RemoteRequestState.NEVER_DISPATCHED, True)]),
        ("ACKED", [(RemoteRequestState.ABORT_ACK, True),
                   (RemoteRequestState.UNCERTAIN, False)]),
        ("UNKNOWN", [(RemoteRequestState.NEVER_DISPATCHED, False),
                     (RemoteRequestState.ABORT_ACK, False),
                     (RemoteRequestState.UNCERTAIN, False)]),
    ]
    for p_state, p_safe in p_states
]
REACHABLE_CLEANUP_DOMAIN += [
    (SeqState.KV_PENDING, RemoteRequestState.ALREADY_COMPLETED, d_state,
     transfer_state, mode, "ACKED", True,
     d_state == RemoteRequestState.ABORT_ACK
     and transfer_state != TransferCleanupState.UNCERTAIN)
    for mode, transfer_states in [
        ("deferred", [TransferCleanupState.DEFERRED_CANCELLED]),
        ("inflight", [TransferCleanupState.FENCED_ACK,
                      TransferCleanupState.UNCERTAIN]),
    ]
    for d_state in [RemoteRequestState.ABORT_ACK, RemoteRequestState.UNCERTAIN]
    for transfer_state in transfer_states
]
REACHABLE_CLEANUP_DOMAIN += [
    (state, RemoteRequestState.ALREADY_COMPLETED, d_state, transfer_state,
     "none", "ACKED", True, d_state == RemoteRequestState.ABORT_ACK)
    for state, transfer_state in [
        (SeqState.DECODING, TransferCleanupState.NONE),
        (SeqState.RECOMPUTING, TransferCleanupState.FENCED_ACK),
    ]
    for d_state in [RemoteRequestState.ABORT_ACK, RemoteRequestState.UNCERTAIN]
]


@pytest.mark.parametrize(
    ("state", "p_state", "d_state", "transfer_state", "mode", "outcome",
     "p_safe", "d_safe"),
    REACHABLE_CLEANUP_DOMAIN,
)
def test_cleanup_proof_exhaustive_reachable_domain(
    state, p_state, d_state, transfer_state, mode, outcome, p_safe, d_safe,
):
    proof = CleanupProof(
        request_state=state,
        p_remote_request_state=p_state,
        d_remote_request_state=d_state,
        transfer_state=transfer_state,
        transfer_mode=mode,
        publish_outcome=outcome,
        request_owns_p_load=state == SeqState.PREFILLING,
        request_owns_original_d_slot=state != SeqState.WAITING,
    )
    validate_cleanup_proof(proof)
    assert p_load_releasable(proof) is p_safe
    assert d_slot_releasable(proof) is d_safe


def test_cleanup_proof_rejects_na_transfer_combination():
    proof = CleanupProof(
        request_state=SeqState.DECODING,
        p_remote_request_state=RemoteRequestState.ALREADY_COMPLETED,
        d_remote_request_state=RemoteRequestState.ABORT_ACK,
        transfer_state=TransferCleanupState.FENCED_ACK,
        transfer_mode="inflight",
        publish_outcome="ACKED",
        request_owns_p_load=False,
        request_owns_original_d_slot=True,
    )
    with pytest.raises(ValueError, match="invalid cleanup proof"):
        validate_cleanup_proof(proof)


@pytest.mark.asyncio
async def test_unknown_cleanup_keeps_pair_credit_until_remote_release_proof():
    scheduler, governor, tracker, _, _, _ = make_components()
    req = RequestInfo(
        req_id="R-unknown",
        state=SeqState.KV_PENDING,
        decode_instance="d-0",
        decode_instance_epoch="d-epoch",
        transfer_operation_id="op-transfer",
    )
    tracker.add(req)
    task = TransferTask(
        req_id=req.req_id,
        src="p-0",
        dst="d-0",
        kv_size=28 * 1024**2,
        operation_id=req.transfer_operation_id,
    )
    governor._inflight_tasks[req.req_id] = task
    governor._pair_bytes_inflight["p-0--d-0"] = task.kv_size
    governor._bytes_inflight["d-0"] = task.kv_size
    governor.infer_client = MagicMock(
        week12_network_control=True,
        cleanup_request=AsyncMock(return_value=False),
    )

    await _canonical_cleanup(
        req,
        tracker,
        scheduler,
        governor,
        "gateway:owner",
        0.01,
        0.01,
    )

    assert tracker.get(req.req_id) is req
    assert req.state == SeqState.ABORTED
    assert governor.owns(req.req_id, req.transfer_operation_id)
    assert governor.bytes_inflight_for_pair("p-0", "d-0") == task.kv_size
    assert governor.bytes_inflight("d-0") == task.kv_size


@pytest.mark.asyncio
async def test_epoch_switch_during_recompute_trigger_is_local_only():
    scheduler, governor, _, _, _, _ = make_components()
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="e1")
    req = RequestInfo(
        req_id="R1", state=SeqState.KV_PENDING,
        decode_instance="d-0", decode_instance_epoch="e1",
    )

    def flip(_req_id, _dst):
        record = scheduler.quarantine_instance("d-0")
        scheduler.reconcile_instance(
            "d-0", "e2", record.reconciliation_token, "decode", 1,
            [], [], [],
        )

    governor.trigger_recompute = MagicMock(side_effect=flip)
    assert not await _trigger_recompute_epoch_fenced(req, scheduler, governor)
    assert req.state == SeqState.KV_PENDING
    assert scheduler.decode_free_slots()["d-0"] == 1
    assert scheduler.quarantine_record("d-0") is None


@pytest.mark.asyncio
async def test_epoch_switch_during_recompute_retry_is_local_only():
    scheduler, governor, _, _, _, _ = make_components()
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="e1")
    started = time.monotonic() - 1.0
    req = RequestInfo(
        req_id="R1", state=SeqState.RECOMPUTING,
        decode_instance="d-0", decode_instance_epoch="e1",
        recompute_start=started,
    )

    async def flip(_req_id, _dst):
        await asyncio.sleep(0)
        record = scheduler.quarantine_instance("d-0")
        scheduler.reconcile_instance(
            "d-0", "e2", record.reconciliation_token, "decode", 1,
            [], [], [],
        )

    governor.trigger_recompute = MagicMock(side_effect=flip)
    assert not await _trigger_recompute_epoch_fenced(req, scheduler, governor)
    assert req.state == SeqState.RECOMPUTING
    assert req.recompute_start == started
    assert scheduler.decode_free_slots()["d-0"] == 1


def make_components(config: dict | None = None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    metrics = NullMetrics()
    infer_client = MagicMock()
    infer_client.transfer.side_effect = (
        lambda src, dst, req_id, operation_id, on_complete=None:
        on_complete() if on_complete else None
    )
    infer_client.reset_to_waiting = MagicMock()
    infer_client.abort_request = AsyncMock(return_value={"success": True})
    infer_client.abort_transfer = AsyncMock(return_value={"success": True})

    governor = TransferGovernor(cfg, infer_client, metrics)
    scheduler = PDScheduler(cfg)
    original_register = scheduler.register_instance

    def register_with_fresh_usage(*args, **kwargs):
        original_register(*args, **kwargs)
        instance_id = args[0]
        role = args[1]
        if role == "decode":
            epoch = scheduler.instance_epoch(instance_id)
            sample = KVUsageSample(0.0, epoch, time.monotonic())
            scheduler.update_kv_usage(instance_id, sample)
            governor.set_expected_epochs(scheduler.decode_instance_epochs())
            governor.update_kv_usage(instance_id, sample)

    scheduler.register_instance = register_with_fresh_usage
    tracker = RequestTracker(metrics)
    queue = NATSQueue(cfg, use_mock=True)
    original_put_mock = queue._put_mock

    async def put_with_epoch(subject: str, data: dict) -> None:
        payload = dict(data)
        request = tracker.get(payload.get("req_id", ""))
        if subject == "prefill_done":
            payload.setdefault(
                "instance_epoch",
                request.prefill_instance_epoch if request is not None else "",
            )
        elif subject in {"recompute_done", "first_token", "decode_done"}:
            payload.setdefault(
                "instance_epoch",
                request.decode_instance_epoch if request is not None else "",
            )
        await original_put_mock(subject, payload)

    queue._put_mock = put_with_epoch
    return scheduler, governor, tracker, queue, metrics, infer_client


async def run_ticks(
    scheduler, governor, tracker, queue, metrics, config,
    *,
    n_ticks: int = 20,
    tick_s: float = 0.001,   # 1 ms per tick in tests (faster than 10 ms)
) -> None:
    ticks_done = 0
    orig_sleep = asyncio.sleep

    async def fast_sleep(delay):
        nonlocal ticks_done
        ticks_done += 1
        if ticks_done >= n_ticks:
            raise asyncio.CancelledError
        await orig_sleep(0)

    import prism_serve.scheduler.main_loop as ml
    ml_sleep_orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(asyncio, "sleep", fast_sleep)
        try:
            await asyncio.wait_for(loop_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        finally:
            loop_task.cancel()
            try:
                await loop_task
            except (asyncio.CancelledError, Exception):
                pass

    ml.TICK_INTERVAL_S = ml_sleep_orig


@pytest.mark.asyncio
async def test_schedule_loop_uses_configured_tick_interval(monkeypatch):
    config = {**DEFAULT_CONFIG, "schedule_loop_tick_ms": 123}
    scheduler, governor, tracker, queue, metrics, _ = make_components(config)
    delays = []

    async def capture_sleep(delay):
        delays.append(delay)
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", capture_sleep)
    with pytest.raises(asyncio.CancelledError):
        await schedule_loop(scheduler, governor, tracker, queue, metrics, config)

    assert delays
    assert 0.100 < delays[0] <= 0.123


@pytest.mark.asyncio
async def test_schedule_loop_default_tick_is_exactly_ten_milliseconds(monkeypatch):
    config = {**DEFAULT_CONFIG, "schedule_loop_tick_ms": 10}
    scheduler, governor, tracker, queue, metrics, _ = make_components(config)
    delays = []

    async def capture_sleep(delay):
        delays.append(delay)
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "prism_serve.scheduler.main_loop.time.monotonic", lambda: 0.0,
    )
    monkeypatch.setattr(asyncio, "sleep", capture_sleep)
    with pytest.raises(asyncio.CancelledError):
        await schedule_loop(scheduler, governor, tracker, queue, metrics, config)

    assert delays == [10 / 1000.0]


async def drive_loop(
    scheduler, governor, tracker, queue, metrics, config,
    *,
    max_ticks: int = 50,
    stop_when=None,      # callable(tracker) -> bool; stop when True
) -> None:
    import prism_serve.scheduler.main_loop as ml

    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    tick = 0
    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )

    while tick < max_ticks:
        await asyncio.sleep(0)   # yield to let loop_task run one iteration
        tick += 1
        if stop_when and stop_when(tracker):
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig


@pytest.mark.asyncio
async def test_happy_path_single():
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    req = RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK)
    tracker.add(req)

    injected = False

    async def _inject_prefill_done_once(tracker_ref):
        nonlocal injected
        if not injected and tracker_ref.get("R1") and \
                tracker_ref.get("R1").state == SeqState.PREFILLING:
            await queue._put_mock("prefill_done", {
                "req_id": "R1",
                "kv_size_bytes": KV_SIZE_1BLOCK,
                "block_table": [3],
            })
            injected = True

    decode_injected = False
    orig_transfer = infer_client.transfer.side_effect

    def transfer_and_decode(src, dst, req_id, operation_id, on_complete=None):
        nonlocal decode_injected
        if on_complete:
            on_complete()
        decode_injected = True

    infer_client.transfer.side_effect = transfer_and_decode

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )

    for _ in range(60):
        await asyncio.sleep(0)
        await _inject_prefill_done_once(tracker)
        if decode_injected and tracker.get("R1") and \
                tracker.get("R1").state == SeqState.DECODING:
            await queue._put_mock("decode_done", {"req_id": "R1"})
            decode_injected = False
        if "R1" not in tracker:
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    assert "R1" not in tracker
    assert scheduler._decode_free_slots["d-0"] == 10
    assert scheduler._prefill_load["p-0"] == 0


@pytest.mark.asyncio
async def test_happy_path_four_concurrent():
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("p-1", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=127)
    scheduler.register_instance("d-1", "decode", max_slots=127)

    req_sizes = {
        "R1": 2 * KV_SIZE_1BLOCK,   # 512-token, 2 blocks
        "R2": 1 * KV_SIZE_1BLOCK,   # 256-token, 1 block
        "R3": 4 * KV_SIZE_1BLOCK,   # 1024-token, 4 blocks
        "R4": 2 * KV_SIZE_1BLOCK,
    }
    for rid, sz in req_sizes.items():
        tracker.add(RequestInfo(req_id=rid, kv_size_bytes=sz))

    prefill_injected: set[str] = set()
    decode_injected: set[str] = set()

    def transfer_side_effect(src, dst, req_id, operation_id, on_complete=None):
        if on_complete:
            on_complete()
        decode_injected.add(req_id)

    infer_client.transfer.side_effect = transfer_side_effect

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )

    for _ in range(200):
        await asyncio.sleep(0)

        for rid, sz in req_sizes.items():
            r = tracker.get(rid)
            if r and r.state == SeqState.PREFILLING and rid not in prefill_injected:
                await queue._put_mock("prefill_done", {
                    "req_id": rid,
                    "kv_size_bytes": sz,
                    "block_table": [0],
                })
                prefill_injected.add(rid)

        for rid in list(decode_injected):
            r = tracker.get(rid)
            if r and r.state == SeqState.DECODING:
                await queue._put_mock("decode_done", {"req_id": rid})
                decode_injected.discard(rid)

        if len(tracker) == 0:
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    assert len(tracker) == 0
    assert scheduler._decode_free_slots["d-0"] + \
           scheduler._decode_free_slots["d-1"] == 254
    assert scheduler._prefill_load["p-0"] == 0
    assert scheduler._prefill_load["p-1"] == 0


@pytest.mark.asyncio
async def test_kv_transfer_deferred_then_flushed():
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    # Set D congested before prefill_done arrives
    sample = KVUsageSample(0.90, "legacy:d-0", time.monotonic())
    governor.update_kv_usage("d-0", sample)

    infer_client.transfer.side_effect = (
        lambda src, dst, req_id, operation_id, on_complete=None:
        on_complete() if on_complete else None
    )

    req = RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK)
    tracker.add(req)

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )

    prefill_injected = False
    for tick in range(100):
        await asyncio.sleep(0)

        r = tracker.get("R1")
        if r and r.state == SeqState.PREFILLING and not prefill_injected:
            await queue._put_mock("prefill_done", {
                "req_id": "R1",
                "kv_size_bytes": KV_SIZE_1BLOCK,
                "block_table": [3],
            })
            prefill_injected = True

        if prefill_injected and tick == 20:
            r2 = tracker.get("R1")
            assert r2 is not None and r2.state == SeqState.KV_PENDING, (
                f"expected KV_PENDING, got {r2.state if r2 else 'gone'}"
            )
            assert governor.deferred_depth("d-0") == 1
            assert infer_client.transfer.call_count == 0

            # Simulate watermark drop → governor should flush
            sample = KVUsageSample(0.60, "legacy:d-0", time.monotonic())
            governor.update_kv_usage("d-0", sample)

        if tick == 30:
            assert infer_client.transfer.call_count == 1
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    assert infer_client.transfer.call_count == 1
    assert governor.deferred_depth("d-0") == 0


@pytest.mark.asyncio
async def test_recompute_fallback_on_timeout():
    """Keep recompute on the decode instance selected before transfer."""
    config = {**DEFAULT_CONFIG, "kv_transfer_timeout_s": 0.01}  # 10 ms

    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    infer_client.transfer.side_effect = (
        lambda src, dst, req_id, operation_id, on_complete=None: None
    )

    req = RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK)
    tracker.add(req)

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )

    prefill_injected = False
    waited = False
    for _ in range(300):
        await asyncio.sleep(0)

        r = tracker.get("R1")
        if r and r.state == SeqState.PREFILLING and not prefill_injected:
            await queue._put_mock("prefill_done", {
                "req_id": "R1",
                "kv_size_bytes": KV_SIZE_1BLOCK,
                "block_table": [3],
            })
            prefill_injected = True

        if prefill_injected and not waited:
            r2 = tracker.get("R1")
            if r2 and r2.state == SeqState.KV_PENDING and r2.kv_sent_at > 0:
                r2.kv_sent_at = time.monotonic() - 1.0
                waited = True

        if r and r.state == SeqState.RECOMPUTING and prefill_injected and waited:
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    r = tracker.get("R1")
    assert r is not None, "request should remain tracked during recompute"
    assert r.state == SeqState.RECOMPUTING
    assert governor._recompute_counts["R1"] == 1
    infer_client.reset_to_waiting.assert_called_once_with("d-0", "R1")
    infer_client.abort_transfer.assert_awaited_once()
    assert scheduler._decode_free_slots["d-0"] == 9
    assert governor.deferred_depth("d-0") == 0


@pytest.mark.asyncio
async def test_transfer_abort_failure_quarantines_before_recompute():
    """Unknown remote writes must quarantine D instead of starting recompute."""
    config = {**DEFAULT_CONFIG, "kv_transfer_timeout_s": 0.01}
    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    infer_client.transfer.side_effect = lambda **_kwargs: None
    infer_client.abort_transfer = AsyncMock(return_value={"success": False})
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    try:
        for _ in range(200):
            await asyncio.sleep(0)
            request = tracker.get("R1")
            if request and request.state == SeqState.PREFILLING:
                await queue._put_mock("prefill_done", {
                    "req_id": "R1",
                    "kv_size_bytes": KV_SIZE_1BLOCK,
                    "block_table": [3],
                })
            if request and request.state == SeqState.KV_PENDING:
                request.kv_sent_at = time.monotonic() - 1.0
            if "R1" not in tracker:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "R1" not in tracker
    assert scheduler.quarantine_record("d-0") is not None
    assert "d-0" not in scheduler.decode_free_slots()
    infer_client.reset_to_waiting.assert_not_called()


@pytest.mark.asyncio
async def test_late_completion_during_transfer_abort_await_wins_once():
    """A completion committed during abort must stop fallback."""
    config = {**DEFAULT_CONFIG, "kv_transfer_timeout_s": 0.01}
    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    infer_client.transfer.side_effect = lambda **_kwargs: None
    abort_started = asyncio.Event()
    allow_abort_reply = asyncio.Event()

    async def abort_transfer(**_kwargs):
        abort_started.set()
        await allow_abort_reply.wait()
        return {"success": True}

    infer_client.abort_transfer.side_effect = abort_transfer
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    prefill_injected = False
    try:
        for _ in range(200):
            await asyncio.sleep(0)
            req = tracker.get("R1")
            if req and req.state == SeqState.PREFILLING and not prefill_injected:
                await queue._put_mock("prefill_done", {
                    "req_id": "R1",
                    "kv_size_bytes": KV_SIZE_1BLOCK,
                    "block_table": [3],
                })
                prefill_injected = True
            if req and req.state == SeqState.KV_PENDING:
                req.kv_sent_at = time.monotonic() - 1.0
            if abort_started.is_set():
                break

        await asyncio.wait_for(abort_started.wait(), timeout=1.0)
        completion = infer_client.transfer.call_args.kwargs["on_complete"]
        completion()
        assert tracker.get("R1").state == SeqState.DECODING

        allow_abort_reply.set()
        for _ in range(20):
            await asyncio.sleep(0)
        assert not task.done()
        assert tracker.get("R1").state == SeqState.DECODING
        assert governor._recompute_counts.get("R1", 0) == 0
        infer_client.reset_to_waiting.assert_not_called()

        await queue._put_mock("decode_done", {"req_id": "R1"})
        for _ in range(100):
            await asyncio.sleep(0)
            if "R1" not in tracker:
                break
    finally:
        allow_abort_reply.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "R1" not in tracker
    assert scheduler.decode_free_slots() == {"d-0": 1}
    assert governor.is_drained()


@pytest.mark.asyncio
async def test_transfer_ownership_loss_quarantines_and_terminates():
    """Lost governor ownership must quarantine the bound decode instance."""
    config = {**DEFAULT_CONFIG, "kv_transfer_timeout_s": 0.01}
    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    infer_client.transfer.side_effect = lambda **_kwargs: None
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    prefill_injected = False
    try:
        for _ in range(200):
            await asyncio.sleep(0)
            req = tracker.get("R1")
            if req and req.state == SeqState.PREFILLING and not prefill_injected:
                await queue._put_mock("prefill_done", {
                    "req_id": "R1",
                    "kv_size_bytes": KV_SIZE_1BLOCK,
                    "block_table": [3],
                })
                prefill_injected = True
            if req and req.state == SeqState.KV_PENDING:
                operation_id = req.transfer_operation_id
                assert governor.cancel("R1", operation_id)
                req.kv_sent_at = time.monotonic() - 1.0
                break

        for _ in range(100):
            await asyncio.sleep(0)
            if "R1" not in tracker:
                break

        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "R1" not in tracker
    assert scheduler.quarantine_record("d-0") is not None
    assert "d-0" not in scheduler.decode_free_slots()
    assert governor.is_drained()
    infer_client.abort_transfer.assert_not_awaited()
    infer_client.reset_to_waiting.assert_not_called()


@pytest.mark.asyncio
async def test_transfer_dispatch_error_does_not_stop_schedule_loop():
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    infer_client.transfer.side_effect = ConnectionError("transport unavailable")
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )
    try:
        for _ in range(100):
            await asyncio.sleep(0)
            request = tracker.get("R1")
            if request and request.state == SeqState.PREFILLING:
                await queue._put_mock("prefill_done", {
                    "req_id": "R1",
                    "kv_size_bytes": KV_SIZE_1BLOCK,
                    "block_table": [3],
                })
            if request and request.state == SeqState.RECOMPUTING:
                break
        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    request = tracker.get("R1")
    assert request is not None and request.state == SeqState.RECOMPUTING
    assert governor.is_drained()
    infer_client.reset_to_waiting.assert_called_once_with("d-0", "R1")


@pytest.mark.asyncio
async def test_abort_after_max_recompute():
    config = {**DEFAULT_CONFIG,
              "kv_transfer_timeout_s": 0.01,
              "recompute_timeout_s": 0.01,
              "max_recompute_attempts": 1}

    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    infer_client.transfer.side_effect = (
        lambda src, dst, req_id, operation_id, on_complete=None: None
    )

    req = RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK)
    tracker.add(req)

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )

    prefill_injected = False
    for _ in range(500):
        await asyncio.sleep(0)

        r = tracker.get("R1")

        if r and r.state == SeqState.PREFILLING:
            await queue._put_mock("prefill_done", {
                "req_id": "R1",
                "kv_size_bytes": KV_SIZE_1BLOCK,
                "block_table": [3],
            })
            prefill_injected = True

        if r and r.state == SeqState.KV_PENDING and r.kv_sent_at > 0:
            if r.kv_sent_at > time.monotonic() - 0.5:
                r.kv_sent_at = time.monotonic() - 1.0

        if r and r.state == SeqState.RECOMPUTING:
            r.recompute_start = time.monotonic() - 1.0

        if "R1" not in tracker and prefill_injected:
            break

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    assert "R1" not in tracker
    assert scheduler._decode_free_slots["d-0"] == 10


@pytest.mark.asyncio
async def test_no_prefill_instance_stays_waiting():
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("d-0", "decode", max_slots=10)

    for i in range(3):
        tracker.add(RequestInfo(req_id=f"R{i}", kv_size_bytes=KV_SIZE_1BLOCK))

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )

    for _ in range(20):
        await asyncio.sleep(0)

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    assert len(tracker) == 3
    for i in range(3):
        r = tracker.get(f"R{i}")
        assert r is not None and r.state == SeqState.WAITING

@pytest.mark.asyncio
async def test_recompute_done_finishes_and_clears_retry_state():
    config = {**DEFAULT_CONFIG, "kv_transfer_timeout_s": 0.01}
    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    infer_client.transfer.side_effect = lambda **kwargs: None
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    try:
        for _ in range(100):
            await asyncio.sleep(0)
            req = tracker.get("R1")
            if req and req.state == SeqState.PREFILLING:
                await queue._put_mock("prefill_done", {
                    "req_id": "R1",
                    "kv_size_bytes": KV_SIZE_1BLOCK,
                    "block_table": [3],
                })
            if req and req.state == SeqState.KV_PENDING:
                req.kv_sent_at = time.monotonic() - 1.0
            if req and req.state == SeqState.RECOMPUTING:
                break

        await queue._put_mock("recompute_done", {"req_id": "R1"})
        for _ in range(100):
            await asyncio.sleep(0)
            if tracker.get("R1").state == SeqState.DECODING:
                break
        req = tracker.get("R1")
        assert req.prefill_instance == "p-0"
        assert req.decode_instance == "d-0"
        assert governor._recompute_counts["R1"] == 1

        await queue._put_mock("decode_done", {"req_id": "R1"})
        for _ in range(100):
            await asyncio.sleep(0)
            if "R1" not in tracker:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "R1" not in tracker
    assert "R1" not in governor._recompute_counts
    assert scheduler.decode_free_slots() == {"d-0": 1}


@pytest.mark.asyncio
async def test_prefill_timeout_retries_same_assignment_then_aborts():
    config = {
        **DEFAULT_CONFIG,
        "prefill_timeout_s": 0.01,
        "max_dispatch_attempts": 2,
    }
    scheduler, governor, tracker, queue, metrics, _ = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))
    published = []

    async def capture_publish(subject, data):
        published.append((subject, data))

    queue.publish = capture_publish
    allocator = EndpointSequenceAllocator("world-a", queue.owner_id)
    task = asyncio.create_task(
        schedule_loop(
            scheduler, governor, tracker, queue, metrics, config,
            operation_allocator=allocator,
        )
    )
    try:
        for _ in range(100):
            await asyncio.sleep(0)
            req = tracker.get("R1")
            if req is None:
                break
            if req.state == SeqState.PREFILLING:
                req.prefill_start = time.monotonic() - 1.0
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "R1" not in tracker
    assert [subject for subject, _ in published] == [
        "dispatch_prefill.p-0", "dispatch_prefill.p-0",
    ]
    assert all(data["schema_version"] == 1 for _, data in published)
    assert published[0][1]["endpoint_ref"] == published[1][1]["endpoint_ref"]
    assert published[0][1]["payload"] == published[1][1]["payload"]
    payloads = [data["payload"] for _, data in published]
    command_ids = {data["command_id"] for data in payloads}
    assert len(command_ids) == 1
    assert next(iter(command_ids)).endswith(":R1")
    assert all(data["instance_id"] == "p-0" for data in payloads)
    assert all(data["first_token_subject"].startswith("first_token.")
               for data in payloads)
    assert all(data["recompute_done_subject"].startswith("recompute_done.")
               for data in payloads)
    assert scheduler.prefill_queue_depths() == {"p-0": 0}
    assert scheduler.decode_free_slots() == {"d-0": 1}


@pytest.mark.asyncio
async def test_cold_dispatch_uses_versioned_endpoint_envelope():
    config = {**DEFAULT_CONFIG, "prefill_timeout_s": 10.0}
    scheduler, governor, tracker, queue, metrics, _ = make_components(config)
    scheduler.register_instance("p-0", "prefill", instance_epoch="p-pod:boot")
    scheduler.register_instance(
        "d-0", "decode", max_slots=1, instance_epoch="d-pod:boot"
    )
    request = RequestInfo(req_id="R1", token_ids=[1, 2], sampling_params={})
    tracker.add(request)
    published = []

    async def capture_publish(subject, data):
        published.append((subject, data))

    queue.publish = capture_publish
    allocator = EndpointSequenceAllocator("world-a", queue.owner_id)
    task = asyncio.create_task(schedule_loop(
        scheduler, governor, tracker, queue, metrics, config,
        operation_allocator=allocator,
    ))
    try:
        for _ in range(20):
            await asyncio.sleep(0)
            if published:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    command = published[0][1]
    assert command["schema_version"] == 1
    assert command["endpoint_ref"]["target_worker_epoch"] == "p-pod:boot"
    assert command["endpoint_ref"]["operation_id"] == "R1"
    assert command["payload"]["token_ids"] == [1, 2]
    assert request.dispatch_operation_ref.operation_seq == 1


@pytest.mark.asyncio
async def test_waiting_until_decode_slot_freed():
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)

    scheduler._decode_free_slots["d-0"] = 0

    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )

    for _ in range(20):
        await asyncio.sleep(0)

    r = tracker.get("R1")
    assert r is not None and r.state == SeqState.WAITING

    scheduler.on_decode_finished("d-0")
    assert scheduler._decode_free_slots["d-0"] == 1

    for _ in range(20):
        await asyncio.sleep(0)
        r2 = tracker.get("R1")
        if r2 and r2.state == SeqState.PREFILLING:
            break

    r = tracker.get("R1")
    assert r is not None and r.state == SeqState.PREFILLING
    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    assert "R1" not in tracker


@pytest.mark.asyncio
async def test_phase_ordering_single_advance_per_tick():
    """A request cannot leave PREFILLING in its dispatch tick."""
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    req = RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK)
    tracker.add(req)

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    r = tracker.get("R1")
    assert r is not None and r.state == SeqState.PREFILLING, (
        f"expected PREFILLING after first tick, got {r.state if r else 'gone'}"
    )

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig


def test_governor_kv_usage_update_unblocks_deferred():
    """Synchronous check: update_kv_usage below LOW_WATERMARK flushes deferred."""
    from prism_serve.scheduler.sequence_state import TransferTask
    infer_client = MagicMock()
    metrics = NullMetrics()
    config = {**DEFAULT_CONFIG, "MAX_BYTES_INFLIGHT": 512 * 1024 ** 2}
    governor = TransferGovernor(config, infer_client, metrics)

    governor.set_expected_epochs({"d-0": "e1"})
    governor.update_kv_usage(
        "d-0", KVUsageSample(0.90, "e1", time.monotonic())
    )
    task = TransferTask(req_id="R1", src="p-0", dst="d-0", kv_size=KV_SIZE_1BLOCK)
    governor.submit(task)

    assert governor.deferred_depth("d-0") == 1
    infer_client.transfer.assert_not_called()

    # Watermark recovery
    governor.update_kv_usage(
        "d-0", KVUsageSample(0.65, "e1", time.monotonic())
    )

    assert governor.deferred_depth("d-0") == 0
    infer_client.transfer.assert_called_once()


@pytest.mark.asyncio
async def test_sync_abort_client_is_bounded_by_timeout():
    """A blocking synchronous RPC must not stall the event loop."""
    import time as wall_time

    client = MagicMock()

    def blocking_abort(**_kwargs):
        wall_time.sleep(0.1)
        return {"success": True}

    client.abort_request.side_effect = blocking_abort
    started = wall_time.monotonic()
    acknowledged = await _abort_remote_request(
        client, "d-0", "owner-uid", "R1", timeout_s=0.01
    )

    assert acknowledged is False
    assert wall_time.monotonic() - started < 0.08


@pytest.mark.asyncio
async def test_metrics_gauges_updated_each_tick():
    scheduler, governor, tracker, queue, _, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=10)

    metrics = MagicMock()

    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    import prism_serve.scheduler.main_loop as ml
    orig = ml.TICK_INTERVAL_S
    ml.TICK_INTERVAL_S = 0

    loop_task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )

    for _ in range(5):
        await asyncio.sleep(0)

    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass

    ml.TICK_INTERVAL_S = orig

    gauge_names = [call.args[0] for call in metrics.gauge.call_args_list]
    assert "active_requests"     in gauge_names
    assert "waiting_requests"    in gauge_names
    assert "kv_pending_requests" in gauge_names


@pytest.mark.asyncio
async def test_decode_timeout_releases_reserved_slot():
    config = {**DEFAULT_CONFIG, "decode_timeout_s": 0.01}
    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))

    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    try:
        for _ in range(100):
            await asyncio.sleep(0)
            req = tracker.get("R1")
            if req and req.state == SeqState.PREFILLING:
                await queue._put_mock("prefill_done", {
                    "req_id": "R1",
                    "kv_size_bytes": KV_SIZE_1BLOCK,
                    "block_table": [3],
                })
            if req and req.state == SeqState.DECODING:
                req.decode_start = time.monotonic() - 1.0
            if "R1" not in tracker:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "R1" not in tracker
    assert scheduler.decode_free_slots() == {"d-0": 1}
    infer_client.abort_request.assert_called_once_with(
        instance_id="d-0", owner_id=queue.owner_id, req_id="R1"
    )


@pytest.mark.asyncio
async def test_dispatch_publish_response_loss_quarantines_and_terminates():
    """Publish failure after task creation is UNKNOWN, not safely retryable."""
    scheduler, governor, tracker, queue, metrics, _ = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))
    calls = 0

    async def flaky_publish(_subject, _data):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("NATS reconnecting")

    queue.publish = flaky_publish
    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )
    try:
        for _ in range(100):
            await asyncio.sleep(0)
            if calls == 1 and "R1" not in tracker:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert calls == 1
    assert "R1" not in tracker
    assert scheduler.quarantine_record("p-0") is not None
    assert scheduler.decode_free_slots() == {"d-0": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fault_kind", "expected_publish_count"),
    [("nats_drop", 0), ("nats_duplicate", 2), ("nats_publish_unknown", 1)],
)
async def test_command_fault_dispatch_is_unknown_and_uses_canonical_cleanup(
    fault_kind: str,
    expected_publish_count: int,
) -> None:
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill", instance_epoch="p-e1")
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="d-e1")
    request = RequestInfo(
        req_id="R1",
        kv_size_bytes=KV_SIZE_1BLOCK,
        correctness_path="cold",
    )
    tracker.add(request)
    published: list[tuple[str, dict[str, object]]] = []

    async def capture_publish(subject, command):
        published.append((subject, command))

    queue.publish = capture_publish
    infer_client.correctness_fault_checkpoint = AsyncMock(return_value={
        "fault_kind": fault_kind,
        "state": "RELEASED",
    })
    infer_client.record_correctness_fault_event = MagicMock()
    infer_client.wait_nats_command_fault_authority = AsyncMock(return_value={
        "delivery_count": expected_publish_count,
        "execution_count": 1 if expected_publish_count else 0,
    })
    allocator = EndpointSequenceAllocator("world-a", queue.owner_id)
    task = asyncio.create_task(schedule_loop(
        scheduler,
        governor,
        tracker,
        queue,
        metrics,
        DEFAULT_CONFIG,
        operation_allocator=allocator,
    ))
    try:
        for _ in range(100):
            await asyncio.sleep(0)
            if "R1" not in tracker:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert request.publish_outcome == "UNKNOWN"
    assert "R1" not in tracker
    assert scheduler.quarantine_record("p-0") is not None
    assert scheduler.decode_free_slots()["d-0"] == 1
    assert len(published) == expected_publish_count
    assert all(subject == "dispatch_prefill.p-0" for subject, _ in published)
    assert all(command is published[0][1] for _, command in published) if published else True
    event = infer_client.record_correctness_fault_event.call_args.args
    assert event[0] == "fault_injected"
    assert event[1]["fault_kind"] == fault_kind
    assert event[1]["publish_count"] == expected_publish_count
    assert event[1]["delivery_count"] == expected_publish_count
    assert event[1]["execution_count"] == (1 if expected_publish_count else 0)
    assert event[1]["endpoint_ref"] == asdict(request.dispatch_operation_ref)


@pytest.mark.asyncio
async def test_duplicate_waits_for_worker_authority_before_abort_can_win() -> None:
    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance("p-0", "prefill", instance_epoch="p-e1")
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="d-e1")
    request = RequestInfo(
        req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK, correctness_path="cold"
    )
    tracker.add(request)
    observed = asyncio.Event()
    release_observation = asyncio.Event()

    async def wait_authority(_endpoint_ref, _fault_kind):
        observed.set()
        await release_observation.wait()
        return {"delivery_count": 2, "execution_count": 1}

    queue.publish = AsyncMock()
    infer_client.correctness_fault_checkpoint = AsyncMock(return_value={
        "fault_kind": "nats_duplicate", "state": "RELEASED",
    })
    infer_client.wait_nats_command_fault_authority = wait_authority
    infer_client.record_correctness_fault_event = MagicMock()
    task = asyncio.create_task(schedule_loop(
        scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG,
        operation_allocator=EndpointSequenceAllocator("world-a", queue.owner_id),
    ))
    await asyncio.wait_for(observed.wait(), 1.0)

    assert tracker.get("R1") is request
    assert request.publish_outcome == "NOT_STARTED"
    infer_client.abort_request.assert_not_awaited()

    release_observation.set()
    for _ in range(100):
        await asyncio.sleep(0)
        if "R1" not in tracker:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert request.publish_outcome == "UNKNOWN"
    infer_client.abort_request.assert_awaited()


@pytest.mark.asyncio
async def test_prefill_publish_cancelled_is_visible_to_shutdown():
    scheduler, governor, tracker, queue, metrics, _ = make_components()
    scheduler.register_instance("p-0", "prefill")
    scheduler.register_instance("d-0", "decode", max_slots=1)
    tracker.add(RequestInfo(req_id="R1", kv_size_bytes=KV_SIZE_1BLOCK))
    entered = asyncio.Event()

    async def blocked_publish(_subject, _data):
        entered.set()
        await asyncio.Event().wait()

    queue.publish = blocked_publish
    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, DEFAULT_CONFIG)
    )
    await asyncio.wait_for(entered.wait(), 1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _abort_remaining_requests(
        tracker, scheduler, governor, queue.owner_id, 0.1, 0.1,
    )
    assert "R1" not in tracker
    assert scheduler.quarantine_record("p-0") is not None
    assert scheduler.decode_free_slots()["d-0"] == 1


@pytest.mark.asyncio
async def test_decode_abort_failure_quarantines_instance_capacity():
    """Unconfirmed remote abort must remove the D instance from scheduling."""
    config = {**DEFAULT_CONFIG, "decode_timeout_s": 0.01}
    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("d-0", "decode", max_slots=1)
    scheduler._decode_free_slots["d-0"] = 0
    infer_client.abort_request.return_value = {"success": False}
    request = RequestInfo(
        req_id="R1", state=SeqState.DECODING, decode_instance="d-0",
        decode_instance_epoch=scheduler.instance_epoch("d-0"),
    )
    request.decode_start = time.monotonic() - 1.0
    tracker.add(request)

    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    try:
        for _ in range(100):
            await asyncio.sleep(0)
            if "R1" not in tracker:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "R1" not in tracker
    assert "d-0" not in scheduler.decode_free_slots()
    assert scheduler.quarantine_record("d-0") is not None


@pytest.mark.asyncio
async def test_old_epoch_decode_done_and_timeout_cannot_mutate_reconciled_capacity():
    """e1 late paths must not release or quarantine slots rebuilt for e2."""
    scheduler, governor, tracker, queue, metrics, infer_client = make_components({
        "decode_timeout_s": 0.01,
    })
    scheduler.register_instance("d-0", "decode", max_slots=2, instance_epoch="e1")
    record = scheduler.quarantine_instance("d-0")
    scheduler.reconcile_instance(
        "d-0", "e2", record.reconciliation_token, "decode", 2, [], [],
    )
    sample = KVUsageSample(0.1, "e2", time.monotonic())
    scheduler.update_kv_usage("d-0", sample)
    governor.set_expected_epochs({"d-0": "e2"})
    governor.update_kv_usage("d-0", sample)

    assert scheduler.pick_decode_instance("new", 0) == "d-0"
    tracker.add(RequestInfo(
        req_id="old", state=SeqState.DECODING, decode_instance="d-0",
        decode_instance_epoch="e1", decode_start=time.monotonic() - 1.0,
    ))
    tracker.add(RequestInfo(
        req_id="new", state=SeqState.DECODING, decode_instance="d-0",
        decode_instance_epoch="e2",
    ))
    await queue._put_mock("decode_done", {
        "req_id": "old", "instance_epoch": "e1",
    })
    await queue._put_mock("decode_done", {
        "req_id": "new", "instance_epoch": "e2",
    })

    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, {
            **DEFAULT_CONFIG, "decode_timeout_s": 0.01,
        })
    )
    try:
        for _ in range(100):
            await asyncio.sleep(0)
            if len(tracker) == 0:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert scheduler.decode_free_slots()["d-0"] == 2
    assert scheduler.quarantine_record("d-0") is None
    infer_client.abort_request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", [SeqState.PREFIX_PREFILLING, SeqState.KV_PENDING],
)
async def test_malformed_decode_progress_does_not_query_before_decode(state):
    from prism_serve.gateway.output import GatewayOutputBuffer

    scheduler, governor, tracker, queue, metrics, infer_client = make_components()
    scheduler.register_instance(
        "d-0", "decode", max_slots=1, instance_epoch="d-e1"
    )
    infer_client.week12_network_control = True
    infer_client.request_output = AsyncMock(return_value={
        "req_id": "R1",
        "instance_epoch": "d-e1",
        "operation_id": "op-1",
        "output_seq_no": 0,
        "token_ids": [],
        "terminal": False,
    })
    now = time.monotonic()
    tracker.add(RequestInfo(
        req_id="R1",
        state=state,
        decode_instance="d-0",
        decode_instance_epoch="d-e1",
        active_operation_id="op-1",
        transfer_operation_id="op-1",
        kv_sent_at=now,
        suffix_prefill_started_at=now,
    ))
    output = GatewayOutputBuffer()
    await queue._put_mock("decode_progress", {
        "req_id": "R1",
        "instance_epoch": "d-e1",
        "operation_id": "op-1",
        "output_seq_no": 2,
        "token_ids": [7],
    })

    task = asyncio.create_task(schedule_loop(
        scheduler,
        governor,
        tracker,
        queue,
        metrics,
        DEFAULT_CONFIG,
        output_buffer=output,
    ))
    try:
        for _ in range(100):
            await asyncio.sleep(0)
            if queue._inbox["decode_progress"].empty():
                break
        assert queue._inbox["decode_progress"].empty()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    infer_client.request_output.assert_not_awaited()


@pytest.mark.asyncio
async def test_e1_to_e2_kv_abort_await_cannot_trigger_recompute():
    config = {**DEFAULT_CONFIG, "kv_transfer_timeout_s": 0.01}
    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill", instance_epoch="p-e1")
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="e1")
    scheduler._decode_free_slots["d-0"] = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def abort_transfer(**_kwargs):
        entered.set()
        await release.wait()
        return {"success": True}

    infer_client.abort_transfer = AsyncMock(side_effect=abort_transfer)
    infer_client.transfer.side_effect = lambda **_kwargs: None
    governor.submit(TransferTask(
        req_id="R1", operation_id="op-e1", src="p-0", dst="d-0", kv_size=1,
    ))
    tracker.add(RequestInfo(
        req_id="R1", state=SeqState.KV_PENDING,
        prefill_instance="p-0", decode_instance="d-0",
        prefill_instance_epoch="p-e1", decode_instance_epoch="e1",
        transfer_operation_id="op-e1",
        kv_sent_at=time.monotonic() - 1.0,
    ))
    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    await asyncio.wait_for(entered.wait(), 1.0)
    record = scheduler.quarantine_instance("d-0")
    scheduler.reconcile_instance(
        "d-0", "e2", record.reconciliation_token, "decode", 1, [], [], [],
    )
    release.set()
    for _ in range(100):
        await asyncio.sleep(0)
        if "R1" not in tracker:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert infer_client.reset_to_waiting.call_count == 0
    assert scheduler.decode_free_slots()["d-0"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc_result", [True, False, "response_lost"])
async def test_phase3_source_epoch_flip_quarantines_current_d(rpc_result):
    config = {**DEFAULT_CONFIG, "kv_transfer_timeout_s": 0.01}
    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("p-0", "prefill", instance_epoch="p-e1")
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="d-e1")
    scheduler._decode_free_slots["d-0"] = 0

    async def flip_source(**_kwargs):
        record = scheduler.quarantine_instance("p-0")
        scheduler.reconcile_instance(
            "p-0", "p-e2", record.reconciliation_token, "prefill", 0,
            [], [], [],
        )
        if rpc_result == "response_lost":
            raise ConnectionError("response lost after source flip")
        return {"success": rpc_result}

    infer_client.abort_transfer = AsyncMock(side_effect=flip_source)
    infer_client.transfer.side_effect = lambda **_kwargs: None
    governor.submit(TransferTask(
        req_id="R1", operation_id="op", src="p-0", dst="d-0", kv_size=1,
    ))
    tracker.add(RequestInfo(
        req_id="R1", state=SeqState.KV_PENDING,
        prefill_instance="p-0", decode_instance="d-0",
        prefill_instance_epoch="p-e1", decode_instance_epoch="d-e1",
        transfer_operation_id="op", kv_sent_at=time.monotonic() - 1.0,
    ))
    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    for _ in range(200):
        await asyncio.sleep(0)
        if "R1" not in tracker:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    record = scheduler.quarantine_record("d-0")
    assert record is not None
    assert record.uncertain_transfer_operations == (
        (queue.owner_id, "R1", "op", "d-e1"),
    )
    assert governor.is_drained()


@pytest.mark.asyncio
async def test_e1_to_e2_recompute_timeout_cannot_retry_or_mutate_e2():
    config = {**DEFAULT_CONFIG, "recompute_timeout_s": 0.01}
    scheduler, governor, tracker, queue, metrics, infer_client = make_components(config)
    scheduler.register_instance("d-0", "decode", max_slots=1, instance_epoch="e1")
    record = scheduler.quarantine_instance("d-0")
    scheduler.reconcile_instance(
        "d-0", "e2", record.reconciliation_token, "decode", 1, [], [], [],
    )
    tracker.add(RequestInfo(
        req_id="R1", state=SeqState.RECOMPUTING,
        decode_instance="d-0", decode_instance_epoch="e1",
        recompute_start=time.monotonic() - 1.0,
    ))
    task = asyncio.create_task(
        schedule_loop(scheduler, governor, tracker, queue, metrics, config)
    )
    for _ in range(100):
        await asyncio.sleep(0)
        if "R1" not in tracker:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert infer_client.reset_to_waiting.call_count == 0
    assert scheduler.decode_free_slots()["d-0"] == 1

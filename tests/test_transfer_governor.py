"""Unit tests for TransferGovernor (scheduler/transfer_governor.py)."""

from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock, call

from prism_serve.scheduler.transfer_governor import (
    TransferDispatchError,
    TransferGovernor,
)
from prism_serve.scheduler.sequence_state import TransferTask
from prism_serve.scheduler.scheduler import KVUsageSample


def make_governor(
    high_wm: float = 0.85,
    low_wm: float = 0.70,
    max_bytes: int = 256 * 1024 ** 2,   # 256 MB
    max_recompute: int = 2,
    timeout_s: float = 30.0,
) -> tuple[TransferGovernor, MagicMock, MagicMock]:
    config = {
        "HIGH_WATERMARK":       high_wm,
        "LOW_WATERMARK":        low_wm,
        "MAX_BYTES_INFLIGHT":   max_bytes,
        "max_recompute_attempts": max_recompute,
        "kv_transfer_timeout_s":  timeout_s,
    }
    infer_client = MagicMock()
    metrics = MagicMock()
    gov = TransferGovernor(config, infer_client, metrics)
    _set_usage(gov, 0.0)
    return gov, infer_client, metrics


def _set_usage(gov: TransferGovernor, ratio: float, epoch: str = "e1") -> None:
    gov.set_expected_epochs({"d-0": epoch})
    gov.update_kv_usage("d-0", KVUsageSample(ratio, epoch, time.monotonic()))


def _task(req_id: str, dst: str, kv_size: int, priority: int = 1) -> TransferTask:
    return TransferTask(req_id=req_id, src="p-0", dst=dst,
                        kv_size=kv_size, priority=priority,
                        operation_id=f"op-{req_id}")


def test_can_send_normal():
    gov, _, _ = make_governor()
    _set_usage(gov, 0.50)
    assert gov.can_send("d-0", 100 * 1024 ** 2)   # 100 MB


def test_can_send_blocked_by_high_watermark():
    gov, _, _ = make_governor()
    _set_usage(gov, 0.90)
    assert not gov.can_send("d-0", 1)


def test_can_send_blocked_by_bytes_cap():
    gov, _, _ = make_governor(max_bytes=256 * 1024 ** 2)
    _set_usage(gov, 0.50)
    gov._bytes_inflight["d-0"] = 200 * 1024 ** 2
    assert not gov.can_send("d-0", 100 * 1024 ** 2)


def test_week12_bytes_cap_is_independent_per_pair():
    gov, infer_client, _ = make_governor(max_bytes=1024 ** 3)
    _set_usage(gov, 0.50)
    block_bytes = 29_360_128

    for index in range(36):
        gov.submit(_task(f"p0-{index}", "d-0", block_bytes))
    assert gov.bytes_inflight_for_pair("p-0", "d-0") == 36 * block_bytes
    assert not gov.can_send("d-0", block_bytes, src="p-0")


    assert gov.can_send("d-0", block_bytes, src="p-1")
    assert infer_client.transfer.call_count == 36


def test_oversized_task_can_own_idle_destination():
    gov, infer_client, _ = make_governor(max_bytes=256 * 1024 ** 2)
    gov.submit(_task("R1", "d-0", 448 * 1024 ** 2))

    infer_client.transfer.assert_called_once()
    assert gov.deferred_depth("d-0") == 0
    assert gov.bytes_inflight("d-0") == 448 * 1024 ** 2


def test_oversized_task_waits_until_destination_is_idle():
    gov, infer_client, _ = make_governor(max_bytes=256 * 1024 ** 2)
    gov.submit(_task("R0", "d-0", 112 * 1024 ** 2))
    first_done = infer_client.transfer.call_args.kwargs["on_complete"]
    gov.submit(_task("R1", "d-0", 448 * 1024 ** 2))

    assert gov.deferred_depth("d-0") == 1
    first_done()

    assert gov.deferred_depth("d-0") == 0
    assert infer_client.transfer.call_count == 2


def test_can_send_low_priority_stricter_cap():
    gov, _, _ = make_governor(max_bytes=256 * 1024 ** 2)
    _set_usage(gov, 0.50)
    # Low-priority cap = 256 * 0.3 = 76.8 MB
    gov._bytes_inflight["d-0"] = 50 * 1024 ** 2   # 50 MB in-flight
    # 50 + 40 = 90 > 76.8 MB → blocked for priority=0
    assert not gov.can_send("d-0", 40 * 1024 ** 2, priority=0)
    assert gov.can_send("d-0", 40 * 1024 ** 2, priority=1)


def test_can_send_unknown_dst_fails_closed():
    gov, _, _ = make_governor()
    assert not gov.can_send("d-new", 1)


def test_kv_usage_stale_fails_closed():
    gov, _, _ = make_governor()
    gov.set_expected_epochs({"d-0": "e1"})
    gov.update_kv_usage(
        "d-0", KVUsageSample(0.1, "e1", time.monotonic() - 31.0)
    )
    assert not gov.can_send("d-0", 1)


def test_kv_usage_epoch_change_fails_closed():
    gov, _, _ = make_governor()
    gov.set_expected_epochs({"d-0": "e2"})
    gov.update_kv_usage("d-0", KVUsageSample(0.1, "e1", time.monotonic()))
    assert not gov.can_send("d-0", 1)


def test_submit_dispatches_when_clear():
    gov, infer_client, _ = make_governor()
    _set_usage(gov, 0.50)
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))
    infer_client.transfer.assert_called_once()
    assert gov._bytes_inflight["d-0"] == 112 * 1024 ** 2


def test_dispatch_exception_rolls_back_local_ledger():
    gov, infer_client, _ = make_governor()
    infer_client.transfer.side_effect = ConnectionError("transport unavailable")

    with pytest.raises(TransferDispatchError):
        gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))

    assert gov.bytes_inflight("d-0") == 0
    assert gov.task_state("R1") == "none"
    assert gov.deferred_depth("d-0") == 0
    assert gov.is_drained()


def test_dispatch_passes_operation_id():
    gov, infer_client, _ = make_governor()

    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))

    assert infer_client.transfer.call_args.kwargs["operation_id"] == "op-R1"


def test_submit_deferred_when_congested():
    gov, infer_client, _ = make_governor()
    _set_usage(gov, 0.90)
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))
    infer_client.transfer.assert_not_called()
    assert len(gov._deferred["d-0"]) == 1


def test_on_complete_releases_inflight():
    gov, infer_client, _ = make_governor()
    _set_usage(gov, 0.50)
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))
    kwargs = infer_client.transfer.call_args.kwargs
    kwargs["on_complete"]()
    assert gov._bytes_inflight["d-0"] == 0


def test_on_complete_fires_user_callback():
    gov, infer_client, _ = make_governor()
    _set_usage(gov, 0.50)
    cb = MagicMock()
    task = _task("R1", "d-0", 112 * 1024 ** 2)
    task.on_complete = cb
    gov.submit(task)
    kwargs = infer_client.transfer.call_args.kwargs
    kwargs["on_complete"]()
    cb.assert_called_once()


def test_on_complete_does_not_own_success_metric():
    """The domain callback owns success metrics, avoiding double counting."""
    gov, infer_client, metrics = make_governor()
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))

    infer_client.transfer.call_args.kwargs["on_complete"]()

    metrics.increment.assert_not_called()


def test_cancel_removes_deferred_task():
    gov, infer_client, _ = make_governor()
    _set_usage(gov, 0.90)
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))

    assert gov.cancel("R1") is True
    _set_usage(gov, 0.0)
    gov.tick()

    assert gov.deferred_depth("d-0") == 0
    infer_client.transfer.assert_not_called()


def test_cancel_inflight_ignores_late_callback():
    """A late timed-out RPC must not advance state or release capacity twice."""
    gov, infer_client, _ = make_governor()
    callback = MagicMock()
    task = _task("R1", "d-0", 112 * 1024 ** 2)
    task.on_complete = callback
    gov.submit(task)
    late_callback = infer_client.transfer.call_args.kwargs["on_complete"]

    assert gov.cancel("R1") is True
    late_callback()

    assert gov.bytes_inflight("d-0") == 0
    callback.assert_not_called()


def test_operation_scoped_ownership_and_cancel_reject_stale_operation():
    """An old operation must not cancel the request's current transfer."""
    gov, _, _ = make_governor()
    task = _task("R1", "d-0", 112 * 1024 ** 2)
    task.operation_id = "op-current"
    gov.submit(task)

    assert gov.owns("R1", "op-current")
    assert not gov.owns("R1", "op-stale")
    assert gov.task_state("R1", "op-stale") == "none"
    assert not gov.cancel("R1", "op-stale")
    assert gov.owns("R1", "op-current")
    assert gov.cancel("R1", "op-current")
    assert gov.is_drained()


def test_flush_deferred_on_low_watermark():
    gov, infer_client, _ = make_governor()
    _set_usage(gov, 0.90)
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))
    assert len(gov._deferred["d-0"]) == 1
    infer_client.transfer.assert_not_called()

    _set_usage(gov, 0.60)
    gov.tick()
    assert len(gov._deferred["d-0"]) == 0
    infer_client.transfer.assert_called_once()


def test_flush_deferred_multiple_tasks():
    gov, infer_client, _ = make_governor(max_bytes=512 * 1024 ** 2)
    _set_usage(gov, 0.90)
    for i in range(3):
        gov.submit(_task(f"R{i}", "d-0", 100 * 1024 ** 2))
    assert len(gov._deferred["d-0"]) == 3

    _set_usage(gov, 0.50)
    gov.tick()
    assert len(gov._deferred["d-0"]) == 0
    assert infer_client.transfer.call_count == 3


def test_flush_stops_when_still_congested():
    gov, infer_client, _ = make_governor()
    _set_usage(gov, 0.90)
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))
    gov.submit(_task("R2", "d-0", 112 * 1024 ** 2))

    # Still congested — nothing should flush
    _set_usage(gov, 0.88)
    gov.tick()
    assert len(gov._deferred["d-0"]) == 2
    infer_client.transfer.assert_not_called()


def test_update_kv_usage_triggers_flush_on_recovery():
    gov, infer_client, _ = make_governor()
    _set_usage(gov, 0.90)
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))
    assert len(gov._deferred["d-0"]) == 1

    # Simulate kv_usage dropping from high → below low watermark
    _set_usage(gov, 0.60)
    assert len(gov._deferred["d-0"]) == 0
    infer_client.transfer.assert_called_once()


def test_recompute_first_failure():
    gov, infer_client, metrics = make_governor(max_recompute=2)
    result = gov.on_transfer_failure("R1", "d-0", "timeout")
    assert result == "recompute"
    assert gov._recompute_counts["R1"] == 1


def test_recompute_second_failure():
    gov, _, _ = make_governor(max_recompute=2)
    gov._recompute_counts["R1"] = 1
    result = gov.on_transfer_failure("R1", "d-0", "timeout")
    assert result == "recompute"
    assert gov._recompute_counts["R1"] == 2


def test_abort_after_max_recompute():
    gov, _, _ = make_governor(max_recompute=2)
    gov._recompute_counts["R1"] = 2
    result = gov.on_transfer_failure("R1", "d-0", "timeout")
    assert result == "abort"
    assert "R1" not in gov._recompute_counts


def test_trigger_recompute_calls_rpc():
    gov, infer_client, _ = make_governor()
    gov.trigger_recompute("R1", "d-0")
    infer_client.reset_to_waiting.assert_called_once_with("d-0", "R1")


def test_on_transfer_failure_metrics_recompute():
    gov, _, metrics = make_governor()
    gov.on_transfer_failure("R1", "d-0", "timeout")
    metrics.increment.assert_called_with(
        "kv_transfer_recompute_total",
        labels={"reason": "timeout", "attempt": "1"},
    )


def test_on_transfer_failure_metrics_abort():
    gov, _, metrics = make_governor(max_recompute=0)
    gov.on_transfer_failure("R1", "d-0", "network")
    metrics.increment.assert_called_with(
        "kv_transfer_abort_total", labels={"reason": "network"}
    )


def test_all_inflight_zero_initially():
    gov, _, _ = make_governor()
    assert gov.all_inflight_zero()


def test_all_inflight_zero_false_when_in_flight():
    gov, infer_client, _ = make_governor()
    _set_usage(gov, 0.50)
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))
    assert not gov.all_inflight_zero()


def test_is_drained_includes_deferred_queue():
    gov, _, _ = make_governor()
    _set_usage(gov, 0.90)
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))
    assert gov.all_inflight_zero()
    assert not gov.is_drained()


def test_finish_request_clears_retry_and_transfer_state():
    gov, infer_client, _ = make_governor()
    gov._recompute_counts["R1"] = 2
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))

    gov.finish_request("R1")

    assert "R1" not in gov._recompute_counts
    assert gov.bytes_inflight("d-0") == 0
    assert gov.is_drained()
    late_callback = infer_client.transfer.call_args.kwargs["on_complete"]
    late_callback()
    assert gov.is_drained()

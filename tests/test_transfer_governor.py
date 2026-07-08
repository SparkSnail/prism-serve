"""Unit tests for TransferGovernor (scheduler/transfer_governor.py)."""
import pytest
from unittest.mock import MagicMock, call

from prism_serve.scheduler.transfer_governor import TransferGovernor
from prism_serve.scheduler.sequence_state import TransferTask


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
    return gov, infer_client, metrics


def _task(req_id: str, dst: str, kv_size: int, priority: int = 1) -> TransferTask:
    return TransferTask(req_id=req_id, src="p-0", dst=dst,
                        kv_size=kv_size, priority=priority)


# ---------------------------------------------------------------------------
# can_send
# ---------------------------------------------------------------------------

def test_can_send_normal():
    gov, _, _ = make_governor()
    gov._kv_usage["d-0"] = 0.50
    assert gov.can_send("d-0", 100 * 1024 ** 2)   # 100 MB


def test_can_send_blocked_by_high_watermark():
    gov, _, _ = make_governor()
    gov._kv_usage["d-0"] = 0.90
    assert not gov.can_send("d-0", 1)


def test_can_send_blocked_by_bytes_cap():
    gov, _, _ = make_governor(max_bytes=256 * 1024 ** 2)
    gov._kv_usage["d-0"] = 0.50
    gov._bytes_inflight["d-0"] = 200 * 1024 ** 2
    # 200 + 100 = 300 > 256 MB
    assert not gov.can_send("d-0", 100 * 1024 ** 2)


def test_can_send_low_priority_stricter_cap():
    gov, _, _ = make_governor(max_bytes=256 * 1024 ** 2)
    gov._kv_usage["d-0"] = 0.50
    # Low-priority cap = 256 * 0.3 = 76.8 MB
    gov._bytes_inflight["d-0"] = 50 * 1024 ** 2   # 50 MB in-flight
    # 50 + 40 = 90 > 76.8 MB → blocked for priority=0
    assert not gov.can_send("d-0", 40 * 1024 ** 2, priority=0)
    # But fine for priority=1 (256 MB cap)
    assert gov.can_send("d-0", 40 * 1024 ** 2, priority=1)


def test_can_send_unknown_dst_defaults_zero_usage():
    gov, _, _ = make_governor()
    # No kv_usage entry → defaults to 0.0 → can send
    assert gov.can_send("d-new", 1)


# ---------------------------------------------------------------------------
# submit → dispatch vs deferred
# ---------------------------------------------------------------------------

def test_submit_dispatches_when_clear():
    gov, infer_client, _ = make_governor()
    gov._kv_usage["d-0"] = 0.50
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))
    infer_client.transfer.assert_called_once()
    assert gov._bytes_inflight["d-0"] == 112 * 1024 ** 2


def test_submit_deferred_when_congested():
    gov, infer_client, _ = make_governor()
    gov._kv_usage["d-0"] = 0.90   # congested
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))
    infer_client.transfer.assert_not_called()
    assert len(gov._deferred["d-0"]) == 1


def test_on_complete_releases_inflight():
    gov, infer_client, _ = make_governor()
    gov._kv_usage["d-0"] = 0.50
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))
    # Trigger on_complete by calling what _dispatch registered
    kwargs = infer_client.transfer.call_args.kwargs
    kwargs["on_complete"]()
    assert gov._bytes_inflight["d-0"] == 0


def test_on_complete_fires_user_callback():
    gov, infer_client, _ = make_governor()
    gov._kv_usage["d-0"] = 0.50
    cb = MagicMock()
    task = _task("R1", "d-0", 112 * 1024 ** 2)
    task.on_complete = cb
    gov.submit(task)
    kwargs = infer_client.transfer.call_args.kwargs
    kwargs["on_complete"]()
    cb.assert_called_once()


# ---------------------------------------------------------------------------
# deferred queue flush
# ---------------------------------------------------------------------------

def test_flush_deferred_on_low_watermark():
    gov, infer_client, _ = make_governor()
    gov._kv_usage["d-0"] = 0.90   # congested
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))
    assert len(gov._deferred["d-0"]) == 1
    infer_client.transfer.assert_not_called()

    gov._kv_usage["d-0"] = 0.60   # below LOW_WATERMARK → triggers flush
    gov.tick()
    assert len(gov._deferred["d-0"]) == 0
    infer_client.transfer.assert_called_once()


def test_flush_deferred_multiple_tasks():
    gov, infer_client, _ = make_governor(max_bytes=512 * 1024 ** 2)
    gov._kv_usage["d-0"] = 0.90
    for i in range(3):
        gov.submit(_task(f"R{i}", "d-0", 100 * 1024 ** 2))
    assert len(gov._deferred["d-0"]) == 3

    gov._kv_usage["d-0"] = 0.50
    gov.tick()
    # All three should be flushed (3 × 100 = 300 < 512 MB)
    assert len(gov._deferred["d-0"]) == 0
    assert infer_client.transfer.call_count == 3


def test_flush_stops_when_still_congested():
    gov, infer_client, _ = make_governor()
    gov._kv_usage["d-0"] = 0.90
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))
    gov.submit(_task("R2", "d-0", 112 * 1024 ** 2))

    # Still congested — nothing should flush
    gov._kv_usage["d-0"] = 0.88   # > HIGH_WATERMARK (0.85)
    gov.tick()
    assert len(gov._deferred["d-0"]) == 2
    infer_client.transfer.assert_not_called()


# ---------------------------------------------------------------------------
# update_kv_usage triggers flush
# ---------------------------------------------------------------------------

def test_update_kv_usage_triggers_flush_on_recovery():
    gov, infer_client, _ = make_governor()
    gov._kv_usage["d-0"] = 0.90
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))
    assert len(gov._deferred["d-0"]) == 1

    # Simulate kv_usage dropping from high → below low watermark
    gov.update_kv_usage("d-0", 0.60)
    assert len(gov._deferred["d-0"]) == 0
    infer_client.transfer.assert_called_once()


# ---------------------------------------------------------------------------
# Recompute fallback
# ---------------------------------------------------------------------------

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
    gov._recompute_counts["R1"] = 2   # already at max
    result = gov.on_transfer_failure("R1", "d-0", "timeout")
    assert result == "abort"
    # Counter cleared on abort
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


# ---------------------------------------------------------------------------
# all_inflight_zero
# ---------------------------------------------------------------------------

def test_all_inflight_zero_initially():
    gov, _, _ = make_governor()
    assert gov.all_inflight_zero()


def test_all_inflight_zero_false_when_in_flight():
    gov, infer_client, _ = make_governor()
    gov._kv_usage["d-0"] = 0.50
    gov.submit(_task("R1", "d-0", 112 * 1024 ** 2))
    assert not gov.all_inflight_zero()

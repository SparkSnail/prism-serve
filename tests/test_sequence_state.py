"""Unit tests for sequence_state.py (SeqState, RequestInfo, RequestTracker)."""
import time
import pytest
from unittest.mock import MagicMock

from prism_serve.scheduler.sequence_state import (
    SeqState, RequestInfo, RequestTracker,
    InstanceSlot, TransferTask,
)


def make_tracker():
    return RequestTracker(metrics=MagicMock())


def test_seqstate_all_values():
    names = {s.name for s in SeqState}
    expected = {
        "WAITING", "PREFILLING", "KV_PENDING", "RECOMPUTING",
        "DECODING", "FINISHED", "ABORTED",
    }
    assert names == expected, f"missing states: {expected - names}"


def test_valid_transitions_full_path():
    t = make_tracker()
    req = RequestInfo(req_id="R1")
    t.add(req)
    t.transition("R1", SeqState.PREFILLING)
    t.transition("R1", SeqState.KV_PENDING)
    t.transition("R1", SeqState.DECODING)
    t.transition("R1", SeqState.FINISHED)
    assert t._requests["R1"].state == SeqState.FINISHED


def test_recompute_transition_keeps_original_decode_assignment():
    t = make_tracker()
    req = RequestInfo(req_id="R1")
    t.add(req)
    t.transition("R1", SeqState.PREFILLING)
    t.transition("R1", SeqState.KV_PENDING)
    t.transition("R1", SeqState.RECOMPUTING)
    assert t._requests["R1"].state == SeqState.RECOMPUTING
    t.transition("R1", SeqState.DECODING)
    assert t._requests["R1"].state == SeqState.DECODING


def test_any_state_can_transition_to_aborted():
    for start, path in [
        (SeqState.WAITING,    []),
        (SeqState.PREFILLING, [SeqState.PREFILLING]),
        (SeqState.KV_PENDING, [SeqState.PREFILLING, SeqState.KV_PENDING]),
        (SeqState.DECODING,   [SeqState.PREFILLING, SeqState.KV_PENDING, SeqState.DECODING]),
        (SeqState.RECOMPUTING, [
            SeqState.PREFILLING, SeqState.KV_PENDING, SeqState.RECOMPUTING,
        ]),
    ]:
        t = make_tracker()
        req = RequestInfo(req_id="R1")
        t.add(req)
        for s in path:
            t.transition("R1", s)
        t.transition("R1", SeqState.ABORTED)
        assert t._requests["R1"].state == SeqState.ABORTED


def test_invalid_transition_raises():
    t = make_tracker()
    req = RequestInfo(req_id="R1")
    t.add(req)
    with pytest.raises(AssertionError, match="illegal transition"):
        t.transition("R1", SeqState.DECODING)


def test_duplicate_add_raises():
    t = make_tracker()
    req = RequestInfo(req_id="R1")
    t.add(req)
    with pytest.raises(AssertionError, match="duplicate"):
        t.add(RequestInfo(req_id="R1"))


def test_transition_missing_req_raises():
    t = make_tracker()
    with pytest.raises(KeyError):
        t.transition("nonexistent", SeqState.PREFILLING)


def test_timestamps_set_on_transition():
    t = make_tracker()
    req = RequestInfo(req_id="R1")
    t.add(req)

    t.transition("R1", SeqState.PREFILLING)
    assert t._requests["R1"].prefill_start > 0

    t.transition("R1", SeqState.KV_PENDING)
    assert t._requests["R1"].kv_sent_at > 0

    t.transition("R1", SeqState.DECODING)
    assert t._requests["R1"].decode_start > 0

    t.transition("R1", SeqState.FINISHED)
    assert t._requests["R1"].finished_at > 0


def test_ttft_ms():
    req = RequestInfo(req_id="R1")
    req.arrived_at = 0.0
    req.first_token_at = 0.010
    assert abs(req.ttft_ms() - 10.0) < 0.001


def test_ttft_ms_not_started():
    req = RequestInfo(req_id="R1")
    assert req.ttft_ms() == -1.0


def test_record_first_token_is_idempotent():
    metrics = MagicMock()
    tracker = RequestTracker(metrics)
    req = RequestInfo(req_id="R1")
    tracker.add(req)
    tracker.transition("R1", SeqState.PREFILLING)
    tracker.transition("R1", SeqState.KV_PENDING)
    tracker.transition("R1", SeqState.DECODING)

    assert tracker.record_first_token("R1") is True
    first_token_at = req.first_token_at
    assert tracker.record_first_token("R1") is False
    assert req.first_token_at == first_token_at
    metrics.observe.assert_called_once()


def test_is_stuck_kv_pending():
    req = RequestInfo(req_id="R1")
    req.state = SeqState.KV_PENDING
    req.kv_sent_at = time.monotonic() - 60.0   # 60 s ago
    assert req.is_stuck(timeout_s=30.0)


def test_is_not_stuck_non_kv_pending():
    req = RequestInfo(req_id="R1")
    req.state = SeqState.DECODING
    req.kv_sent_at = time.monotonic() - 60.0
    assert not req.is_stuck(timeout_s=30.0)


def test_get_stuck_requests():
    t = make_tracker()
    for rid in ("R1", "R2", "R3"):
        r = RequestInfo(req_id=rid)
        t.add(r)
        t.transition(rid, SeqState.PREFILLING)
        t.transition(rid, SeqState.KV_PENDING)

    t._requests["R1"].kv_sent_at = time.monotonic() - 60.0
    t._requests["R3"].kv_sent_at = time.monotonic() - 45.0

    stuck = t.get_stuck_requests(timeout_s=30.0)
    assert {r.req_id for r in stuck} == {"R1", "R3"}
    assert stuck[0].req_id == "R1"


def test_remove():
    t = make_tracker()
    req = RequestInfo(req_id="R1")
    t.add(req)
    removed = t.remove("R1")
    assert removed is req
    assert "R1" not in t


def test_len_and_contains():
    t = make_tracker()
    assert len(t) == 0
    t.add(RequestInfo(req_id="R1"))
    t.add(RequestInfo(req_id="R2"))
    assert len(t) == 2
    assert "R1" in t
    assert "XX" not in t


def test_instance_slot_compute_max_slots():
    # TP=1 reference: four 28 MiB blocks use 112 MiB per average sequence.
    n = InstanceSlot.compute_max_slots(
        gpu_memory_gb=80.0,
        model_weight_gb=14.0,
        avg_seq_kv_gb=0.109375,
        safety_margin=0.85,
    )
    assert 510 <= n <= 513, f"expected ~512, got {n}"


def test_instance_slot_is_stale():
    slot = InstanceSlot(instance_id="d-0", seq_id="R1")
    slot.allocated_at = time.monotonic() - 400.0   # 400 s ago
    assert slot.is_stale(timeout_s=300.0)


def test_instance_slot_not_stale_when_idle():
    slot = InstanceSlot(instance_id="d-0", seq_id=None)
    slot.allocated_at = time.monotonic() - 400.0
    assert not slot.is_stale(timeout_s=300.0)

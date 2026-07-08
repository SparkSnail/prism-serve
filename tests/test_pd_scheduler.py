"""Unit tests for PDScheduler (scheduler/scheduler.py)."""
import pytest
from prism_serve.scheduler.scheduler import PDScheduler


def make_scheduler(extra: dict | None = None) -> PDScheduler:
    config = {
        "HIGH_WATERMARK": 0.85,
        "min_decode_instances": 1,
        "max_decode_instances": 10,
        "kv_per_instance_bytes": 56 * 1024 ** 3,
    }
    if extra:
        config.update(extra)
    return PDScheduler(config)


# ---------------------------------------------------------------------------
# register_instance
# ---------------------------------------------------------------------------

def test_register_prefill():
    sch = make_scheduler()
    sch.register_instance("p-0", "prefill")
    assert "p-0" in sch._prefill_load
    assert sch._prefill_load["p-0"] == 0


def test_register_decode():
    sch = make_scheduler()
    sch.register_instance("d-0", "decode", max_slots=100)
    assert "d-0" in sch._decode_free_slots
    assert sch._decode_free_slots["d-0"] == 100
    assert sch._kv_usage["d-0"] == 0.0


def test_register_decode_no_slots_raises():
    sch = make_scheduler()
    with pytest.raises(AssertionError, match="max_slots"):
        sch.register_instance("d-0", "decode", max_slots=0)


def test_register_unknown_role_raises():
    sch = make_scheduler()
    with pytest.raises(ValueError, match="unknown role"):
        sch.register_instance("x-0", "unknown_role")


def test_deregister():
    sch = make_scheduler()
    sch.register_instance("p-0", "prefill")
    sch.register_instance("d-0", "decode", max_slots=50)
    sch.deregister_instance("p-0")
    sch.deregister_instance("d-0")
    assert "p-0" not in sch._prefill_load
    assert "d-0" not in sch._decode_free_slots


# ---------------------------------------------------------------------------
# pick_prefill_instance
# ---------------------------------------------------------------------------

def test_pick_prefill_shortest_queue():
    sch = make_scheduler()
    sch.register_instance("p-0", "prefill")
    sch.register_instance("p-1", "prefill")
    sch._prefill_load["p-0"] = 3
    sch._prefill_load["p-1"] = 1
    result = sch.pick_prefill_instance("req-1")
    assert result == "p-1"


def test_pick_prefill_increments_load():
    sch = make_scheduler()
    sch.register_instance("p-0", "prefill")
    sch.pick_prefill_instance("req-1")
    assert sch._prefill_load["p-0"] == 1


def test_pick_prefill_none_when_empty():
    sch = make_scheduler()
    assert sch.pick_prefill_instance("req-1") is None


# ---------------------------------------------------------------------------
# pick_decode_instance
# ---------------------------------------------------------------------------

def test_pick_decode_most_slots():
    sch = make_scheduler()
    sch.register_instance("d-0", "decode", max_slots=100)
    sch.register_instance("d-1", "decode", max_slots=100)
    sch._decode_free_slots["d-0"] = 50
    sch._decode_free_slots["d-1"] = 80
    result = sch.pick_decode_instance("req-1", kv_size_bytes=0)
    assert result == "d-1"


def test_pick_decode_decrements_slot():
    sch = make_scheduler()
    sch.register_instance("d-0", "decode", max_slots=100)
    sch.pick_decode_instance("req-1", kv_size_bytes=0)
    assert sch._decode_free_slots["d-0"] == 99


def test_pick_decode_excludes_congested():
    sch = make_scheduler()
    sch.register_instance("d-0", "decode", max_slots=100)
    sch.register_instance("d-1", "decode", max_slots=100)
    sch._kv_usage["d-0"] = 0.90   # above HIGH_WATERMARK
    sch._kv_usage["d-1"] = 0.50
    result = sch.pick_decode_instance("req-1", kv_size_bytes=0)
    assert result == "d-1"


def test_pick_decode_excludes_full():
    sch = make_scheduler()
    sch.register_instance("d-0", "decode", max_slots=100)
    sch._decode_free_slots["d-0"] = 0
    result = sch.pick_decode_instance("req-1", kv_size_bytes=0)
    assert result is None


def test_pick_decode_all_congested_returns_none():
    sch = make_scheduler()
    sch.register_instance("d-0", "decode", max_slots=100)
    sch._kv_usage["d-0"] = 0.90
    result = sch.pick_decode_instance("req-1", kv_size_bytes=0)
    assert result is None


# ---------------------------------------------------------------------------
# Feedback callbacks
# ---------------------------------------------------------------------------

def test_on_prefill_done_decrements():
    sch = make_scheduler()
    sch.register_instance("p-0", "prefill")
    sch._prefill_load["p-0"] = 3
    sch.on_prefill_done("p-0")
    assert sch._prefill_load["p-0"] == 2


def test_on_prefill_done_no_underflow():
    sch = make_scheduler()
    sch.register_instance("p-0", "prefill")
    sch._prefill_load["p-0"] = 0
    sch.on_prefill_done("p-0")   # should not go negative
    assert sch._prefill_load["p-0"] == 0


def test_on_decode_finished_increments():
    sch = make_scheduler()
    sch.register_instance("d-0", "decode", max_slots=100)
    sch._decode_free_slots["d-0"] = 99
    sch.on_decode_finished("d-0")
    assert sch._decode_free_slots["d-0"] == 100


def test_update_kv_usage_clamps():
    sch = make_scheduler()
    sch.register_instance("d-0", "decode", max_slots=100)
    sch.update_kv_usage("d-0", 1.5)   # above 1.0
    assert sch._kv_usage["d-0"] == 1.0
    sch.update_kv_usage("d-0", -0.1)  # below 0.0
    assert sch._kv_usage["d-0"] == 0.0


# ---------------------------------------------------------------------------
# decide_decode_instance_count (← Flink AdaptiveBatch)
# ---------------------------------------------------------------------------

def test_decide_count_flink_formula_low():
    sch = make_scheduler({
        "kv_per_instance_bytes": 56 * 1024 ** 3,
        "min_decode_instances": 1,
        "max_decode_instances": 10,
    })
    # 1.76 GB active KV → 1.76 / 56 ≈ 0.03 → ceil = 1
    n = sch.decide_decode_instance_count(int(1.76 * 1024 ** 3))
    assert n == 1


def test_decide_count_flink_formula_multiple():
    sch = make_scheduler({
        "kv_per_instance_bytes": 56 * 1024 ** 3,
        "min_decode_instances": 1,
        "max_decode_instances": 10,
    })
    # 88 GB → ceil(88/56) = 2
    n = sch.decide_decode_instance_count(int(88 * 1024 ** 3))
    assert n == 2


def test_decide_count_respects_max():
    sch = make_scheduler({
        "kv_per_instance_bytes": 1 * 1024 ** 3,
        "min_decode_instances": 1,
        "max_decode_instances": 5,
    })
    # Would be 1000 without cap
    n = sch.decide_decode_instance_count(int(1000 * 1024 ** 3))
    assert n == 5


def test_decide_count_respects_min():
    sch = make_scheduler({
        "kv_per_instance_bytes": 56 * 1024 ** 3,
        "min_decode_instances": 3,
        "max_decode_instances": 10,
    })
    n = sch.decide_decode_instance_count(0)
    assert n == 3

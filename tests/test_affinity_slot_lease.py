from __future__ import annotations

import pytest

from prism_serve.scheduler.scheduler import PDScheduler


def _scheduler() -> PDScheduler:
    scheduler = PDScheduler({})
    scheduler.register_instance("d0", "decode", max_slots=1, instance_epoch="e1")
    return scheduler


def test_only_one_operation_reserves_last_slot() -> None:
    scheduler = _scheduler()
    first = scheduler.reserve_decode_slot("d0", "r1", "op1")
    assert first is not None
    assert scheduler.reserve_decode_slot("d0", "r2", "op2") is None
    assert scheduler.reserve_decode_slot("d0", "r1", "op1") is first


def test_reserved_release_is_idempotent_and_bounded() -> None:
    scheduler = _scheduler()
    scheduler.reserve_decode_slot("d0", "r1", "op1")
    assert scheduler.release_decode_slot("op1") is True
    assert scheduler.release_decode_slot("op1") is False
    assert scheduler.decode_free_slots()["d0"] == 1


def test_active_cannot_rollback_and_unknown_is_quarantined() -> None:
    scheduler = _scheduler()
    scheduler.reserve_decode_slot("d0", "r1", "op1")
    scheduler.commit_decode_slot("op1")
    with pytest.raises(ValueError):
        scheduler.release_decode_slot("op1")

    other = _scheduler()
    other.reserve_decode_slot("d0", "r2", "op2")
    other.quarantine_decode_slot("op2")
    assert other.release_decode_slot("op2") is False
    assert other.decode_free_slots()["d0"] == 0


def test_decode_completion_releases_active_operation_once() -> None:
    scheduler = _scheduler()
    scheduler.reserve_decode_slot("d0", "r1", "op1")
    scheduler.commit_decode_slot("op1")
    assert scheduler.on_decode_finished("d0", "e1", "op1") is True
    assert scheduler.on_decode_finished("d0", "e1", "op1") is False


def test_authoritative_abort_can_release_quarantined_lease() -> None:
    scheduler = _scheduler()
    lease = scheduler.reserve_decode_slot("d0", "r", "op")
    assert lease is not None
    scheduler.quarantine_decode_slot("op")
    assert scheduler.release_quarantined_decode_slot("op") is True
    assert scheduler.decode_slot_lease("op").state == "RELEASED"
    assert scheduler.decode_free_slots()["d0"] == 1
    assert scheduler.decode_free_slots()["d0"] == 1

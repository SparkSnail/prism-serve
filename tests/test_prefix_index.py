from __future__ import annotations

import pytest

from prism_serve.router.fingerprint import PromptFingerprint
from prism_serve.router.prefix_index import CacheLocation, FullReportRequired, PrefixEvent, PrefixIndex


def _location(*, epoch: str = "e1", block: int = 0, updated: float = 10.0) -> CacheLocation:
    return CacheLocation("d0", epoch, "ns", "compat", "text", 11 + block, block, block, 4 * (block + 1), updated)


def test_event_apply_duplicate_gap_and_eviction() -> None:
    index = PrefixIndex(max_age_s=100)
    owner = ("d0", "e1")
    first = PrefixEvent("hash_added", _location(), 1)
    assert index.apply_events(owner, [first, first]) == 1
    with pytest.raises(FullReportRequired):
        index.apply_events(owner, [PrefixEvent("hash_added", _location(block=1), 3)])
    assert index.apply_events(owner, [PrefixEvent("evicted", _location(), 2)]) == 2
    index.assert_consistent()


def test_full_report_epoch_replacement_and_stale_exclusion() -> None:
    index = PrefixIndex(max_age_s=5)
    old = _location(epoch="e1", updated=10)
    index.install_full_report(("d0", "e1"), 4, [old])
    index.remove_instance("d0")
    new = _location(epoch="e2", updated=20)
    index.install_full_report(("d0", "e2"), 1, [new])
    fp = PromptFingerprint("ns", "compat", "text", (1, 2, 3, 9, 10), (11,), 4)
    assert list(index.iter_matches(fp, now=24))[0][1] == {new}
    assert list(index.iter_matches(fp, now=26)) == []


def test_world_full_report_validation_failure_preserves_old_directory() -> None:
    index = PrefixIndex(max_age_s=100)
    old = _location(epoch="old", updated=10)
    index.install_full_report(("d0", "old"), 4, [old])

    with pytest.raises(ValueError, match="expected owners"):
        index.install_world_full_reports(
            [(("d0", "new"), 1, [_location(epoch="new", updated=20)])],
            expected_owners={("p0", "p-new"), ("d0", "new")},
        )

    assert index._last_seq == {("d0", "old"): 4}
    index.assert_consistent()

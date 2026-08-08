from __future__ import annotations

import time

from prism_serve.router.fingerprint import PromptFingerprint
from prism_serve.router.prefix_index import CacheLocation, PrefixIndex
from prism_serve.router.router import AffinityRouter
from prism_serve.router.topology import LinkInfo, TopologyMatrix


def test_router_prefers_longest_executable_positive_gain() -> None:
    index = PrefixIndex(max_age_s=100)
    now = time.monotonic()
    long = CacheLocation("d0", "e1", "ns", "c", "text", 22, 1, 8, 8, now)
    short = CacheLocation("d1", "e1", "ns", "c", "text", 11, 0, 4, 4, now)
    index.install_full_report(("d0", "e1"), 1, [long])
    index.install_full_report(("d1", "e1"), 1, [short])
    topology = TopologyMatrix()
    topology.set_link("d0", "d2", LinkInfo("PCIE", 32))
    router = AffinityRouter(index, topology, block_bytes=1024, safety_margin_ms=0)
    fp = PromptFingerprint("ns", "c", "text", tuple(range(10)), (11, 22), 4)
    decisions = router.iter_decisions(fp, ["d2"], lambda _i, e: e == "e1", 10, 0.1)
    assert decisions[0].cached_prefix_blocks == 2
    assert decisions[0].decode_instance == "d2"


def test_router_requires_gain_strictly_above_safety_margin() -> None:
    index = PrefixIndex(max_age_s=100)
    now = time.monotonic()
    location = CacheLocation("d0", "e1", "ns", "c", "text", 22, 1, 8, 8, now)
    index.install_full_report(("d0", "e1"), 1, [location])
    router = AffinityRouter(
        index, TopologyMatrix(), block_bytes=1024, safety_margin_ms=1.0,
    )
    fingerprint = PromptFingerprint(
        "ns", "c", "text", tuple(range(10)), (11, 22), 4,
    )

    at_boundary = router.iter_decisions(
        fingerprint, ["d0"], lambda _instance, epoch: epoch == "e1",
        full_prefill_ms=3.0, suffix_prefill_ms_per_token=1.0,
    )
    above_boundary = router.iter_decisions(
        fingerprint, ["d0"], lambda _instance, epoch: epoch == "e1",
        full_prefill_ms=3.000001, suffix_prefill_ms_per_token=1.0,
    )

    assert at_boundary == []
    assert len(above_boundary) == 1
    assert above_boundary[0].estimated_gain_ms > 1.0

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

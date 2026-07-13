"""Deterministic CPU benchmark for affinity routing policy."""

from __future__ import annotations

import argparse
import json
import time

from prism_serve.router.fingerprint import PromptFingerprint
from prism_serve.router.prefix_index import CacheLocation, PrefixIndex
from prism_serve.router.router import AffinityRouter
from prism_serve.router.topology import LinkInfo, TopologyMatrix


def run_synthetic(requests: int, block_size: int, block_bytes: int) -> dict:
    assert requests > 0
    shared = list(range(block_size * 2))
    first = PromptFingerprint.create(
        namespace="bench", kv_compatibility_id="cpu",
        request_context_digest="text", token_ids=shared + [1000],
        block_size=block_size,
    )
    index = PrefixIndex(max_age_s=60)
    now = time.monotonic()
    index.install_full_report(("src", "epoch"), 1, [CacheLocation(
        "src", "epoch", "bench", "cpu", "text",
        first.chain_hashes[1], 1, 7, block_size * 2, now,
    )])
    topology = TopologyMatrix()
    topology.set_link("src", "dst", LinkInfo("SYNTHETIC", 32.0, 0.01))
    router = AffinityRouter(
        index, topology, block_bytes=block_bytes, safety_margin_ms=0
    )
    hits = 0
    mapped_bytes = 0
    cold_bytes = 0
    started = time.perf_counter()
    for suffix in range(requests):
        fingerprint = PromptFingerprint.create(
            namespace="bench", kv_compatibility_id="cpu",
            request_context_digest="text",
            token_ids=shared + [10_000 + suffix], block_size=block_size,
        )
        decisions = router.iter_decisions(
            fingerprint, ["dst"], lambda _instance, epoch: epoch == "epoch",
            full_prefill_ms=10.0, suffix_prefill_ms_per_token=0.01,
        )
        cold_blocks = (
            len(fingerprint.token_ids) + fingerprint.block_size - 1
        ) // fingerprint.block_size
        cold_bytes += cold_blocks * block_bytes
        if decisions:
            hits += 1
            mapped_bytes += decisions[0].transfer_bytes
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "kind": "synthetic_cpu_policy",
        "requests": requests,
        "route_hit_rate": hits / requests,
        "mapped_to_cold_bytes_ratio": mapped_bytes / cold_bytes,
        "router_elapsed_ms": elapsed_ms,
        "claims_not_made": ["GPU parity", "TTFT speedup", "multi-process E2E"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--block-bytes", type=int, default=28 * 1024 * 1024)
    args = parser.parse_args()
    print(json.dumps(
        run_synthetic(args.requests, args.block_size, args.block_bytes),
        indent=2,
    ))


if __name__ == "__main__":
    main()

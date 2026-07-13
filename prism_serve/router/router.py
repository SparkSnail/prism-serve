"""Route to the longest executable prefix with positive gain."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from prism_serve.router.fingerprint import PromptFingerprint
from prism_serve.router.prefix_index import CacheLocation, PrefixIndex
from prism_serve.router.protocol import CachedPrefixDecision
from prism_serve.router.topology import TopologyMatrix


class AffinityRouter:
    def __init__(
        self,
        index: PrefixIndex,
        topology: TopologyMatrix,
        *,
        block_bytes: int,
        safety_margin_ms: float = 1.0,
    ) -> None:
        assert block_bytes > 0, f"invalid block bytes: {block_bytes=}"
        self.index = index
        self.topology = topology
        self.block_bytes = block_bytes
        self.safety_margin_ms = safety_margin_ms

    def iter_decisions(
        self,
        fingerprint: PromptFingerprint,
        decode_candidates: Iterable[str],
        source_epoch_is_current: Callable[[str, str], bool],
        full_prefill_ms: float,
        suffix_prefill_ms_per_token: float,
    ) -> list[CachedPrefixDecision]:
        decisions: list[CachedPrefixDecision] = []
        targets = tuple(decode_candidates)
        for blocks, locations in self.index.iter_matches(fingerprint):
            prefix_tokens = blocks * fingerprint.block_size
            suffix_tokens = len(fingerprint.token_ids) - prefix_tokens
            for location in locations:
                if not source_epoch_is_current(location.instance_id, location.instance_epoch):
                    continue
                for target in targets:
                    transfer_bytes = 0 if target == location.instance_id else blocks * self.block_bytes
                    transfer_ms = self.topology.transfer_cost_ms(
                        location.instance_id, target, transfer_bytes
                    )
                    cost = transfer_ms + suffix_tokens * suffix_prefill_ms_per_token
                    gain = full_prefill_ms - cost
                    if gain <= self.safety_margin_ms:
                        continue
                    decisions.append(CachedPrefixDecision(
                        source_instance=location.instance_id,
                        source_epoch=location.instance_epoch,
                        decode_instance=target,
                        cached_prefix_blocks=blocks,
                        cached_prefix_tokens=prefix_tokens,
                        transfer_bytes=transfer_bytes,
                        estimated_gain_ms=gain,
                    ))
            if decisions:
                break
        return sorted(decisions, key=lambda item: (-item.cached_prefix_blocks, -item.estimated_gain_ms))

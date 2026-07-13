"""Process-local, rebuildable view of infer prefix cache locations."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator

from prism_serve.router.fingerprint import PromptFingerprint


class FullReportRequired(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class CacheLocation:
    instance_id: str
    instance_epoch: str
    namespace: str
    kv_compatibility_id: str
    request_context_digest: str
    chain_hash: int
    block_index: int
    block_id: int
    prefix_tokens: int
    last_updated: float


@dataclass(slots=True, frozen=True)
class PrefixEvent:
    kind: str
    location: CacheLocation
    seq_no: int


Key = tuple[str, str, str, int]
Owner = tuple[str, str]


class PrefixIndex:
    def __init__(self, max_age_s: float = 600.0) -> None:
        assert max_age_s > 0, f"invalid max age: {max_age_s=}"
        self.max_age_s = max_age_s
        self._index: dict[Key, set[CacheLocation]] = defaultdict(set)
        self._forward: dict[Owner, set[Key]] = defaultdict(set)
        self._last_seq: dict[Owner, int] = {}

    @staticmethod
    def _key(location: CacheLocation) -> Key:
        return (
            location.namespace,
            location.kv_compatibility_id,
            location.request_context_digest,
            location.chain_hash,
        )

    def apply_events(self, owner: Owner, events: Iterable[PrefixEvent]) -> int:
        cursor = self._last_seq.get(owner, 0)
        for event in events:
            location = event.location
            if (location.instance_id, location.instance_epoch) != owner:
                raise ValueError("prefix event owner mismatch")
            if event.seq_no <= cursor:
                continue
            if event.seq_no != cursor + 1:
                raise FullReportRequired(
                    f"event gap: expected={cursor + 1}, got={event.seq_no}"
                )
            if event.kind == "hash_added":
                self._add(location)
            elif event.kind == "evicted":
                self._remove_block(self._key(location), owner, location.block_id)
            else:
                raise ValueError(f"unsupported prefix event: {event.kind!r}")
            cursor = event.seq_no
        self._last_seq[owner] = cursor
        return cursor

    def install_full_report(
        self, owner: Owner, snapshot_seq_no: int, locations: Iterable[CacheLocation]
    ) -> None:
        self.remove_owner(owner)
        for location in locations:
            if (location.instance_id, location.instance_epoch) != owner:
                raise ValueError("full report owner mismatch")
            self._add(location)
        self._last_seq[owner] = snapshot_seq_no

    def iter_matches(
        self, fingerprint: PromptFingerprint, now: float | None = None
    ) -> Iterator[tuple[int, set[CacheLocation]]]:
        current = time.monotonic() if now is None else now
        for blocks in range(fingerprint.max_reusable_blocks(), 0, -1):
            key = (
                fingerprint.namespace,
                fingerprint.kv_compatibility_id,
                fingerprint.request_context_digest,
                fingerprint.chain_hashes[blocks - 1],
            )
            fresh = {
                location for location in self._index.get(key, set())
                if current - location.last_updated <= self.max_age_s
            }
            if fresh:
                yield blocks, fresh

    def remove_instance(self, instance_id: str) -> None:
        for owner in [owner for owner in self._forward if owner[0] == instance_id]:
            self.remove_owner(owner)

    def remove_owner(self, owner: Owner) -> None:
        for key in list(self._forward.get(owner, set())):
            self._remove_owner(key, owner)
        self._forward.pop(owner, None)
        self._last_seq.pop(owner, None)

    def _add(self, location: CacheLocation) -> None:
        key = self._key(location)
        owner = (location.instance_id, location.instance_epoch)
        self._index[key].add(location)
        self._forward[owner].add(key)

    def _remove_block(self, key: Key, owner: Owner, block_id: int) -> None:
        locations = self._index.get(key, set())
        self._index[key] = {
            item for item in locations
            if (item.instance_id, item.instance_epoch) != owner or item.block_id != block_id
        }
        if not self._index[key]:
            self._index.pop(key, None)
        if not any((item.instance_id, item.instance_epoch) == owner for item in self._index.get(key, set())):
            self._forward[owner].discard(key)

    def _remove_owner(self, key: Key, owner: Owner) -> None:
        self._index[key] = {
            item for item in self._index.get(key, set())
            if (item.instance_id, item.instance_epoch) != owner
        }
        if not self._index[key]:
            self._index.pop(key, None)

    def assert_consistent(self) -> None:
        for key, locations in self._index.items():
            assert locations, f"empty inverted entry: {key=}"
            for location in locations:
                owner = (location.instance_id, location.instance_epoch)
                assert key in self._forward.get(owner, set()), f"missing forward entry: {key=} {owner=}"

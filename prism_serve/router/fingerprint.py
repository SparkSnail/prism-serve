"""KV prefix fingerprints with compatibility identities."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass

import xxhash


def _stable_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_namespace(
    model_id: str,
    tokenizer_revision: str,
    chat_template_version: str,
    block_size: int,
    hash_version: str,
    kv_compatibility_id: str,
) -> str:
    assert block_size > 0, f"invalid block size: {block_size=}"
    return _stable_digest({
        "block_size": block_size,
        "chat_template_version": chat_template_version,
        "hash_version": hash_version,
        "kv_compatibility_id": kv_compatibility_id,
        "model_id": model_id,
        "tokenizer_revision": tokenizer_revision,
    })


def build_kv_compatibility_id(**identity: object) -> str:
    """Hash the complete engine identity supplied by worker registration."""
    return _stable_digest(identity)


def compute_chain_hashes(token_ids: list[int], block_size: int) -> list[int]:
    """Compute stable 64-bit hashes for complete chained token blocks."""
    assert block_size > 0, f"invalid block size: {block_size=}"
    result: list[int] = []
    previous = -1
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        block = token_ids[start:start + block_size]
        digest = xxhash.xxh64()
        if previous != -1:
            digest.update(previous.to_bytes(8, "little"))
        digest.update(struct.pack(f"<{len(block)}q", *block))
        previous = digest.intdigest()
        result.append(previous)
    return result


@dataclass(slots=True, frozen=True)
class PromptFingerprint:
    namespace: str
    kv_compatibility_id: str
    request_context_digest: str
    token_ids: tuple[int, ...]
    chain_hashes: tuple[int, ...]
    block_size: int

    @classmethod
    def create(
        cls,
        *,
        namespace: str,
        kv_compatibility_id: str,
        request_context_digest: str,
        token_ids: list[int],
        block_size: int,
    ) -> "PromptFingerprint":
        return cls(
            namespace=namespace,
            kv_compatibility_id=kv_compatibility_id,
            request_context_digest=request_context_digest,
            token_ids=tuple(token_ids),
            chain_hashes=tuple(compute_chain_hashes(token_ids, block_size)),
            block_size=block_size,
        )

    def max_reusable_blocks(self) -> int:
        complete = len(self.token_ids) // self.block_size
        keep_last_complete = int(len(self.token_ids) % self.block_size == 0)
        return max(0, min(len(self.chain_hashes), complete - keep_last_complete))

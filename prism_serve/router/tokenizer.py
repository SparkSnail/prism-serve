"""Process-local tokenizer owner for stable request fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from prism_serve.router.fingerprint import PromptFingerprint, build_namespace


class TokenEncoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(slots=True, frozen=True)
class TokenizerIdentity:
    model_id: str
    tokenizer_revision: str
    chat_template_version: str
    block_size: int
    hash_version: str
    kv_compatibility_id: str


class TokenizerAdapter:
    def __init__(self, encoder: TokenEncoder, identity: TokenizerIdentity):
        assert identity.model_id, "model_id is required"
        assert identity.tokenizer_revision, "tokenizer_revision is required"
        assert identity.chat_template_version, "chat_template_version is required"
        assert identity.kv_compatibility_id, "kv_compatibility_id is required"
        assert identity.block_size > 0, f"invalid block size: {identity.block_size}"
        self.encoder = encoder
        self.identity = identity
        self.namespace = build_namespace(
            identity.model_id,
            identity.tokenizer_revision,
            identity.chat_template_version,
            identity.block_size,
            identity.hash_version,
            identity.kv_compatibility_id,
        )

    def fingerprint_request(
        self, text: str, *, request_context_digest: str = "text-only"
    ) -> PromptFingerprint:
        token_ids = self.encoder.encode(text)
        return PromptFingerprint.create(
            namespace=self.namespace,
            kv_compatibility_id=self.identity.kv_compatibility_id,
            request_context_digest=request_context_digest,
            token_ids=token_ids,
            block_size=self.identity.block_size,
        )

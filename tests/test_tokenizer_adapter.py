from __future__ import annotations

from prism_serve.router.tokenizer import TokenizerAdapter, TokenizerIdentity


class Encoder:
    def encode(self, text):
        return [ord(char) for char in text]


def _identity(**overrides):
    values = {
        "model_id": "model@rev1",
        "tokenizer_revision": "tok1",
        "chat_template_version": "chat1",
        "block_size": 4,
        "hash_version": "xxh64v1",
        "kv_compatibility_id": "compat1",
    }
    values.update(overrides)
    return TokenizerIdentity(**values)


def test_adapter_builds_stable_namespaced_fingerprint():
    adapter = TokenizerAdapter(Encoder(), _identity())
    first = adapter.fingerprint_request("abcdefghij")
    second = adapter.fingerprint_request("abcdWXYZij")
    assert first.namespace == second.namespace
    assert first.chain_hashes[0] == second.chain_hashes[0]
    assert first.chain_hashes[1] != second.chain_hashes[1]


def test_revision_or_compatibility_change_rotates_namespace():
    base = TokenizerAdapter(Encoder(), _identity()).namespace
    changed_tokenizer = TokenizerAdapter(
        Encoder(), _identity(tokenizer_revision="tok2")
    ).namespace
    changed_compat = TokenizerAdapter(
        Encoder(), _identity(kv_compatibility_id="compat2")
    ).namespace
    assert len({base, changed_tokenizer, changed_compat}) == 3

from __future__ import annotations

from prism_serve.router.fingerprint import PromptFingerprint, build_namespace, compute_chain_hashes


def test_max_reusable_blocks_preserves_logits_input() -> None:
    namespace = build_namespace("m", "t", "c", 256, "v1", "compat")
    for token_count, expected in [(256, 0), (512, 1), (530, 2)]:
        fp = PromptFingerprint.create(
            namespace=namespace,
            kv_compatibility_id="compat",
            request_context_digest="text",
            token_ids=list(range(token_count)),
            block_size=256,
        )
        assert fp.max_reusable_blocks() == expected


def test_namespace_changes_with_compatibility_inputs() -> None:
    base = build_namespace("m", "t", "c", 256, "v1", "compat-a")
    changed = build_namespace("m", "t", "c", 256, "v1", "compat-b")
    assert base != changed


def test_chain_hash_is_prefix_dependent() -> None:
    hashes = compute_chain_hashes([1, 2, 3, 4, 5, 6, 7, 8], 4)
    standalone = compute_chain_hashes([5, 6, 7, 8], 4)[0]
    assert len(hashes) == 2
    assert hashes[1] != standalone

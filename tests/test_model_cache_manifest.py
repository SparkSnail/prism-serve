"""Offline model-cache manifest contract tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.create_model_cache_manifest import (
    build_manifest,
    verify_model_cache,
    write_model_cache_identity,
)


def _cache(tmp_path: Path) -> tuple[Path, str]:
    model = tmp_path / "Qwen3-8B"
    model.mkdir()
    config = model / "config.json"
    config.write_text('{"hidden_size": 1}\n', encoding="utf-8")
    for name in (
        "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    ):
        (model / name).write_text(name, encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    return model, hashlib.sha256(config.read_bytes()).hexdigest()


def test_manifest_records_all_model_files(tmp_path: Path):
    model, config_sha = _cache(tmp_path)
    manifest = build_manifest(
        model,
        model_id="Qwen/Qwen3-8B",
        revision="a" * 40,
        tokenizer_revision="a" * 40,
        config_sha256=config_sha,
    )
    assert manifest["schema_version"] == "prism.local_model_cache/v1"
    assert "model.safetensors" in manifest["files"]
    assert ".prism-model-manifest.json" not in manifest["files"]


def test_manifest_rejects_config_drift(tmp_path: Path):
    model, _ = _cache(tmp_path)
    with pytest.raises(ValueError, match="config.json SHA-256 mismatch"):
        build_manifest(
            model,
            model_id="Qwen/Qwen3-8B",
            revision="a" * 40,
            tokenizer_revision="a" * 40,
            config_sha256="b" * 64,
        )


def test_cache_identity_writer_creates_the_marker_and_fails_closed_on_drift(
    tmp_path: Path,
):
    model, config_sha = _cache(tmp_path)
    revision = "a" * 40
    manifest = build_manifest(
        model,
        model_id="Qwen/Qwen3-8B",
        revision=revision,
        tokenizer_revision=revision,
        config_sha256=config_sha,
    )

    write_model_cache_identity(model, manifest, revision)
    assert (model / ".prism-model-revision").read_text(encoding="utf-8") == revision + "\n"
    verify_model_cache(
        model,
        model_id="Qwen/Qwen3-8B",
        revision=revision,
        tokenizer_revision=revision,
        config_sha256=config_sha,
    )

    (model / "model.safetensors").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="does not match the current cache identity"):
        verify_model_cache(
            model,
            model_id="Qwen/Qwen3-8B",
            revision=revision,
            tokenizer_revision=revision,
            config_sha256=config_sha,
        )

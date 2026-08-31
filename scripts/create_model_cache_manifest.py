#!/usr/bin/env python3
"""Create an offline integrity manifest for a local model cache directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


SCHEMA_VERSION = "prism.local_model_cache/v1"
_SHA40 = re.compile(r"^[0-9a-f]{40}$", re.ASCII)
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
CONTROL_FILES = {".prism-model-manifest.json", ".prism-model-revision"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha(value: str, label: str, pattern: re.Pattern[str]) -> str:
    if pattern.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase hexadecimal digest")
    return value


def _resolve_model_dir(model_dir: Path) -> Path:
    expanded = model_dir.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"model cache root must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise ValueError(f"model directory not found: {resolved}")
    return resolved


def _cache_file_hashes(model_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(model_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"model cache must not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(model_dir).as_posix()
        if relative in CONTROL_FILES:
            continue
        files[relative] = _sha256(path)
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("model directory is missing safetensors weights")
    return files


def build_manifest(
    model_dir: Path,
    *,
    model_id: str,
    revision: str,
    tokenizer_revision: str,
    config_sha256: str,
) -> dict[str, object]:
    model_dir = _resolve_model_dir(model_dir)
    _validate_sha(revision, "revision", _SHA40)
    _validate_sha(tokenizer_revision, "tokenizer_revision", _SHA40)
    _validate_sha(config_sha256, "config_sha256", _SHA256)
    config = model_dir / "config.json"
    if not config.is_file():
        raise ValueError("model directory is missing config.json")
    actual_config_sha256 = _sha256(config)
    if actual_config_sha256 != config_sha256:
        raise ValueError(
            "config.json SHA-256 mismatch: "
            f"expected {config_sha256}, got {actual_config_sha256}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "revision": revision,
        "tokenizer_revision": tokenizer_revision,
        "config_sha256": config_sha256,
        "files": _cache_file_hashes(model_dir),
    }


def verify_model_cache(
    model_dir: Path,
    *,
    model_id: str,
    revision: str,
    tokenizer_revision: str,
    config_sha256: str,
) -> None:
    """Fail closed unless the marker and manifest describe this exact cache."""
    model_dir = _resolve_model_dir(model_dir)
    revision_marker = model_dir / ".prism-model-revision"
    if revision_marker.is_symlink() or not revision_marker.is_file():
        raise ValueError("model directory is missing .prism-model-revision")
    if revision_marker.read_text(encoding="utf-8").strip() != revision:
        raise ValueError(".prism-model-revision does not match the expected revision")

    manifest_path = model_dir / ".prism-model-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("model directory is missing .prism-model-manifest.json")
    try:
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("model cache manifest is not valid JSON") from exc
    expected = build_manifest(
        model_dir,
        model_id=model_id,
        revision=revision,
        tokenizer_revision=tokenizer_revision,
        config_sha256=config_sha256,
    )
    if actual != expected:
        raise ValueError("model cache manifest does not match the current cache identity")


def _atomic_write(destination: Path, content: str) -> None:
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_model_cache_identity(
    model_dir: Path, manifest: dict[str, object], revision: str,
) -> None:
    """Write the duplicated marker first; the manifest is the completion signal."""
    model_dir = _resolve_model_dir(model_dir)
    _atomic_write(model_dir / ".prism-model-revision", revision + "\n")
    _atomic_write(
        model_dir / ".prism-model-manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument(
        "--verify", action="store_true",
        help="verify existing marker and manifest without changing the cache",
    )
    args = parser.parse_args()
    try:
        if args.verify:
            verify_model_cache(
                args.model_dir,
                model_id=args.model_id,
                revision=args.revision,
                tokenizer_revision=args.tokenizer_revision,
                config_sha256=args.config_sha256,
            )
            print(args.model_dir.expanduser().resolve())
            return 0
        manifest = build_manifest(
            args.model_dir,
            model_id=args.model_id,
            revision=args.revision,
            tokenizer_revision=args.tokenizer_revision,
            config_sha256=args.config_sha256,
        )
    except ValueError as exc:
        parser.error(str(exc))
    write_model_cache_identity(args.model_dir, manifest, args.revision)
    print(args.model_dir.expanduser().resolve() / ".prism-model-manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

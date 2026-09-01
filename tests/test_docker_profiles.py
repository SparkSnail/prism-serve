"""Static contract tests for the two immutable Gateway image variants."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
DOCKERFILE = ROOT / "docker" / "Dockerfile"
DOCKER_GUIDE = DOCKERFILE.parent / "README.md"
MODEL_NOTICE = ROOT / "licenses" / "QWEN3-MODEL-NOTICE.txt"


def _stage(text: str, name: str, next_name: str) -> str:
    return text.split(f"FROM common AS {name}", 1)[1].split(
        f"FROM common AS {next_name}", 1
    )[0]


def test_docker_definition_and_guide_have_one_dedicated_location() -> None:
    assert DOCKERFILE.is_file()
    assert DOCKER_GUIDE.is_file()
    assert not (ROOT / "Dockerfile").exists()


def test_dockerfile_has_one_closed_variant_selector() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert text.startswith("# syntax=docker/dockerfile:1.7\n")
    assert text.count("ARG PRISM_IMAGE_VARIANT") == 2
    assert "ARG PRISM_IMAGE_VARIANT=correctness" in text
    assert "FROM common AS profile-correctness" in text
    assert "FROM common AS profile-performance" in text
    assert "FROM profile-${PRISM_IMAGE_VARIANT} AS selected" in text
    assert "PRISM_IMAGE_VARIANT must be correctness or performance" in text
    assert not re.search(
        r"^ARG PRISM_(?!IMAGE_VARIANT(?:=|$)|RELEASE(?:=|$))",
        text,
        re.MULTILINE,
    )


def test_dockerfile_hard_codes_both_complete_profile_bundles() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    correctness = _stage(text, "profile-correctness", "profile-performance")
    performance = text.split("FROM common AS profile-performance", 1)[1].split(
        "FROM profile-${PRISM_IMAGE_VARIANT} AS selected", 1
    )[0]

    common = {
        "PRISM_TP_SIZE": "1",
        "PRISM_TOKENS_PER_BLOCK": "256",
        "PRISM_MODEL_NUM_KEY_VALUE_HEADS": "8",
        "PRISM_MODEL_HEAD_DIM": "128",
        "PRISM_MODEL_ROPE_THETA": "1000000.0",
        "PRISM_MAX_MODEL_LEN": "4096",
        "PRISM_MAX_NUM_BATCHED_TOKENS": "16384",
        "PRISM_GPU_MEMORY_UTILIZATION": "0.9",
        "PRISM_ENFORCE_EAGER": "false",
    }
    bundles = (
        (
            correctness,
            {
                "PRISM_IMAGE_VARIANT": "correctness",
                "PRISM_MODEL_PROFILE": "week12-qwen3-0.6b",
                "PRISM_MODEL_ID": "Qwen/Qwen3-0.6B",
                "PRISM_MODEL_REVISION": "9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439",
                "PRISM_MODEL_CONFIG_SHA256": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
                "PRISM_KV_BLOCK_BYTES": "29360128",
                "PRISM_KV_COMPATIBILITY_ID": "a305c48442086f050a0f703c9e79e6e4596c52eaf5dd9f9015cc1a744c1b5b19",
                "PRISM_MODEL_NUM_HIDDEN_LAYERS": "28",
                "PRISM_MAX_NUM_SEQS": "512",
            },
        ),
        (
            performance,
            {
                "PRISM_IMAGE_VARIANT": "performance",
                "PRISM_MODEL_PROFILE": "qwen3-8b-bf16-tp1",
                "PRISM_MODEL_ID": "Qwen/Qwen3-8B",
                "PRISM_MODEL_REVISION": "b968826d9c46dd6066d109eabc6255188de91218",
                "PRISM_MODEL_CONFIG_SHA256": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
                "PRISM_KV_BLOCK_BYTES": "37748736",
                "PRISM_KV_COMPATIBILITY_ID": "2647222531d143800a56551dbe8d030c535a1bb2e0e47ff2a12f0781964f4c6c",
                "PRISM_MODEL_NUM_HIDDEN_LAYERS": "36",
                "PRISM_MAX_NUM_SEQS": "128",
            },
        ),
    )
    for stage, profile in bundles:
        for name, value in {**common, **profile}.items():
            assert f"{name}={value}" in stage
        for name in (
            "MODEL_NUM_HIDDEN_LAYERS",
            "MODEL_NUM_KEY_VALUE_HEADS",
            "MODEL_HEAD_DIM",
            "MODEL_ROPE_THETA",
            "MAX_MODEL_LEN",
            "MAX_NUM_BATCHED_TOKENS",
            "MAX_NUM_SEQS",
            "GPU_MEMORY_UTILIZATION",
            "ENFORCE_EAGER",
        ):
            assert f"PRISM_SERVE_{name}=" in stage


def test_dockerfile_exposes_full_oci_provenance_and_smoke_import() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    for key in (
        "org.opencontainers.image.source",
        "org.opencontainers.image.revision",
        "org.opencontainers.image.title",
        "ai.sparksnail.prism.image.variant",
        "ai.sparksnail.prism.model.profile",
        "ai.sparksnail.prism.model.id",
        "ai.sparksnail.prism.model.revision",
        "ai.sparksnail.prism.model.tokenizer-revision",
        "ai.sparksnail.prism.model.config-sha256",
        "ai.sparksnail.prism.model.kv-compatibility-id",
    ):
        assert f"{key}=" in text
    assert "performance_harness import PerformanceTraceRegistry" in text
    assert "ARG GIT_SHA=local" in text
    assert "ARG PRISM_RELEASE=false" in text
    assert "PRISM_RELEASE=true requires a full lowercase commit SHA" in text
    assert "USER prism" in text
    assert "COPY licenses/QWEN3-MODEL-NOTICE.txt /opt/prism/licenses/QWEN3-MODEL-NOTICE.txt" in text


def test_dockerfile_carries_qwen_model_attribution_into_the_image() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert MODEL_NOTICE.is_file()
    notice = MODEL_NOTICE.read_text(encoding="utf-8")
    assert "Qwen3-0.6B" in notice
    assert "Qwen3-8B" in notice
    assert "Apache License, Version 2.0" in notice
    assert "COPY licenses/QWEN3-MODEL-NOTICE.txt" in text


def test_dockerfile_stages_tokenizer_from_local_named_context() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    guide = DOCKER_GUIDE.read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "FROM profile-${PRISM_IMAGE_VARIANT} AS model-staging" in text
    assert (
        "--mount=type=bind,from=model-cache,source=.,target=/mnt/model-cache,ro"
        in text
    )
    assert "COPY --link --from=model-staging /opt/models/ /opt/models/" in text
    assert "snapshot_download" not in text
    assert "huggingface_hub" not in text
    assert "model-cache must be a model directory" in text
    assert "-f docker/Dockerfile" in guide
    assert "--build-context model-cache=" in guide
    assert "never downloads a model" in guide
    assert "<release-tag>" in guide
    assert not re.search(r"\bv\d+\.\d+\.\d+\b", guide)
    assert "[Docker guide](docker/README.md)" in readme
    assert "model-cache is missing .prism-model-manifest.json" in text
    assert '"prism.local_model_cache/v1"' in text
    assert "file hash mismatch" in text
    assert "model-cache is missing .prism-model-revision" in text
    assert ".prism-model-manifest.json" in text
    assert "write_text" not in text.split("FROM profile-${PRISM_IMAGE_VARIANT} AS model-staging", 1)[1].split("PY", 1)[0]
    assert (ROOT / "scripts" / "create_model_cache_manifest.py").is_file()

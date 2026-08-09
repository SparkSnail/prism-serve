# syntax=docker/dockerfile:1.7

ARG PRISM_IMAGE_VARIANT=correctness

FROM python:3.12.4-slim-bookworm@sha256:a074fac67aa01841fee592d00bae14d25dcaf98ef6e12a683ecceb7e0147e2d1 AS common

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade --retries 20 --timeout 120 \
      "pip==25.1.1" "setuptools==82.0.1" "wheel==0.47.0" && \
    python -m pip install --resume-retries 20 --retries 20 --timeout 120 \
      "fastapi==0.139.0" \
      "uvicorn[standard]==0.51.0" \
      "pydantic==2.13.4" \
      "pydantic-settings==2.14.2" \
      "httpx==0.28.1" \
      "nats-py==2.15.0" \
      "prometheus-client==0.25.0" \
      "xxhash==3.7.0" \
      "transformers==4.51.3"

COPY pyproject.toml README.md LICENSE ./
COPY prism_serve ./prism_serve

RUN python -m pip install --no-build-isolation --no-deps .

FROM common AS profile-correctness

ENV PRISM_IMAGE_VARIANT=correctness \
    PRISM_MODEL_PROFILE=week12-qwen3-0.6b \
    PRISM_MODEL=/opt/models/Qwen3-0.6B \
    PRISM_MODEL_ID=Qwen/Qwen3-0.6B \
    PRISM_MODEL_REVISION=9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439 \
    PRISM_TOKENIZER_REVISION=9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439 \
    PRISM_MODEL_CONFIG_SHA256=660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd \
    PRISM_DTYPE=bfloat16 \
    PRISM_TP_SIZE=1 \
    PRISM_TOKENS_PER_BLOCK=256 \
    PRISM_KV_BLOCK_BYTES=29360128 \
    PRISM_KV_LAYOUT=NHDC \
    PRISM_KV_COMPATIBILITY_ID=a305c48442086f050a0f703c9e79e6e4596c52eaf5dd9f9015cc1a744c1b5b19 \
    PRISM_MODEL_NUM_HIDDEN_LAYERS=28 \
    PRISM_MODEL_NUM_KEY_VALUE_HEADS=8 \
    PRISM_MODEL_HEAD_DIM=128 \
    PRISM_MODEL_ROPE_THETA=1000000.0 \
    PRISM_MAX_MODEL_LEN=4096 \
    PRISM_MAX_NUM_BATCHED_TOKENS=16384 \
    PRISM_MAX_NUM_SEQS=512 \
    PRISM_GPU_MEMORY_UTILIZATION=0.9 \
    PRISM_ENFORCE_EAGER=false \
    PRISM_SERVE_TOKENIZER_MODEL=/opt/models/Qwen3-0.6B \
    PRISM_SERVE_TOKENIZER_REVISION=9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439 \
    PRISM_SERVE_CHAT_TEMPLATE_VERSION=v1 \
    PRISM_SERVE_MODEL_PROFILE_ID=week12-qwen3-0.6b \
    PRISM_SERVE_MODEL_ID=Qwen/Qwen3-0.6B \
    PRISM_SERVE_MODEL_REVISION=9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439 \
    PRISM_SERVE_MODEL_CONFIG_SHA256=660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd \
    PRISM_SERVE_RUNTIME_DTYPE=bfloat16 \
    PRISM_SERVE_TENSOR_PARALLEL_SIZE=1 \
    PRISM_SERVE_KV_LAYOUT=NHDC \
    PRISM_SERVE_KV_BLOCK_BYTES=29360128 \
    PRISM_SERVE_KV_COMPATIBILITY_ID=a305c48442086f050a0f703c9e79e6e4596c52eaf5dd9f9015cc1a744c1b5b19 \
    PRISM_SERVE_MODEL_NUM_HIDDEN_LAYERS=28 \
    PRISM_SERVE_MODEL_NUM_KEY_VALUE_HEADS=8 \
    PRISM_SERVE_MODEL_HEAD_DIM=128 \
    PRISM_SERVE_MODEL_ROPE_THETA=1000000.0 \
    PRISM_SERVE_MAX_MODEL_LEN=4096 \
    PRISM_SERVE_MAX_NUM_BATCHED_TOKENS=16384 \
    PRISM_SERVE_MAX_NUM_SEQS=512 \
    PRISM_SERVE_GPU_MEMORY_UTILIZATION=0.9 \
    PRISM_SERVE_ENFORCE_EAGER=false \
    PRISM_SERVE_PREFIX_BLOCK_SIZE=256 \
    PRISM_SERVE_PREFIX_BLOCK_BYTES=29360128

FROM common AS profile-performance

ENV PRISM_IMAGE_VARIANT=performance \
    PRISM_MODEL_PROFILE=qwen3-8b-bf16-tp1 \
    PRISM_MODEL=/opt/models/Qwen3-8B \
    PRISM_MODEL_ID=Qwen/Qwen3-8B \
    PRISM_MODEL_REVISION=b968826d9c46dd6066d109eabc6255188de91218 \
    PRISM_TOKENIZER_REVISION=b968826d9c46dd6066d109eabc6255188de91218 \
    PRISM_MODEL_CONFIG_SHA256=f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30 \
    PRISM_DTYPE=bfloat16 \
    PRISM_TP_SIZE=1 \
    PRISM_TOKENS_PER_BLOCK=256 \
    PRISM_KV_BLOCK_BYTES=37748736 \
    PRISM_KV_LAYOUT=NHDC \
    PRISM_KV_COMPATIBILITY_ID=2647222531d143800a56551dbe8d030c535a1bb2e0e47ff2a12f0781964f4c6c \
    PRISM_MODEL_NUM_HIDDEN_LAYERS=36 \
    PRISM_MODEL_NUM_KEY_VALUE_HEADS=8 \
    PRISM_MODEL_HEAD_DIM=128 \
    PRISM_MODEL_ROPE_THETA=1000000.0 \
    PRISM_MAX_MODEL_LEN=4096 \
    PRISM_MAX_NUM_BATCHED_TOKENS=16384 \
    PRISM_MAX_NUM_SEQS=128 \
    PRISM_GPU_MEMORY_UTILIZATION=0.9 \
    PRISM_ENFORCE_EAGER=false \
    PRISM_SERVE_TOKENIZER_MODEL=/opt/models/Qwen3-8B \
    PRISM_SERVE_TOKENIZER_REVISION=b968826d9c46dd6066d109eabc6255188de91218 \
    PRISM_SERVE_CHAT_TEMPLATE_VERSION=v1 \
    PRISM_SERVE_MODEL_PROFILE_ID=qwen3-8b-bf16-tp1 \
    PRISM_SERVE_MODEL_ID=Qwen/Qwen3-8B \
    PRISM_SERVE_MODEL_REVISION=b968826d9c46dd6066d109eabc6255188de91218 \
    PRISM_SERVE_MODEL_CONFIG_SHA256=f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30 \
    PRISM_SERVE_RUNTIME_DTYPE=bfloat16 \
    PRISM_SERVE_TENSOR_PARALLEL_SIZE=1 \
    PRISM_SERVE_KV_LAYOUT=NHDC \
    PRISM_SERVE_KV_BLOCK_BYTES=37748736 \
    PRISM_SERVE_KV_COMPATIBILITY_ID=2647222531d143800a56551dbe8d030c535a1bb2e0e47ff2a12f0781964f4c6c \
    PRISM_SERVE_MODEL_NUM_HIDDEN_LAYERS=36 \
    PRISM_SERVE_MODEL_NUM_KEY_VALUE_HEADS=8 \
    PRISM_SERVE_MODEL_HEAD_DIM=128 \
    PRISM_SERVE_MODEL_ROPE_THETA=1000000.0 \
    PRISM_SERVE_MAX_MODEL_LEN=4096 \
    PRISM_SERVE_MAX_NUM_BATCHED_TOKENS=16384 \
    PRISM_SERVE_MAX_NUM_SEQS=128 \
    PRISM_SERVE_GPU_MEMORY_UTILIZATION=0.9 \
    PRISM_SERVE_ENFORCE_EAGER=false \
    PRISM_SERVE_PREFIX_BLOCK_SIZE=256 \
    PRISM_SERVE_PREFIX_BLOCK_BYTES=37748736

FROM profile-${PRISM_IMAGE_VARIANT} AS selected

ARG PRISM_IMAGE_VARIANT
ARG GIT_SHA
ARG SOURCE_URL=https://github.com/SparkSnail/prism-serve

RUN case "${PRISM_IMAGE_VARIANT}" in \
      correctness|performance) ;; \
      *) echo "PRISM_IMAGE_VARIANT must be correctness or performance" >&2; exit 64 ;; \
    esac

ENV PRISM_IMAGE_GIT_SHA=${GIT_SHA}

# Gateway needs the pinned tokenizer/config snapshot, but no model weights.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python - <<'PY'
import hashlib
import os
from pathlib import Path

from huggingface_hub import snapshot_download

target = Path(os.environ["PRISM_MODEL"])
snapshot_download(
    repo_id=os.environ["PRISM_MODEL_ID"],
    revision=os.environ["PRISM_MODEL_REVISION"],
    local_dir=target,
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template*",
        "*.model",
        "merges.txt",
        "vocab.json",
    ],
)
actual = hashlib.sha256((target / "config.json").read_bytes()).hexdigest()
expected = os.environ["PRISM_MODEL_CONFIG_SHA256"]
if actual != expected:
    raise SystemExit(f"config.json SHA-256 mismatch: expected {expected}, got {actual}")
if not (target / "tokenizer.json").is_file():
    raise SystemExit("pinned snapshot is missing tokenizer.json")
(target / ".prism-model-revision").write_text(
    os.environ["PRISM_MODEL_REVISION"] + "\n", encoding="utf-8"
)
PY

RUN python -c "import re,sys; assert re.fullmatch(r'[0-9a-f]{40}', sys.argv[1]), 'GIT_SHA must be a full lowercase commit SHA'" "${GIT_SHA}" && \
    python -c "from prism_serve.gateway.app import main; from prism_serve.gateway.performance_harness import PerformanceTraceRegistry; from prism_serve.process_identity import assert_pidfd_support; assert_pidfd_support()" && \
    mkdir -p /opt/prism/build && \
    python -m pip freeze --all > /tmp/prism-pip-freeze.txt && \
    LC_ALL=C sort /tmp/prism-pip-freeze.txt > /opt/prism/build/pip-freeze.txt && \
    rm /tmp/prism-pip-freeze.txt

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PIP_NO_CACHE_DIR=1

LABEL org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.title="prism-serve experimental 2P2D gateway" \
      ai.sparksnail.prism.image.variant="${PRISM_IMAGE_VARIANT}" \
      ai.sparksnail.prism.model.profile="${PRISM_MODEL_PROFILE}" \
      ai.sparksnail.prism.model.id="${PRISM_MODEL_ID}" \
      ai.sparksnail.prism.model.revision="${PRISM_MODEL_REVISION}" \
      ai.sparksnail.prism.model.tokenizer-revision="${PRISM_TOKENIZER_REVISION}" \
      ai.sparksnail.prism.model.config-sha256="${PRISM_MODEL_CONFIG_SHA256}" \
      ai.sparksnail.prism.model.kv-compatibility-id="${PRISM_KV_COMPATIBILITY_ID}"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3)" || exit 1

STOPSIGNAL SIGTERM
ENTRYPOINT ["prism-serve"]

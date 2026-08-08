# syntax=docker/dockerfile:1.7

FROM python:3.12.4-slim-bookworm@sha256:a074fac67aa01841fee592d00bae14d25dcaf98ef6e12a683ecceb7e0147e2d1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
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
    PRISM_KV_COMPATIBILITY_ID=a305c48442086f050a0f703c9e79e6e4596c52eaf5dd9f9015cc1a744c1b5b19

ENV PRISM_SERVE_TOKENIZER_MODEL=/opt/models/Qwen3-0.6B \
    PRISM_SERVE_TOKENIZER_REVISION=9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439 \
    PRISM_SERVE_MODEL_PROFILE_ID=week12-qwen3-0.6b \
    PRISM_SERVE_MODEL_ID=Qwen/Qwen3-0.6B \
    PRISM_SERVE_MODEL_REVISION=9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439 \
    PRISM_SERVE_MODEL_CONFIG_SHA256=660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd \
    PRISM_SERVE_RUNTIME_DTYPE=bfloat16 \
    PRISM_SERVE_TENSOR_PARALLEL_SIZE=1 \
    PRISM_SERVE_KV_LAYOUT=NHDC \
    PRISM_SERVE_KV_BLOCK_BYTES=29360128 \
    PRISM_SERVE_KV_COMPATIBILITY_ID=a305c48442086f050a0f703c9e79e6e4596c52eaf5dd9f9015cc1a744c1b5b19

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

# Gateway needs the exact tokenizer/config snapshot for affinity tokenization,
# but it does not need model weights. Keep this image CPU-only and small.
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

COPY pyproject.toml README.md LICENSE ./
COPY prism_serve ./prism_serve

ARG GIT_SHA
ARG SOURCE_URL=https://github.com/SparkSnail/prism-serve

ENV PRISM_IMAGE_GIT_SHA=${GIT_SHA}

RUN python -c "import re,sys; assert re.fullmatch(r'[0-9a-f]{40}', sys.argv[1]), 'GIT_SHA must be a full lowercase commit SHA'" "${GIT_SHA}" && \
    python -m pip install --no-build-isolation --no-deps . && \
    python -c "from prism_serve.gateway.app import main; from prism_serve.process_identity import assert_pidfd_support; from transformers import AutoTokenizer; assert_pidfd_support()" && \
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
      ai.sparksnail.prism.model.id="Qwen/Qwen3-0.6B" \
      ai.sparksnail.prism.model.revision="9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439" \
      ai.sparksnail.prism.model.config-sha256="660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3)" || exit 1

STOPSIGNAL SIGTERM
ENTRYPOINT ["prism-serve"]

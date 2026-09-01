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

FROM profile-${PRISM_IMAGE_VARIANT} AS model-staging

# ``model-cache`` is a required BuildKit named context. The Gateway only
# stages tokenizer/config metadata and never downloads from the Hub.
RUN --mount=type=bind,from=model-cache,source=.,target=/mnt/model-cache,ro \
    python - <<'PY'
import hashlib
import json
import os
import shutil
from pathlib import Path

cache_root = Path("/mnt/model-cache").resolve()
model_name = Path(os.environ["PRISM_MODEL"]).name
candidates = (cache_root / model_name, cache_root)
source = next(
    (candidate.resolve() for candidate in candidates if (candidate / "config.json").is_file()),
    None,
)
if source is None:
    raise SystemExit(
        "model-cache must be a model directory or contain "
        f"{model_name}/config.json"
    )
try:
    source.relative_to(cache_root)
except ValueError as exc:
    raise SystemExit("model-cache model path must stay inside the named context") from exc

expected_revision = os.environ["PRISM_MODEL_REVISION"]
expected_tokenizer_revision = os.environ["PRISM_TOKENIZER_REVISION"]
expected_model_id = os.environ["PRISM_MODEL_ID"]
revision_marker = source / ".prism-model-revision"
if revision_marker.is_symlink() or not revision_marker.is_file():
    raise SystemExit("model-cache is missing .prism-model-revision")
actual_revision = revision_marker.read_text(encoding="utf-8").strip()
if actual_revision != expected_revision:
    raise SystemExit(
        "model-cache revision mismatch: "
        f"expected {expected_revision}, got {actual_revision}"
    )

manifest_path = source / ".prism-model-manifest.json"
if manifest_path.is_symlink() or not manifest_path.is_file():
    raise SystemExit("model-cache is missing .prism-model-manifest.json")
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    raise SystemExit("model-cache manifest is not valid JSON") from exc
if not isinstance(manifest, dict) or set(manifest) != {
    "schema_version", "model_id", "revision", "tokenizer_revision",
    "config_sha256", "files",
}:
    raise SystemExit("model-cache manifest has an invalid shape")
if manifest["schema_version"] != "prism.local_model_cache/v1":
    raise SystemExit("model-cache manifest schema_version is unsupported")
if manifest["model_id"] != expected_model_id:
    raise SystemExit("model-cache manifest model_id mismatch")
if manifest["revision"] != expected_revision:
    raise SystemExit("model-cache manifest revision mismatch")
if manifest["tokenizer_revision"] != expected_tokenizer_revision:
    raise SystemExit("model-cache manifest tokenizer_revision mismatch")
expected_config_sha = os.environ["PRISM_MODEL_CONFIG_SHA256"]
if manifest["config_sha256"] != expected_config_sha:
    raise SystemExit("model-cache manifest config_sha256 mismatch")
manifest_files = manifest["files"]
if not isinstance(manifest_files, dict):
    raise SystemExit("model-cache manifest files must be an object")

target = Path(os.environ["PRISM_MODEL"])
required_files = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
optional_names = {
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
    "tokenizer.model",
    "model.safetensors.index.json",
}

def resolve_cache_file(relative_path):
    relative_path = Path(relative_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise SystemExit("model-cache manifest contains an unsafe path")
    source_file = (source / relative_path).resolve()
    try:
        source_file.relative_to(cache_root)
    except ValueError as exc:
        raise SystemExit("model-cache contains a file outside the named context") from exc
    if not source_file.is_file():
        raise SystemExit(f"model-cache is missing required file: {relative_path}")
    return relative_path, source_file

def verify_hash(relative_path, source_file):
    manifest_key = relative_path.as_posix()
    expected_hash = manifest_files.get(manifest_key)
    if not isinstance(expected_hash, str) or len(expected_hash) != 64 \
            or any(character not in "0123456789abcdef" for character in expected_hash):
        raise SystemExit(f"model-cache manifest is missing {manifest_key}")
    digest = hashlib.sha256()
    with source_file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise SystemExit(f"model-cache file hash mismatch: {manifest_key}")

def copy_from_cache(relative_path):
    relative_path, source_file = resolve_cache_file(relative_path)
    verify_hash(relative_path, source_file)
    destination = target / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, destination)

for relative_path in required_files:
    copy_from_cache(relative_path)
weight_files = sorted(source.glob("*.safetensors"))
if not weight_files:
    raise SystemExit("model-cache is missing safetensors weights")
for weight_file in weight_files:
    # The Gateway does not copy weights, but it still verifies every shard in
    # the cache so its runtime identity cannot hide a mismatched worker image.
    weight_key, resolved_weight = resolve_cache_file(weight_file.name)
    verify_hash(weight_key, resolved_weight)
for candidate in source.iterdir():
    if candidate.is_file() and (
        candidate.name in optional_names or candidate.name.startswith("chat_template")
    ):
        copy_from_cache(candidate.name)

actual_config_sha = hashlib.sha256((target / "config.json").read_bytes()).hexdigest()
if actual_config_sha != expected_config_sha:
    raise SystemExit(
        "config.json SHA-256 mismatch: "
        f"expected {expected_config_sha}, got {actual_config_sha}"
    )
shutil.copy2(revision_marker, target / ".prism-model-revision")
shutil.copy2(manifest_path, target / ".prism-model-manifest.json")
PY

FROM profile-${PRISM_IMAGE_VARIANT} AS selected

ARG PRISM_IMAGE_VARIANT
ARG GIT_SHA=local
ARG PRISM_RELEASE=false
ARG SOURCE_URL=https://github.com/SparkSnail/prism-serve

RUN case "${PRISM_IMAGE_VARIANT}" in \
      correctness|performance) ;; \
      *) echo "PRISM_IMAGE_VARIANT must be correctness or performance" >&2; exit 64 ;; \
    esac

ENV PRISM_IMAGE_GIT_SHA=${GIT_SHA} \
    PRISM_SERVE_IMAGE_SOURCE_URL=${SOURCE_URL} \
    PRISM_SERVE_IMAGE_SOURCE_COMMIT=${GIT_SHA}

# The Gateway image carries tokenizer/config metadata only; workers own model
# weights. Keeping this as a separate copy also makes the dependency explicit.
COPY --link --from=model-staging /opt/models/ /opt/models/
COPY licenses/QWEN3-MODEL-NOTICE.txt /opt/prism/licenses/QWEN3-MODEL-NOTICE.txt

RUN python -c "import re,sys; sha,release=sys.argv[1:]; valid_sha=bool(re.fullmatch(r'[0-9a-f]{40}', sha)); assert release in ('true','false'), 'PRISM_RELEASE must be true or false'; assert valid_sha or (release == 'false' and sha == 'local'), 'release images require a full lowercase commit SHA'; assert not (release == 'true' and sha == 'local'), 'PRISM_RELEASE=true requires a full lowercase commit SHA'" "${GIT_SHA}" "${PRISM_RELEASE}" && \
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

# The Gateway only needs read access to the image and writes runtime state to
# the mounted replacement store or /tmp. Keep the published image non-root.
RUN addgroup --system --gid 10001 prism && \
    adduser --system --uid 10001 --ingroup prism --no-create-home prism && \
    mkdir -p /var/lib/prism/replacement && \
    chown -R prism:prism /var/lib/prism
USER prism

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3)" || exit 1

STOPSIGNAL SIGTERM
ENTRYPOINT ["prism-serve"]

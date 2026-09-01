# Container images

This directory contains the Docker build definition for the Prism Gateway. The build is profile-bound: the `correctness` profile packages Qwen3-0.6B tokenizer/configuration metadata for the fixed correctness path, and the `performance` profile packages the corresponding Qwen3-8B metadata for the paired 2P2D performance profile.

Run every command from the repository root. `docker/Dockerfile` is intentionally separate from the application entry point, but the build context must remain `.` because the image copies the package, metadata, license, and README from the repository root. The root `.dockerignore` continues to filter that context.

## Image profiles

| Image | Profile |
| --- | --- |
| `sparksnail/prism-serve:<release-tag>` | Qwen3-0.6B correctness Gateway |
| `sparksnail/prism-serve:<release-tag>-qwen3-8b` | Qwen3-8B performance Gateway |

Select an existing version tag from the container registry. Tags are release aliases; pin an image digest together with the matching source commit when installing the chart or collecting paired benchmark evidence. Do not use `latest`.

## Build from source

The build requires Docker BuildKit and a local model-cache named context. It never downloads a model: the cache must already contain the selected model plus its `.prism-model-manifest.json` and `.prism-model-revision` identity files.

Build the correctness Gateway with a cache parent that contains `Qwen3-0.6B`, or pass that model directory directly:

```bash
MODEL_CACHE="$HOME/models"
docker build \
  -f docker/Dockerfile \
  --build-context model-cache="$MODEL_CACHE" \
  --build-arg PRISM_IMAGE_VARIANT=correctness \
  --build-arg GIT_SHA="$(git rev-parse HEAD)" \
  -t prism-serve:local .
```

Build the performance Gateway with a cache that contains `Qwen3-8B`:

```bash
docker build \
  -f docker/Dockerfile \
  --build-context model-cache="$HOME/models" \
  --build-arg PRISM_IMAGE_VARIANT=performance \
  --build-arg GIT_SHA="$(git rev-parse HEAD)" \
  -t prism-serve:qwen3-8b .
```

The Gateway image contains verified configuration and tokenizer metadata only; the paired `prism-infer` worker image owns the safetensors weights. The cache manifest records the model and tokenizer revisions and SHA-256 values for its files. A missing marker, mismatched revision, missing file, or SHA-256 mismatch fails the build. Keep model bytes outside the application build context so they are supplied only through `model-cache`.

## Create cache identity

After acquiring a model by your own approved process, create the offline cache identity once. This command writes both required identity files and records a hash for every model file:

```bash
python scripts/create_model_cache_manifest.py \
  --model-dir "$HOME/models/Qwen3-8B" \
  --model-id Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --config-sha256 f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30
```

Use the corresponding Qwen3-0.6B revision and config hash when preparing the correctness cache. The build does not create a marker or accept an unverified cache.

## Publish a versioned image

Set `PRISM_RELEASE=true` for a published image and pass an exact lowercase 40-character source revision. Release mode rejects `local`:

```bash
docker build \
  -f docker/Dockerfile \
  --build-context model-cache="$HOME/models" \
  --build-arg PRISM_IMAGE_VARIANT=correctness \
  --build-arg PRISM_RELEASE=true \
  --build-arg GIT_SHA="$(git rev-parse HEAD)" \
  -t prism-serve:<release> .
```

After verifying the source and image, create a Git tag with the same version on the clean source commit used for the image. Create a GitHub Release only for an intentional user-facing milestone.

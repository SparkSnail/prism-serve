<p align="center">
  <img src="assets/banner.png" width="640" alt="Prism-Serve"/>
</p>

<h3 align="center">A Kubernetes control plane for disaggregated LLM serving</h3>

<p align="center">
  <a href="#features"><b>Features</b></a> &middot;
  <a href="#reference-performance-snapshot"><b>Performance</b></a> &middot;
  <a href="#installation"><b>Installation</b></a> &middot;
  <a href="#quick-start"><b>Quick Start</b></a> &middot;
  <a href="#testing"><b>Testing</b></a> &middot;
  <a href="#deployment"><b>Deployment</b></a>
</p>

<p align="center">
  <a href="https://github.com/SparkSnail/prism-serve/actions/workflows/ci.yml">
    <img src="https://github.com/SparkSnail/prism-serve/actions/workflows/ci.yml/badge.svg" alt="CI"/>
  </a>
</p>

**prism-serve** provides the Kubernetes Gateway and control plane for the Prism stack. It schedules and prefix-routes [prism-infer](https://github.com/SparkSnail/prism-infer) workers, which execute model inference and KV operations. The current code covers cluster-level prefill/decode (PD) scheduling, KV transfer flow control, recompute fallback, and a 10 ms reconcile loop modelled on Ray Serve.

## Features

- **FastAPI gateway**: liveness/readiness probes and a Prism token-id streaming route
- **Fixed 2P2D deployment**: Helm templates for one active Gateway and an atomic four-worker world
- **PD scheduler**: shortest-queue prefill selection, most-free-slots decode selection, runtime instance-count recommendation
- **KV transfer flow control**: dynamic high/low watermark, per-dst byte cap, FIFO deferred queue with automatic flush
- **Recompute fallback**: timeout detection, `reset_to_waiting` on D instance, abort after max attempts
- **Request state machine**: per-request `SeqState` lifecycle, illegal-transition guard, stuck-request detection, TTFT timestamps
- **10 ms schedule loop**: Phase 1-6 reconcile - assign P/D, submit KV, stuck check, deferred flush, collect finished, metrics
- **NATS queue**: publish/subscribe wrapper, bounded inbox, queue-group load balancing, wildcard `kv_usage.*` subscription
- **Metrics**: Prometheus counters/gauges/histograms for TTFT, KV transfer, congestion, deferred depth, slot utilisation

## Reference Performance Snapshot

This immutable historical reference measures the end-to-end Prism stack, not an isolated benchmark of either repository. `prism-serve` provides prefix-affinity routing and coordination, while `prism-infer` runs the prefill/decode workers and KV-cache runtime. On the recorded model, hardware, topology, request mix, and concurrency, enabling affinity lowered time-to-first-token and end-to-end latency while increasing completed-request throughput. The table keeps the decode trade-off visible. It is a controlled paired benchmark for the fixed 2P2D setup, not a production SLO or a claim about the current working tree.

**Headline:** With affinity enabled, TTFT p50 is 64.9% lower, E2E p50 is 33.4% lower, and successful request throughput is 35.1% higher on this workload. TPOT rises, so this is a workload-specific prefix-reuse result, not a blanket speedup.

The machine-readable snapshot and immutable input provenance are in [`bench/results/performance_snapshot.json`](bench/results/performance_snapshot.json).

| Benchmark setup | Value |
|---|---|
| Model | Qwen3-8B, BF16 |
| Hardware | 2 nodes, 4 NVIDIA L20 GPUs |
| Parallelism | fixed 2P2D, TP=1 |
| Workload | 512 shared + 257 unique input tokens; 32 output tokens |
| Concurrency | 50 |
| Sampling | streaming; 3 repetitions of 100 warm-up + 250 measured requests (1,050 total per configuration) |
| Transport | NCCL Socket |

| Metric | Affinity OFF (baseline) | Affinity ON | Change vs OFF |
|---|---:|---:|---:|
| TTFT p50 / p95 / p99 (ms) | 6,527.661 / 9,975.492 / 10,729.448 | 2,293.325 / 4,305.316 / 5,522.773 | 64.868% / 56.841% / 48.527% lower |
| TPOT p50 / p95 / p99 (ms) | 25.798 / 29.938 / 31.989 | 81.959 / 145.253 / 151.019 | 217.696% / 385.173% / 372.103% higher |
| E2E p50 / p95 / p99 (ms) | 7,387.104 / 10,750.170 / 11,610.063 | 4,916.720 / 5,442.772 / 5,581.826 | 33.442% / 49.370% / 51.923% lower |
| Successful requests/s | 6.274584 | 8.475600 | 35.078% higher |
| Successful output tokens/s | 200.7867 | 271.2192 | 35.078% higher |

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/SparkSnail/prism-serve.git
cd prism-serve
pip install -e .
```

The Gateway uses HTTP to communicate with `prism-infer`; the engine is not a required install-time dependency. For local in-process integration tests, add the optional adapter explicitly:

```bash
pip install -e ".[infer]"
```

The optional operator benchmark/correctness harness also needs the tokenizer extra:

```bash
pip install -e ".[harness]"
```

Container images, source builds, model-cache verification, and release tags are documented in the [Docker guide](docker/README.md).

## Quick Start

Start the gateway:

```bash
prism-serve            # requires NATS at nats://localhost:4222
# or: uvicorn prism_serve.gateway.app:app --host 0.0.0.0 --port 8080
```

For gateway-only local development without NATS, explicitly enable the mock queue:

```bash
PRISM_SERVE_NATS_REQUIRED=false prism-serve
```

This mode intentionally has no infer workers. It is suitable for checking the Gateway health endpoints and CPU control-plane tests only.

Check it is alive:

```bash
curl localhost:8080/healthz   # {"status":"ok","version":"..."}
curl localhost:8080/readyz    # {"status":"ready"}
```

After the fixed worker world is ready, send a normal token-id streaming request:

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello from Prism"}],
    "stream": true,
    "temperature": 0,
    "max_tokens": 32
  }'
```

The stream intentionally exposes generated `token_id` values. It is a Prism runtime endpoint, not a drop-in replacement for an OpenAI text response.

For a fixed 2P2D deployment, worker identity and readiness are established by the Helm topology and `scripts/pd_worldctl.py`. The legacy `/internal/register_instance` endpoint is only a single-process compatibility endpoint and is rejected when the Week 12 worker registry is active; it is not the worker bootstrap path.


## Testing

```bash
pip install -e ".[test]"
python -m pytest tests/ -q
```

## Deployment

The chart is a fixed 2P2D reference deployment. It does not install NATS or create the startup permit that authorizes a worker world. Prepare those two dependencies first:

1. Provide a reachable NATS service and set its URL. For example, an existing service named `nats` in the `prism` namespace is `nats://nats:4222`.
2. Install the chart with the published image tags (or pin both images to digests):

```bash
helm upgrade --install prism-serve k8s/helm/prism-serve \
  -n prism --create-namespace \
  --set-string nats.url=nats://nats:4222
```

The default chart references the release tags `v0.2.0` (Gateway) and `v0.3.0` (worker). Verify that the tags are present in your registry before installing; the chart does not build or publish images:

```bash
docker manifest inspect sparksnail/prism-serve:v0.2.0
docker manifest inspect sparksnail/prism-infer:v0.3.0
```

Pin both images to their registry digests for a reproducible deployment:

```bash
helm install prism-serve k8s/helm/prism-serve \
  -n prism --create-namespace \
  --set-string gateway.image.digest=sha256:REPLACE_WITH_GATEWAY_DIGEST \
  --set-string worker.image.digest=sha256:REPLACE_WITH_WORKER_DIGEST \
  --set-string gateway.image.sourceCommit=REPLACE_WITH_GATEWAY_COMMIT \
  --set-string worker.image.sourceCommit=REPLACE_WITH_WORKER_COMMIT
```

Workers intentionally wait for a controller-issued startup permit before loading the model or initializing NCCL. After the Gateway Service exists, run the controller from this repository (with `kubectl`/Helm access to the same cluster):

```bash
mkdir -p .prism-state
python scripts/pd_worldctl.py initialize \
  --release prism-serve \
  --namespace prism \
  --chart k8s/helm/prism-serve \
  --generation 00000000-0000-4000-8000-000000000001 \
  --gateway-url http://prism-serve-prism-serve-gateway.prism.svc:80 \
  --run-state .prism-state/initialize.json \
  --execute
```

The deployment becomes ready only after all four workers report the same generation and the controller accepts the resulting evidence. Node labels `prism.sparksnail.ai/pd-node-group=a|b`, one NVIDIA GPU per worker, and a dynamic storage class for the replacement PVC are required by the default values.

The 8B benchmark profile is an explicit overlay and keeps affinity opt-in:

```bash
docker manifest inspect sparksnail/prism-serve:v0.2.0-qwen3-8b
docker manifest inspect sparksnail/prism-infer:v0.3.0-qwen3-8b
helm install prism-serve k8s/helm/prism-serve \
  -f k8s/helm/prism-serve/values-performance.yaml \
  --set-string nats.url=nats://nats:4222 \
  --set-string gateway.image.digest=sha256:REPLACE_WITH_GATEWAY_DIGEST \
  --set-string worker.image.digest=sha256:REPLACE_WITH_WORKER_DIGEST \
  --set-string gateway.image.sourceCommit=REPLACE_WITH_GATEWAY_COMMIT \
  --set-string worker.image.sourceCommit=REPLACE_WITH_WORKER_COMMIT \
  -n prism --create-namespace
```

For a paired public benchmark, the image digests and commits must describe the same published Gateway and worker images. The chart passes the full `repository@sha256:...` references to the authenticated runtime identity endpoint; a mutable tag or incomplete identity deliberately cannot produce a paired-provenance result.

## License

[Apache-2.0](LICENSE)

<p align="center">
  <img src="assets/banner.png" width="640" alt="Prism-Serve"/>
</p>

<h3 align="center">A Kubernetes control plane for disaggregated LLM serving</h3>

<p align="center">
  <a href="#features"><b>Features</b></a> &middot;
  <a href="#performance-snapshot"><b>Performance</b></a> &middot;
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

- **FastAPI gateway**: liveness/readiness probes, OpenAI-compatible route stub, infer instance registration
- **PD scheduler**: shortest-queue prefill selection, most-free-slots decode selection, runtime instance-count recommendation
- **KV transfer flow control**: dynamic high/low watermark, per-dst byte cap, FIFO deferred queue with automatic flush
- **Recompute fallback**: timeout detection, `reset_to_waiting` on D instance, abort after max attempts
- **Request state machine**: per-request `SeqState` lifecycle, illegal-transition guard, stuck-request detection, TTFT timestamps
- **10 ms schedule loop**: Phase 1-6 reconcile - assign P/D, submit KV, stuck check, deferred flush, collect finished, metrics
- **NATS queue**: publish/subscribe wrapper, bounded inbox, queue-group load balancing, wildcard `kv_usage.*` subscription
- **Metrics**: Prometheus counters/gauges/histograms for TTFT, KV transfer, congestion, deferred depth, slot utilisation

## Performance Snapshot

This paired 2P2D snapshot measures the end-to-end Prism stack, not an isolated benchmark of either repository. `prism-serve` provides prefix-affinity routing and coordination, while `prism-infer` runs the prefill/decode workers and KV-cache runtime. On the same model, hardware, topology, request mix, and concurrency, enabling affinity lowers time-to-first-token and end-to-end latency while increasing completed-request throughput. The table keeps the decode trade-off visible instead of presenting a single best-case number. This is a controlled paired benchmark for the fixed 2P2D setup, not a production SLO.

**Headline:** With affinity enabled, TTFT p50 is 64.9% lower, E2E p50 is 33.4% lower, and successful request throughput is 35.1% higher on this workload. TPOT rises, so this is a workload-specific prefix-reuse result, not a blanket speedup.

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

Check it is alive:

```bash
curl localhost:8080/healthz   # {"status":"ok","version":"..."}
curl localhost:8080/readyz    # {"status":"ready"}
```

Register an infer instance once it is up:

```bash
curl -X POST localhost:8080/internal/register_instance \
  -H "Content-Type: application/json" \
  -d '{"instance_id":"p-0","role":"prefill"}'

curl -X POST localhost:8080/internal/register_instance \
  -H "Content-Type: application/json" \
  -d '{"instance_id":"d-0","role":"decode","max_slots":127}'
```


## Testing

```bash
pip install -e ".[test]"
python -m pytest tests/ -q
```

## Deployment

Kubernetes deployment via Helm:

```bash
helm install prism-serve k8s/helm/prism-serve -n prism --create-namespace
```

## License

[Apache-2.0](LICENSE)

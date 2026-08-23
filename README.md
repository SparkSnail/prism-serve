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

**prism-serve** coordinates disaggregated LLM inference workers on Kubernetes. The current
code covers cluster-level prefill/decode (PD) scheduling, KV transfer flow control,
recompute fallback, and a 10 ms reconcile loop modelled on Ray Serve.

> [!WARNING]
> This is an experimental fixed 2P2D snapshot, not a production-ready release, and it
> does not claim complete E2E success. A separate frozen correctness campaign ran
> 31/35 cases passed (31 passed, 3 failed, and 1 was blocked); 4/5 evidence packets passed;
> final-clean failed. Known limitations are a decode-worker SIGSEGV during gateway
> restart cleanup and tunnel recovery failure while the local forwarding port remained bound.

## Features

- [x] **FastAPI gateway**: liveness/readiness probes, OpenAI-compatible route stub, infer instance registration
- [x] **PD scheduler**: shortest-queue prefill selection, most-free-slots decode selection, runtime instance-count recommendation
- [x] **KV transfer flow control**: dynamic high/low watermark, per-dst byte cap, FIFO deferred queue with automatic flush
- [x] **Recompute fallback**: timeout detection, `reset_to_waiting` on D instance, abort after max attempts
- [x] **Request state machine**: per-request `SeqState` lifecycle, illegal-transition guard, stuck-request detection, TTFT timestamps
- [x] **10 ms schedule loop**: Phase 1-6 reconcile - assign P/D, submit KV, stuck check, deferred flush, collect finished, metrics
- [x] **NATS queue**: publish/subscribe wrapper, bounded inbox, queue-group load balancing, wildcard `kv_usage.*` subscription
- [x] **Metrics**: Prometheus counters/gauges/histograms for TTFT, KV transfer, congestion, deferred depth, slot utilisation

## Performance Snapshot

A frozen absolute-baseline run on 2026-08-18 passed the canonical performance
validator (`headline_valid=true`) and its final resource-clean proof. These numbers
cover only affinity disabled (`PERF_OFF`). The affinity-enabled `PERF_ON` comparison
is `NOT_RUN`; no optimization gain is claimed.

| Run setup | Value |
|---|---|
| Environment | Alibaba Cloud ACK, 2 nodes, 4 x NVIDIA L20 GPUs |
| Model / topology | Qwen3-8B, BF16, TP=1, fixed 2P2D |
| Routing / transport | Affinity disabled; `NCCL_SOCKET` on 1,050/1,050 request traces |
| Load shape | Concurrency 50; 769 input tokens; 32 output tokens |
| Samples | 300/300 warm-up requests passed; 750/750 measured requests passed across 3 repetitions |

| Latency | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| TTFT | 6,921.109 | 9,615.555 | 10,908.582 |
| TPOT | 27.199 | 29.338 | 31.013 |
| Inter-chunk | 22.795 | 84.418 | 139.210 |
| End-to-end | 7,790.814 | 10,505.132 | 11,755.448 |

| Throughput / GPU telemetry | Result |
|---|---:|
| Successful requests/s | 6.005 |
| Successful output tokens/s | 192.152 |
| GPU utilization, mean / p95 | 54.664% / 92.000% |
| GPU memory, mean / p95 | 39,931 / 40,008 MiB |
| GPU power, mean / p95 | 165.540 / 225.330 W |
| GPU SM clock, mean / p95 | 2,520 / 2,520 MHz |

Evidence identity: run `20260818T135933Z`; final performance-packet SHA-256
`5951db093ca55124632c0ddd4f197b46534ea9e43cf0b094eb7d2cf43cd35836`.

### Supplemental `PERF_ON` cell evidence

Collected on 2026-08-22 with the same ACK/Qwen3-8B BF16 fixed 2P2D setup.
These cell-level results have zero failed requests and `headline_valid=true`,
but remain separate from the canonical `PERF_OFF` packet:

| Cell | Warmup / measured | TTFT p50 / p95 / p99 (ms) | TPOT p50 (ms) | Output tok/s |
|---|---:|---:|---:|---:|
| `balanced-1024x128-c100` | 600 / 1,500 | 17,157.092 / 24,832.803 / 29,436.452 | 30.011 | 582.598 |
| `prefill-2048x32-c50` | 300 / 750 | 13,706.317 / 20,176.845 / 23,585.095 | 30.099 | 108.156 |
| `decode-128x256-c50` | 300 / 750 | 2,569.163 / 5,107.496 / 6,417.857 | 36.844 | 883.593 |

`unified_comparison=NOT_RUN`; no optimization gain is claimed.

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


## Test

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

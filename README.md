<p align="center">
  <img src="assets/banner.png" width="640" alt="Prism-Serve"/>
</p>

<h3 align="center">A Kubernetes control plane for disaggregated LLM serving</h3>

<p align="center">
  <a href="#features"><b>Features</b></a> &middot;
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
> does not claim complete E2E success. The latest frozen two-node/four-GPU campaign ran
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

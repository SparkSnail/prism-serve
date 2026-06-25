<p align="center">
  <img src="assets/banner.png" width="640" alt="Prism-Serve"/>
</p>

<h3 align="center">A Kubernetes control plane for disaggregated LLM serving</h3>

<p align="center">
  <a href="#features"><b>Features</b></a> ·
  <a href="#installation"><b>Installation</b></a> ·
  <a href="#quick-start"><b>Quick Start</b></a> ·
  <a href="#deployment"><b>Deployment</b></a>
</p>

**prism-serve** is a control plane that turns single-instance LLM inference engines into a fault-tolerant, elastic cluster service on Kubernetes: KV-affinity routing, predictive autoscaling, stateful rescale, and cluster-level prefill/decode (PD) scheduling.

## Features

- [x] **`prism_serve` package skeleton**: gateway / scheduler / router / engine / metrics
- [x] **FastAPI gateway** with `/healthz` and `/readyz` probes and an OpenAI-compatible route stub
- [x] **Helm chart skeleton**: gateway Deployment + Service, worker StatefulSet

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
prism-serve            # serves the FastAPI gateway on :8080
# or: uvicorn prism_serve.gateway.app:app --host 0.0.0.0 --port 8080
```

Check it is alive:

```bash
curl localhost:8080/healthz   # {"status":"ok","version":"0.0.1"}
curl localhost:8080/readyz    # {"status":"ready"}
```

Configuration is read from environment variables (prefix `PRISM_SERVE_`), e.g. `PRISM_SERVE_PORT=9090 prism-serve`.

## Deployment

Kubernetes deployment via Helm:

```bash
helm install prism-serve k8s/helm/prism-serve -n prism --create-namespace
```

## License

Apache-2.0, see [LICENSE](LICENSE).

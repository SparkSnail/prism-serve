# Benchmarks

This directory contains reproducible benchmark entry points and schemas. It also contains the committed summary of the paired GPU result shown in the top-level README. Raw samples remain in a separate evidence archive with the image digest, model revision, topology, workload, and capture logs that produced the summary.

## CPU routing policy microbenchmark

`bench_affinity.py` is deterministic and exercises only the local prefix-index and routing policy. It does not load a model, start NATS, contact a worker, or make GPU latency/throughput claims.

```bash
python bench/bench_affinity.py --requests 1000 --block-size 256 --block-bytes 29360128
```

The JSON result includes its request count, route-hit ratio, mapped-to-cold byte ratio, elapsed CPU time, and an explicit list of claims it does not make. The schema is `bench/schemas/synthetic_cpu_policy-v1.json`.

## Public endpoint client

`bench_endpoint.py` is a small operator-side client for the Gateway's token-id streaming endpoint. It records one row per request and computes p50, p95, and p99 TTFT, TPOT, and end-to-end latency from requests that received the expected number of valid token chunks, a `finish_reason: "stop"` terminal chunk, and a terminal `[DONE]` event. It does not hide failed rows or turn a partial run into a passing result.

The client targets the opt-in performance harness, so the deployment must have the pinned tokenizer and the operator harness enabled. Supply the operator token through `PRISM_OPERATOR_TOKEN`; never put that token in a prompt file or commit it to the repository. An exact input-token count is required because the Gateway validates the pinned tokenizer result. A JSONL prompt file keeps the input workload explicit:

```json
{"content":"<prompt whose tokenized length is 512>","expected_input_tokens":512}
```

```bash
python bench/bench_endpoint.py \
  --url http://127.0.0.1:8080/v1/chat/completions \
  --model Qwen/Qwen3-8B \
  --prompt-file prompts.jsonl \
  --expected-output-tokens 32 \
  --requests 10 --concurrency 2 \
  --output endpoint-result.json
```

Without `--provenance`, the output is deliberately labelled `timing-only` and must not be used as a reproducible paired result. A provenance file alone is not enough: it must contain the exact endpoint/workload binding and the benchmark must read the runtime identity from that same Gateway origin. The client labels a result `paired-provenance` only after both documents match field-for-field. The manifest follows `bench/schemas/public_endpoint_provenance-v1.json`; the authenticated runtime response follows `bench/schemas/public_endpoint_runtime_identity-v1.json`.

```bash
PRISM_OPERATOR_TOKEN="$PRISM_OPERATOR_TOKEN" \
PRISM_RUNTIME_IDENTITY_URL="<authenticated-runtime-identity-url>" \
python bench/bench_endpoint.py \
  --url http://127.0.0.1:8080/v1/chat/completions \
  --model Qwen/Qwen3-8B --prompt-file prompts.jsonl \
  --provenance provenance.json \
  --runtime-identity-url "$PRISM_RUNTIME_IDENTITY_URL" \
  --output endpoint-result.json
```

The operator is responsible for choosing prompts whose declared token count matches the pinned tokenizer. The performance harness exposes an authenticated runtime identity endpoint and requires the same operator token. Set `PRISM_RUNTIME_IDENTITY_URL` to that endpoint. Install the performance overlay with immutable `gateway.image.digest` and `worker.image.digest` values plus the matching 40-character `gateway.image.sourceCommit` and `worker.image.sourceCommit`; the chart then exports full `repository@sha256:...` references to the endpoint. The client rejects an incomplete runtime identity before sending requests.

The client rejects a different origin, endpoint path, model/revision, topology generation, source commit, source URL, or image digest before sending any benchmark request. Compare only runs with the same model, image digests, topology, prompt mix, concurrency, and warm-up policy.

## 2P2D performance campaign

The paired GPU result shown in the top-level README is recorded as an immutable historical reference in [`results/performance_snapshot.json`](results/performance_snapshot.json) and validated against `schemas/performance_snapshot-v1.json`. It was collected by the operator-only performance harness with the fixed `qwen3-8b-bf16-tp1` profile. It requires the `values-performance.yaml` overlay, a Secret-backed harness token, real `prism-infer` workers, and an immutable Gateway/worker image pair. The normal control path is `scripts/pd_worldctl.py initialize`; workers remain blocked before model and NCCL initialization until that controller publishes a matching startup permit.

The campaign producer and validator are maintained separately from this repository. The public evidence record includes the model revision, source commits, image digests, topology, workload, aggregation percentiles, and route counters needed to interpret the headline numbers.

Use the CPU microbenchmark to validate local routing changes. Do not use it as a substitute for model parity, cross-process transfer, fault recovery, or GPU performance evidence.

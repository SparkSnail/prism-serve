"""Small public client for collecting Prism token-stream endpoint timings."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx


AUTH_HEADER = "x-prism-week12-token"
PERFORMANCE_SCHEMA = "prism.performance_request/v1"
PROVENANCE_SCHEMA = "prism.public_endpoint_provenance/v1"
RUNTIME_IDENTITY_SCHEMA = "prism.public_endpoint_runtime_identity/v1"
RUNTIME_IDENTITY_PATH = "/internal/week12/performance/runtime-identity"
_SHA40 = re.compile(r"^[0-9a-f]{40}$", re.ASCII)
_IMAGE_DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$", re.ASCII)
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prompt_digest(prompts: list[dict[str, Any]]) -> str:
    return _canonical_digest(prompts)


def _load_prompts(
    path: Path | None,
    default: str | None,
    default_expected_input_tokens: int | None = None,
) -> list[dict[str, Any]]:
    if path is None:
        if not isinstance(default, str) or not default:
            raise ValueError(
                "provide --prompt-file or --prompt with --expected-input-tokens"
            )
        if type(default_expected_input_tokens) is not int:
            raise ValueError(
                "--expected-input-tokens is required when --prompt is used"
            )
        if not 1 <= default_expected_input_tokens <= 4096:
            raise ValueError("expected_input_tokens must be 1..4096")
        return [{
            "content": default,
            "expected_input_tokens": default_expected_input_tokens,
        }]
    prompts: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"prompt file line {line_number} is not JSON") from exc
        if not isinstance(value, dict) or not isinstance(value.get("content"), str):
            raise ValueError(
                f"prompt file line {line_number} must contain a string content"
            )
        expected = value.get(
            "expected_input_tokens", default_expected_input_tokens
        )
        if expected is None or (
            type(expected) is not int or not 1 <= expected <= 4096
        ):
            raise ValueError(
                f"prompt file line {line_number} expected_input_tokens must be 1..4096"
            )
        prompts.append({
            "content": value["content"],
            "expected_input_tokens": expected,
        })
    if not prompts:
        raise ValueError("prompt file does not contain any prompts")
    return prompts


def _load_provenance(
    path: Path | None,
    *,
    expected_binding: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Load immutable provenance, or explicitly mark a run timing-only."""
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"provenance file is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("provenance must be a JSON object")
    if value.get("schema_version") != PROVENANCE_SCHEMA:
        raise ValueError(f"provenance schema_version must be {PROVENANCE_SCHEMA}")
    if set(value) != {
        "schema_version", "gateway", "worker", "model",
        "topology_generation", "binding",
    }:
        raise ValueError("provenance has an invalid shape")
    for component in ("gateway", "worker"):
        record = value.get(component)
        if not isinstance(record, dict):
            raise ValueError(f"provenance.{component} must be an object")
        if set(record) != {"source_commit", "image", "source_url"}:
            raise ValueError(
                f"provenance.{component} must contain only immutable identity fields"
            )
        source_commit = record.get("source_commit")
        image = record.get("image")
        source_url = record.get("source_url")
        if not isinstance(source_commit, str) or _SHA40.fullmatch(source_commit) is None:
            raise ValueError(f"provenance.{component}.source_commit must be a 40-char SHA")
        if not isinstance(image, str) or _IMAGE_DIGEST.fullmatch(image) is None:
            raise ValueError(
                f"provenance.{component}.image must use repository@sha256:<64-hex>"
            )
        if not isinstance(source_url, str) \
                or not source_url.startswith(("http://", "https://")):
            raise ValueError(
                f"provenance.{component}.source_url must be an http(s) URL"
            )
    model = value.get("model")
    if isinstance(model, dict) and set(model) != {"id", "revision"}:
        raise ValueError("provenance.model must contain only id and revision")
    if not isinstance(model, dict) or not isinstance(model.get("id"), str) \
            or not model["id"]:
        raise ValueError("provenance.model.id must be a non-empty string")
    if not isinstance(model.get("revision"), str) \
            or _SHA40.fullmatch(model["revision"]) is None:
        raise ValueError("provenance.model.revision must be a 40-char SHA")
    topology_generation = value.get("topology_generation")
    if not isinstance(topology_generation, str) or not topology_generation:
        raise ValueError("provenance.topology_generation must be non-empty")
    binding = value.get("binding")
    if not isinstance(binding, dict):
        raise ValueError(
            "provenance.binding is required; an unbound manifest cannot claim paired evidence"
        )
    required_binding = {
        "endpoint", "model", "cell", "requests", "concurrency",
        "expected_input_tokens", "expected_output_tokens", "prompt_sha256",
    }
    if set(binding) != required_binding:
        raise ValueError(
            "provenance.binding must contain exactly the endpoint workload fields"
        )
    if not isinstance(binding["endpoint"], str) or not binding["endpoint"]:
        raise ValueError("provenance.binding.endpoint must be non-empty")
    if not isinstance(binding["model"], str) or binding["model"] != model["id"]:
        raise ValueError("provenance.binding.model must match provenance.model.id")
    if not isinstance(binding["cell"], str) or not binding["cell"]:
        raise ValueError("provenance.binding.cell must be non-empty")
    for field in ("requests", "concurrency", "expected_output_tokens"):
        if type(binding[field]) is not int or binding[field] <= 0:
            raise ValueError(f"provenance.binding.{field} must be a positive integer")
    if binding["expected_output_tokens"] < 2:
        raise ValueError(
            "provenance.binding.expected_output_tokens must be at least 2"
        )
    expected_input = binding["expected_input_tokens"]
    if expected_input is not None and (
        type(expected_input) is not int or not 1 <= expected_input <= 4096
    ):
        raise ValueError(
            "provenance.binding.expected_input_tokens must be null or 1..4096"
        )
    if (
        not isinstance(binding["prompt_sha256"], str)
        or _SHA256.fullmatch(binding["prompt_sha256"]) is None
    ):
        raise ValueError("provenance.binding.prompt_sha256 must be a SHA-256")
    if expected_binding is not None:
        for field in required_binding:
            if binding[field] != expected_binding[field]:
                raise ValueError(
                    f"provenance binding does not match invocation: {field}"
                )
    return value


def _validate_runtime_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("runtime identity must be a JSON object")
    if value.get("schema_version") != RUNTIME_IDENTITY_SCHEMA:
        raise ValueError(
            f"runtime identity schema_version must be {RUNTIME_IDENTITY_SCHEMA}"
        )
    if value.get("endpoint_path") != "/v1/chat/completions":
        raise ValueError("runtime identity endpoint_path is not the benchmark endpoint")
    if set(value) != {
        "schema_version", "endpoint_path", "gateway", "worker", "model",
        "topology_generation",
    }:
        raise ValueError("runtime identity has an invalid shape")
    for component in ("gateway", "worker"):
        record = value.get(component)
        if not isinstance(record, dict):
            raise ValueError(f"runtime identity.{component} must be an object")
        if set(record) != {"source_commit", "image", "source_url"}:
            raise ValueError(
                f"runtime identity.{component} has an invalid shape"
            )
        for field in ("source_commit", "image", "source_url"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(
                    f"runtime identity.{component}.{field} must be non-empty"
                )
        if not record["source_url"].startswith(("http://", "https://")):
            raise ValueError(
                f"runtime identity.{component}.source_url must be an http(s) URL"
            )
        if _SHA40.fullmatch(record["source_commit"]) is None:
            raise ValueError(
                f"runtime identity.{component}.source_commit must be a 40-char SHA"
            )
        if _IMAGE_DIGEST.fullmatch(record["image"]) is None:
            raise ValueError(
                f"runtime identity.{component}.image must use repository@sha256:<64-hex>"
            )
    model = value.get("model")
    if isinstance(model, dict) and set(model) != {"id", "revision"}:
        raise ValueError("runtime identity.model has an invalid shape")
    if not isinstance(model, dict) or not isinstance(model.get("id"), str) \
            or not model["id"]:
        raise ValueError("runtime identity.model.id must be non-empty")
    if not isinstance(model.get("revision"), str) \
            or _SHA40.fullmatch(model["revision"]) is None:
        raise ValueError("runtime identity.model.revision must be a 40-char SHA")
    generation = value.get("topology_generation")
    if not isinstance(generation, str) or not generation:
        raise ValueError("runtime identity.topology_generation must be non-empty")
    return value


def _runtime_identity_matches(
    provenance: dict[str, Any], runtime: dict[str, Any],
) -> bool:
    for component in ("gateway", "worker"):
        for field in ("source_commit", "image", "source_url"):
            if runtime[component][field] != provenance[component][field]:
                return False
    return (
        runtime["model"]["id"] == provenance["model"]["id"]
        and runtime["model"]["revision"] == provenance["model"]["revision"]
        and runtime["topology_generation"] == provenance["topology_generation"]
    )


def _endpoint_origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint URL must use http or https")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("endpoint URL contains an invalid port") from exc
    return parsed.scheme.lower(), parsed.hostname.lower(), port


def _validate_runtime_identity_url(endpoint_url: str, identity_url: str) -> None:
    if not isinstance(identity_url, str) or not identity_url:
        raise ValueError("runtime identity URL must be non-empty")
    if urlsplit(identity_url).path.rstrip("/") != RUNTIME_IDENTITY_PATH:
        raise ValueError(
            "runtime identity URL must use "
            f"{RUNTIME_IDENTITY_PATH}"
        )
    if _endpoint_origin(endpoint_url) != _endpoint_origin(identity_url):
        raise ValueError(
            "runtime identity URL must have the same origin as the benchmark endpoint"
        )


async def _fetch_runtime_identity(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    """Fetch and validate the identity from the service being measured."""
    try:
        response = await client.get(url, headers=headers, timeout=timeout_s)
        response.raise_for_status()
        value = response.json()
    except httpx.HTTPError as exc:
        raise ValueError(f"runtime identity request failed: {exc}") from exc
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("runtime identity response is not valid JSON") from exc
    return _validate_runtime_identity(value)


def _invocation_binding(
    args: argparse.Namespace, prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "endpoint": args.url,
        "model": args.model,
        "cell": args.cell,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "expected_input_tokens": args.expected_input_tokens,
        "expected_output_tokens": args.expected_output_tokens,
        "prompt_sha256": _prompt_digest(prompts),
    }


def build_request(
    *,
    request_id: str,
    model: str,
    prompt: dict[str, Any],
    expected_input_tokens: int,
    expected_output_tokens: int,
) -> dict[str, Any]:
    """Build the exact performance envelope accepted by the Gateway."""
    if type(expected_input_tokens) is not int or not 1 <= expected_input_tokens <= 4096:
        raise ValueError("expected_input_tokens must be an integer in 1..4096")
    if type(expected_output_tokens) is not int or not 2 <= expected_output_tokens <= 256:
        raise ValueError("expected_output_tokens must be an integer in 2..256")
    return {
        "request_id": request_id,
        "model": model,
        "messages": [{"role": "user", "content": prompt["content"]}],
        "stream": True,
        "temperature": 0,
        "ignore_eos": True,
        "max_tokens": expected_output_tokens,
        "prism_performance": {
            "schema_version": PERFORMANCE_SCHEMA,
            "expected_input_tokens": expected_input_tokens,
            "expected_output_tokens": expected_output_tokens,
        },
    }


def _parse_sse_line(line: str) -> tuple[dict[str, Any] | None, bool]:
    if not line or line.startswith(":"):
        return None, False
    if not line.startswith("data: "):
        raise ValueError("unexpected SSE line")
    payload = line[6:]
    if payload == "[DONE]":
        return None, True
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("SSE data payload is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("SSE data payload must be an object")
    return value, False


def _validate_stream_event(
    event: dict[str, Any],
    *,
    request_id: str,
    model: str,
    terminal_seen: bool,
) -> tuple[str, int | None]:
    """Validate one Gateway chunk and classify it as a token or terminal frame."""
    if event.get("id") != request_id:
        raise ValueError("SSE chunk id does not match request id")
    if event.get("model") != model:
        raise ValueError("SSE chunk model does not match request model")
    if event.get("object") != "chat.completion.chunk":
        raise ValueError("SSE chunk object is invalid")
    choices = event.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("SSE chunk must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("index") != 0:
        raise ValueError("SSE chunk choice is invalid")
    if "finish_reason" not in choice:
        raise ValueError("SSE chunk is missing finish_reason")
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        raise ValueError("SSE chunk delta must be an object")

    finish_reason = choice["finish_reason"]
    if finish_reason is None:
        if terminal_seen:
            raise ValueError("SSE token arrived after a terminal frame")
        if set(delta) != {"token_id"}:
            raise ValueError("SSE token delta must contain only token_id")
        token_id = delta["token_id"]
        if type(token_id) is not int or not 0 <= token_id < 2**64:
            raise ValueError("SSE token_id must be an unsigned 64-bit integer")
        return "token", token_id

    if finish_reason not in {"stop", "error"}:
        raise ValueError("SSE chunk finish_reason is invalid")
    if terminal_seen:
        raise ValueError("SSE stream contains multiple terminal frames")
    if delta:
        raise ValueError("SSE terminal delta must be empty")
    return finish_reason, None


async def _request(
    client: httpx.AsyncClient,
    *,
    url: str,
    body: dict[str, Any],
    request_index: int,
    timeout_s: float,
    semaphore: asyncio.Semaphore,
    headers: dict[str, str],
) -> dict[str, Any]:
    async with semaphore:
        started = time.perf_counter_ns()
        first_token_ns: int | None = None
        token_count = 0
        done = False
        terminal_seen = False
        terminal_stop_seen = False
        error: str | None = None
        status_code: int | None = None
        expected = body["prism_performance"]["expected_output_tokens"]
        try:
            async with client.stream(
                "POST", url, json=body, headers=headers, timeout=timeout_s
            ) as response:
                status_code = response.status_code
                if response.status_code != 200:
                    error = (await response.aread()).decode("utf-8", "replace")[:512]
                elif response.headers.get("content-type", "").split(
                    ";", 1
                )[0].strip().lower() != "text/event-stream":
                    error = "SSE response content type must be text/event-stream"
                else:
                    async for line in response.aiter_lines():
                        event, is_done = _parse_sse_line(line)
                        if is_done:
                            if not terminal_seen:
                                raise ValueError(
                                    "SSE stream ended before a terminal frame"
                                )
                            done = True
                            break
                        if event is None:
                            continue
                        kind, _token_id = _validate_stream_event(
                            event,
                            request_id=body["request_id"],
                            model=body["model"],
                            terminal_seen=terminal_seen,
                        )
                        if kind == "token":
                            token_count += 1
                            if token_count > expected:
                                raise ValueError(
                                    "SSE stream emitted more tokens than expected"
                                )
                            if first_token_ns is None:
                                first_token_ns = time.perf_counter_ns()
                        else:
                            terminal_seen = True
                            if kind == "stop":
                                if token_count != expected:
                                    raise ValueError(
                                        "SSE stop terminal token count does not match "
                                        "expected output tokens"
                                    )
                                terminal_stop_seen = True
                            else:
                                error = "SSE stream terminated with error"
                    if error is None and not done:
                        error = "SSE stream ended without [DONE]"
        except Exception as exc:  # benchmark rows must remain materialized
            error = f"{type(exc).__name__}: {exc}"
        finished = time.perf_counter_ns()
        passed = (
            error is None
            and done
            and terminal_stop_seen
            and token_count == expected
        )
        return {
            "request_index": request_index,
            "request_id": body["request_id"],
            "status_code": status_code,
            "status": "PASS" if passed else "FAIL",
            "error": error,
            "terminal_observed": done,
            "output_tokens": token_count,
            "ttft_ms": (
                (first_token_ns - started) / 1_000_000 if first_token_ns else None
            ),
            "tpot_ms": (
                (finished - first_token_ns) / 1_000_000 / (token_count - 1)
                if first_token_ns and token_count > 1
                else None
            ),
            "e2e_ms": (finished - started) / 1_000_000,
        }


async def collect(args: argparse.Namespace) -> dict[str, Any]:
    global_expected_input_tokens = getattr(args, "expected_input_tokens", None)
    prompts = _load_prompts(
        getattr(args, "prompt_file", None),
        getattr(args, "prompt", None),
        global_expected_input_tokens,
    )
    if type(args.requests) is not int or args.requests <= 0 \
            or type(args.concurrency) is not int or args.concurrency <= 0:
        raise ValueError("requests and concurrency must be positive")
    if type(args.expected_output_tokens) is not int \
            or not 2 <= args.expected_output_tokens <= 256:
        raise ValueError("expected_output_tokens must be an integer in 2..256")
    if not isinstance(args.timeout_s, (int, float)) \
            or isinstance(args.timeout_s, bool) \
            or not math.isfinite(float(args.timeout_s)) \
            or args.timeout_s <= 0:
        raise ValueError("timeout_s must be a finite positive number")
    expected_binding = _invocation_binding(args, prompts)
    provenance = _load_provenance(
        getattr(args, "provenance", None), expected_binding=expected_binding
    )
    runtime_identity_url = getattr(args, "runtime_identity_url", None)
    if provenance is not None and not runtime_identity_url:
        raise ValueError(
            "--runtime-identity-url is required when --provenance is supplied"
        )
    if runtime_identity_url and provenance is None:
        raise ValueError("--runtime-identity-url requires --provenance")
    rows: list[dict[str, Any]] = []
    semaphore = asyncio.Semaphore(args.concurrency)
    headers = {AUTH_HEADER: args.operator_token} if args.operator_token else {}
    started = time.perf_counter_ns()
    runtime_identity: dict[str, Any] | None = None
    async with httpx.AsyncClient() as client:
        if runtime_identity_url:
            _validate_runtime_identity_url(args.url, runtime_identity_url)
            runtime_identity = await _fetch_runtime_identity(
                client,
                url=runtime_identity_url,
                headers=headers,
                # Identity is a local control-plane read; it must not inherit a
                # multi-minute stream deadline from the benchmark itself.
                timeout_s=min(float(args.timeout_s), 30.0),
            )
            if provenance is None or not _runtime_identity_matches(
                provenance, runtime_identity
            ):
                raise ValueError(
                    "runtime identity does not match the supplied provenance"
                )
        tasks = []
        for index in range(args.requests):
            prompt = prompts[index % len(prompts)]
            expected_input = prompt["expected_input_tokens"]
            if expected_input is None:
                expected_input = args.expected_input_tokens
            body = build_request(
                request_id=f"{args.cell}.r{index:06d}",
                model=args.model,
                prompt=prompt,
                expected_input_tokens=expected_input,
                expected_output_tokens=args.expected_output_tokens,
            )
            tasks.append(_request(
                client,
                url=args.url,
                body=body,
                request_index=index,
                timeout_s=args.timeout_s,
                semaphore=semaphore,
                headers=headers,
            ))
        rows = await asyncio.gather(*tasks)
    elapsed_s = (time.perf_counter_ns() - started) / 1_000_000_000
    passed = [row for row in rows if row["status"] == "PASS"]
    metrics = {}
    for name in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [row[name] for row in passed if row[name] is not None]
        metrics[name] = {
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
        }
    return {
        "schema_version": "prism.public_endpoint_benchmark/v1",
        "evidence_level": (
            "paired-provenance"
            if provenance is not None and runtime_identity is not None
            else "timing-only"
        ),
        "claims_not_made": ([] if provenance is not None and runtime_identity is not None else [
            "source/image reproducibility",
            "runtime image/model identity attestation",
            "paired GPU comparison",
            "production SLO",
        ]),
        "endpoint": args.url,
        "model": args.model,
        "cell": args.cell,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "expected_input_tokens": args.expected_input_tokens,
        "expected_output_tokens": args.expected_output_tokens,
        "provenance": provenance,
        "runtime_identity": runtime_identity,
        "successful_requests": len(passed),
        "elapsed_s": elapsed_s,
        "successful_requests_per_s": len(passed) / elapsed_s if elapsed_s > 0 else None,
        "metrics": metrics,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Gateway /v1/chat/completions URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cell", default="public")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--expected-input-tokens", type=int)
    parser.add_argument("--expected-output-tokens", type=int, default=32)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--operator-token", default=os.environ.get("PRISM_OPERATOR_TOKEN", ""),
        help=f"operator token for the performance harness ({AUTH_HEADER})",
    )
    parser.add_argument(
        "--provenance", type=Path,
        help=(
            "JSON provenance manifest; requires --runtime-identity-url for a paired result"
        ),
    )
    parser.add_argument(
        "--runtime-identity-url",
        help=(
            "authenticated runtime identity URL on the same Gateway origin; "
            "required with --provenance"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        _load_prompts(args.prompt_file, args.prompt, args.expected_input_tokens)
        _load_provenance(args.provenance)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        result = asyncio.run(collect(args))
    except ValueError as exc:
        parser.error(str(exc))
    payload = json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

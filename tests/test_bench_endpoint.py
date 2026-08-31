"""Contract tests for the public endpoint benchmark client."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import bench.bench_endpoint as benchmark
from bench.bench_endpoint import (
    _fetch_runtime_identity,
    _load_prompts,
    _load_provenance,
    _prompt_digest,
    _request,
    _runtime_identity_matches,
    collect,
)


def test_prompt_loader_requires_an_exact_token_count():
    with pytest.raises(ValueError, match="expected-input-tokens"):
        _load_prompts(None, "benchmark prompt")


def test_prompt_file_can_supply_per_prompt_token_counts(tmp_path: Path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps({"content": "hello", "expected_input_tokens": 1}) + "\n",
        encoding="utf-8",
    )
    assert _load_prompts(path, None) == [
        {"content": "hello", "expected_input_tokens": 1}
    ]


def test_prompt_file_requires_counts_when_no_global_count_is_given(tmp_path: Path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"content": "hello"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected_input_tokens"):
        _load_prompts(path, None)


def test_provenance_loader_requires_immutable_inputs(tmp_path: Path):
    path = tmp_path / "provenance.json"
    value = {
        "schema_version": "prism.public_endpoint_provenance/v1",
        "gateway": {
            "source_commit": "a" * 40,
            "image": "registry.example/prism-serve@sha256:" + "b" * 64,
            "source_url": "https://github.com/SparkSnail/prism-serve",
        },
        "worker": {
            "source_commit": "c" * 40,
            "image": "registry.example/prism-infer@sha256:" + "d" * 64,
            "source_url": "https://github.com/SparkSnail/prism-infer",
        },
        "model": {"id": "Qwen/Qwen3-8B", "revision": "e" * 40},
        "topology_generation": "generation-1",
        "binding": {
            "endpoint": "http://gateway/v1/chat/completions",
            "model": "Qwen/Qwen3-8B",
            "cell": "public",
            "requests": 1,
            "concurrency": 1,
            "expected_input_tokens": 1,
            "expected_output_tokens": 2,
            "prompt_sha256": "f" * 64,
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    assert _load_provenance(path) == value
    value["gateway"]["source_commit"] = "working-tree"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="source_commit"):
        _load_provenance(path)


def test_provenance_binding_must_match_the_invocation(tmp_path: Path):
    path = tmp_path / "provenance.json"
    prompts = [{"content": "hello", "expected_input_tokens": 1}]
    args = SimpleNamespace(
        url="http://gateway/v1/chat/completions",
        model="Qwen/Qwen3-8B",
        cell="public",
        requests=1,
        concurrency=1,
        expected_input_tokens=None,
        expected_output_tokens=2,
    )
    value = {
        "schema_version": "prism.public_endpoint_provenance/v1",
        "gateway": {
            "source_commit": "a" * 40,
            "image": "registry.example/prism-serve@sha256:" + "b" * 64,
            "source_url": "https://github.com/SparkSnail/prism-serve",
        },
        "worker": {
            "source_commit": "c" * 40,
            "image": "registry.example/prism-infer@sha256:" + "d" * 64,
            "source_url": "https://github.com/SparkSnail/prism-infer",
        },
        "model": {"id": args.model, "revision": "e" * 40},
        "topology_generation": "generation-1",
        "binding": {
            **benchmark._invocation_binding(args, prompts),
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    assert _load_provenance(
        path, expected_binding=benchmark._invocation_binding(args, prompts)
    ) == value
    value["binding"]["concurrency"] = 2
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match invocation"):
        _load_provenance(
            path, expected_binding=benchmark._invocation_binding(args, prompts)
        )


def _runtime_identity() -> dict[str, object]:
    return {
        "schema_version": "prism.public_endpoint_runtime_identity/v1",
        "endpoint_path": "/v1/chat/completions",
        "gateway": {
            "source_commit": "a" * 40,
            "image": "registry.example/prism-serve@sha256:" + "b" * 64,
            "source_url": "https://github.com/SparkSnail/prism-serve",
        },
        "worker": {
            "source_commit": "c" * 40,
            "image": "registry.example/prism-infer@sha256:" + "d" * 64,
            "source_url": "https://github.com/SparkSnail/prism-infer",
        },
        "model": {"id": "Qwen/Qwen3-8B", "revision": "e" * 40},
        "topology_generation": "generation-1",
    }


@pytest.mark.asyncio
async def test_runtime_identity_fetch_requires_a_valid_response():
    value = _runtime_identity()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=value, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await _fetch_runtime_identity(
            client,
            url="http://gateway/internal/week12/performance/runtime-identity",
            headers={},
            timeout_s=5,
        ) == value


def test_runtime_identity_matches_all_immutable_fields():
    provenance = {
        **_runtime_identity(),
        "binding": {},
    }
    runtime = _runtime_identity()
    assert _runtime_identity_matches(provenance, runtime)
    runtime["worker"]["image"] = "registry.example/prism-infer@sha256:" + "0" * 64
    assert not _runtime_identity_matches(provenance, runtime)


def _paired_args(provenance: Path) -> SimpleNamespace:
    return SimpleNamespace(
        url="http://gateway/v1/chat/completions",
        model="Qwen/Qwen3-8B",
        cell="public",
        requests=1,
        concurrency=1,
        prompt_file=None,
        prompt="hello",
        expected_input_tokens=1,
        expected_output_tokens=2,
        timeout_s=5.0,
        operator_token="",
        provenance=provenance,
        runtime_identity_url=(
            "http://gateway/internal/week12/performance/runtime-identity"
        ),
    )


@pytest.mark.asyncio
async def test_collect_marks_paired_only_after_runtime_identity_match(tmp_path: Path,
                                                                       monkeypatch):
    args = _paired_args(tmp_path / "provenance.json")
    prompts = [{"content": "hello", "expected_input_tokens": 1}]
    identity = _runtime_identity()
    value = {
        "schema_version": "prism.public_endpoint_provenance/v1",
        "gateway": identity["gateway"],
        "worker": identity["worker"],
        "model": identity["model"],
        "topology_generation": identity["topology_generation"],
        "binding": {
            "endpoint": args.url,
            "model": args.model,
            "cell": args.cell,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "expected_input_tokens": args.expected_input_tokens,
            "expected_output_tokens": args.expected_output_tokens,
            "prompt_sha256": _prompt_digest(prompts),
        },
    }
    # The provenance and runtime documents use different schema-specific keys
    # but otherwise carry the same immutable identity.
    args.provenance.write_text(json.dumps(value), encoding="utf-8")
    posts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.method == "GET":
            return httpx.Response(200, json=identity, request=request)
        posts += 1
        body = json.loads(request.content)
        events = [
            _stream_chunk(
                request_id=body["request_id"],
                model=body["model"],
                delta={"token_id": 7},
                finish_reason=None,
            ),
            _stream_chunk(
                request_id=body["request_id"],
                model=body["model"],
                delta={"token_id": 8},
                finish_reason=None,
            ),
            _stream_chunk(
                request_id=body["request_id"],
                model=body["model"],
                delta={},
                finish_reason="stop",
            ),
            "data: [DONE]",
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=("\n\n".join(events) + "\n\n").encode(),
            request=request,
        )

    real_async_client = httpx.AsyncClient

    def client_factory(*_args, **_kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(benchmark.httpx, "AsyncClient", client_factory)
    result = await collect(args)
    assert result["evidence_level"] == "paired-provenance"
    assert result["runtime_identity"] == identity
    assert posts == 1


def _stream_chunk(
    *,
    request_id: str = "public.r1",
    model: str = "test-model",
    delta: dict[str, object],
    finish_reason: str | None,
) -> str:
    return "data: " + json.dumps({
        "id": request_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }, separators=(",", ":"))


async def _request_row(events: list[str]) -> dict[str, object]:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["prism_performance"]["expected_output_tokens"] == 2
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream; charset=utf-8"},
            content=("\n\n".join(events) + "\n\n").encode(),
            request=request,
        )

    body = {
        "request_id": "public.r1",
        "model": "test-model",
        "prism_performance": {"expected_output_tokens": 2},
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await _request(
            client,
            url="http://gateway/v1/chat/completions",
            body=body,
            request_index=0,
            timeout_s=5,
            semaphore=asyncio.Semaphore(1),
            headers={},
        )


@pytest.mark.asyncio
async def test_request_marks_complete_token_stream_as_pass():
    row = await _request_row([
        _stream_chunk(delta={"token_id": 7}, finish_reason=None),
        _stream_chunk(delta={"token_id": 8}, finish_reason=None),
        _stream_chunk(delta={}, finish_reason="stop"),
        "data: [DONE]",
    ])
    assert row["status"] == "PASS"
    assert row["output_tokens"] == 2
    assert row["terminal_observed"] is True


@pytest.mark.asyncio
async def test_request_rejects_error_terminal_after_exact_token_count():
    row = await _request_row([
        _stream_chunk(delta={"token_id": 7}, finish_reason=None),
        _stream_chunk(delta={"token_id": 8}, finish_reason=None),
        _stream_chunk(delta={}, finish_reason="error"),
        "data: [DONE]",
    ])
    assert row["status"] == "FAIL"
    assert row["output_tokens"] == 2
    assert row["terminal_observed"] is True
    assert "terminated with error" in str(row["error"])


@pytest.mark.asyncio
@pytest.mark.parametrize("token_id", [None, True, "not-a-token"])
async def test_request_rejects_malformed_token_id(token_id: object):
    row = await _request_row([
        _stream_chunk(delta={"token_id": token_id}, finish_reason=None),
        "data: [DONE]",
    ])
    assert row["status"] == "FAIL"
    assert row["output_tokens"] == 0
    assert "token_id must be an unsigned 64-bit integer" in str(row["error"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_id", "model", "expected_error"),
    [
        ("different-request", "test-model", "id does not match request id"),
        ("public.r1", "different-model", "model does not match request model"),
    ],
)
async def test_request_rejects_chunk_for_a_different_request_or_model(
    request_id: str,
    model: str,
    expected_error: str,
):
    row = await _request_row([
        _stream_chunk(
            request_id=request_id,
            model=model,
            delta={"token_id": 7},
            finish_reason=None,
        ),
        "data: [DONE]",
    ])
    assert row["status"] == "FAIL"
    assert expected_error in str(row["error"])


@pytest.mark.asyncio
async def test_request_rejects_done_without_a_stop_terminal():
    row = await _request_row(["data: [DONE]"])
    assert row["status"] == "FAIL"
    assert row["terminal_observed"] is False
    assert "ended before a terminal frame" in str(row["error"])

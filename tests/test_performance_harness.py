"""Performance admission, evidence terminalization, and tombstone tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import prism_serve.gateway.app as gateway_module
from prism_serve.gateway.app import app
from prism_serve.gateway.correctness_harness import AUTH_HEADER
from prism_serve.gateway.performance_harness import (
    PERFORMANCE_EVIDENCE_SCHEMA,
    PERFORMANCE_REQUEST_SCHEMA,
    PERFORMANCE_TRACE_CAP,
    PerformanceRequest,
    PerformanceTraceCapacity,
    PerformanceTraceConflict,
    PerformanceTraceRegistry,
    parse_performance_request,
)


SECRET = "performance-operator-secret-value-0001"


class Encoder:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) for char in text]


def request_body(**updates) -> dict[str, object]:
    body: dict[str, object] = {
        "request_id": "perf.r1",
        "model": "Qwen/Qwen3-8B",
        "messages": [{"role": "user", "content": "abc"}],
        "stream": True,
        "temperature": 0,
        "ignore_eos": True,
        "max_tokens": 2,
        "prism_performance": {
            "schema_version": PERFORMANCE_REQUEST_SCHEMA,
            "expected_input_tokens": 3,
            "expected_output_tokens": 2,
        },
    }
    body.update(updates)
    return body


def parsed_request(request_id: str = "perf.r1") -> PerformanceRequest:
    body = request_body(request_id=request_id)
    return parse_performance_request(
        body,
        encoder=Encoder(),
        runtime_model="Qwen/Qwen3-8B",
        model_profile={"profile_id": "qwen3-8b-bf16-tp1"},
    )


def world_identity(*, affinity_enabled: bool = True) -> dict[str, object]:
    return {
        "gateway": {"pod_uid": "gateway-pod", "clock_epoch": "gateway-e1"},
        "topology_generation": "topology-a",
        "affinity_enabled": affinity_enabled,
        "workers": {
            instance_id: {
                "pod_uid": f"pod-{instance_id}",
                "instance_epoch": f"pod-{instance_id}:process-{instance_id}",
                "global_rank": rank,
            }
            for instance_id, rank in (("p0", 0), ("p1", 1), ("d0", 2), ("d1", 3))
        },
    }


def local_route() -> dict[str, object]:
    return {
        "path": "same_instance",
        "route": {"source": "d0", "target": "d0"},
        "src_block_ids": [],
        "dst_block_ids": [],
        "cached_prefix_tokens": 256,
        "suffix_tokens": 3,
        "mapping": [],
        "transport": {
            "selected_mode": "NO_TRANSFER",
            "completed_bytes": 0,
            "work_terminal": True,
            "cuda_terminal": True,
            "gateway_clock_epoch": "gateway-e1",
            "transfer_started_ns": None,
            "transfer_terminal_ns": None,
        },
    }


def test_parser_freezes_raw_tokenization_and_exact_sampling() -> None:
    parsed = parsed_request()

    assert parsed.request_id == "perf.r1"
    assert parsed.input_token_ids == (97, 98, 99)
    assert parsed.expected_output_tokens == 2
    assert parsed.raw_content_sha256.startswith("sha256:")
    assert parsed.input_token_ids_sha256.startswith("sha256:")


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"request_id": "bad id"}, "request_id"),
        ({"model": "Qwen/Qwen3-0.6B"}, "model"),
        ({"stream": False}, "stream=true"),
        ({"temperature": 0.1}, "temperature=0"),
        ({"ignore_eos": False}, "ignore_eos=true"),
        ({"max_tokens": 3}, "max_tokens"),
        ({"input_token_ids": [1, 2, 3]}, "raw text"),
        ({"messages": [{"role": "system", "content": "abc"}]}, "raw user"),
    ],
)
def test_parser_rejects_contract_drift(update, message) -> None:
    with pytest.raises(ValueError, match=message):
        parse_performance_request(
            request_body(**update),
            encoder=Encoder(),
            runtime_model="Qwen/Qwen3-8B",
            model_profile={"profile_id": "qwen3-8b-bf16-tp1"},
        )


def test_trace_registry_retains_terminal_and_idempotent_tombstone() -> None:
    registry = PerformanceTraceRegistry()
    registry.reserve(parsed_request(), world_identity=world_identity())
    registry.observe_stream_terminal("perf.r1")

    evidence = registry.finalize(
        "perf.r1",
        output_token_ids=[7, 8],
        runtime_error=None,
        route_evidence=local_route(),
    )

    assert evidence["schema_version"] == PERFORMANCE_EVIDENCE_SCHEMA
    assert evidence["status"] == "PASS"
    assert evidence["world_identity"] == world_identity()
    assert evidence["request"]["actual_output_tokens"] == 2
    assert evidence["resources"]["terminal"] is True
    assert registry.acknowledge("perf.r1") == "acked"
    assert registry.acknowledge("perf.r1") == "acked"
    assert registry.state("perf.r1") == "tombstone"
    assert registry.counts() == {
        "active": 0,
        "unacked_terminal": 0,
        "acked_tombstones": 1,
        "total": 1,
        "capacity": PERFORMANCE_TRACE_CAP,
    }


def test_trace_registry_records_short_output_disconnect_and_capacity() -> None:
    registry = PerformanceTraceRegistry()
    registry.reserve(parsed_request("short"), world_identity=world_identity())
    short = registry.finalize(
        "short", output_token_ids=[7], runtime_error=None, route_evidence=local_route()
    )
    assert short["status"] == "FAILED"
    assert short["error_code"] == "OUTPUT_TOKEN_COUNT_MISMATCH"

    registry.reserve(parsed_request("detached"), world_identity=world_identity())
    registry.mark_detached("detached")
    detached = registry.finalize(
        "detached", output_token_ids=[7, 8], runtime_error=None,
        route_evidence=local_route(),
    )
    assert detached["status"] == "CANCELLED"
    assert detached["error_code"] == "CLIENT_DISCONNECTED"

    bounded = PerformanceTraceRegistry()
    bounded._tombstones.update((f"old-{index}", None) for index in range(PERFORMANCE_TRACE_CAP))
    with pytest.raises(PerformanceTraceCapacity):
        bounded.reserve(parsed_request("overflow"), world_identity=world_identity())
    with pytest.raises(PerformanceTraceConflict):
        registry.reserve(parsed_request("short"), world_identity=world_identity())


def test_trace_registry_rejects_complete_output_without_stream_terminal() -> None:
    registry = PerformanceTraceRegistry()
    registry.reserve(parsed_request("missing-terminal"), world_identity=world_identity())

    evidence = registry.finalize(
        "missing-terminal",
        output_token_ids=[7, 8],
        runtime_error=None,
        route_evidence=local_route(),
    )

    assert evidence["status"] == "FAILED"
    assert evidence["error_code"] == "STREAM_TERMINAL_NOT_OBSERVED"
    assert evidence["stream"]["terminal_observed"] is False


def test_performance_mode_retains_backing_snapshots_for_the_trace_cap(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gateway_module.settings, "terminal_snapshot_cap", 4096)
    monkeypatch.setattr(gateway_module.settings, "performance_trace_cap", 8192)
    monkeypatch.setattr(gateway_module.settings, "performance_harness_enabled", True)

    assert gateway_module._build_config()["terminal_snapshot_cap"] == 8192


def test_performance_off_runtime_keeps_pinned_tokenizer_without_affinity() -> None:
    config = gateway_module._build_config()
    config.update({
        "correctness_harness_enabled": False,
        "performance_harness_enabled": True,
        "route_parity_harness_enabled": True,
        "correctness_harness_secret": SECRET,
        "multinode_e2e_enabled": True,
        "affinity_enabled": False,
    })
    assert gateway_module._validate_operator_harness_config(config) is True

    tokenizer = SimpleNamespace(
        encoder=Encoder(), fingerprint_request=lambda _text: object()
    )
    dummy = SimpleNamespace(state=SimpleNamespace(tokenizer_adapter=tokenizer))
    assert gateway_module._install_pinned_tokenizer(dummy, config) is tokenizer

    config["performance_harness_enabled"] = False
    config["route_parity_harness_enabled"] = False
    config["correctness_harness_enabled"] = True
    with pytest.raises(RuntimeError, match="requires affinity"):
        gateway_module._validate_operator_harness_config(config)


def test_performance_world_identity_is_exact_and_fail_closed(monkeypatch) -> None:
    generation = "topology-a"
    members = {
        instance_id: SimpleNamespace(
            instance_id=instance_id,
            role=role,
            topology_generation=generation,
            pod_uid=f"pod-{instance_id}",
            instance_epoch=f"pod-{instance_id}:process-{instance_id}",
            global_rank=rank,
        )
        for instance_id, role, rank in (
            ("p0", "prefill", 0), ("p1", "prefill", 1),
            ("d0", "decode", 2), ("d1", "decode", 3),
        )
    }
    registry = SimpleNamespace(
        world_fresh=lambda: True,
        members=members,
        expected_topology_generation=generation,
    )
    monkeypatch.setattr(app.state, "worker_registry", registry, raising=False)
    monkeypatch.setattr(
        app.state,
        "network_control",
        SimpleNamespace(gateway_clock_epoch="gateway-e1"),
        raising=False,
    )
    monkeypatch.setattr(
        app.state,
        "runtime_config",
        {
            "scheduler_id": "gateway-pod",
            "scheduler_generation": "gateway-e1",
            "affinity_enabled": False,
        },
        raising=False,
    )

    assert gateway_module._performance_world_identity(app) == world_identity(
        affinity_enabled=False
    )
    members["d1"] = SimpleNamespace(**{
        **vars(members["d1"]), "global_rank": 2,
    })
    with pytest.raises(RuntimeError, match="drifted"):
        gateway_module._performance_world_identity(app)


def test_performance_trace_http_get_delete_semantics(monkeypatch) -> None:
    registry = PerformanceTraceRegistry()
    registry.reserve(parsed_request(), world_identity=world_identity())
    registry.observe_stream_terminal("perf.r1")
    evidence = {
        "request_id": "perf.r1",
        "operation_id": "op-1",
        "path": "same_instance",
        "route": {"source": "d0", "target": "d0"},
        "src_block_ids": [],
        "dst_block_ids": [],
        "cached_prefix_tokens": 256,
        "suffix_tokens": 3,
        "completed_bytes": 0,
        "work_terminal": True,
        "cuda_terminal": True,
        "gateway_clock_epoch": "gateway-e1",
        "transfer_started_ns": None,
        "transfer_terminal_ns": None,
    }
    runtime = {
        "performance_harness_enabled": True,
        "correctness_harness_secret": SECRET,
    }
    monkeypatch.setattr(app.state, "runtime_config", runtime, raising=False)
    monkeypatch.setattr(app.state, "performance_trace_registry", registry, raising=False)
    monkeypatch.setattr(
        app.state, "tracker", SimpleNamespace(get=lambda _req_id: None), raising=False
    )
    monkeypatch.setattr(
        app.state,
        "output_buffer",
        SimpleNamespace(snapshot=lambda _req_id: ([7, 8], True, None)),
        raising=False,
    )
    monkeypatch.setattr(
        app.state,
        "network_control",
        SimpleNamespace(request_evidence=lambda _req_id: evidence),
        raising=False,
    )
    monkeypatch.setattr(
        app.state,
        "worker_registry",
        SimpleNamespace(capabilities={}, members={}),
        raising=False,
    )
    headers = {AUTH_HEADER: SECRET}
    client = TestClient(app, raise_server_exceptions=True)

    assert client.get(
        "/internal/week12/performance/requests/perf.r1", headers=headers
    ).status_code == 200
    assert client.delete(
        "/internal/week12/performance/requests/perf.r1", headers=headers
    ).status_code == 204
    assert client.delete(
        "/internal/week12/performance/requests/perf.r1", headers=headers
    ).status_code == 204
    assert client.get(
        "/internal/week12/performance/requests/perf.r1", headers=headers
    ).status_code == 410


def test_performance_trace_http_auth_and_active_ack(monkeypatch) -> None:
    registry = PerformanceTraceRegistry()
    registry.reserve(parsed_request(), world_identity=world_identity())
    monkeypatch.setattr(
        app.state,
        "runtime_config",
        {"performance_harness_enabled": True, "correctness_harness_secret": SECRET},
        raising=False,
    )
    monkeypatch.setattr(app.state, "performance_trace_registry", registry, raising=False)
    monkeypatch.setattr(
        app.state, "tracker", SimpleNamespace(get=lambda _req_id: object()), raising=False
    )
    monkeypatch.setattr(
        app.state,
        "output_buffer",
        SimpleNamespace(snapshot=lambda _req_id: ([], False, None)),
        raising=False,
    )
    client = TestClient(app, raise_server_exceptions=True)

    assert client.get(
        "/internal/week12/performance/requests/perf.r1"
    ).status_code == 403
    headers = {AUTH_HEADER: SECRET}
    assert client.get(
        "/internal/week12/performance/requests/perf.r1", headers=headers
    ).status_code == 202
    assert client.delete(
        "/internal/week12/performance/requests/perf.r1", headers=headers
    ).status_code == 409


def test_performance_resources_are_read_only_and_fault_isolated(monkeypatch) -> None:
    registry = PerformanceTraceRegistry()
    registry.reserve(parsed_request("active"), world_identity=world_identity())
    registry.reserve(parsed_request("terminal"), world_identity=world_identity())
    registry.finalize(
        "terminal", output_token_ids=[7, 8], runtime_error=None,
        route_evidence=local_route(),
    )
    registry.reserve(parsed_request("acked"), world_identity=world_identity())
    registry.finalize(
        "acked", output_token_ids=[7, 8], runtime_error=None,
        route_evidence=local_route(),
    )
    assert registry.acknowledge("acked") == "acked"

    async def resource_snapshot(_app):
        return {
            "resources": {
                "slots": 0,
                "source_retain": 0,
                "source_pins": 0,
                "target_pending": 0,
                "sequence_blocks": 0,
                "pair_bytes": 0,
                "quarantine_operation_ids": [],
            },
            "active_requests": 0,
            "worker_reports": {},
            "decode_slot_lease_counts": {},
            "pair_bytes_by_pair": {},
        }

    runtime = {
        "performance_harness_enabled": True,
        "correctness_harness_enabled": False,
        "correctness_harness_secret": SECRET,
        "expected_model_profile": {
            "profile_id": "qwen3-8b-bf16-tp1",
            "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "kv_compatibility_id": "2647222531d143800a56551dbe8d030c535a1bb2e0e47ff2a12f0781964f4c6c",
        },
    }
    monkeypatch.setattr(app.state, "runtime_config", runtime, raising=False)
    monkeypatch.setattr(app.state, "performance_trace_registry", registry, raising=False)
    monkeypatch.setattr(
        gateway_module, "_correctness_resource_snapshot", resource_snapshot
    )
    monkeypatch.setattr(
        gateway_module,
        "_performance_world_identity",
        lambda _app: world_identity(affinity_enabled=False),
    )
    client = TestClient(app, raise_server_exceptions=True)
    headers = {AUTH_HEADER: SECRET}

    assert client.get("/internal/week12/performance/resources").status_code == 403
    response = client.get(
        "/internal/week12/performance/resources", headers=headers
    )
    assert response.status_code == 200
    assert response.json()["resources"] == {
        "slots": 0,
        "source_retain": 0,
        "source_pins": 0,
        "target_pending": 0,
        "sequence_blocks": 0,
        "pair_bytes": 0,
        "quarantine_operation_ids": [],
    }
    assert response.json()["active_requests"] == 0
    assert response.json()["world_identity"] == world_identity(
        affinity_enabled=False
    )
    assert response.json()["model_profile"] == runtime["expected_model_profile"]
    assert response.json()["trace_counts"] == {
        "active": 1,
        "unacked_terminal": 1,
        "acked_tombstones": 1,
        "total": 3,
        "capacity": PERFORMANCE_TRACE_CAP,
    }
    assert client.get(
        "/internal/week12/correctness/resources", headers=headers
    ).status_code == 404
    assert client.post(
        "/internal/week12/correctness/faults",
        headers=headers,
        json={"fault_kind": "gateway_restart"},
    ).status_code == 404

    runtime["performance_harness_enabled"] = False
    assert client.get(
        "/internal/week12/performance/resources", headers=headers
    ).status_code == 404


def test_performance_chat_rolls_back_local_admission_failure(monkeypatch) -> None:
    from prism_serve.gateway.output import GatewayOutputBuffer
    from prism_serve.metrics.collector import NullMetrics
    from prism_serve.scheduler.sequence_state import RequestTracker

    class Control:
        def __init__(self) -> None:
            self.required: list[str] = []
            self.cancelled: list[str] = []

        def require_request_evidence(self, request_id: str) -> None:
            self.required.append(request_id)

        def cancel_request_evidence(self, request_id: str) -> None:
            self.cancelled.append(request_id)

    registry = PerformanceTraceRegistry()
    control = Control()
    output = GatewayOutputBuffer(active_operation_cap=1, terminal_snapshot_cap=1)
    output.ensure("held")
    tracker = RequestTracker(NullMetrics())
    tokenizer = SimpleNamespace(
        encoder=Encoder(),
        namespace="performance-test",
        identity=SimpleNamespace(
            kv_compatibility_id="kv-performance-test", block_size=256
        ),
    )
    runtime = {
        "model_id": "Qwen/Qwen3-8B",
        "performance_harness_enabled": True,
        "correctness_harness_secret": SECRET,
        "expected_model_profile": {"profile_id": "qwen3-8b-bf16-tp1"},
        "multinode_e2e_enabled": False,
    }
    monkeypatch.setattr(app.state, "accepting", True, raising=False)
    monkeypatch.setattr(
        app.state, "queue", SimpleNamespace(is_connected=True), raising=False
    )
    monkeypatch.setattr(app.state, "control_plane_failed", False, raising=False)
    monkeypatch.setattr(app.state, "worker_registry", None, raising=False)
    monkeypatch.setattr(app.state, "runtime_config", runtime, raising=False)
    monkeypatch.setattr(app.state, "tokenizer_adapter", tokenizer, raising=False)
    monkeypatch.setattr(app.state, "tracker", tracker, raising=False)
    monkeypatch.setattr(app.state, "output_buffer", output, raising=False)
    monkeypatch.setattr(app.state, "network_control", control, raising=False)
    monkeypatch.setattr(app.state, "performance_trace_registry", registry, raising=False)
    monkeypatch.setattr(
        gateway_module, "_performance_world_identity", lambda _app: world_identity()
    )
    client = TestClient(app, raise_server_exceptions=True)
    headers = {AUTH_HEADER: SECRET}

    mismatch = client.post(
        "/v1/chat/completions",
        json=request_body(model="Qwen/Qwen3-0.6B"),
        headers=headers,
    )
    rejected = client.post(
        "/v1/chat/completions", json=request_body(), headers=headers
    )

    assert mismatch.status_code == 422
    assert rejected.status_code == 503
    assert registry.state("perf.r1") == "unknown"
    assert tracker.get("perf.r1") is None
    assert control.required == ["perf.r1"]
    assert control.cancelled == ["perf.r1"]


def test_performance_off_exposes_only_cold_route_parity(monkeypatch) -> None:
    runtime = {
        "model_id": "Qwen/Qwen3-8B",
        "performance_harness_enabled": True,
        "route_parity_harness_enabled": True,
        "correctness_harness_enabled": False,
        "correctness_harness_secret": SECRET,
        "affinity_enabled": False,
        "multinode_e2e_enabled": False,
    }
    monkeypatch.setattr(app.state, "accepting", True, raising=False)
    monkeypatch.setattr(
        app.state, "queue", SimpleNamespace(is_connected=True), raising=False
    )
    monkeypatch.setattr(app.state, "control_plane_failed", False, raising=False)
    monkeypatch.setattr(app.state, "worker_registry", None, raising=False)
    monkeypatch.setattr(app.state, "runtime_config", runtime, raising=False)
    monkeypatch.setattr(app.state, "tokenizer_adapter", None, raising=False)
    monkeypatch.setattr(app.state, "tracker", None, raising=False)
    monkeypatch.setattr(app.state, "output_buffer", None, raising=False)
    client = TestClient(app, raise_server_exceptions=True)
    headers = {AUTH_HEADER: SECRET}

    def route_body(path: str) -> dict[str, object]:
        return {
            "request_id": f"parity-{path}",
            "model": "Qwen/Qwen3-8B",
            "input_token_ids": [1] * 769,
            "stream": False,
            "temperature": 0,
            "ignore_eos": True,
            "max_tokens": 32,
            "week12_correctness": {"path": path},
        }

    for path in ("same_instance", "cross_instance", "seed_d0", "seed_d1"):
        assert client.post(
            "/v1/chat/completions", json=route_body(path), headers=headers
        ).status_code == 404
    assert client.post(
        "/v1/chat/completions", json=route_body("cold"), headers=headers
    ).status_code == 422

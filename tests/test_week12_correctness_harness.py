from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from prism_serve.gateway.app import _correctness_auth_error
from prism_serve.gateway.correctness_harness import (
    AUTH_HEADER,
    FaultInjectionGate,
    parse_route,
    validate_fixture,
)
from prism_serve.router.fingerprint import PromptFingerprint
from prism_serve.router.http_rpc import EndpointSequenceAllocator
from prism_serve.router.network_rpc import NetworkControlRPC
from prism_serve.router.prefix_index import CacheLocation, PrefixIndex
from prism_serve.router.router import AffinityRouter
from prism_serve.router.topology import LinkInfo, TopologyMatrix


def _request(*, enabled: bool, configured: str, supplied: str = "") -> Request:
    headers = [] if not supplied else [(AUTH_HEADER.encode(), supplied.encode())]
    state = SimpleNamespace(runtime_config={
        "correctness_harness_enabled": enabled,
        "correctness_harness_secret": configured,
    })
    return Request({"type": "http", "app": SimpleNamespace(state=state), "headers": headers})


def test_correctness_harness_is_hidden_by_default_and_forbidden_without_auth() -> None:
    secret = "s" * 32
    disabled = _correctness_auth_error(
        _request(enabled=False, configured=secret, supplied=secret)
    )
    forbidden = _correctness_auth_error(
        _request(enabled=True, configured=secret, supplied="wrong")
    )

    assert disabled is not None and disabled.status_code == 404
    assert forbidden is not None and forbidden.status_code == 403
    assert secret.encode() not in disabled.body
    assert secret.encode() not in forbidden.body
    assert _correctness_auth_error(
        _request(enabled=True, configured=secret, supplied=secret)
    ) is None


def test_correctness_fixture_is_exact_and_seeds_leave_p0_cold() -> None:
    acceptance = parse_route({"path": "same_instance"})
    validate_fixture(
        route=acceptance,
        token_ids=list(range(769)),
        sampling={"temperature": 0.0, "max_tokens": 32, "ignore_eos": True},
    )
    with pytest.raises(ValueError, match="769 input"):
        validate_fixture(
            route=acceptance,
            token_ids=list(range(768)),
            sampling={"temperature": 0.0, "max_tokens": 32, "ignore_eos": True},
        )
    with pytest.raises(ValueError, match="ignore_eos=true"):
        validate_fixture(
            route=acceptance,
            token_ids=list(range(769)),
            sampling={"temperature": 0.0, "max_tokens": 32, "ignore_eos": False},
        )
    assert parse_route({"path": "seed_d1"}).source_instance == "p1"
    assert parse_route({"path": "seed_d0"}).source_instance == "p1"


@pytest.mark.asyncio
async def test_fault_gate_pauses_only_its_observed_checkpoint_and_keeps_refs() -> None:
    gate = FaultInjectionGate(timeout_s=1.0)
    armed = await gate.arm("nats_disconnect")
    await gate.arrive("before_nccl_source_start", {"ignored": True})

    reached = asyncio.create_task(gate.arrive(
        "before_nats_dispatch",
        {
            "source_endpoint_ref": {
                "operation_id": "request-1",
                "owner_generation": "gateway-owner-1",
            },
            "target_endpoint_ref": {"operation_id": "request-1"},
        },
    ))
    for _ in range(20):
        snapshot = await gate.snapshot()
        if snapshot is not None and snapshot["state"] == "REACHED":
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("fault checkpoint was not reached")

    assert snapshot["fault_run_id"] == armed["fault_run_id"]
    assert snapshot["details"]["source_endpoint_ref"]["operation_id"] == "request-1"
    gate.record_event("endpoint_query_observed", {"operation_id": "request-1"})
    snapshot = await gate.snapshot()
    assert snapshot["events"][0]["producer_epoch"] == "gateway-owner-1"
    await gate.release(str(armed["fault_run_id"]))
    await reached
    assert (await gate.snapshot())["state"] == "RELEASED"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault_kind",
    ["nats_drop", "nats_duplicate", "nats_publish_unknown"],
)
async def test_nats_command_fault_gate_returns_released_one_shot_directive(
    fault_kind: str,
) -> None:
    gate = FaultInjectionGate(timeout_s=1.0)
    armed = await gate.arm(fault_kind)
    waiter = asyncio.create_task(gate.arrive(
        "before_nats_dispatch",
        {
            "request_id": "request-1",
            "source_endpoint_ref": {
                "operation_id": "request-1",
                "owner_generation": "gateway-owner-1",
            },
        },
    ))
    for _ in range(20):
        snapshot = await gate.snapshot()
        if snapshot is not None and snapshot["state"] == "REACHED":
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("fault checkpoint was not reached")

    await gate.release(str(armed["fault_run_id"]))
    directive = await waiter

    assert directive is not None
    assert directive["fault_run_id"] == armed["fault_run_id"]
    assert directive["fault_kind"] == fault_kind
    assert directive["checkpoint"] == "before_nats_dispatch"
    assert directive["state"] == "RELEASED"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fault_kind", "path", "route_role"),
    [
        ("rpc_response_loss_source", "/v1/transfers/start", "source"),
        (
            "rpc_response_loss_target",
            "/v1/transfers/prepare-receive",
            "target",
        ),
        ("finalize_response_loss_source", "/v1/cleanup/finalize", "source"),
        ("finalize_response_loss_target", "/v1/cleanup/finalize", "target"),
    ],
)
async def test_post_success_fault_gate_matches_exact_mutation_and_role(
    fault_kind: str, path: str, route_role: str,
) -> None:
    gate = FaultInjectionGate(timeout_s=1.0)
    armed = await gate.arm(fault_kind)
    finalize_details = (
        {
            "cleanup_id": "cleanup-1",
            "cleanup_operation_id": "transfer-1",
            "cleanup_lease_id": "lease-1",
            "cleanup_payload_digest": "sha256:cleanup",
        }
        if fault_kind.startswith("finalize_response_loss_") else {}
    )
    endpoint_operation_id = (
        "transfer-1"
        if fault_kind.startswith("finalize_response_loss_")
        else "request-1"
    )
    ignored = await gate.arrive(
        "after_infer_success_before_control_observe",
        {
            "request_id": "request-1",
            "path": path,
            "route_role": "target" if route_role == "source" else "source",
            "endpoint_ref": {"operation_id": endpoint_operation_id},
            "response_status": 200,
            "request_digest": "sha256:request",
            "response_digest": "sha256:response",
            **finalize_details,
        },
    )
    assert ignored is None
    assert (await gate.snapshot())["state"] == "ARMED"

    waiter = asyncio.create_task(gate.arrive(
        "after_infer_success_before_control_observe",
        {
            "request_id": "request-1",
            "path": path,
            "route_role": route_role,
            "endpoint_ref": {
                "operation_id": endpoint_operation_id,
                "owner_generation": "gateway-owner-1",
            },
            "request_digest": "sha256:request",
            "response_digest": "sha256:response",
            "response_status": 200,
            **finalize_details,
        },
    ))
    for _ in range(20):
        snapshot = await gate.snapshot()
        if snapshot is not None and snapshot["state"] == "REACHED":
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("post-success checkpoint was not reached")
    await gate.release(str(armed["fault_run_id"]))
    directive = await waiter
    assert directive is not None and directive["state"] == "RELEASED"


@pytest.mark.asyncio
async def test_finalize_fault_gate_rejects_cleanup_operation_for_another_ref():
    gate = FaultInjectionGate(timeout_s=1.0)
    await gate.arm("finalize_response_loss_source")

    result = await gate.arrive(
        "after_infer_success_before_control_observe",
        {
            "request_id": "logical-request",
            "path": "/v1/cleanup/finalize",
            "route_role": "source",
            "endpoint_ref": {"operation_id": "transfer-1"},
            "response_status": 200,
            "request_digest": "sha256:request",
            "response_digest": "sha256:response",
            "cleanup_id": "cleanup-1",
            "cleanup_operation_id": "transfer-2",
            "cleanup_lease_id": "lease-1",
            "cleanup_payload_digest": "sha256:cleanup",
        },
    )

    assert result is None
    assert (await gate.snapshot())["state"] == "ARMED"


@pytest.mark.asyncio
async def test_fault_gate_rejects_overlap_and_release_before_reached() -> None:
    gate = FaultInjectionGate(timeout_s=1.0)
    armed = await gate.arm("worker_crash")
    with pytest.raises(RuntimeError, match="already active"):
        await gate.arm("nats_disconnect")
    with pytest.raises(RuntimeError, match="not reached"):
        await gate.release(str(armed["fault_run_id"]))


@pytest.mark.asyncio
async def test_fault_gate_slot_seal_releases_waiter_and_old_waiter_cannot_clobber_new_run():
    gate = FaultInjectionGate(timeout_s=1.0)
    first = await gate.arm("nats_disconnect")
    waiter = asyncio.create_task(gate.arrive(
        "before_nats_dispatch",
        {
            "request_id": "first",
            "source_endpoint_ref": {
                "operation_id": "first", "owner_generation": "gateway-1"
            },
        },
    ))
    for _ in range(20):
        snapshot = await gate.snapshot()
        if snapshot is not None and snapshot["state"] == "REACHED":
            break
        await asyncio.sleep(0)
    gate.record_event("slot_released", {"operation_id": "first"})
    await waiter
    assert (await gate.snapshot())["state"] == "SEALED"
    event_count = len((await gate.snapshot())["events"])
    gate.record_event("late_event", {"operation_id": "first"})
    assert len((await gate.snapshot())["events"]) == event_count

    second = await gate.arm("worker_crash")
    assert second["fault_run_id"] != first["fault_run_id"]
    assert (await gate.snapshot())["state"] == "ARMED"


@pytest.mark.asyncio
async def test_fault_gate_unrelated_operation_list_cannot_seal_active_run():
    gate = FaultInjectionGate(timeout_s=1.0)
    await gate.arm("worker_crash")
    waiter = asyncio.create_task(gate.arrive(
        "before_nccl_source_start",
        {
            "request_id": "request-1",
            "source_endpoint_ref": {
                "operation_id": "transfer-1", "owner_generation": "gateway-1"
            },
            "target_endpoint_ref": {"operation_id": "transfer-1"},
        },
    ))
    for _ in range(20):
        snapshot = await gate.snapshot()
        if snapshot is not None and snapshot["state"] == "REACHED":
            break
        await asyncio.sleep(0)
    gate.record_event("slot_released", {"operation_ids": ["unrelated"]})
    assert (await gate.snapshot())["state"] == "REACHED"
    await gate.release(str((await gate.snapshot())["fault_run_id"]))
    await waiter


def test_forced_correctness_route_bypasses_only_gain_gate() -> None:
    index = PrefixIndex(max_age_s=100)
    location = CacheLocation(
        "d0", "d0-epoch", "ns", "compat", "text", 22, 1, 2, 2,
        time.monotonic(),
    )
    index.install_full_report(("d0", "d0-epoch"), 1, [location])
    topology = TopologyMatrix()
    topology.set_link("d0", "d1", LinkInfo("TCP", 0.001))
    router = AffinityRouter(
        index, topology, block_bytes=29_360_128, safety_margin_ms=1000
    )
    fingerprint = PromptFingerprint(
        "ns", "compat", "text", tuple(range(769)), (11, 22, 33), 256
    )

    assert router.iter_decisions(
        fingerprint, ["d1"], lambda _instance, epoch: epoch == "d0-epoch",
        full_prefill_ms=1.0, suffix_prefill_ms_per_token=1.0,
    ) == []
    forced = router.iter_decisions(
        fingerprint, ["d1"], lambda _instance, epoch: epoch == "d0-epoch",
        full_prefill_ms=1.0, suffix_prefill_ms_per_token=1.0,
        required_source="d0", required_target="d1",
        required_cached_prefix_blocks=2,
    )
    assert len(forced) == 1
    assert forced[0].source_instance == "d0"
    assert forced[0].decode_instance == "d1"
    assert forced[0].cached_prefix_tokens == 512


def test_correctness_evidence_is_bounded_and_terminal_facts_are_strict() -> None:
    rpc = NetworkControlRPC(
        SimpleNamespace(), EndpointSequenceAllocator("world", "owner"),
        {"d0": "epoch"}, active_operation_cap=2, terminal_snapshot_cap=2,
    )
    for req_id in ("r0", "r1", "r2"):
        rpc.require_correctness_evidence(req_id)
        rpc._store_request_evidence(req_id, {"request_id": req_id})
    assert tuple(rpc._request_evidence) == ("r1", "r2")
    assert rpc._correctness_required == set()
    rpc.require_correctness_evidence("cancelled")
    rpc.cancel_correctness_evidence("cancelled")
    assert "cancelled" not in rpc._correctness_required

    terminal = {
        "state": "COMPLETED",
        "result": {"completed_bytes": 123, "work_terminal": True, "cuda_terminal": True},
    }
    assert rpc._terminal_transfer_facts(terminal, terminal) == (123, True, True)
    with pytest.raises(RuntimeError, match="completed bytes disagree"):
        rpc._terminal_transfer_facts(
            terminal,
            {"state": "COMPLETED", "result": {
                "completed_bytes": 124, "work_terminal": True, "cuda_terminal": True,
            }},
        )


def test_helm_correctness_secret_is_disabled_by_default_and_secret_backed() -> None:
    chart = Path(__file__).parents[1] / "k8s" / "helm" / "prism-serve"
    default = subprocess.run(
        ["helm", "template", "week12", str(chart)],
        check=True, capture_output=True, text=True,
    ).stdout
    assert '- name: PRISM_SERVE_CORRECTNESS_HARNESS_ENABLED\n              value: "false"' in default
    assert "PRISM_SERVE_CORRECTNESS_HARNESS_SECRET" not in default
    assert "shareProcessNamespace:" not in default
    assert "PRISM_SERVE_PROCESS_IDENTITY_PATH" not in default
    assert "PRISM_PROCESS_IDENTITY_PATH" not in default

    missing = subprocess.run([
        "helm", "template", "week12", str(chart),
        "--set", "gateway.correctnessHarness.enabled=true",
    ], capture_output=True, text=True)
    assert missing.returncode != 0
    assert "correctnessHarness.secretName is required" in missing.stderr

    sentinel = "this-value-must-never-render"
    enabled = subprocess.run([
        "helm", "template", "week12", str(chart),
        "--set", "gateway.correctnessHarness.enabled=true",
        "--set-string", "gateway.correctnessHarness.secretName=week12-harness",
    ], check=True, capture_output=True, text=True).stdout
    assert 'name: "week12-harness"' in enabled
    assert 'key: "token"' in enabled
    assert "valueFrom:" in enabled
    assert sentinel not in enabled
    assert enabled.count("shareProcessNamespace: true") == 5
    assert enabled.count("PRISM_SERVE_PROCESS_IDENTITY_PATH") == 1
    assert enabled.count("PRISM_PROCESS_IDENTITY_PATH") == 4

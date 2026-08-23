from __future__ import annotations

from types import SimpleNamespace
from dataclasses import asdict
import hashlib
import json

import pytest
import asyncio

from prism_serve.router.http_rpc import (
    AmbiguousRPCError,
    EndpointOperationRef,
    EndpointSequenceAllocator,
    InferRPCError,
)
from prism_serve.router.network_rpc import NetworkControlRPC, activate_replacement_owner
from prism_serve.gateway.correctness_harness import FaultInjectionGate
from prism_serve.router.loader import PrefixLoadContext
from prism_serve.router.protocol import CachedPrefixPlan, MappedTransferStatus
from prism_serve.scheduler.resource_release import ResourceReleaseEvaluator
from prism_serve.scheduler.scheduler import PDScheduler


class FakeHttpClient:
    def __init__(self):
        self.calls = []
        self.correctness_post_success_hook = None

    def set_correctness_post_success_hook(self, hook):
        self.correctness_post_success_hook = hook

    async def prepare_receive(self, instance, ref, payload):
        assert ref.payload_digest == _payload_digest(payload)
        self.calls.append(("prepare_receive", instance, ref, payload))
        return {
            "endpoint_ref": asdict(ref),
            "state": "PREPARED",
            "execution_count": 1,
        }

    async def start_transfer(self, instance, ref, payload):
        assert ref.payload_digest == _payload_digest(payload)
        self.calls.append(("start_transfer", instance, ref, payload))
        return {
            "endpoint_ref": asdict(ref),
            "state": "RUNNING",
            "execution_count": 1,
        }

    async def operation_ref_status(self, instance, ref):
        self.calls.append(("query", instance, ref))
        return {
            "state": "COMPLETED",
            "result": {
                "completed_bytes": 1,
                "work_terminal": True,
                "cuda_terminal": True,
            },
        }

    async def _mutate(self, instance, path, ref, payload):
        self.calls.append(("mutate", instance, path, ref, payload))
        return {"state": "COMPLETED"}


class RecordingMetrics:
    def __init__(self):
        self.observations = []
        self.increments = []

    def observe(self, name, value, *, labels=None):
        self.observations.append((name, value, labels))

    def increment(self, name, amount=1, *, labels=None):
        self.increments.append((name, amount, labels))


def test_one_target_sequence_space_is_shared_across_all_mutation_paths() -> None:
    allocator = EndpointSequenceAllocator("world-a", "gateway-a:boot-a")
    rpc = NetworkControlRPC(
        FakeHttpClient(), allocator, {"d0": "pod-d0:boot-a"}
    )

    request_ref = rpc._allocate(
        "request.prepare", "d0", "request-1", {"path": "request"}
    )
    prefix_ref = rpc._allocate(
        "prefix.prepare", "d0", "prefix-1", {"path": "prefix"}
    )
    transfer_ref = rpc._allocate(
        "transfer.target", "d0", "transfer-1", {"path": "transfer"}
    )
    suffix_ref = allocator.allocate(
        target_instance="d0",
        target_worker_epoch="pod-d0:boot-a",
        operation_id="suffix-1",
        payload={"path": "suffix"},
    )
    rpc.remember_external_ref("suffix", "d0", "suffix-1", suffix_ref)

    assert [
        request_ref.operation_seq,
        prefix_ref.operation_seq,
        transfer_ref.operation_seq,
        suffix_ref.operation_seq,
    ] == [1, 2, 3, 4]
    assert rpc._refs[("suffix", "d0", "suffix-1")] == suffix_ref


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "instance", "expected_role"),
    [
        ("/v1/transfers/prepare-receive", "d1", "target"),
        ("/v1/transfers/start", "d0", "source"),
    ],
)
async def test_correctness_post_success_hook_binds_exact_rpc_role(
    path: str, instance: str, expected_role: str,
) -> None:
    client = FakeHttpClient()
    rpc = NetworkControlRPC(
        client, EndpointSequenceAllocator("world", "owner"),
        {"d0": "d0-e1", "d1": "d1-e1"},
    )
    arrivals = []
    events = []

    class Gate:
        async def arrive(self, checkpoint, details):
            arrivals.append((checkpoint, details))
            return {"state": "RELEASED"}

        def record_event(self, name, details):
            events.append((name, details))

    rpc.set_correctness_fault_gate(Gate())
    rpc.require_correctness_evidence("request-1")
    plan = CachedPrefixPlan(
        operation_id="transfer-1",
        req_id="request-1",
        source_instance="d0",
        target_instance="d1",
        source_epoch="d0-e1",
        target_epoch="d1-e1",
        src_block_ids=(1,),
        dst_block_ids=(2,),
        cached_prefix_tokens=1,
    )
    assert await rpc.transfer_cached_prefix(plan) == MappedTransferStatus.COMPLETED
    arrivals.clear()
    events.clear()
    ref = rpc._refs[(f"transfer.{expected_role}", instance, "transfer-1")]
    discarded = await client.correctness_post_success_hook({
        "path": path, "instance_id": instance, "endpoint_ref": asdict(ref),
        "request_digest": "sha256:request", "response_digest": "sha256:response",
    })

    assert discarded is True
    assert arrivals[0][0] == "after_infer_success_before_control_observe"
    assert arrivals[0][1]["request_id"] == "request-1"
    assert arrivals[0][1]["route_role"] == expected_role
    assert events[0][0] == "response_loss_injected"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("instance", "ref_kind", "expected_role"),
    [
        ("d0", "prefix.resolve", "source"),
        ("d1", "prefix.prepare", "target"),
    ],
)
async def test_finalize_response_loss_hook_uses_request_owner_and_exact_ref(
    instance: str, ref_kind: str, expected_role: str,
) -> None:
    client = FakeHttpClient()
    rpc = NetworkControlRPC(
        client, EndpointSequenceAllocator("world", "owner"),
        {"d0": "d0-e1", "d1": "d1-e1"},
    )
    arrivals = []
    events = []

    class Gate:
        async def arrive(self, checkpoint, details):
            arrivals.append(details)
            return {"state": "RELEASED"}

        def record_event(self, name, details):
            events.append((name, details))

    rpc.set_correctness_fault_gate(Gate())
    rpc.require_correctness_evidence("request-1")
    rpc._allocate("prefix.resolve", "d0", "transfer-1", {"phase": "resolve"})
    rpc._allocate("prefix.prepare", "d1", "transfer-1", {"phase": "prepare"})
    plan = CachedPrefixPlan(
        operation_id="transfer-1",
        req_id="request-1",
        source_instance="d0",
        target_instance="d1",
        source_epoch="d0-e1",
        target_epoch="d1-e1",
        src_block_ids=(1,),
        dst_block_ids=(2,),
        cached_prefix_tokens=1,
    )
    assert await rpc.transfer_cached_prefix(plan) == MappedTransferStatus.COMPLETED
    arrivals.clear()
    events.clear()
    ref = rpc._refs[(ref_kind, instance, "transfer-1")]

    discarded = await client.correctness_post_success_hook({
        "path": "/v1/cleanup/finalize", "instance_id": instance,
        "endpoint_ref": asdict(ref), "request_digest": "sha256:request",
        "response_digest": "sha256:response",
        "cleanup_operation_id": "transfer-1",
    })

    assert discarded is True
    assert arrivals[0]["request_id"] == "request-1"
    assert arrivals[0]["route_role"] == expected_role
    assert events[0][0] == "response_loss_injected"
    rpc._retire_operation("transfer-1")
    assert "transfer-1" not in rpc._correctness_operation_owners
    assert "request-1" not in rpc._correctness_operations
    assert await client.correctness_post_success_hook({
        "path": "/v1/cleanup/finalize", "instance_id": instance,
        "endpoint_ref": asdict(ref),
        "cleanup_operation_id": "transfer-1",
    }) is False


@pytest.mark.asyncio
async def test_correctness_post_success_hook_rejects_unrelated_exact_ref() -> None:
    client = FakeHttpClient()
    rpc = NetworkControlRPC(
        client, EndpointSequenceAllocator("world", "owner"),
        {"d0": "d0-e1", "d1": "d1-e1"},
    )
    arrivals = []

    class Gate:
        async def arrive(self, checkpoint, details):
            arrivals.append((checkpoint, details))
            return {"state": "RELEASED"}

        def record_event(self, name, details):
            raise AssertionError("unrelated ref must not emit a fault event")

    rpc.set_correctness_fault_gate(Gate())
    rpc.require_correctness_evidence("request-1")
    plan = CachedPrefixPlan(
        operation_id="transfer-1",
        req_id="request-1",
        source_instance="d0",
        target_instance="d1",
        source_epoch="d0-e1",
        target_epoch="d1-e1",
        src_block_ids=(1,),
        dst_block_ids=(2,),
        cached_prefix_tokens=1,
    )
    assert await rpc.transfer_cached_prefix(plan) == MappedTransferStatus.COMPLETED
    arrivals.clear()
    source_ref = rpc._refs[("transfer.source", "d0", "transfer-1")]
    unrelated = EndpointOperationRef(
        **{
            **asdict(source_ref),
            "operation_seq": source_ref.operation_seq + 100,
        }
    )

    assert await client.correctness_post_success_hook({
        "path": "/v1/transfers/start",
        "instance_id": "d0",
        "endpoint_ref": asdict(unrelated),
    }) is False
    assert arrivals == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("foreign_ref", "another endpoint ref"),
        ("overdelivery", "too many NATS deliveries"),
    ],
)
async def test_nats_command_authority_semantic_drift_fails_fast(
    mutation: str, message: str,
) -> None:
    ref = EndpointOperationRef(
        topology_generation="world", owner_generation="owner",
        operation_seq=1, target_instance="p0", target_worker_epoch="p0-e1",
        operation_id="request-1", payload_digest="sha256:payload",
    )

    class Client(FakeHttpClient):
        query_calls = 0

        async def operation_ref_status(self, instance, actual_ref):
            self.query_calls += 1
            value = {
                "endpoint_ref": asdict(actual_ref),
                "delivery_count": 2,
                "execution_count": 1,
            }
            if mutation == "foreign_ref":
                value["endpoint_ref"]["operation_seq"] = 2
            else:
                value["delivery_count"] = 3
            return value

    client = Client()
    rpc = NetworkControlRPC(
        client, EndpointSequenceAllocator("world", "owner"), {"p0": "p0-e1"},
        query_interval_s=10, operation_timeout_s=10,
    )

    with pytest.raises(RuntimeError, match=message):
        await rpc.wait_nats_command_fault_authority(
            asdict(ref), "nats_duplicate"
        )
    assert client.query_calls == 1


def test_low_cap_network_state_never_evicts_active_or_unknown_operations():
    rpc = NetworkControlRPC(
        FakeHttpClient(), EndpointSequenceAllocator("world", "owner"),
        {"d0": "epoch"}, active_operation_cap=1, terminal_snapshot_cap=1,
    )
    rpc._allocate("request.prepare", "d0", "held", {})
    with pytest.raises(RuntimeError, match="capacity"):
        rpc._allocate("request.prepare", "d0", "blocked", {})

    rpc._retire_operation("held")
    rpc._allocate("request.prepare", "d0", "next", {})
    rpc._retire_operation("next")

    assert rpc.state_counts() == {
        "active_operations": 0, "refs": 0, "request_metadata": 0,
        "cleanup_plans": 0, "retired_operations": 1,
    }
    assert tuple(rpc._retired_operations) == ("next",)


@pytest.mark.asyncio
async def test_quiesce_joins_cancelled_and_failed_normal_tasks():
    rpc = NetworkControlRPC(
        FakeHttpClient(), EndpointSequenceAllocator("world", "owner"),
        {"d0": "epoch"},
    )

    async def pending():
        await asyncio.Event().wait()

    async def failed():
        raise RuntimeError("old transfer failed")

    pending_task = asyncio.create_task(pending())
    failed_task = asyncio.create_task(failed())
    await asyncio.sleep(0)
    rpc._normal_tasks = {"pending": pending_task, "failed": failed_task}

    errors = await rpc.quiesce()

    assert pending_task.cancelled()
    assert any(isinstance(error, RuntimeError) for error in errors)
    assert rpc._normal_tasks == {}


@pytest.mark.asyncio
async def test_commit_response_loss_queries_exact_ref_and_completes_once():
    class ResponseLossClient(FakeHttpClient):
        commit_ref = None
        commit_calls = 0

        async def operation_ref_status(self, instance, ref):
            if self.commit_ref is ref:
                return {"state": "COMPLETED"}
            return await super().operation_ref_status(instance, ref)

        async def _mutate(self, instance, path, ref, payload):
            self.commit_ref = ref
            self.commit_calls += 1
            raise AmbiguousRPCError(ref, "commit response lost")

    client = ResponseLossClient()
    rpc = NetworkControlRPC(
        client, EndpointSequenceAllocator("world", "owner"),
        {"p0": "p-epoch", "d1": "d-epoch"},
    )
    completed = []
    task = SimpleNamespace(
        operation_id="r1", req_id="r1", src="p0", dst="d1",
        src_epoch="p-epoch", dst_epoch="d-epoch", src_block_ids=(1,),
        dst_block_ids=(2,), kv_size=8, token_ids=(1,), first_token=7,
        transfer_target_ref=None, transfer_source_ref=None,
        target_request_commit_ref=None,
    )

    await rpc._run_normal_transfer(task, lambda: completed.append(True))

    assert completed == [True]
    assert client.commit_calls == 1
    assert task.target_request_commit_ref is client.commit_ref
    assert rpc._refs[("request.commit", "d1", "r1")] is client.commit_ref


@pytest.mark.asyncio
async def test_source_unpin_ambiguous_held_query_replays_identical_cleanup():
    class SourceReleaseClient(FakeHttpClient):
        def __init__(self):
            super().__init__()
            self.finalize_calls = []
            self.queried_ref = None

        async def finalize_release(self, instance, **kwargs):
            self.finalize_calls.append((instance, kwargs))
            if len(self.finalize_calls) == 1:
                raise AmbiguousRPCError(
                    kwargs["endpoint_refs"][0], "response lost before commit"
                )
            return {"cleanup_id": kwargs["cleanup_id"]}

        async def operation_ref_status(self, instance, ref):
            self.queried_ref = ref
            return {
                "state": "COMPLETED",
                "resources_held": True,
                "held_resource_kinds": ["SOURCE_PIN"],
            }

    client = SourceReleaseClient()
    rpc = NetworkControlRPC(
        client,
        EndpointSequenceAllocator("world", "owner"),
        {"p0": "p0-epoch"},
    )
    ref = rpc._allocate("prefix.resolve", "p0", "op-1", {"source": "pin"})

    with pytest.raises(AmbiguousRPCError):
        await rpc.unpin_prefix("p0", "op-1")
    await rpc.unpin_prefix("p0", "op-1")

    assert client.queried_ref is ref
    assert len(client.finalize_calls) == 2
    assert client.finalize_calls[0] == client.finalize_calls[1]
    _, kwargs = client.finalize_calls[0]
    assert kwargs["cleanup_id"] == "prefix-unpin:op-1"
    assert kwargs["endpoint_refs"] == (ref,)
    assert kwargs["resource_kinds"] == ("SOURCE_PIN",)

def _payload_digest(payload):
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.mark.asyncio
async def test_normal_transfer_posts_target_before_source_queries_both_then_commits():
    client = FakeHttpClient()
    metrics = RecordingMetrics()
    rpc = NetworkControlRPC(
        client,
        EndpointSequenceAllocator("world", "owner"),
        {"p0": "p-epoch", "d1": "d-epoch"},
        metrics=metrics,
    )
    completed = []
    task = SimpleNamespace(
        operation_id="transfer-1", req_id="r1", src="p0", dst="d1",
        src_epoch="p-epoch", dst_epoch="d-epoch", src_block_ids=(7, 8),
        dst_block_ids=(17, 18), kv_size=1024, token_ids=(1, 2),
        first_token=42,
        transfer_target_ref=None, transfer_source_ref=None,
    )

    await rpc._run_normal_transfer(task, lambda: completed.append(True))

    assert [call[0] for call in client.calls] == [
        "prepare_receive", "start_transfer", "query", "query", "mutate"
    ]
    assert task.transfer_target_ref.target_instance == "d1"
    assert task.transfer_source_ref.target_instance == "p0"
    assert task.transfer_target_ref.operation_seq == 1
    assert task.transfer_source_ref.operation_seq == 1
    assert completed == [True]
    commit_payload = client.calls[-1][4]
    assert commit_payload["first_token"] == 42
    assert commit_payload["cached_prefix_tokens"] == 2
    assert commit_payload["transfer_endpoint_ref"] == asdict(
        task.transfer_target_ref
    )
    assert any(
        name == "nccl_transfer_latency_ms"
        and labels == {"pair": "p0--d1", "path": "normal"}
        for name, _, labels in metrics.observations
    )
    assert (
        "nccl_transfer_bytes", 1024,
        {"pair": "p0--d1", "path": "normal"},
    ) in metrics.increments


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_phase", ["target_prepare", "source_start"])
async def test_nccl_response_loss_queries_exact_ref_without_repeating_mutation(
    lost_phase: str,
) -> None:
    class Client(FakeHttpClient):
        async def prepare_receive(self, instance, ref, payload):
            await super().prepare_receive(instance, ref, payload)
            if lost_phase == "target_prepare":
                raise AmbiguousRPCError(ref, "target response lost")
            return {"state": "PREPARED"}

        async def start_transfer(self, instance, ref, payload):
            await super().start_transfer(instance, ref, payload)
            if lost_phase == "source_start":
                raise AmbiguousRPCError(ref, "source response lost")
            return {"state": "RUNNING"}

        async def operation_ref_status(self, instance, ref):
            self.calls.append(("query", instance, ref))
            return {
                "endpoint_ref": asdict(ref),
                "state": "COMPLETED",
                "result": {
                    "pair_id": "p0--d0",
                    "completed_bytes": 1,
                    "work_terminal": True,
                    "cuda_terminal": True,
                },
            }

    client = Client()
    allocator = EndpointSequenceAllocator("world", "owner")
    rpc = NetworkControlRPC(
        client, allocator, {"p0": "p-epoch", "d0": "d-epoch"}
    )
    payload = {
        "req_id": "request-1",
        "source_instance": "p0",
        "target_instance": "d0",
        "src_block_ids": [1],
        "dst_block_ids": [2],
        "kv_size_bytes": 1,
    }
    target_ref = allocator.allocate(
        target_instance="d0", target_worker_epoch="d-epoch",
        operation_id="transfer-1", payload=payload,
    )
    source_ref = allocator.allocate(
        target_instance="p0", target_worker_epoch="p-epoch",
        operation_id="transfer-1", payload=payload,
    )

    await rpc._run_nccl_pair(
        source_instance="p0", source_ref=source_ref, source_payload=payload,
        target_instance="d0", target_ref=target_ref, target_payload=payload,
    )

    assert sum(call[0] == "prepare_receive" for call in client.calls) == 1
    assert sum(call[0] == "start_transfer" for call in client.calls) == 1
    lost_ref = target_ref if lost_phase == "target_prepare" else source_ref
    assert ("query", lost_ref.target_instance, lost_ref) in client.calls
    assert rpc._nccl_sequence_poisoned is False


@pytest.mark.asyncio
async def test_nccl_response_loss_with_foreign_query_ref_poisons_sequence() -> None:
    class Client(FakeHttpClient):
        async def prepare_receive(self, instance, ref, payload):
            await super().prepare_receive(instance, ref, payload)
            raise AmbiguousRPCError(ref, "target response lost")

        async def operation_ref_status(self, instance, ref):
            return {
                "endpoint_ref": {**asdict(ref), "operation_seq": ref.operation_seq + 1},
                "state": "COMPLETED",
            }

    client = Client()
    allocator = EndpointSequenceAllocator("world", "owner")
    rpc = NetworkControlRPC(
        client, allocator, {"p0": "p-epoch", "d0": "d-epoch"}
    )
    payload = {
        "req_id": "request-1", "source_instance": "p0",
        "target_instance": "d0", "src_block_ids": [1],
        "dst_block_ids": [2], "kv_size_bytes": 1,
    }
    target_ref = allocator.allocate(
        target_instance="d0", target_worker_epoch="d-epoch",
        operation_id="transfer-1", payload=payload,
    )
    source_ref = allocator.allocate(
        target_instance="p0", target_worker_epoch="p-epoch",
        operation_id="transfer-1", payload=payload,
    )

    with pytest.raises(RuntimeError, match="another endpoint ref"):
        await rpc._run_nccl_pair(
            source_instance="p0", source_ref=source_ref,
            source_payload=payload, target_instance="d0",
            target_ref=target_ref, target_payload=payload,
        )

    assert not any(call[0] == "start_transfer" for call in client.calls)
    assert rpc._nccl_sequence_poisoned is True


@pytest.mark.asyncio
async def test_worker_crash_gate_reaches_after_target_prepare_before_source_start():
    client = FakeHttpClient()
    rpc = NetworkControlRPC(
        client, EndpointSequenceAllocator("world", "owner"),
        {"p0": "p-epoch", "d1": "d-epoch"},
    )
    gate = FaultInjectionGate(timeout_s=1.0)
    rpc.set_correctness_fault_gate(gate)
    rpc.require_correctness_evidence("request-1")
    armed = await gate.arm("worker_crash")
    task = SimpleNamespace(
        operation_id="transfer-1", req_id="request-1", src="p0", dst="d1",
        src_epoch="p-epoch", dst_epoch="d-epoch", src_block_ids=(7,),
        dst_block_ids=(17,), kv_size=1, token_ids=(1,), first_token=2,
        correctness_path="cross_instance", transfer_target_ref=None,
        transfer_source_ref=None, target_request_commit_ref=None,
    )
    running = asyncio.create_task(rpc._run_normal_transfer(task, lambda: None))
    for _ in range(30):
        snapshot = await gate.snapshot()
        if snapshot is not None and snapshot["state"] == "REACHED":
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("worker crash checkpoint was never reached")

    assert [call[0] for call in client.calls] == ["prepare_receive"]
    assert snapshot["details"]["source_endpoint_ref"]["operation_id"] == (
        "transfer-1"
    )
    assert snapshot["details"]["target_endpoint_ref"]["target_instance"] == "d1"
    await gate.release(str(armed["fault_run_id"]))
    await running
    assert any(call[0] == "start_transfer" for call in client.calls)


@pytest.mark.asyncio
async def test_gateway_restart_gate_reaches_only_after_both_mutations_accept():
    client = FakeHttpClient()
    rpc = NetworkControlRPC(
        client, EndpointSequenceAllocator("world", "owner"),
        {"p0": "p-epoch", "d1": "d-epoch"},
    )
    gate = FaultInjectionGate(timeout_s=1.0)
    rpc.set_correctness_fault_gate(gate)
    rpc.require_correctness_evidence("request-1")
    armed = await gate.arm("gateway_restart")
    task = SimpleNamespace(
        operation_id="transfer-1", req_id="request-1", src="p0", dst="d1",
        src_epoch="p-epoch", dst_epoch="d-epoch", src_block_ids=(7,),
        dst_block_ids=(17,), kv_size=1, token_ids=(1,), first_token=2,
        correctness_path="cross_instance", transfer_target_ref=None,
        transfer_source_ref=None, target_request_commit_ref=None,
    )

    running = asyncio.create_task(rpc._run_normal_transfer(task, lambda: None))
    for _ in range(30):
        snapshot = await gate.snapshot()
        if snapshot is not None and snapshot["state"] == "REACHED":
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("gateway restart checkpoint was never reached")

    assert [call[0] for call in client.calls] == [
        "prepare_receive", "start_transfer"
    ]
    assert snapshot["details"]["target_acceptance"]["state"] == "PREPARED"
    assert snapshot["details"]["source_acceptance"]["state"] == "RUNNING"
    assert snapshot["details"]["source_payload"]["src_block_ids"] == [7]
    assert snapshot["details"]["target_payload"]["dst_block_ids"] == [17]
    assert snapshot["details"]["source_payload"]["kv_size_bytes"] == 1
    assert snapshot["details"]["target_payload"]["kv_size_bytes"] == 1
    assert snapshot["details"]["source_acceptance"]["endpoint_ref"] \
        == snapshot["details"]["source_endpoint_ref"]
    assert snapshot["details"]["target_acceptance"]["endpoint_ref"] \
        == snapshot["details"]["target_endpoint_ref"]
    await gate.release(str(armed["fault_run_id"]))
    await running


@pytest.mark.asyncio
async def test_overlapping_nccl_pairs_are_globally_serialized_through_terminal():
    class BlockingClient(FakeHttpClient):
        def __init__(self):
            super().__init__()
            self.first_queried = asyncio.Event()
            self.release_first = asyncio.Event()

        async def operation_ref_status(self, instance, ref):
            self.calls.append(("query", instance, ref))
            if ref.operation_id == "transfer-a":
                self.first_queried.set()
                await self.release_first.wait()
            return {
                "state": "COMPLETED",
                "result": {
                    "completed_bytes": 8,
                    "work_terminal": True,
                    "cuda_terminal": True,
                },
            }

    client = BlockingClient()
    rpc = NetworkControlRPC(
        client,
        EndpointSequenceAllocator("world", "owner"),
        {"p0": "p0e", "p1": "p1e", "d0": "d0e", "d1": "d1e"},
    )

    def task(operation_id, source, target, source_epoch, target_epoch):
        return SimpleNamespace(
            operation_id=operation_id,
            req_id=operation_id,
            src=source,
            dst=target,
            src_epoch=source_epoch,
            dst_epoch=target_epoch,
            src_block_ids=(0,),
            dst_block_ids=(1,),
            kv_size=8,
            token_ids=(1,),
            first_token=2,
            transfer_target_ref=None,
            transfer_source_ref=None,
            target_request_commit_ref=None,
        )

    first = asyncio.create_task(rpc._run_normal_transfer(
        task("transfer-a", "p0", "d0", "p0e", "d0e"), lambda: None
    ))
    await client.first_queried.wait()
    second = asyncio.create_task(rpc._run_normal_transfer(
        task("transfer-b", "p1", "d1", "p1e", "d1e"), lambda: None
    ))
    await asyncio.sleep(0)

    assert not any(
        call[0] == "prepare_receive"
        and call[2].operation_id == "transfer-b"
        for call in client.calls
    )

    client.release_first.set()
    await asyncio.gather(first, second)
    prepares = [
        call[2].operation_id for call in client.calls
        if call[0] == "prepare_receive"
    ]
    assert prepares == ["transfer-a", "transfer-b"]


class OwnerClient:
    def __init__(
        self,
        held=False,
        *,
        listed_state="FENCED",
        query_unknown=False,
        lose_finalize_once=False,
    ):
        self.active = "old"
        self.held = held
        self.listed_state = listed_state
        self.query_unknown = query_unknown
        self.lose_finalize_once = lose_finalize_once
        self.calls = []
        self.ref = EndpointOperationRef(
            "world", "old", 1, "d0", "epoch", "op-1", "sha256:old"
        )

    async def get_identity(self, instance):
        return {"instance_id": instance, "instance_epoch": "epoch"}

    async def owner_status(self, instance):
        return {"active_owner": self.active}

    async def list_operations(self, instance, owner):
        return {"instance_epoch": "epoch", "complete": True, "operations": [{
            "state": self.listed_state, "resources_held": self.held,
            "held_resource_kinds": ["TARGET_PENDING"] if self.held else [],
            "endpoint_ref": asdict(self.ref),
        }]}

    async def operation_ref_status(self, instance, ref):
        if self.query_unknown:
            raise TimeoutError("ambiguous")
        return {
            "state": "FENCED", "resources_held": self.held,
            "held_resource_kinds": ["TARGET_PENDING"] if self.held else [],
            "endpoint_ref": asdict(ref),
        }

    async def abort_request(self, instance, ref, reason):
        self.calls.append(("abort", instance, ref.operation_id))
        self.listed_state = "FENCED"
        return {"state": "FENCED"}

    async def finalize_release(self, instance, **kwargs):
        self.calls.append(("finalize", instance, kwargs["cleanup_id"]))
        self.held = False
        if self.lose_finalize_once:
            self.lose_finalize_once = False
            raise TimeoutError("response lost")
        return _release_snapshot(instance, kwargs)

    async def retire_owner(self, instance, owner):
        self.calls.append(("retire", instance, owner))
        self.active = None

    async def activate_owner(self, instance, owner):
        self.calls.append(("activate", instance, owner))
        self.active = owner


def _list_evidence_digest(value):
    payload = {
        "instance_epoch": value.instance_epoch,
        "complete": value.complete,
        "operations": [
            {
                "endpoint_ref": asdict(operation.endpoint_ref),
                "state": operation.state,
                "resources_held": operation.resources_held,
                "held_resource_kinds": list(operation.held_resource_kinds),
            }
            for operation in value.operations
        ],
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.mark.asyncio
async def test_replacement_owner_retire_then_activate():
    client = OwnerClient()
    audit = await activate_replacement_owner(client, ("d0",), "new")
    assert client.calls == [("retire", "d0", "old"), ("activate", "d0", "new")]
    assert audit.old_owners == (("d0", "old"),)
    assert audit.operation_list_evidence[0].instance_id == "d0"
    assert audit.operation_list_evidence[0].complete is True
    assert audit.operation_list_evidence[0].endpoint_refs == (client.ref,)
    listed = audit.operation_list_evidence[0].operations
    assert len(listed) == 1
    assert listed[0].endpoint_ref == client.ref
    assert listed[0].state == "FENCED"
    assert listed[0].resources_held is False
    assert listed[0].held_resource_kinds == ()
    assert audit.operation_list_evidence[0].report_digest == (
        _list_evidence_digest(audit.operation_list_evidence[0])
    )
    assert audit.operations[0].endpoint_ref == client.ref
    assert audit.operations[0].listed_state == "FENCED"
    assert audit.operations[0].query_confirmed is True
    assert audit.operations[0].query_evidence.endpoint_ref == client.ref
    assert audit.operations[0].query_evidence.owner_generation == "old"
    assert audit.operations[0].query_evidence.operation_id == "op-1"
    assert audit.operations[0].abort_attempted is False
    assert audit.operations[0].finalize_acknowledged is False
    assert audit.operations[0].finalize_evidence is None
    assert audit.activated_instances == ("d0",)
    assert audit.confirmed_active_owners == (("d0", "new"),)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["retire", "activate"])
async def test_replacement_owner_reads_back_committed_409(mutation):
    class Committed409Client(OwnerClient):
        async def retire_owner(self, instance, owner):
            await super().retire_owner(instance, owner)
            if mutation == "retire":
                raise InferRPCError(409, "UNKNOWN_OWNER", "response raced")

        async def activate_owner(self, instance, owner):
            await super().activate_owner(instance, owner)
            if mutation == "activate":
                raise InferRPCError(409, "OWNER_CONFLICT", "response raced")

    client = Committed409Client()
    audit = await activate_replacement_owner(
        client, ("d0",), "new", retry_interval_s=0.001
    )

    assert client.calls == [
        ("retire", "d0", "old"),
        ("activate", "d0", "new"),
    ]
    assert audit.confirmed_active_owners == (("d0", "new"),)


@pytest.mark.asyncio
async def test_replacement_owner_replays_retire_when_409_left_old_active():
    class ReplayRetireClient(OwnerClient):
        def __init__(self):
            super().__init__()
            self.retire_attempts = 0

        async def retire_owner(self, instance, owner):
            self.retire_attempts += 1
            self.calls.append(("retire", instance, owner))
            if self.retire_attempts == 1:
                raise InferRPCError(409, "PRECONDITION_FAILED", "still active")
            self.active = None

    client = ReplayRetireClient()
    await activate_replacement_owner(
        client, ("d0",), "new", retry_interval_s=0.001
    )

    assert client.retire_attempts == 2
    assert client.calls == [
        ("retire", "d0", "old"),
        ("retire", "d0", "old"),
        ("activate", "d0", "new"),
    ]


@pytest.mark.asyncio
async def test_replacement_owner_rejects_foreign_owner_after_mutation_409():
    class ForeignAfterRetireClient(OwnerClient):
        async def retire_owner(self, instance, owner):
            self.calls.append(("retire", instance, owner))
            self.active = "foreign"
            raise InferRPCError(409, "OWNER_CONFLICT", "foreign owner won")

    client = ForeignAfterRetireClient()
    with pytest.raises(RuntimeError, match="foreign owner"):
        await activate_replacement_owner(
            client, ("d0",), "new", retry_interval_s=0.001
        )

    assert client.calls == [("retire", "d0", "old")]


@pytest.mark.asyncio
async def test_partial_activation_does_not_repeat_orphan_abort_or_finalize():
    class PartialActivationClient:
        def __init__(self):
            self.owners = {"d0": "old", "d1": "old"}
            self.held = True
            self.state = "RUNNING"
            self.calls = []
            self.ref = EndpointOperationRef(
                "world", "old", 1, "d0", "d0-epoch", "op-1",
                "sha256:old",
            )

        async def owner_status(self, instance):
            return {"active_owner": self.owners[instance]}

        async def get_identity(self, instance):
            return {
                "instance_id": instance,
                "instance_epoch": f"{instance}-epoch",
            }

        async def list_operations(self, instance, owner):
            operations = []
            if instance == "d0":
                operations.append({
                    "state": self.state,
                    "resources_held": self.held,
                    "held_resource_kinds": (
                        ["TARGET_PENDING"] if self.held else []
                    ),
                    "endpoint_ref": asdict(self.ref),
                })
            return {
                "instance_epoch": f"{instance}-epoch",
                "complete": True,
                "operations": operations,
            }

        async def abort_request(self, instance, ref, reason):
            self.calls.append(("abort", instance, ref.operation_id))
            self.state = "FENCED"
            return {"state": "FENCED"}

        async def operation_ref_status(self, instance, ref):
            return {
                "state": self.state,
                "resources_held": self.held,
                "held_resource_kinds": (
                    ["TARGET_PENDING"] if self.held else []
                ),
                "endpoint_ref": asdict(ref),
            }

        async def finalize_release(self, instance, **kwargs):
            self.calls.append(("finalize", instance, kwargs["cleanup_id"]))
            self.held = False
            return _release_snapshot(instance, kwargs)

        async def retire_owner(self, instance, owner):
            self.calls.append(("retire", instance, owner))
            self.owners[instance] = None

        async def activate_owner(self, instance, owner):
            self.calls.append(("activate", instance, owner))
            self.owners[instance] = owner
            if instance == "d0":
                raise InferRPCError(
                    409, "OWNER_CONFLICT", "committed response lost"
                )

    client = PartialActivationClient()
    audit = await activate_replacement_owner(
        client, ("d0", "d1"), "new", retry_interval_s=0.001
    )

    assert [call[0] for call in client.calls].count("abort") == 1
    assert [call[0] for call in client.calls].count("finalize") == 1
    assert [call[0] for call in client.calls].count("retire") == 2
    assert [call[0] for call in client.calls].count("activate") == 2
    assert audit.confirmed_active_owners == (("d0", "new"), ("d1", "new"))


@pytest.mark.asyncio
async def test_replacement_owner_audit_exposes_four_complete_empty_lists():
    class EmptyOwnerClient:
        def __init__(self):
            self.owners = {
                instance: "old" for instance in ("p0", "p1", "d0", "d1")
            }

        async def owner_status(self, instance):
            return {"active_owner": self.owners[instance]}

        async def get_identity(self, instance):
            return {
                "instance_id": instance,
                "instance_epoch": f"{instance}-epoch",
            }

        async def list_operations(self, instance, owner):
            return {
                "instance_epoch": f"{instance}-epoch",
                "complete": True,
                "operations": [],
            }

        async def finalize_release(self, *args, **kwargs):
            raise AssertionError("empty operation lists must not finalize")

        async def retire_owner(self, instance, owner):
            self.owners[instance] = None

        async def activate_owner(self, instance, owner):
            self.owners[instance] = owner

    client = EmptyOwnerClient()
    audit = await activate_replacement_owner(
        client, ("p1", "d0", "p0", "d1"), "new"
    )

    assert [value.instance_id for value in audit.operation_list_evidence] == [
        "d0", "d1", "p0", "p1"
    ]
    assert audit.operations == ()
    for value in audit.operation_list_evidence:
        assert value.instance_epoch == f"{value.instance_id}-epoch"
        assert value.owner_generation == "old"
        assert value.complete is True
        assert value.endpoint_refs == ()
        assert value.operations == ()
        assert value.report_digest == _list_evidence_digest(value)
        assert len(value.report_digest) == len("sha256:") + 64


@pytest.mark.asyncio
async def test_replacement_owner_unknown_orphan_keeps_admission_closed():
    client = OwnerClient(held=True, query_unknown=True)
    with pytest.raises(RuntimeError, match="UNKNOWN"):
        await activate_replacement_owner(client, ("d0",), "new")
    assert client.calls == []


@pytest.mark.asyncio
async def test_replacement_owner_finalizes_held_terminal_before_retire():
    client = OwnerClient(held=True)
    audit = await activate_replacement_owner(client, ("d0",), "new")
    assert [call[0] for call in client.calls] == [
        "finalize", "retire", "activate"
    ]
    assert audit.finalized_operation_ids == ("op-1",)
    assert audit.operations[0].finalize_acknowledged is True
    finalize = audit.operations[0].finalize_evidence
    assert finalize is not None
    assert finalize.instance_id == "d0"
    assert finalize.operation_id == "op-1"
    assert finalize.endpoint_epoch == "epoch"
    assert finalize.request_endpoint_refs == (client.ref,)
    assert finalize.released_resource_kinds == ("TARGET_PENDING",)
    assert finalize.released_counts == (("TARGET_PENDING", 1),)
    assert finalize.resources_held_after is False


def _release_snapshot(instance, kwargs):
    payload = {
        "cleanup_id": kwargs["cleanup_id"],
        "operation_id": kwargs["operation_id"],
        "lease_id": kwargs["lease_id"],
        "endpoint_refs": [asdict(ref) for ref in kwargs["endpoint_refs"]],
        "resource_kinds": sorted(set(kwargs["resource_kinds"])),
        "release_basis": "ENDPOINT_TERMINAL",
    }
    payload_digest = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "cleanup_id": kwargs["cleanup_id"],
        "operation_id": kwargs["operation_id"],
        "lease_id": kwargs["lease_id"],
        "endpoint_epoch": kwargs["endpoint_refs"][0].target_worker_epoch,
        "released_resource_kinds": payload["resource_kinds"],
        "released_counts": [[kind, 1] for kind in payload["resource_kinds"]],
        "resources_held_after": False,
        "payload_digest": payload_digest,
    }


@pytest.mark.asyncio
async def test_replacement_owner_aborts_listed_nonterminal_then_queries_exact_ref():
    client = OwnerClient(listed_state="RUNNING")

    audit = await activate_replacement_owner(client, ("d0",), "new")

    assert client.calls == [
        ("abort", "d0", "op-1"),
        ("retire", "d0", "old"),
        ("activate", "d0", "new"),
    ]
    operation = audit.operations[0]
    assert operation.listed_state == "RUNNING"
    assert operation.abort_attempted is True
    assert operation.query_confirmed is True
    assert operation.terminal_state == "FENCED"


@pytest.mark.asyncio
async def test_replacement_owner_marks_finalize_ack_only_on_held_exact_ref():
    class MixedHeldOwnerClient:
        def __init__(self):
            self.active = "old"
            self.held = {"d0": True, "d1": False}
            self.refs = {
                instance: EndpointOperationRef(
                    "world", "old", seq, instance, f"{instance}-epoch",
                    "op-1", f"sha256:{instance}",
                )
                for seq, instance in enumerate(("d0", "d1"), start=1)
            }

        async def owner_status(self, instance):
            return {"active_owner": self.active}

        async def get_identity(self, instance):
            return {
                "instance_id": instance,
                "instance_epoch": f"{instance}-epoch",
            }

        async def list_operations(self, instance, owner):
            return {
                "instance_epoch": f"{instance}-epoch",
                "complete": True,
                "operations": [{
                "state": "FENCED",
                "resources_held": self.held[instance],
                "held_resource_kinds": (
                    ["SOURCE_RETAIN"] if self.held[instance] else []
                ),
                "endpoint_ref": asdict(self.refs[instance]),
                }],
            }

        async def operation_ref_status(self, instance, ref):
            return {
                "state": "FENCED",
                "resources_held": self.held[instance],
                "held_resource_kinds": (
                    ["SOURCE_RETAIN"] if self.held[instance] else []
                ),
                "endpoint_ref": asdict(ref),
            }

        async def finalize_release(self, instance, **kwargs):
            self.held[instance] = False
            return _release_snapshot(instance, kwargs)

        async def retire_owner(self, instance, owner):
            self.active = None

        async def activate_owner(self, instance, owner):
            self.active = owner

    client = MixedHeldOwnerClient()

    audit = await activate_replacement_owner(client, ("d0", "d1"), "new")

    operations = {value.instance_id: value for value in audit.operations}
    assert operations["d0"].finalize_acknowledged is True
    assert operations["d1"].finalize_acknowledged is False


@pytest.mark.asyncio
async def test_replacement_owner_finalize_response_loss_replays_same_cleanup():
    client = OwnerClient(held=True, lose_finalize_once=True)
    await activate_replacement_owner(client, ("d0",), "new")
    finalize_calls = [call for call in client.calls if call[0] == "finalize"]
    assert len(finalize_calls) == 2
    assert finalize_calls[0][2] == finalize_calls[1][2]
    assert client.active == "new"


@pytest.mark.asyncio
async def test_replacement_owner_audit_cap_fails_before_mutation():
    class TooManyOperationsClient(OwnerClient):
        async def list_operations(self, instance, owner):
            value = await super().list_operations(instance, owner)
            value["operations"] *= 2
            return value

    client = TooManyOperationsClient()
    with pytest.raises(RuntimeError, match="audit exceeds"):
        await activate_replacement_owner(
            client, ("d0",), "new", max_audit_entries=1
        )
    assert client.calls == []


@pytest.mark.asyncio
async def test_replacement_owner_rejects_incomplete_operation_list():
    class IncompleteListClient(OwnerClient):
        async def list_operations(self, instance, owner):
            value = await super().list_operations(instance, owner)
            value.pop("complete")
            return value

    client = IncompleteListClient()
    with pytest.raises(RuntimeError, match="complete"):
        await activate_replacement_owner(client, ("d0",), "new")
    assert client.calls == []


@pytest.mark.asyncio
async def test_replacement_owner_rejects_list_from_another_worker_epoch():
    class WrongEpochListClient(OwnerClient):
        async def list_operations(self, instance, owner):
            value = await super().list_operations(instance, owner)
            value["instance_epoch"] = "replacement-epoch"
            return value

    client = WrongEpochListClient()
    with pytest.raises(RuntimeError, match="worker epoch"):
        await activate_replacement_owner(client, ("d0",), "new")
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["missing", "foreign"])
async def test_replacement_owner_rejects_unbound_exact_query(mutation):
    class UnboundQueryClient(OwnerClient):
        async def operation_ref_status(self, instance, ref):
            value = await super().operation_ref_status(instance, ref)
            if mutation == "missing":
                value.pop("endpoint_ref")
            else:
                value["endpoint_ref"] = {
                    **asdict(ref), "operation_id": "another-operation"
                }
            return value

    client = UnboundQueryClient()
    with pytest.raises(RuntimeError, match="exact query"):
        await activate_replacement_owner(client, ("d0",), "new")
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "cleanup_id",
        "operation_id",
        "lease_id",
        "endpoint_epoch",
        "payload_digest",
        "resources_held_after",
        "released_resource_kinds",
        "released_counts",
    ],
)
async def test_replacement_owner_rejects_unbound_finalize_ack(mutation):
    class InvalidFinalizeClient(OwnerClient):
        async def finalize_release(self, instance, **kwargs):
            self.calls.append(("finalize", instance, kwargs["cleanup_id"]))
            value = _release_snapshot(instance, kwargs)
            if mutation == "empty":
                return {}
            if mutation == "resources_held_after":
                value[mutation] = True
            elif mutation == "released_resource_kinds":
                value[mutation] = ["OTHER"]
            elif mutation == "released_counts":
                value[mutation] = [["OTHER", 1]]
            else:
                value[mutation] = f"wrong-{mutation}"
            return value

    client = InvalidFinalizeClient(held=True)
    with pytest.raises(RuntimeError, match="finalize did not converge"):
        await activate_replacement_owner(client, ("d0",), "new")
    assert not any(call[0] in {"retire", "activate"} for call in client.calls)


@pytest.mark.asyncio
async def test_replacement_owner_rejects_foreign_ref_before_sweep():
    client = OwnerClient()
    client.ref = EndpointOperationRef(
        "world", "foreign-owner", 1, "d0", "epoch", "op-1", "sha256:old"
    )

    with pytest.raises(RuntimeError, match="foreign endpoint ref"):
        await activate_replacement_owner(client, ("d0",), "new")

    assert client.calls == []


class CleanupClient:
    def __init__(self, fail_query=False):
        self.fail_query = fail_query
        self.finalize_calls = []

    async def _abort(self, instance, path, ref, reason):
        return {"state": "FENCED"}

    async def abort_transfer(self, instance, ref, reason):
        return {"state": "FENCED"}

    async def abort_request(self, instance, ref, reason):
        return {"state": "FENCED"}

    async def operation_ref_status(self, instance, ref):
        if self.fail_query:
            raise TimeoutError("ambiguous")
        kind = "SOURCE_PIN" if instance == "d0" else "TARGET_PENDING"
        return {
            "state": "FENCED", "resources_held": True,
            "held_resource_kinds": [kind],
        }

    async def finalize_release(self, instance, **kwargs):
        self.finalize_calls.append((instance, kwargs))
        return {"cleanup_id": kwargs["cleanup_id"]}


class PreSourceCancellationClient:

    def __init__(self):
        self.calls = []
        self.snapshots = {}
        self.resources = {
            "d0": {"SOURCE_PIN": 0},
            "d1": {"TARGET_PENDING": 0},
        }

    def _accept(self, instance, ref, *, state, held_kinds=()):
        key = (instance, ref)
        self.snapshots[key] = {
            "endpoint_ref": asdict(ref),
            "state": state,
            "resources_held": bool(held_kinds),
            "held_resource_kinds": list(held_kinds),
        }
        for kind in held_kinds:
            self.resources[instance][kind] += 1

    def _snapshot(self, instance, ref):
        value = self.snapshots.get((instance, ref))
        if value is not None:
            return value
        if any(
            stored_ref.operation_id == ref.operation_id
            for stored_instance, stored_ref in self.snapshots
            if stored_instance == instance
        ):
            raise InferRPCError(
                409, "OPERATION_CONFLICT",
                "abort requires the original endpoint ref for operation id",
            )
        raise InferRPCError(412, "PRECONDITION_FAILED", "unknown endpoint ref")

    async def prefix_mutation(self, instance, action, ref, payload):
        self.calls.append((f"prefix.{action}", instance, ref))
        if action == "resolve":
            self._accept(
                instance, ref, state="COMPLETED", held_kinds=("SOURCE_PIN",),
            )
            return {
                "state": "COMPLETED",
                "result": {"miss": False, "src_block_ids": [7]},
            }
        if action == "prepare":
            self._accept(
                instance, ref, state="PREPARED", held_kinds=("TARGET_PENDING",),
            )
            return {
                "state": "PREPARED",
                "result": {"mode": "remote_transfer", "dst_block_ids": [17]},
            }
        raise AssertionError(f"unexpected prefix action: {action}")

    async def prepare_receive(self, instance, ref, payload):
        self.calls.append(("prepare_receive", instance, ref))
        self._accept(instance, ref, state="PREPARED")
        return {"endpoint_ref": asdict(ref), "state": "PREPARED"}

    async def start_transfer(self, instance, ref, payload):
        self.calls.append(("start_transfer", instance, ref))
        raise AssertionError("source transfer must not cross the cancellation gate")

    async def _abort(self, instance, path, ref, reason):
        self.calls.append(("abort.prefix", instance, ref))
        value = self._snapshot(instance, ref)
        value["state"] = "FENCED"
        return dict(value)

    async def abort_transfer(self, instance, ref, reason):
        self.calls.append(("abort.transfer", instance, ref))
        value = self._snapshot(instance, ref)
        value["state"] = "FENCED"
        return dict(value)

    async def operation_ref_status(self, instance, ref):
        self.calls.append(("query", instance, ref))
        return dict(self._snapshot(instance, ref))

    async def finalize_release(self, instance, **kwargs):
        self.calls.append(("finalize", instance, kwargs["endpoint_refs"]))
        for ref in kwargs["endpoint_refs"]:
            value = self._snapshot(instance, ref)
            for kind in value["held_resource_kinds"]:
                self.resources[instance][kind] -= 1
            value["resources_held"] = False
            value["held_resource_kinds"] = []
        return {"cleanup_id": kwargs["cleanup_id"]}


@pytest.mark.asyncio
async def test_cancel_before_source_start_never_cleans_up_unattempted_source_ref():
    client = PreSourceCancellationClient()
    rpc = NetworkControlRPC(
        client, EndpointSequenceAllocator("world", "owner"),
        {"d0": "d0-epoch", "d1": "d1-epoch"},
    )
    gate = FaultInjectionGate(timeout_s=1.0)
    rpc.set_correctness_fault_gate(gate)
    rpc.require_correctness_evidence("r1")
    await rpc.resolve_prefix("d0", "d0-epoch", "op-1", ())
    await rpc.prepare_prefix(
        "d1", "d1-epoch", "op-1", "r1",
        mode="remote_transfer", token_ids=(1,),
    )
    scheduler = PDScheduler({})
    scheduler.register_instance(
        "d1", "decode", max_slots=1, instance_epoch="d1-epoch",
    )
    scheduler.reserve_decode_slot("d1", "r1", "op-1")
    rpc.set_release_evaluator(
        ResourceReleaseEvaluator(scheduler, client.finalize_release)
    )
    plan = CachedPrefixPlan(
        operation_id="op-1", req_id="r1",
        source_instance="d0", target_instance="d1",
        source_epoch="d0-epoch", target_epoch="d1-epoch",
        src_block_ids=(7,), dst_block_ids=(17,), cached_prefix_tokens=1,
        token_ids=(1,),
    )
    await gate.arm("worker_crash")
    running = asyncio.create_task(rpc.transfer_cached_prefix(plan))
    for _ in range(30):
        snapshot = await gate.snapshot()
        if snapshot is not None and snapshot["state"] == "REACHED":
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("before-source-start checkpoint was never reached")

    source_transfer_ref = rpc._refs[("transfer.source", "d0", "op-1")]
    target_transfer_ref = rpc._refs[("transfer.target", "d1", "op-1")]
    assert source_transfer_ref not in rpc._attempted_refs
    assert target_transfer_ref in rpc._attempted_refs
    with pytest.raises(InferRPCError) as collision:
        client._snapshot("d0", source_transfer_ref)
    assert collision.value.status_code == 409
    assert not any(call[0] == "start_transfer" for call in client.calls)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    outcome = await rpc.cleanup_prefix_context(
        scheduler, PrefixLoadContext("op-1", "r1", "d0", "d1")
    )

    assert outcome.action == "ABORTED"
    assert not any(
        call[0] in {"abort.transfer", "query"}
        and call[1] == "d0"
        and call[2] == source_transfer_ref
        for call in client.calls
    )
    assert client.resources == {
        "d0": {"SOURCE_PIN": 0},
        "d1": {"TARGET_PENDING": 0},
    }
    assert scheduler.decode_slot_lease("op-1").state == "RELEASED"
    assert rpc._attempted_refs == set()
    assert rpc._ambiguous_refs == set()


def _cleanup_rpc(client):
    rpc = NetworkControlRPC(
        client, EndpointSequenceAllocator("world", "owner"),
        {"d0": "d0-epoch", "d1": "d1-epoch"},
    )
    resolve_ref = rpc._allocate("prefix.resolve", "d0", "op-1", {"a": 1})
    prepare_ref = rpc._allocate("prefix.prepare", "d1", "op-1", {"a": 2})
    rpc._mark_ref_attempted(resolve_ref)
    rpc._mark_ref_attempted(prepare_ref)
    scheduler = PDScheduler({})
    scheduler.register_instance("d1", "decode", max_slots=1, instance_epoch="d1-epoch")
    scheduler.reserve_decode_slot("d1", "r1", "op-1")
    rpc.set_release_evaluator(ResourceReleaseEvaluator(scheduler, client.finalize_release))
    return rpc, scheduler


@pytest.mark.asyncio
async def test_network_cleanup_unknown_sends_zero_finalize_and_holds_slot():
    client = CleanupClient(fail_query=True)
    rpc, scheduler = _cleanup_rpc(client)
    outcome = await rpc.cleanup_prefix_context(
        scheduler, PrefixLoadContext("op-1", "r1", "d0", "d1")
    )
    assert outcome.action == "QUARANTINED"
    assert client.finalize_calls == []
    assert scheduler.decode_slot_lease("op-1").state == "QUARANTINED"


@pytest.mark.asyncio
async def test_network_cleanup_terminal_remote_acks_then_slot_last():
    client = CleanupClient()
    rpc, scheduler = _cleanup_rpc(client)
    outcome = await rpc.cleanup_prefix_context(
        scheduler, PrefixLoadContext("op-1", "r1", "d0", "d1")
    )
    assert outcome.action == "ABORTED"
    assert [call[0] for call in client.finalize_calls] == ["d0", "d1"]
    assert scheduler.decode_slot_lease("op-1").state == "RELEASED"


class PartialAckCleanupClient(CleanupClient):
    def __init__(self):
        super().__init__()
        self.source_held = True
        self.target_finalize_attempts = 0
        self.scheduler = None

    async def operation_ref_status(self, instance, ref):
        if instance == "d0":
            return {
                "state": "FENCED",
                "resources_held": self.source_held,
                "held_resource_kinds": ["SOURCE_PIN"] if self.source_held else [],
            }
        return {
            "state": "FENCED",
            "resources_held": True,
            "held_resource_kinds": ["TARGET_PENDING"],
        }

    async def finalize_release(self, instance, **kwargs):
        assert self.scheduler.decode_slot_lease("op-1").state == "QUARANTINED"
        self.finalize_calls.append((instance, kwargs))
        if instance == "d0":
            self.source_held = False
            return {"cleanup_id": kwargs["cleanup_id"], "endpoint": instance}
        self.target_finalize_attempts += 1
        if self.target_finalize_attempts == 1:
            raise TimeoutError("target finalize response lost")
        return {"cleanup_id": kwargs["cleanup_id"], "endpoint": instance}


@pytest.mark.asyncio
async def test_prefix_cleanup_freezes_plan_and_replays_only_unacked_endpoint():
    client = PartialAckCleanupClient()
    rpc, scheduler = _cleanup_rpc(client)
    client.scheduler = scheduler
    context = PrefixLoadContext("op-1", "r1", "d0", "d1")
    cleanup_id = "prefix-cleanup:op-1"

    with pytest.raises(TimeoutError, match="response lost"):
        await rpc.cleanup_prefix_context(scheduler, context)

    frozen_plan = rpc._cleanup_plans[cleanup_id]
    frozen_signature = (
        frozen_plan.predicates,
        tuple(
            (endpoint.instance_id, endpoint.endpoint_refs, endpoint.resource_kinds)
            for endpoint in frozen_plan.endpoints
        ),
    )
    first_target_kwargs = next(
        kwargs for instance, kwargs in client.finalize_calls if instance == "d1"
    )
    assert scheduler.decode_slot_lease("op-1").state == "QUARANTINED"

    outcome = await rpc.cleanup_prefix_context(scheduler, context)

    assert outcome.action == "ABORTED"
    assert [instance for instance, _ in client.finalize_calls] == ["d0", "d1", "d1"]
    target_calls = [
        kwargs for instance, kwargs in client.finalize_calls if instance == "d1"
    ]
    assert target_calls[0] == target_calls[1] == first_target_kwargs
    assert frozen_signature == (
        frozen_plan.predicates,
        tuple(
            (endpoint.instance_id, endpoint.endpoint_refs, endpoint.resource_kinds)
            for endpoint in frozen_plan.endpoints
        ),
    )
    assert scheduler.decode_slot_lease("op-1").state == "RELEASED"


class NormalCleanupClient(CleanupClient):
    async def operation_ref_status(self, instance, ref):
        if self.fail_query:
            raise TimeoutError("ambiguous")
        ownership = {
            ("p0", 1): (True, ["SOURCE_BLOCKS"]),
            ("p0", 2): (True, ["SOURCE_RETAIN", "TRANSFER_BYTES"]),
            ("d1", 1): (True, ["TARGET_SEQUENCE"]),
            ("d1", 2): (False, []),
        }
        held, kinds = ownership[(instance, ref.operation_seq)]
        return {
            "state": "COMPLETED", "resources_held": held,
            "held_resource_kinds": kinds,
        }


def _normal_cleanup_rpc(client):
    rpc = NetworkControlRPC(
        client, EndpointSequenceAllocator("world", "owner"),
        {"p0": "p0-epoch", "d1": "d1-epoch"},
    )
    source_request = rpc.allocator.allocate(
        target_instance="p0", target_worker_epoch="p0-epoch",
        operation_id="r1", payload={"source": "request"},
    )
    rpc.remember_external_ref("request.source", "p0", "r1", source_request)
    target_request = rpc._allocate(
        "request.commit", "d1", "r1", {"target": "request"}
    )
    source_transfer = rpc._allocate(
        "transfer.source", "p0", "r1", {"source": "transfer"}
    )
    target_transfer = rpc._allocate(
        "transfer.target", "d1", "r1", {"target": "transfer"}
    )
    rpc._mark_ref_attempted(target_request)
    rpc._mark_ref_attempted(source_transfer)
    rpc._mark_ref_attempted(target_transfer)
    scheduler = PDScheduler({})
    scheduler.register_instance(
        "d1", "decode", max_slots=1, instance_epoch="d1-epoch"
    )
    scheduler.reserve_decode_slot("d1", "r1", "r1")
    scheduler.commit_decode_slot("r1")
    rpc.set_release_evaluator(
        ResourceReleaseEvaluator(scheduler, client.finalize_release)
    )
    req = SimpleNamespace(
        req_id="r1", active_operation_id="r1",
        prefill_instance="p0", dispatch_operation_ref=source_request,
    )
    return rpc, scheduler, req


@pytest.mark.asyncio
async def test_normal_cleanup_remote_acks_then_releases_slot_last():
    client = NormalCleanupClient()
    rpc, scheduler, req = _normal_cleanup_rpc(client)
    assert await rpc.cleanup_request(scheduler, req, abort=False) is True
    assert [value[0] for value in client.finalize_calls] == ["d1", "p0"]
    by_instance = {instance: kwargs for instance, kwargs in client.finalize_calls}
    assert by_instance["p0"]["resource_kinds"] == (
        "SOURCE_BLOCKS", "SOURCE_RETAIN", "TRANSFER_BYTES"
    )
    assert by_instance["d1"]["resource_kinds"] == ("TARGET_SEQUENCE",)
    assert scheduler.decode_slot_lease("r1").state == "RELEASED"


@pytest.mark.asyncio
async def test_normal_cleanup_waits_for_active_network_task_before_finalizing():
    client = NormalCleanupClient()
    rpc, scheduler, req = _normal_cleanup_rpc(client)
    active_task = asyncio.create_task(asyncio.Event().wait())
    rpc._normal_tasks["r1"] = active_task

    try:
        assert await rpc.cleanup_request(scheduler, req, abort=True) is False
        assert client.finalize_calls == []
        assert scheduler.decode_slot_lease("r1").state == "QUARANTINED"

        active_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active_task

        assert await rpc.cleanup_request(scheduler, req, abort=True) is True
        assert scheduler.decode_slot_lease("r1").state == "RELEASED"
    finally:
        if not active_task.done():
            active_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await active_task


@pytest.mark.asyncio
async def test_normal_cleanup_unknown_sends_zero_finalize_and_holds_slot():
    client = NormalCleanupClient(fail_query=True)
    rpc, scheduler, req = _normal_cleanup_rpc(client)
    assert await rpc.cleanup_request(scheduler, req, abort=True) is False
    assert client.finalize_calls == []
    assert scheduler.decode_slot_lease("r1").state == "QUARANTINED"

from __future__ import annotations

from types import SimpleNamespace

import pytest

import prism_serve.gateway.app as gateway
from prism_serve.router.topology import TopologyMatrix
from prism_serve.router.tokenizer import TokenizerAdapter, TokenizerIdentity


class CapableClient:
    def transfer(self, *args, **kwargs):
        pass

    def reset_to_waiting(self, *args, **kwargs):
        pass

    def abort_request(self, *args, **kwargs):
        return {"success": True}

    def abort_transfer(self, *args, **kwargs):
        return {"success": True}

    def get_reconciliation_report(self, instance_id, instance_epoch, challenge):
        return {
            "instance_id": instance_id, "instance_epoch": instance_epoch,
            "challenge": challenge, "active_request_ids": [],
            "active_transfer_operation_ids": [], "pending_dispatch_command_ids": [],
        }

    async def get_kv_usage_all(self):
        return {}

    async def resolve_prefix(self, *args, **kwargs):
        return None

    async def prepare_prefix(self, *args, **kwargs):
        return None

    async def transfer_cached_prefix(self, *args, **kwargs):
        raise AssertionError("no requests expected")

    async def commit_cached_prefix(self, *args, **kwargs):
        pass

    async def abort_mapped_prefix(self, *args, **kwargs):
        raise AssertionError("no requests expected")

    async def get_prefix_operation(self, *args, **kwargs):
        raise AssertionError("no requests expected")

    async def abort_cached_prefix(self, *args, **kwargs):
        pass

    async def unpin_prefix(self, *args, **kwargs):
        pass

    async def abort_suffix_prefill(self, *args, **kwargs):
        return True

    async def get_prefix_resource_counts(self, *args, **kwargs):
        return {"transfer_pins": 0, "pending_allocations": 0}

    async def full_report_and_register(self, *args, **kwargs):
        raise AssertionError("no instances expected")

    async def peek_events(self, *args, **kwargs):
        return []

    async def ack_events(self, *args, **kwargs):
        pass


class Encoder:
    def encode(self, text):
        return [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_lifespan_wires_and_stops_affinity_runtime(monkeypatch):
    config = gateway._build_config()
    config.update({
        "affinity_enabled": True,
        "prefix_block_bytes": 1024,
        "nats_required": False,
        "nats_connect_timeout_s": 0.01,
        "shutdown_drain_timeout_s": 0.01,
        "scheduler_id": "gateway-test",
        "scheduler_generation": "generation-test",
    })
    monkeypatch.setattr(gateway, "_build_config", lambda: config)
    dummy = SimpleNamespace(state=SimpleNamespace(
        infer_client=CapableClient(), topology_matrix=TopologyMatrix(),
        tokenizer_adapter=TokenizerAdapter(Encoder(), TokenizerIdentity(
            "model", "tokenizer", "chat", 4, "xxh64v1", "compat"
        )),
    ))
    async with gateway.lifespan(dummy):
        assert dummy.state.affinity_coordinator is not None
        assert dummy.state.prefix_reconciler is not None
        assert not dummy.state.reconciler_task.done()
        assert dummy.state.accepting is True
    assert dummy.state.reconciler_task.cancelled()
    assert dummy.state.accepting is False

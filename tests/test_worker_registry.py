from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import logging
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from nats.errors import (
    ConnectionClosedError as NATSConnectionClosedError,
    NoServersError as NATSNoServersError,
    StaleConnectionError as NATSStaleConnectionError,
    TimeoutError as NATSTimeoutError,
)

from prism_serve.gateway.app import (
    _bootstrap_replacement_owner, _bootstrap_week12_http_control,
    _collapse_pair_attestations, _refresh_worker_resources,
    _refresh_worker_world_once, _pull_replacement_resource_reports,
    _run_gateway_bootstrap_stage,
)
from prism_serve.router.http_rpc import InferRPCError
from prism_serve.router.worker_registry import (
    PairCapability,
    ResourceSignalState,
    TopologyState,
    WorkerIdentity,
    WorkerRegistry,
)
from prism_serve.scheduler.scheduler import PDScheduler


def _identity(instance_id: str, *, generation: str = "world-a", boot: str = "boot-a"):
    role = "prefill" if instance_id.startswith("p") else "decode"
    rank = {"p0": 0, "p1": 1, "d0": 2, "d1": 3}[instance_id]
    return WorkerIdentity(
        instance_id=instance_id,
        role=role,
        topology_generation=generation,
        pod_uid=f"pod-{instance_id}",
        process_generation=boot,
        rpc_endpoint=f"http://{instance_id}:8001",
        global_rank=rank,
        topology_digest="sha256:topology-a",
        kv_compatibility_id="kv-a",
    )


def _cap(source: str, target: str):
    return PairCapability(
        pair_id=f"{source}--{target}",
        source_epoch=f"pod-{source}:boot-a",
        target_epoch=f"pod-{target}:boot-a",
        transport="NCCL_SOCKET",
        probe_generation="probe-a",
        probe_passed=True,
        evidence_path="logs/nccl-net.log",
    )


def _ready_registry(clock=lambda: 10.0):
    registry = WorkerRegistry(
        expected_topology_generation="world-a",
        require_gpudirect_rdma=False,
        resource_report_stale_after_s=2.0,
        clock=clock,
    )
    identities = [_identity(i) for i in ("p0", "p1", "d0", "d1")]
    caps = [
        _cap("p0", "d0"),
        _cap("p0", "d1"),
        _cap("p1", "d0"),
        _cap("p1", "d1"),
        _cap("d0", "d1"),
    ]
    registry.install_world(identities, caps)
    return registry


def test_worker_world_requires_exact_four_members_and_five_pairs():
    registry = WorkerRegistry(expected_topology_generation="world-a")

    assert registry.install_world([_identity("p0")], []) is False
    assert registry.state == TopologyState.FAILED


def test_worker_world_requires_one_expected_kv_compatibility_id():
    registry = WorkerRegistry(
        expected_topology_generation="world-a",
        expected_kv_compatibility_id="kv-a",
    )
    identities = [_identity(i) for i in ("p0", "p1", "d0", "d1")]
    identities[-1] = WorkerIdentity(
        **{**asdict(identities[-1]), "kv_compatibility_id": "kv-b"}
    )

    assert registry.install_world(identities, [
        _cap("p0", "d0"), _cap("p0", "d1"), _cap("p1", "d0"),
        _cap("p1", "d1"), _cap("d0", "d1"),
    ]) is False
    assert registry.state == TopologyState.FAILED


def test_pair_attestation_requires_both_endpoints_and_exact_agreement():
    capabilities = _ready_registry().capabilities
    attestations = {
        pair_id: {endpoint: capability for endpoint in pair_id.split("--")}
        for pair_id, capability in capabilities.items()
    }
    assert _collapse_pair_attestations(attestations) == capabilities

    missing = {pair: dict(values) for pair, values in attestations.items()}
    missing["p0--d0"].pop("d0")
    with pytest.raises(RuntimeError, match="two endpoint"):
        _collapse_pair_attestations(missing)

    disagree = {pair: dict(values) for pair, values in attestations.items()}
    value = disagree["p0--d0"]["d0"]
    disagree["p0--d0"]["d0"] = PairCapability(
        value.pair_id, value.source_epoch, value.target_epoch, "NCCL_GDR",
        value.probe_generation, value.probe_passed, value.evidence_path,
    )
    with pytest.raises(RuntimeError, match="disagree"):
        _collapse_pair_attestations(disagree)


def test_worker_epoch_change_invalidates_signal_and_topology():
    registry = _ready_registry()
    registry.update_resource_report(
        "d0",
        {"instance_epoch": "pod-d0:boot-a", "complete": True, "resources": {}},
    )

    registry.observe_identity(_identity("d0", boot="boot-b"))

    assert registry.state == TopologyState.FAILED
    assert registry.resource_signal("d0").state == ResourceSignalState.UNKNOWN


def test_partial_resource_response_preserves_last_validated_sample_age():
    now = [10.0]
    registry = _ready_registry(clock=lambda: now[0])
    registry.update_resource_report(
        "d0",
        {"instance_epoch": "pod-d0:boot-a", "complete": True, "resources": {}},
    )
    now[0] = 11.0

    assert registry.update_resource_report("d0", {"complete": True}) is False
    signal = registry.resource_signal("d0")
    assert signal.state == ResourceSignalState.FRESH
    assert signal.received_at == 10.0
    assert signal.age_s == 1.0
    now[0] = 12.1
    assert registry.resource_signal("d0").state == ResourceSignalState.STALE


def test_partial_resource_response_without_prior_sample_remains_unknown():
    registry = _ready_registry()

    assert registry.update_resource_report("d0", {"complete": True}) is False
    assert registry.resource_signal("d0").state == ResourceSignalState.UNKNOWN


def test_fixed_profile_and_block_conservation_fail_closed():
    expected_profile = {
        "profile_id": "week12-qwen3-0.6b",
        "kv_compatibility_id": "kv-a",
        "kv_block_bytes": 29_360_128,
    }
    registry = WorkerRegistry(
        expected_topology_generation="world-a",
        expected_kv_compatibility_id="kv-a",
        expected_model_profile=expected_profile,
    )
    identities = [_identity(i) for i in ("p0", "p1", "d0", "d1")]
    caps = [
        _cap("p0", "d0"), _cap("p0", "d1"), _cap("p1", "d0"),
        _cap("p1", "d1"), _cap("d0", "d1"),
    ]
    assert registry.install_world(identities, caps)
    report = {
        "instance_epoch": "pod-d0:boot-a", "complete": True,
        "resources": {}, "model_profile": expected_profile,
        "num_gpu_blocks": 10, "free_blocks": 4,
        "block_buckets": {
            "free": 4, "pending": 1, "sequence": 2,
            "evictable": 2, "quarantined": 1,
        },
        "block_conservation_valid": True,
    }
    assert registry.update_resource_report("d0", report)

    report["block_buckets"] = {**report["block_buckets"], "pending": 2}
    assert registry.update_resource_report("d0", report) is False
    assert registry.resource_signal("d0").state == ResourceSignalState.FRESH


def test_resource_exception_does_not_refresh_monotonic_age():
    now = [10.0]
    registry = _ready_registry(clock=lambda: now[0])
    registry.update_resource_report(
        "d0",
        {"instance_epoch": "pod-d0:boot-a", "complete": True, "resources": {}},
    )
    now[0] = 12.1

    registry.resource_report_failed("d0")

    assert registry.resource_signal("d0").state == ResourceSignalState.STALE


def test_unreachable_observation_preserves_last_sample_until_natural_stale():
    now = [10.0]
    registry = _ready_registry(clock=lambda: now[0])
    registry.update_resource_report(
        "d0",
        {"instance_epoch": "pod-d0:boot-a", "complete": True, "resources": {}},
    )

    now[0] = 11.0
    registry.observe_unreachable("d0")

    assert registry.state == TopologyState.READY
    assert registry.resource_signal("d0").state == ResourceSignalState.FRESH
    now[0] = 12.1
    assert registry.resource_signal("d0").state == ResourceSignalState.STALE


def test_resource_signal_unknown_and_stale_block_all_consumers():
    now = [10.0]
    registry = _ready_registry(clock=lambda: now[0])
    assert registry.can_admit("d0") is False
    registry.update_resource_report(
        "d0",
        {"instance_epoch": "pod-d0:boot-a", "complete": True, "resources": {}},
    )
    assert registry.can_admit("d0") is True
    now[0] = 12.01
    assert registry.can_admit("d0") is False


def test_capability_fail_closed_when_gdr_is_required():
    registry = WorkerRegistry(
        expected_topology_generation="world-a",
        require_gpudirect_rdma=True,
    )
    identities = [_identity(i) for i in ("p0", "p1", "d0", "d1")]
    caps = [
        _cap("p0", "d0"), _cap("p0", "d1"), _cap("p1", "d0"),
        _cap("p1", "d1"), _cap("d0", "d1"),
    ]

    assert registry.install_world(identities, caps) is False
    assert registry.state == TopologyState.FAILED


def test_world_fresh_requires_all_four_current_resource_reports():
    now = [10.0]
    registry = _ready_registry(clock=lambda: now[0])
    for instance in ("p0", "p1", "d0"):
        assert registry.update_resource_report(instance, {
            "instance_epoch": f"pod-{instance}:boot-a",
            "complete": True, "resources": {},
        })
    assert registry.world_fresh() is False
    assert registry.update_resource_report("d1", {
        "instance_epoch": "pod-d1:boot-a", "complete": True, "resources": {},
    })
    assert registry.world_fresh() is True
    now[0] = 12.01
    assert registry.world_fresh() is False


class _Metrics:
    def __init__(self):
        self.values = []

    def increment(self, name, amount=1, *, labels=None):
        self.values.append(("increment", name, amount, labels))

    def gauge(self, name, value, *, labels=None):
        self.values.append(("gauge", name, value, labels))


class _RefreshClient:
    def __init__(
        self, *, identity_drift=False, capability_drift=False,
        unreachable_instance=None, capability_not_ready_instance=None,
        incomplete_identity_instance=None, incomplete_capability_instance=None,
        incomplete_resource_instance=None,
    ):
        self.identity_drift = identity_drift
        self.capability_drift = capability_drift
        self.unreachable_instance = unreachable_instance
        self.capability_not_ready_instance = capability_not_ready_instance
        self.incomplete_identity_instance = incomplete_identity_instance
        self.incomplete_capability_instance = incomplete_capability_instance
        self.incomplete_resource_instance = incomplete_resource_instance

    async def get_identity(self, instance):
        if instance == self.unreachable_instance:
            raise ConnectionError("worker is unreachable")
        identity = _identity(
            instance,
            boot="boot-b" if self.identity_drift and instance == "d0" else "boot-a",
        )
        value = {**asdict(identity), "instance_epoch": identity.instance_epoch}
        if instance == self.incomplete_identity_instance:
            value.pop("process_generation")
        return value

    async def get_capabilities(self, instance):
        if instance == self.capability_not_ready_instance:
            return {"ready": False, "pairs": []}
        capabilities = [
            value for value in _ready_registry().capabilities.values()
            if instance in set(value.pair_id.split("--"))
        ]
        if self.capability_drift:
            first = capabilities[0]
            capabilities[0] = PairCapability(
                first.pair_id, first.source_epoch, first.target_epoch,
                "NCCL_GDR", first.probe_generation, first.probe_passed,
                first.evidence_path,
            )
        pairs = [asdict(value) for value in capabilities]
        if instance == self.incomplete_capability_instance:
            pairs[0].pop("probe_generation")
        return {"ready": True, "pairs": pairs}

    async def get_resources(self, instance):
        if instance == self.incomplete_resource_instance:
            return {"complete": True}
        return {
            "instance_epoch": (
                f"pod-{instance}:boot-b"
                if self.identity_drift and instance == "d0"
                else f"pod-{instance}:boot-a"
            ),
            "complete": True,
            "resources": {"TARGET_PENDING": 0},
        }


def _refresh_app(client):
    app = FastAPI()
    scheduler = PDScheduler({})
    app.state.worker_registry = _ready_registry()
    app.state.http_infer_client = client
    app.state.metrics = _Metrics()
    app.state.scheduler = scheduler
    app.state.governor = SimpleNamespace(
        quarantined_transfer_totals=lambda operation_ids: (0, {})
    )
    app.state.queue = SimpleNamespace(owner_id="gateway-a")
    app.state.accepting = True
    return app


@pytest.mark.asyncio
async def test_refresh_identity_drift_latches_failed_and_emits_epoch_change():
    app = _refresh_app(_RefreshClient(identity_drift=True))

    await _refresh_worker_world_once(app)

    assert app.state.worker_registry.state == TopologyState.FAILED
    assert app.state.accepting is False
    assert any(
        value[1] == "worker_epoch_change_total"
        and value[3] == {"instance": "d0"}
        for value in app.state.metrics.values
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transient_field",
    [
        "capability_not_ready_instance",
        "incomplete_identity_instance",
        "incomplete_capability_instance",
        "incomplete_resource_instance",
    ],
)
async def test_refresh_transient_incomplete_evidence_ages_then_recovers(
    transient_field,
):
    now = [0.0]
    client = _RefreshClient()
    app = _refresh_app(client)
    app.state.worker_registry = _ready_registry(clock=lambda: now[0])

    await _refresh_worker_world_once(app)
    before = app.state.worker_registry.resource_signal("d0")
    assert app.state.worker_registry.world_fresh() is True

    setattr(client, transient_field, "d0")
    now[0] = 1.0
    await _refresh_worker_world_once(app)
    during = app.state.worker_registry.resource_signal("d0")
    assert app.state.worker_registry.state == TopologyState.READY
    assert during.received_at == before.received_at
    assert during.age_s == 1.0
    assert app.state.worker_registry.world_fresh() is True
    assert app.state.accepting is True

    now[0] = 2.000001
    await _refresh_worker_world_once(app)
    assert app.state.worker_registry.state == TopologyState.READY
    assert app.state.worker_registry.world_fresh() is False
    assert app.state.accepting is False

    setattr(client, transient_field, None)
    await _refresh_worker_world_once(app)
    assert app.state.worker_registry.world_fresh() is True
    assert app.state.accepting is True


@pytest.mark.asyncio
async def test_refresh_capability_drift_latches_failed_and_pair_not_ready():
    app = _refresh_app(_RefreshClient(capability_drift=True))

    await _refresh_worker_world_once(app)

    assert app.state.worker_registry.state == TopologyState.FAILED
    assert app.state.accepting is False
    assert any(
        value[1] == "pair_capability_ready" and value[2] == 0
        for value in app.state.metrics.values
    )


@pytest.mark.asyncio
async def test_refresh_transport_failure_ages_cached_report_before_fail_closed():
    now = [0.0]
    client = _RefreshClient()
    app = _refresh_app(client)
    app.state.worker_registry = _ready_registry(clock=lambda: now[0])

    await _refresh_worker_world_once(app)
    assert app.state.worker_registry.world_fresh() is True

    client.unreachable_instance = "d0"
    now[0] = 1.0
    await _refresh_worker_world_once(app)
    assert app.state.worker_registry.state == TopologyState.READY
    assert app.state.worker_registry.world_fresh() is True
    assert app.state.accepting is True

    now[0] = 2.0
    await _refresh_worker_world_once(app)
    assert app.state.worker_registry.resource_signal("d0").age_s == 2.0
    assert app.state.worker_registry.world_fresh() is True

    now[0] = 2.000001
    await _refresh_worker_world_once(app)
    assert app.state.worker_registry.state == TopologyState.READY
    assert app.state.worker_registry.world_fresh() is False
    assert app.state.accepting is False

    client.unreachable_instance = None
    await _refresh_worker_world_once(app)
    assert app.state.worker_registry.world_fresh() is True
    assert app.state.accepting is True

    client.identity_drift = True
    await _refresh_worker_world_once(app)
    assert app.state.worker_registry.state == TopologyState.FAILED
    assert app.state.accepting is False


@pytest.mark.asyncio
async def test_refresh_uses_resource_response_completion_time_not_slow_peer_evidence():
    now = [0.0]
    resource_completed = asyncio.Event()

    class Client(_RefreshClient):
        async def get_capabilities(self, instance):
            if instance == "p0":
                await resource_completed.wait()
                await asyncio.sleep(0)
                now[0] = 2.1
            return await super().get_capabilities(instance)

        async def get_resources(self, instance):
            report = await super().get_resources(instance)
            if instance == "p0":
                resource_completed.set()
            return report

    app = _refresh_app(Client())
    app.state.worker_registry = _ready_registry(clock=lambda: now[0])

    await _refresh_worker_world_once(app)

    signal = app.state.worker_registry.resource_signal("p0")
    assert signal.received_at == 0.0
    assert signal.state == ResourceSignalState.STALE
    assert app.state.worker_registry.world_fresh() is False
    assert app.state.accepting is False


@pytest.mark.asyncio
async def test_resource_refresh_cadence_reaches_stale_on_fifth_half_second_tick(
    monkeypatch,
):
    now = [0.0]
    registry = _ready_registry(clock=lambda: now[0])
    for instance in ("p0", "p1", "d0", "d1"):
        assert registry.update_resource_report(instance, {
            "instance_epoch": f"pod-{instance}:boot-a",
            "complete": True,
            "resources": {},
        })
    app = _refresh_app(_RefreshClient())
    app.state.worker_registry = registry
    observations = []
    sleeps = []

    async def fail_one_report(_app):
        registry.resource_report_failed("d0")
        observations.append((now[0], registry.world_fresh()))
        if len(observations) == 6:
            raise asyncio.CancelledError

    async def advance_clock(delay):
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr(
        "prism_serve.gateway.app._refresh_worker_world_once", fail_one_report,
    )
    monkeypatch.setattr(asyncio, "sleep", advance_clock)

    with pytest.raises(asyncio.CancelledError):
        await _refresh_worker_resources(
            app, {"resource_report_stale_after_s": 2.0},
        )

    assert sleeps == [0.5] * 5
    assert observations == [
        (0.0, True), (0.5, True), (1.0, True),
        (1.5, True), (2.0, True), (2.5, False),
    ]

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("accepted_generation", "acceptance_required"),
    [
        pytest.param("world-a", True, id="pending-acceptance"),
        pytest.param("world-b", False, id="persisted-acceptance"),
        pytest.param(None, True, id="legacy-env-fallback"),
    ],
)
async def test_fresh_gateway_bootstrap_uses_replacement_topology_artifact_generation(
    tmp_path, monkeypatch, accepted_generation, acceptance_required,
):
    import json

    topology = tmp_path / "topology.json"
    topology_payload = {
        "topology_generation": "world-b",
        "endpoints": {name: f"http://{name}:8001" for name in ("p0", "p1", "d0", "d1")},
    }
    if accepted_generation is not None:
        topology_payload["accepted_topology_generation"] = accepted_generation
    topology.write_text(json.dumps(topology_payload), encoding="utf-8")
    identities = {
        name: _identity(name, generation="world-b")
        for name in ("p0", "p1", "d0", "d1")
    }
    capabilities = {
        cap.pair_id: cap for cap in (
            _cap("p0", "d0"), _cap("p0", "d1"), _cap("p1", "d0"),
            _cap("p1", "d1"), _cap("d0", "d1"),
        )
    }

    class Client:
        def __init__(self, endpoints, timeout_s):
            self.endpoints = endpoints

        async def get_identity(self, instance):
            return {**asdict(identities[instance]),
                    "instance_epoch": identities[instance].instance_epoch}

        async def get_capabilities(self, instance):
            values = [
                asdict(value) for value in capabilities.values()
                if instance in set(value.pair_id.split("--"))
            ]
            return {"ready": True, "pairs": values}

        async def get_resources(self, instance):
            return {
                "instance_epoch": identities[instance].instance_epoch,
                "complete": True, "resources": {},
            }

        async def close(self):
            pass

    monkeypatch.setattr("prism_serve.router.http_rpc.HttpInferClient", Client)
    app = FastAPI()
    config = {
        "worker_topology_path": str(topology),
        "topology_generation": "world-a",
        "infer_rpc_timeout_s": 1.0,
        "require_gpudirect_rdma": False,
        "resource_report_stale_after_s": 2.0,
    }

    await _bootstrap_week12_http_control(app, config)

    assert config["topology_generation"] == "world-b"
    assert app.state.worker_registry.expected_topology_generation == "world-b"
    assert app.state.worker_registry.world_fresh() is True
    assert app.state.topology_acceptance_required is acceptance_required
    assert app.state.pending_topology_generation == (
        "world-b" if acceptance_required else None
    )

    app.state.metrics = _Metrics()
    app.state.scheduler = PDScheduler({})
    app.state.governor = SimpleNamespace(
        quarantined_transfer_totals=lambda operation_ids: (0, {})
    )
    app.state.queue = SimpleNamespace(owner_id="gateway-b")
    app.state.accepting = False
    await _refresh_worker_world_once(app)
    assert app.state.worker_registry.world_fresh() is True
    assert app.state.accepting is not acceptance_required


def _bootstrap_config(tmp_path, *, generation="world-b"):
    import json

    topology = tmp_path / "topology.json"
    topology.write_text(json.dumps({
        "topology_generation": generation,
        "accepted_topology_generation": generation,
        "endpoints": {
            name: f"http://{name}:8001"
            for name in ("p0", "p1", "d0", "d1")
        },
    }), encoding="utf-8")
    return {
        "worker_topology_path": str(topology),
        "topology_generation": generation,
        "infer_rpc_timeout_s": 1.0,
        "operation_query_interval_ms": 1,
        "require_gpudirect_rdma": False,
        "resource_report_stale_after_s": 2.0,
    }


class _BootstrapClient:
    transient_error = None
    not_ready_sequence = False
    mutation = None

    def __init__(self, endpoints, timeout_s):
        self.endpoints = endpoints
        self.attempts = {name: 0 for name in endpoints}
        self.closed = False
        self.identities = {
            name: _identity(name, generation="world-b")
            for name in endpoints
        }
        self.capabilities = {
            cap.pair_id: cap for cap in (
                _cap("p0", "d0"), _cap("p0", "d1"), _cap("p1", "d0"),
                _cap("p1", "d1"), _cap("d0", "d1"),
            )
        }

    async def get_identity(self, instance):
        self.attempts[instance] += 1
        attempt = self.attempts[instance]
        if instance == "p0" and attempt == 1 and self.transient_error is not None:
            raise self.transient_error
        value = {
            **asdict(self.identities[instance]),
            "instance_epoch": self.identities[instance].instance_epoch,
        }
        if self.mutation == "malformed" and instance == "d1":
            value.pop("pod_uid")
        elif self.mutation == "wrong_generation" and instance == "d1":
            value["topology_generation"] = "world-foreign"
        elif self.mutation == "wrong_digest" and instance == "d1":
            value["topology_digest"] = "sha256:foreign"
        elif self.mutation == "duplicate_member" and instance == "d1":
            value["instance_id"] = "d0"
        return value

    async def get_capabilities(self, instance):
        attempt = self.attempts[instance]
        if self.not_ready_sequence and instance == "p0" and attempt == 1:
            return {"ready": False, "pairs": []}
        values = [
            asdict(value) for value in self.capabilities.values()
            if instance in set(value.pair_id.split("--"))
        ]
        if self.mutation == "wrong_epoch" and instance == "p0":
            values[0]["source_epoch"] = "pod-p0:foreign"
        elif self.mutation == "attestation_conflict" and instance == "d0":
            values[0]["transport"] = "NCCL_GDR"
        return {"ready": True, "pairs": values}

    async def get_resources(self, instance):
        attempt = self.attempts[instance]
        if self.not_ready_sequence and instance == "p0" and attempt == 2:
            return {
                "instance_epoch": self.identities[instance].instance_epoch,
                "complete": False,
                "resources": {},
            }
        return {
            "instance_epoch": self.identities[instance].instance_epoch,
            "complete": True,
            "resources": {},
        }

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ConnectionRefusedError("refused"),
        ConnectionResetError("reset"),
        TimeoutError("timeout"),
        InferRPCError(503, "NOT_READY", "worker not ready"),
    ],
)
async def test_gateway_bootstrap_retries_transient_transport_in_same_process(
    tmp_path, monkeypatch, error,
):
    class Client(_BootstrapClient):
        transient_error = error

    clients = []

    def factory(*args, **kwargs):
        client = Client(*args, **kwargs)
        clients.append(client)
        return client

    async def no_wait(_delay):
        return None

    monkeypatch.setattr("prism_serve.router.http_rpc.HttpInferClient", factory)
    monkeypatch.setattr(asyncio, "sleep", no_wait)
    app = FastAPI()

    await _bootstrap_week12_http_control(app, _bootstrap_config(tmp_path))

    assert clients[0].attempts["p0"] == 2
    assert clients[0].closed is False
    assert app.state.worker_registry.world_fresh() is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        NATSNoServersError(),
        NATSConnectionClosedError(),
        NATSStaleConnectionError(),
        NATSTimeoutError(),
    ],
)
async def test_gateway_bootstrap_retries_explicit_nats_transients(
    monkeypatch, error,
):
    attempts = 0

    async def connect():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error
        return "connected"

    async def no_wait(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    deadline = asyncio.get_running_loop().time() + 1.0

    assert await _run_gateway_bootstrap_stage(
        connect,
        stage="NATS connect",
        deadline=deadline,
        retry_interval_s=0.01,
    ) == "connected"
    assert attempts == 2


@pytest.mark.asyncio
async def test_gateway_bootstrap_retry_diagnostic_preserves_semantics(
    monkeypatch, caplog,
):
    secret = "bootstrap-secret-value"
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionResetError(f"reset token={secret}")
        return "ready"

    async def no_wait(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    deadline = asyncio.get_running_loop().time() + 1.0
    with caplog.at_level(
        logging.WARNING, logger="prism_serve.gateway.app"
    ):
        result = await _run_gateway_bootstrap_stage(
            operation,
            stage="owner reconciliation",
            deadline=deadline,
            retry_interval_s=0.01,
        )

    assert result == "ready"
    assert attempts == 2
    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.getMessage().startswith("{")
    ]
    assert len(events) == 1
    assert events[0]["event"] == "gateway_bootstrap.retry"
    assert events[0]["stage"] == "owner reconciliation"
    assert events[0]["retry_index"] == 0
    assert type(events[0]["remaining_ms"]) is int
    assert events[0]["remaining_ms"] > 0
    assert events[0]["error_chain"][0]["type"] == "ConnectionResetError"
    assert secret not in json.dumps(events, sort_keys=True)


@pytest.mark.asyncio
async def test_gateway_bootstrap_deadline_diagnostic_preserves_cause(caplog):
    blocker = asyncio.Event()
    deadline = asyncio.get_running_loop().time() + 0.01

    with caplog.at_level(
        logging.ERROR, logger="prism_serve.gateway.app"
    ):
        with pytest.raises(
            RuntimeError,
            match="gateway bootstrap deadline exceeded during NATS connect",
        ) as error:
            await _run_gateway_bootstrap_stage(
                blocker.wait,
                stage="NATS connect",
                deadline=deadline,
                retry_interval_s=0.01,
            )

    assert isinstance(error.value.__cause__, TimeoutError)
    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.getMessage().startswith("{")
    ]
    assert len(events) == 1
    assert events[0]["event"] == "gateway_bootstrap.deadline_exceeded"
    assert events[0]["stage"] == "NATS connect"
    assert events[0]["retry_index"] == 0
    assert events[0]["remaining_ms"] == 0
    assert events[0]["error_chain"][0]["type"] == "TimeoutError"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ImportError("nats dependency missing"),
        ValueError("invalid NATS configuration"),
        TypeError("NATS client shape mismatch"),
    ],
)
async def test_gateway_bootstrap_does_not_retry_nats_setup_errors(
    monkeypatch, error,
):
    attempts = 0

    async def connect():
        nonlocal attempts
        attempts += 1
        raise error

    async def forbidden_sleep(_delay):
        raise AssertionError("fatal NATS setup error must not retry")

    monkeypatch.setattr(asyncio, "sleep", forbidden_sleep)
    deadline = asyncio.get_running_loop().time() + 1.0

    with pytest.raises(type(error), match=str(error)):
        await _run_gateway_bootstrap_stage(
            connect,
            stage="NATS connect",
            deadline=deadline,
            retry_interval_s=0.01,
        )

    assert attempts == 1


@pytest.mark.asyncio
async def test_gateway_bootstrap_retries_not_ready_and_incomplete_report(
    tmp_path, monkeypatch,
):
    class Client(_BootstrapClient):
        not_ready_sequence = True

    clients = []

    def factory(*args, **kwargs):
        client = Client(*args, **kwargs)
        clients.append(client)
        return client

    async def no_wait(_delay):
        return None

    monkeypatch.setattr("prism_serve.router.http_rpc.HttpInferClient", factory)
    monkeypatch.setattr(asyncio, "sleep", no_wait)
    app = FastAPI()

    await _bootstrap_week12_http_control(app, _bootstrap_config(tmp_path))

    assert clients[0].attempts["p0"] == 3
    assert app.state.worker_registry.world_fresh() is True


@pytest.mark.asyncio
async def test_gateway_bootstrap_rejects_round_when_earliest_response_is_stale(
    tmp_path, monkeypatch,
):
    now = [0.0]
    resource_completed = asyncio.Event()

    class ClockedRegistry(WorkerRegistry):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, clock=lambda: now[0])

    class Client(_BootstrapClient):
        async def get_capabilities(self, instance):
            if instance == "p0":
                await resource_completed.wait()
                await asyncio.sleep(0)
                now[0] = 2.1
            return await super().get_capabilities(instance)

        async def get_resources(self, instance):
            report = await super().get_resources(instance)
            if instance == "p0":
                resource_completed.set()
            return report

    clients = []

    def factory(*args, **kwargs):
        client = Client(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("prism_serve.router.http_rpc.HttpInferClient", factory)
    monkeypatch.setattr(
        "prism_serve.router.worker_registry.WorkerRegistry", ClockedRegistry,
    )

    with pytest.raises(RuntimeError, match="resource reports.*stale"):
        await _bootstrap_week12_http_control(
            FastAPI(), _bootstrap_config(tmp_path),
        )

    assert clients[0].closed is True


@pytest.mark.asyncio
async def test_gateway_bootstrap_retries_stale_round_with_shared_deadline(
    tmp_path, monkeypatch,
):
    now = [0.0]
    clients = []

    class ClockedRegistry(WorkerRegistry):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, clock=lambda: now[0])

    class Client(_BootstrapClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.first_round = not clients
            self.resource_completed = asyncio.Event()

        async def get_capabilities(self, instance):
            if self.first_round and instance == "p0":
                await self.resource_completed.wait()
                await asyncio.sleep(0)
                now[0] = 2.1
            return await super().get_capabilities(instance)

        async def get_resources(self, instance):
            report = await super().get_resources(instance)
            if self.first_round and instance == "p0":
                self.resource_completed.set()
            return report

    def factory(*args, **kwargs):
        client = Client(*args, **kwargs)
        clients.append(client)
        return client

    async def no_wait(_delay):
        return None

    monkeypatch.setattr("prism_serve.router.http_rpc.HttpInferClient", factory)
    monkeypatch.setattr(
        "prism_serve.router.worker_registry.WorkerRegistry", ClockedRegistry,
    )
    monkeypatch.setattr(asyncio, "sleep", no_wait)
    app = FastAPI()
    deadline = asyncio.get_running_loop().time() + 1.0

    await _run_gateway_bootstrap_stage(
        lambda: _bootstrap_week12_http_control(
            app, _bootstrap_config(tmp_path), deadline=deadline,
        ),
        stage="worker world bootstrap",
        deadline=deadline,
        retry_interval_s=0.01,
    )

    assert len(clients) == 2
    assert clients[0].closed is True
    assert clients[1].closed is False
    assert app.state.worker_registry.world_fresh() is True


@pytest.mark.asyncio
async def test_replacement_report_round_rejects_slow_peer_skew():
    now = [0.0]
    first_completed = asyncio.Event()
    registry = _ready_registry(clock=lambda: now[0])

    class Client:
        async def get_resources(self, instance):
            if instance == "p0":
                first_completed.set()
            elif instance == "p1":
                await first_completed.wait()
                await asyncio.sleep(0)
                now[0] = 2.1
            return {
                "instance_epoch": f"pod-{instance}:boot-a",
                "complete": True,
                "resources": {},
                "operation_ids": [],
                "resource_ids": [],
            }

    app = FastAPI()
    app.state.http_infer_client = Client()

    with pytest.raises(ValueError, match="world is not fresh"):
        await _pull_replacement_resource_reports(
            app,
            registry,
            excluded_operation_ids=set(),
            excluded_resource_ids=set(),
        )

    assert registry.resource_signal("p0").received_at == 0.0
    assert registry.world_fresh() is False


@pytest.mark.asyncio
async def test_post_owner_resource_round_retries_stale_then_fresh(
    monkeypatch,
):
    now = [0.0]
    first_completed = asyncio.Event()
    calls = {name: 0 for name in ("p0", "p1", "d0", "d1")}
    registry = _ready_registry(clock=lambda: now[0])

    class Client:
        async def get_resources(self, instance):
            calls[instance] += 1
            if calls[instance] == 1 and instance == "p0":
                first_completed.set()
            elif calls[instance] == 1 and instance == "p1":
                await first_completed.wait()
                await asyncio.sleep(0)
                now[0] = 2.1
            return {
                "instance_epoch": f"pod-{instance}:boot-a",
                "complete": True,
                "resources": {},
                "operation_ids": [],
                "resource_ids": [],
            }

    async def no_wait(_delay):
        return None

    app = FastAPI()
    app.state.http_infer_client = Client()
    monkeypatch.setattr(asyncio, "sleep", no_wait)
    deadline = asyncio.get_running_loop().time() + 1.0

    reports = await _run_gateway_bootstrap_stage(
        lambda: _pull_replacement_resource_reports(
            app,
            registry,
            excluded_operation_ids=set(),
            excluded_resource_ids=set(),
            incomplete_is_not_ready=True,
        ),
        stage="post-owner resource reports",
        deadline=deadline,
        retry_interval_s=0.01,
    )

    assert set(reports) == {"p0", "p1", "d0", "d1"}
    assert calls == {"p0": 2, "p1": 2, "d0": 2, "d1": 2}
    assert registry.world_fresh() is True


@pytest.mark.asyncio
async def test_gateway_bootstrap_stages_share_one_absolute_deadline(
    monkeypatch,
):
    now = [0.0]

    class Clock:
        def time(self):
            return now[0]

    async def advance(delay):
        now[0] += delay

    first_attempts = 0

    async def first_stage():
        nonlocal first_attempts
        first_attempts += 1
        if first_attempts == 1:
            raise ConnectionError("worker not listening")
        return "ready"

    second_attempts = 0

    async def second_stage():
        nonlocal second_attempts
        second_attempts += 1
        raise ConnectionError("owner status unavailable")

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: Clock())
    monkeypatch.setattr(asyncio, "sleep", advance)

    assert await _run_gateway_bootstrap_stage(
        first_stage,
        stage="worker evidence",
        deadline=1.0,
        retry_interval_s=0.6,
    ) == "ready"
    with pytest.raises(RuntimeError, match="owner reconciliation"):
        await _run_gateway_bootstrap_stage(
            second_stage,
            stage="owner reconciliation",
            deadline=1.0,
            retry_interval_s=0.6,
        )

    assert first_attempts == 2
    assert second_attempts == 1
    assert now[0] == 1.0


@pytest.mark.asyncio
async def test_gateway_bootstrap_treats_404_as_fatal_without_retry(
    tmp_path, monkeypatch,
):
    error = InferRPCError(404, "NOT_FOUND", "missing endpoint")

    class Client(_BootstrapClient):
        transient_error = error

    clients = []

    def factory(*args, **kwargs):
        client = Client(*args, **kwargs)
        clients.append(client)
        return client

    async def forbidden_sleep(_delay):
        raise AssertionError("fatal bootstrap error must not sleep or retry")

    monkeypatch.setattr("prism_serve.router.http_rpc.HttpInferClient", factory)
    monkeypatch.setattr(asyncio, "sleep", forbidden_sleep)

    with pytest.raises(InferRPCError) as caught:
        await _bootstrap_week12_http_control(
            FastAPI(), _bootstrap_config(tmp_path),
        )

    assert caught.value.status_code == 404
    assert clients[0].attempts["p0"] == 1
    assert clients[0].closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "malformed",
        "wrong_generation",
        "wrong_digest",
        "duplicate_member",
        "wrong_epoch",
        "attestation_conflict",
    ],
)
async def test_gateway_bootstrap_structural_conflict_is_fatal_without_retry(
    tmp_path, monkeypatch, mutation,
):
    class Client(_BootstrapClient):
        pass

    Client.mutation = mutation
    clients = []

    def factory(*args, **kwargs):
        client = Client(*args, **kwargs)
        clients.append(client)
        return client

    async def forbidden_sleep(_delay):
        raise AssertionError("structural bootstrap error must not retry")

    monkeypatch.setattr("prism_serve.router.http_rpc.HttpInferClient", factory)
    monkeypatch.setattr(asyncio, "sleep", forbidden_sleep)

    with pytest.raises((KeyError, TypeError, ValueError, RuntimeError)):
        await _bootstrap_week12_http_control(
            FastAPI(), _bootstrap_config(tmp_path),
        )

    assert all(value == 1 for value in clients[0].attempts.values())
    assert clients[0].closed is True


@pytest.mark.asyncio
async def test_gateway_bootstrap_propagates_cancellation_and_closes_client(
    tmp_path, monkeypatch,
):
    class Client(_BootstrapClient):
        async def get_identity(self, instance):
            raise asyncio.CancelledError

    clients = []

    def factory(*args, **kwargs):
        client = Client(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("prism_serve.router.http_rpc.HttpInferClient", factory)

    with pytest.raises(asyncio.CancelledError):
        await _bootstrap_week12_http_control(
            FastAPI(), _bootstrap_config(tmp_path),
        )

    assert clients[0].closed is True


@pytest.mark.asyncio
async def test_gateway_owner_bootstrap_retries_transient_but_rejects_foreign_owner(
    monkeypatch,
):
    class Client:
        def __init__(self):
            self.owners = {name: "old" for name in ("p0", "p1", "d0", "d1")}

        async def owner_status(self, instance):
            return {"active_owner": self.owners[instance]}

    client = Client()
    calls = []

    async def activate(_client, _instances, new_owner, **_kwargs):
        calls.append("activate")
        if len(calls) == 1:
            raise TimeoutError("transient owner status")
        for instance in client.owners:
            client.owners[instance] = new_owner
        return "audit"

    async def no_wait(_delay):
        return None

    monkeypatch.setattr(
        "prism_serve.router.network_rpc.activate_replacement_owner", activate,
    )
    monkeypatch.setattr(asyncio, "sleep", no_wait)

    audit = await _bootstrap_replacement_owner(
        client,
        ("p0", "p1", "d0", "d1"),
        "new",
        retry_interval_s=0.001,
    )
    assert audit == "audit"
    assert calls == ["activate", "activate"]

    foreign_client = Client()
    foreign_calls = []

    async def activate_then_drift(fenced_client, _instances, _new_owner, **_kwargs):
        foreign_calls.append("activate")
        if len(foreign_calls) == 1:
            foreign_client.owners["d1"] = "foreign"
            raise TimeoutError("response lost after owner drift")
        await fenced_client.owner_status("d1")
        raise AssertionError("foreign owner must fail before activation")

    monkeypatch.setattr(
        "prism_serve.router.network_rpc.activate_replacement_owner",
        activate_then_drift,
    )

    with pytest.raises(RuntimeError, match="foreign owner"):
        await _bootstrap_replacement_owner(
            foreign_client,
            ("p0", "p1", "d0", "d1"),
            "new",
            retry_interval_s=0.001,
        )
    assert foreign_calls == ["activate", "activate"]

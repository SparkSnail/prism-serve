from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import tempfile
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from prism_serve.gateway.app import (
    _accept_replacement_topology,
    _background_control_plane_tasks_healthy,
    _begin_topology_acceptance,
    _finalize_topology_acceptance,
    _refresh_worker_world_once,
    chat_completions,
    healthz,
    readyz,
    topology_status,
)
from prism_serve.gateway.output import GatewayOutputBuffer
from prism_serve.gateway.topology_admin import (
    RestartRunRecord, TopologyAcceptanceLedger, parse_worker_identity,
    worker_identity_wire,
)
from prism_serve.metrics.collector import NullMetrics
from prism_serve.router.prefix_index import PrefixIndex
from prism_serve.router.worker_registry import PairCapability, WorkerIdentity, WorkerRegistry
from prism_serve.scheduler.resource_release import ResourceReleaseEvaluator
from prism_serve.scheduler.replacement_store import ReplacementDecisionStore
from prism_serve.scheduler.replacement_store import RetiredReplacementRun
from prism_serve.scheduler.replacement_store import UnknownReplacementRun
from prism_serve.scheduler.scheduler import PDScheduler
from prism_serve.scheduler.sequence_state import RequestInfo, RequestTracker
from prism_serve.scheduler.transfer_governor import TransferGovernor


MEMBERS = ("p0", "p1", "d0", "d1")
PAIRS = (("p0", "d0"), ("p0", "d1"), ("p1", "d0"), ("p1", "d1"), ("d0", "d1"))


def _identity(name, generation, suffix):
    return WorkerIdentity(
        name, "prefill" if name.startswith("p") else "decode", generation,
        f"pod-{suffix}-{name}", f"boot-{suffix}", f"http://{name}:8001",
        {"p0": 0, "p1": 1, "d0": 2, "d1": 3}[name],
        f"sha256:topology-{generation}", "kv-a",
    )


def _cap(source, target, identities):
    return PairCapability(
        f"{source}--{target}", identities[source].instance_epoch,
        identities[target].instance_epoch, "NCCL_SOCKET", "probe", True,
        f"evidence/{source}--{target}.json",
    )


def _ledger():
    identities = {name: _identity(name, "world-a", "old") for name in MEMBERS}
    registry = WorkerRegistry(expected_topology_generation="world-a")
    assert registry.install_world(
        list(identities.values()), [_cap(a, b, identities) for a, b in PAIRS]
    )
    return TopologyAcceptanceLedger(registry), identities


def _evidence(
    old,
    *,
    restart_run_id="run-1",
    old_generation="world-a",
    new_generation="world-b",
    new_suffix="new",
):
    new = {
        name: _identity(name, new_generation, new_suffix)
        for name in MEMBERS
    }
    old_ops = ["op-old"]
    old_resources = ["block-old"]
    reports = {
        name: {
            "instance_epoch": identity.instance_epoch, "complete": True,
            "resources": {}, "operation_ids": [], "resource_ids": [],
            "excluded_operation_ids": old_ops,
            "excluded_resource_ids": old_resources,
        }
        for name, identity in new.items()
    }
    return {
        "restart_run_id": restart_run_id,
        "old_topology_generation": old_generation,
        "new_topology_generation": new_generation,
        "termination_records": [
            _termination_record(
                name,
                identity,
                sequence_base=(index * 2) + 1,
                topology_generation=old_generation,
            )
            for index, (name, identity) in enumerate(old.items())
        ],
        "identities": [worker_identity_wire(value) for value in new.values()],
        "pair_capabilities": [asdict(_cap(a, b, new)) for a, b in PAIRS],
        "resource_reports": reports,
        "old_operation_ids": old_ops, "old_resource_ids": old_resources,
    }


def _termination_record(
    name, identity, *, sequence_base, topology_generation="world-a"
):
    record = {
        "logical_instance_id": name,
        "topology_generation": topology_generation,
        "pod_uid": identity.pod_uid,
        "node_uid": f"node-{name}",
        "container_name": "worker",
        "captured_container_id": f"container-{name}",
        "process_generation": identity.process_generation,
        "watch_start_resource_version": "100",
        "observed_resource_version": "101",
        "deletion_resource_version": "102",
        "restart_count_before": 0,
        "restart_count_observed": 0,
        "termination_source": "state.terminated",
        "termination_event_type": "MODIFIED",
        "deletion_event_type": "DELETED",
        "terminated": {
            "exit_code": 0,
            "reason": "Completed",
            "signal": 0,
            "started_at": "2026-07-23T11:59:59Z",
            "finished_at": "2026-07-23T12:00:00Z",
        },
        "adjacent_current_container_id": None,
        "pod_deletion_observed": True,
        "raw_pod_json_sha256": (
            "sha256:" + f"{sequence_base:064x}"
        ),
        "termination_raw_observation_sequence": sequence_base,
        "deletion_raw_pod_json_sha256": (
            "sha256:" + f"{sequence_base + 1:064x}"
        ),
        "deletion_raw_observation_sequence": sequence_base + 1,
    }
    record["observation_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            record, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return record


def _assert_accept_rejected_atomically(ledger, evidence, message):
    registry = ledger.registry
    records = dict(ledger.records)

    with pytest.raises(ValueError, match=message):
        ledger.accept(evidence)

    assert ledger.registry is registry
    assert ledger.registry.expected_topology_generation == "world-a"
    assert dict(ledger.records) == records


def test_topology_accept_requires_four_fresh_exclusion_reports():
    ledger, old = _ledger()
    evidence = _evidence(old)
    evidence["resource_reports"].pop("d1")
    with pytest.raises(ValueError, match="four resource reports"):
        ledger.accept(evidence)
    assert ledger.registry.expected_topology_generation == "world-a"


def test_topology_accept_atomically_installs_new_world_and_replays():
    ledger, old = _ledger()
    evidence = _evidence(old)
    first, registry = ledger.accept(evidence)
    replay, replay_registry = ledger.accept(evidence)
    assert first == replay
    assert registry is replay_registry
    assert registry.expected_topology_generation == "world-b"
    assert set(registry.members) == set(MEMBERS)


def test_topology_accept_parses_real_wire_identity_with_derived_epoch():
    ledger, old = _ledger()
    evidence = _evidence(old)
    for identity in evidence["identities"]:
        identity["instance_epoch"] = (
            f"{identity['pod_uid']}:{identity['process_generation']}"
        )

    _, registry = ledger.accept(evidence)

    assert registry.expected_topology_generation == "world-b"
    assert {
        name: identity.instance_epoch
        for name, identity in registry.members.items()
    } == {
        value["instance_id"]: value["instance_epoch"]
        for value in evidence["identities"]
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda identity: identity.update(instance_epoch="wrong:epoch"),
            "instance_epoch does not match",
        ),
        (
            lambda identity: identity.update(unexpected="value"),
            "unknown fields",
        ),
        (
            lambda identity: identity.pop("pod_uid"),
            "missing required fields",
        ),
        (
            lambda identity: identity.update(pod_uid=123),
            "string fields must be strings",
        ),
        (
            lambda identity: identity.update(global_rank=True),
            "global_rank must be an integer",
        ),
        (
            lambda identity: identity.update(instance_epoch=123),
            "instance_epoch must be a string",
        ),
    ],
)
def test_topology_accept_rejects_invalid_wire_identity_atomically(
    mutate, message,
):
    ledger, old = _ledger()
    evidence = _evidence(old)
    identity = evidence["identities"][0]
    identity["instance_epoch"] = (
        f"{identity['pod_uid']}:{identity['process_generation']}"
    )
    mutate(identity)

    _assert_accept_rejected_atomically(ledger, evidence, message)


def test_topology_accept_rejects_non_object_wire_identity_atomically():
    ledger, old = _ledger()
    evidence = _evidence(old)
    evidence["identities"][0] = "not-an-object"

    _assert_accept_rejected_atomically(ledger, evidence, "must be an object")


def test_topology_accept_same_run_different_evidence_conflicts():
    ledger, old = _ledger()
    evidence = _evidence(old)
    ledger.accept(evidence)
    evidence["old_resource_ids"] = ["other"]
    with pytest.raises(ValueError, match="different evidence"):
        ledger.accept(evidence)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.update(node_uid=""),
            "node_uid",
        ),
        (
            lambda value: value.update(captured_container_id=""),
            "captured_container_id",
        ),
        (
            lambda value: value.update(process_generation=""),
            "process_generation",
        ),
        (
            lambda value: value["terminated"].update(exit_code="0"),
            "exit_code",
        ),
    ],
)
def test_topology_accept_requires_full_physical_termination_proof(
    mutate, message,
):
    ledger, old = _ledger()
    evidence = _evidence(old)
    mutate(evidence["termination_records"][0])
    _reseal_termination_record(evidence["termination_records"][0])

    with pytest.raises(ValueError, match=message):
        ledger.accept(evidence)


def _reseal_termination_record(record):
    payload = {
        key: value
        for key, value in record.items()
        if key != "observation_sha256"
    }
    record["observation_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


@pytest.mark.parametrize(
    ("mutate", "reseal", "message"),
    [
        (
            lambda value: value.update(
                observation_sha256="sha256:" + "f" * 64
            ),
            False,
            "observation_sha256",
        ),
        (
            lambda value: value.update(
                topology_generation="foreign-world"
            ),
            True,
            "topology generation",
        ),
        (
            lambda value: value.update(
                deletion_resource_version=value[
                    "watch_start_resource_version"
                ],
            ),
            True,
            "resourceVersion",
        ),
        (
            lambda value: value.update(
                termination_source="lastState.terminated",
                restart_count_observed=99,
                adjacent_current_container_id="new-container",
            ),
            True,
            "restartCount",
        ),
        (
            lambda value: value.update(
                termination_source="lastState.terminated",
                restart_count_observed=1,
                adjacent_current_container_id=None,
            ),
            True,
            "adjacent current container",
        ),
        (
            lambda value: value.update(
                deletion_raw_observation_sequence=(
                    value["termination_raw_observation_sequence"]
                )
            ),
            True,
            "raw transition",
        ),
        (
            lambda value: value.update(pod_uid="foreign-pod"),
            True,
            "old pod uid",
        ),
    ],
    ids=[
        "digest-tamper",
        "generation-drift",
        "rv-not-monotonic",
        "restart-jump",
        "missing-adjacent",
        "raw-transitions-not-linked",
        "old-pod-drift",
    ],
)
def test_gateway_recomputes_and_validates_full_termination_authority(
    mutate,
    reseal,
    message,
):
    ledger, old = _ledger()
    evidence = _evidence(old)
    record = evidence["termination_records"][0]
    mutate(record)
    if reseal:
        _reseal_termination_record(record)

    _assert_accept_rejected_atomically(
        ledger,
        evidence,
        message,
    )


def test_gateway_accepts_opaque_resource_versions_and_equal_pod_digests():
    ledger, old = _ledger()
    evidence = _evidence(old)
    for index, record in enumerate(evidence["termination_records"]):
        record.update({
            "watch_start_resource_version": f"rv/start/{index}",
            "observed_resource_version":
                f"rv/terminated/{index}",
            "deletion_resource_version": f"rv/terminated/{index}",
            "deletion_raw_pod_json_sha256":
                record["raw_pod_json_sha256"],
        })
        _reseal_termination_record(record)

    accepted, registry = ledger.accept(evidence)

    assert accepted.accepted is True
    assert registry.expected_topology_generation == "world-b"


def test_topology_accept_allows_explicitly_empty_idle_world_snapshot():
    ledger, old = _ledger()
    evidence = _evidence(old)
    evidence["old_operation_ids"] = []
    evidence["old_resource_ids"] = []
    for report in evidence["resource_reports"].values():
        report["excluded_operation_ids"] = []
        report["excluded_resource_ids"] = []

    record, registry = ledger.accept(evidence)

    assert record.accepted is True
    assert registry.expected_topology_generation == "world-b"


@pytest.mark.parametrize("field", ["identities", "pair_capabilities"])
def test_topology_accept_rejects_duplicate_identity_or_pair_evidence(field):
    ledger, old = _ledger()
    evidence = _evidence(old)
    evidence[field].append(dict(evidence[field][0]))

    with pytest.raises(ValueError, match="four unique|five pair"):
        ledger.accept(evidence)


class _ReplacementClient:
    def __init__(
        self,
        *,
        fail_activate=False,
        lose_activate_response_once=False,
        resource_suffix="new",
    ):
        self.fail_activate = fail_activate
        self.lose_activate_response_once = lose_activate_response_once
        self.resource_suffix = resource_suffix
        self.owners = {name: None for name in MEMBERS}
        self.mutations = []
        self.resource_report_calls = 0

    async def owner_status(self, instance):
        return {"active_owner": self.owners[instance]}

    async def activate_owner(self, instance, owner):
        if self.fail_activate:
            raise RuntimeError("owner activation failed")
        self.owners[instance] = owner
        if self.lose_activate_response_once:
            self.lose_activate_response_once = False
            raise TimeoutError("activation response lost")
        return {"active_owner": owner}

    async def list_operations(self, instance, owner):
        return {"operations": []}

    async def get_resources(self, instance):
        self.resource_report_calls += 1
        return {
            "instance_epoch": (
                f"pod-{self.resource_suffix}-{instance}:"
                f"boot-{self.resource_suffix}"
            ),
            "complete": True,
            "resources": {},
            "operation_ids": [],
            "resource_ids": [],
        }

    async def retire_owner(self, instance, owner):
        self.owners[instance] = None
        return {"active_owner": None}

    async def finalize_release(self, *args, **kwargs):
        return {"resources_held_after": False}

    async def prepare_request(self, instance, ref, payload):
        self.mutations.append((instance, ref, payload))
        return {"state": "RUNNING", "result": {"dst_block_ids": [1]}}


def _runtime(ledger, client, *, replacement_store=None):
    async def empty_poll(_subject):
        return []

    app = FastAPI()
    scheduler = PDScheduler({})
    for name, identity in ledger.registry.members.items():
        scheduler.register_instance(
            name, identity.role,
            max_slots=2 if identity.role == "decode" else 0,
            instance_epoch=identity.instance_epoch,
        )
    app.state.accepting = True
    app.state.topology_admin = ledger
    app.state.worker_registry = ledger.registry
    app.state.scheduler = scheduler
    app.state.http_infer_client = client
    app.state.metrics = NullMetrics()
    app.state.queue = SimpleNamespace(
        owner_id="gateway-a:boot-a", poll=empty_poll
    )
    app.state.operation_allocator = SimpleNamespace(
        owner_generation="gateway-a:boot-a"
    )
    app.state.network_control = object()
    app.state.governor = SimpleNamespace(infer_client=object())
    if replacement_store is None:
        app.state._replacement_store_tempdir = tempfile.TemporaryDirectory()
        replacement_store = ReplacementDecisionStore(
            app.state._replacement_store_tempdir.name
        )
    app.state.replacement_store = replacement_store
    app.state.resource_release_evaluator = ResourceReleaseEvaluator(
        scheduler, client.finalize_release, app.state.metrics,
        replacement_store=app.state.replacement_store,
    )
    app.state.tracker = RequestTracker(app.state.metrics)
    app.state.output_buffer = GatewayOutputBuffer()
    return app


def _registry_from_replacement_evidence(evidence):
    registry = WorkerRegistry(
        expected_topology_generation=evidence["new_topology_generation"]
    )
    identities = [
        parse_worker_identity(value) for value in evidence["identities"]
    ]
    capabilities = [
        PairCapability(**value) for value in evidence["pair_capabilities"]
    ]
    assert registry.install_world(identities, capabilities)
    assert all(
        registry.update_resource_report(name, report)
        for name, report in evidence["resource_reports"].items()
    )
    return registry


CONFIG = {"operation_query_interval_ms": 1, "kv_transfer_timeout_s": 1.0}


@pytest.mark.asyncio
async def test_topology_status_pulls_live_reports_instead_of_cached_snapshot():
    ledger, _old = _ledger()

    class LiveClient(_ReplacementClient):
        async def get_resources(self, instance):
            identity = ledger.registry.members[instance]
            return {
                "instance_epoch": identity.instance_epoch,
                "complete": True,
                "resources": {"TARGET_PENDING": 1 if instance == "d1" else 0},
                "operation_ids": ["fresh-op"] if instance == "d1" else [],
                "resource_ids": ["block-fresh"] if instance == "d1" else [],
            }

    app = _runtime(ledger, LiveClient())
    cached = ledger.registry.resource_signal("d1").report
    assert cached is None or "fresh-op" not in cached.get("operation_ids", ())
    request = Request({"type": "http", "app": app, "headers": []})

    value = await topology_status(request)

    assert isinstance(value, dict)
    assert value["resource_reports"]["d1"]["operation_ids"] == ["fresh-op"]
    assert ledger.registry.resource_signal("d1").report["operation_ids"] == [
        "fresh-op"
    ]


@pytest.mark.asyncio
async def test_runtime_accept_rebuilds_generation_consumers_and_releases_old_quarantine():
    ledger, old = _ledger()
    evidence = _evidence(old)
    client = _ReplacementClient()
    app = _runtime(ledger, client)
    lease = app.state.scheduler.reserve_decode_slot("d0", "r-old", "op-old")
    app.state.scheduler.quarantine_decode_slot("op-old")
    old_evaluator = app.state.resource_release_evaluator

    record = await _accept_replacement_topology(app, evidence, CONFIG)

    assert record.new_topology_generation == "world-b"
    assert app.state.worker_registry.expected_topology_generation == "world-b"
    assert app.state.operation_allocator.topology_generation == "world-b"
    assert app.state.network_control.worker_epochs == {
        name: identity.instance_epoch
        for name, identity in app.state.worker_registry.members.items()
    }
    assert lease.state == "RELEASED"
    assert old_evaluator._replacement_records == {}
    assert app.state.replacement_store.seal_count == 1
    assert app.state.replacement_store.transition_closed is True
    assert app.state.accepting is True
    assert client.resource_report_calls == 12

    ref, _ = await app.state.network_control.prepare_normal_request(
        "d0", app.state.network_control.worker_epochs["d0"],
        "request-new", [1], {},
    )
    assert ref.topology_generation == "world-b"
    assert ref.target_worker_epoch == app.state.worker_registry.members["d0"].instance_epoch
    assert ref.operation_seq == 1


@pytest.mark.asyncio
async def test_runtime_accept_publishes_four_prefix_reports_after_fresh_resources():
    class PrefixReplacementClient(_ReplacementClient):
        def __init__(self):
            super().__init__()
            self.events = []

        async def get_resources(self, instance):
            self.events.append(("resource", instance))
            report = await super().get_resources(instance)
            report["max_slots"] = 7 if instance.startswith("d") else 0
            return report

        async def _request(self, method, instance, path, **kwargs):
            if method == "GET" and path.startswith("/v1/prefix/events?"):
                return {"events": []}
            if method == "POST" and path == "/v1/prefix/events/ack":
                return {}
            assert method == "POST"
            assert path == "/v1/prefix/reports/register"
            self.events.append(("prefix", instance))
            return {
                "instance_id": instance,
                "instance_epoch": f"pod-new-{instance}:boot-new",
                "snapshot_seq_no": 0,
                "locations": [],
            }

    class Coordinator:
        def __init__(self):
            self.rpc = object()
            self.operation_allocator = object()
            self._contexts = {}

        async def shutdown(self):
            return None

    ledger, old = _ledger()
    evidence = _evidence(old)
    for instance, report in evidence["resource_reports"].items():
        report["max_slots"] = 2 if instance.startswith("d") else 0
    client = PrefixReplacementClient()
    runtime = _runtime(ledger, client)
    runtime.state.runtime_config = {
        "multinode_e2e_enabled": True, "affinity_enabled": True
    }
    runtime.state.resource_refresh_task = asyncio.create_task(
        asyncio.Event().wait()
    )
    runtime.state.prefix_index = PrefixIndex()
    runtime.state.affinity_coordinator = Coordinator()
    config = {
        **CONFIG,
        "prefix_block_size": 256,
        "prefix_block_bytes": 1024,
        "prefix_event_poll_interval_ms": 1000,
    }

    await _accept_replacement_topology(runtime, evidence, config)
    try:
        publication = runtime.state.prefix_world_publication
        assert len(publication.reports) == 4
        assert {
            value.instance_id: value.instance_epoch for value in publication.reports
        } == {
            name: identity.instance_epoch
            for name, identity in runtime.state.worker_registry.members.items()
        }
        assert runtime.state.prefix_reconciler.world_ready({
            name: identity.instance_epoch
            for name, identity in runtime.state.worker_registry.members.items()
        })
        assert runtime.state.owner_takeover_audit.instances == tuple(sorted(MEMBERS))
        assert runtime.state.scheduler.decode_free_slots() == {"d0": 7, "d1": 7}
        assert runtime.state.accepting is True
        kinds = [kind for kind, _ in client.events]
        prefix_indexes = [
            index for index, kind in enumerate(kinds) if kind == "prefix"
        ]
        assert len(prefix_indexes) == 4
        assert kinds[:prefix_indexes[0]].count("resource") == 4
        assert kinds[prefix_indexes[-1] + 1:].count("resource") >= 8
    finally:
        runtime.state.reconciler_task.cancel()
        runtime.state.resource_refresh_task.cancel()
        await asyncio.gather(
            runtime.state.reconciler_task,
            runtime.state.resource_refresh_task,
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_replacement_rejects_nonpositive_prefix_poll_before_pending():
    ledger, old = _ledger()
    app = _runtime(ledger, _ReplacementClient())
    app.state.affinity_coordinator = object()
    original_registry = app.state.worker_registry

    with pytest.raises(
        ValueError,
        match="prefix_event_poll_interval_ms must be positive",
    ):
        await _accept_replacement_topology(
            app,
            _evidence(old),
            {**CONFIG, "prefix_event_poll_interval_ms": 0},
        )

    assert app.state.worker_registry is original_registry
    assert ledger.registry is original_registry
    assert getattr(app.state, "topology_acceptance_required", False) is False
    assert getattr(app.state, "pending_topology_generation", None) is None
    assert app.state.accepting is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param("success", id="success"),
        pytest.param("failure", id="failure-before-new-task"),
        pytest.param("cancel", id="cancel-before-new-task"),
    ],
)
async def test_reconciler_replacement_window_keeps_only_liveness(
    outcome,
):
    activation_started = asyncio.Event()
    release_activation = asyncio.Event()

    class PausedPrefixClient(_ReplacementClient):
        async def activate_owner(self, instance, owner):
            activation_started.set()
            await release_activation.wait()
            if outcome == "failure":
                raise RuntimeError("owner activation failed")
            return await super().activate_owner(instance, owner)

        async def _request(self, method, instance, path, **kwargs):
            if method == "GET" and path.startswith("/v1/prefix/events?"):
                return {"events": []}
            if method == "POST" and path == "/v1/prefix/events/ack":
                return {}
            assert method == "POST"
            assert path == "/v1/prefix/reports/register"
            return {
                "instance_id": instance,
                "instance_epoch": f"pod-new-{instance}:boot-new",
                "snapshot_seq_no": 0,
                "locations": [],
            }

    class Coordinator:
        def __init__(self):
            self.rpc = object()
            self.operation_allocator = object()
            self._contexts = {}

        async def shutdown(self):
            return None

    ledger, old = _ledger()
    runtime = _runtime(ledger, PausedPrefixClient())
    runtime.state.runtime_config = {
        "multinode_e2e_enabled": True,
        "affinity_enabled": True,
    }
    runtime.state.queue.is_connected = True
    runtime.state.prefix_index = PrefixIndex()
    runtime.state.affinity_coordinator = Coordinator()
    runtime.state.resource_refresh_task = asyncio.create_task(
        asyncio.Event().wait()
    )
    old_reconciler_task = asyncio.create_task(asyncio.Event().wait())
    runtime.state.reconciler_task = old_reconciler_task
    config = {
        **CONFIG,
        "prefix_block_size": 256,
        "prefix_block_bytes": 1024,
        "prefix_event_poll_interval_ms": 1000,
    }
    request = Request({"type": "http", "app": runtime, "headers": []})
    acceptance_task = asyncio.create_task(
        _accept_replacement_topology(runtime, _evidence(old), config)
    )

    try:
        await asyncio.wait_for(activation_started.wait(), timeout=1.0)
        await asyncio.sleep(0)

        assert old_reconciler_task.done()
        assert runtime.state.topology_acceptance_task is acceptance_task
        assert runtime.state.reconciler_replacement_task is acceptance_task
        assert _background_control_plane_tasks_healthy(runtime) is False
        assert healthz(request).status_code == 200
        assert readyz(request).status_code == 503
        assert (await chat_completions(request)).status_code == 503
        with pytest.raises(
            RuntimeError,
            match="another topology acceptance is already running",
        ):
            await _accept_replacement_topology(
                runtime,
                _evidence(old),
                config,
            )
        assert runtime.state.topology_acceptance_task is acceptance_task
        assert runtime.state.reconciler_replacement_task is acceptance_task

        if outcome == "cancel":
            acceptance_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await acceptance_task
            assert runtime.state.topology_acceptance_task is None
            assert runtime.state.reconciler_replacement_task is None
            assert healthz(request).status_code == 503
        elif outcome == "failure":
            release_activation.set()
            with pytest.raises(RuntimeError, match="owner activation failed"):
                await acceptance_task
            assert runtime.state.topology_acceptance_task is None
            assert runtime.state.reconciler_replacement_task is None
            assert healthz(request).status_code == 503
        else:
            release_activation.set()
            await acceptance_task
            assert runtime.state.topology_acceptance_task is None
            assert runtime.state.reconciler_replacement_task is None
            assert healthz(request).status_code == 200
    finally:
        release_activation.set()
        if not acceptance_task.done():
            acceptance_task.cancel()
        tasks = {
            acceptance_task,
            old_reconciler_task,
            runtime.state.resource_refresh_task,
            getattr(runtime.state, "reconciler_task", None),
        }
        for task in tasks:
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in tasks if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failed_component", "message"),
    [
        pytest.param(
            "prefix_reconciler",
            "prefix world is not ready",
            id="prefix-reconciler",
        ),
        pytest.param(
            "resource_refresh",
            "background control plane task failed",
            id="resource-refresh",
        ),
    ],
)
async def test_topology_finalizer_requires_live_background_tasks(
    failed_component,
    message,
):
    _, old = _ledger()
    evidence = _evidence(old)
    registry = _registry_from_replacement_evidence(evidence)
    runtime = _runtime(
        TopologyAcceptanceLedger(registry),
        _ReplacementClient(),
    )
    expected_epochs = {
        name: identity.instance_epoch
        for name, identity in registry.members.items()
    }

    class Publication:
        def matches(self, observed):
            return observed == expected_epochs

    class Reconciler:
        world_publication = Publication()

        def world_ready(self, observed):
            return observed == expected_epochs

    live_task = asyncio.create_task(asyncio.Event().wait())
    stopped_task = asyncio.create_task(asyncio.sleep(0))
    await stopped_task
    runtime.state.runtime_config = {
        "multinode_e2e_enabled": True,
        "affinity_enabled": True,
    }
    runtime.state.prefix_reconciler = Reconciler()
    runtime.state.prefix_world_publication = (
        runtime.state.prefix_reconciler.world_publication
    )
    runtime.state.reconciler_task = (
        stopped_task
        if failed_component == "prefix_reconciler"
        else live_task
    )
    runtime.state.resource_refresh_task = (
        stopped_task
        if failed_component == "resource_refresh"
        else live_task
    )
    _begin_topology_acceptance(runtime, "world-b")

    try:
        with pytest.raises(RuntimeError, match=message):
            _finalize_topology_acceptance(runtime, "world-b")
        assert runtime.state.topology_acceptance_required is True
        assert runtime.state.pending_topology_generation == "world-b"
        assert runtime.state.accepting is False
    finally:
        live_task.cancel()
        await asyncio.gather(live_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_final_stale_round_keeps_candidate_starting_until_fresh_replay(
    monkeypatch,
):
    now = [0.0]

    class SlowFinalRoundClient(_ReplacementClient):
        def __init__(self):
            super().__init__()
            self.calls_by_instance = {name: 0 for name in MEMBERS}
            self.final_first_completed = asyncio.Event()

        async def get_resources(self, instance):
            self.calls_by_instance[instance] += 1
            call = self.calls_by_instance[instance]
            if call == 3 and instance == "p0":
                self.final_first_completed.set()
            elif call == 3 and instance == "p1":
                await self.final_first_completed.wait()
                await asyncio.sleep(0)
                now[0] = 2.1
            return await super().get_resources(instance)

    ledger, old = _ledger()
    original_stage = ledger.stage

    def stage_with_clock(body):
        record, candidate, replay = original_stage(body)
        candidate._clock = lambda: now[0]
        return record, candidate, replay

    monkeypatch.setattr(ledger, "stage", stage_with_clock)
    evidence = _evidence(old)
    client = SlowFinalRoundClient()
    app = _runtime(ledger, client)
    app.state.topology_acceptance_required = True
    app.state.pending_topology_generation = "world-b"

    with pytest.raises(ValueError, match="not fresh"):
        await _accept_replacement_topology(app, evidence, CONFIG)

    assert app.state.worker_registry.expected_topology_generation == "world-b"
    assert ledger.registry is app.state.worker_registry
    assert app.state.replacement_store.transition_closed is True
    assert app.state.replacement_store.seal_count == 1
    assert app.state.topology_acceptance_required is True
    assert app.state.pending_topology_generation == "world-b"
    assert app.state.accepting is False
    assert client.calls_by_instance == {
        "p0": 3, "p1": 3, "d0": 3, "d1": 3,
    }

    record = await _accept_replacement_topology(app, evidence, CONFIG)

    assert record.restart_run_id == "run-1"
    assert app.state.replacement_store.seal_count == 1
    assert app.state.topology_acceptance_required is False
    assert app.state.pending_topology_generation is None
    assert app.state.accepting is True

    assert client.calls_by_instance == {
        "p0": 4, "p1": 4, "d0": 4, "d1": 4,
    }


@pytest.mark.asyncio
async def test_background_refresh_cannot_open_pending_candidate():
    _, old = _ledger()
    evidence = _evidence(old)
    registry = _registry_from_replacement_evidence(evidence)

    class RefreshClient(_ReplacementClient):
        async def get_identity(self, instance):
            return worker_identity_wire(registry.members[instance])

        async def get_capabilities(self, instance):
            return {
                "ready": True,
                "pairs": [
                    asdict(capability)
                    for pair_id, capability in registry.capabilities.items()
                    if instance in pair_id.split("--")
                ],
            }

    app = _runtime(TopologyAcceptanceLedger(registry), RefreshClient())
    app.state.governor = SimpleNamespace(
        quarantined_transfer_totals=lambda _operation_ids: (0, {})
    )
    app.state.topology_acceptance_required = True
    app.state.pending_topology_generation = "world-b"
    app.state.accepting = False

    await _refresh_worker_world_once(app)

    assert app.state.worker_registry.world_fresh() is True
    assert app.state.accepting is False

    app.state.topology_acceptance_required = False
    app.state.pending_topology_generation = None
    await _refresh_worker_world_once(app)
    assert app.state.accepting is True


@pytest.mark.asyncio
async def test_prefix_invalidation_during_final_pull_blocks_acceptance(
    monkeypatch,
):
    runtime = None
    calls_by_instance = {name: 0 for name in MEMBERS}

    class PrefixReplacementClient(_ReplacementClient):
        async def get_resources(self, instance):
            calls_by_instance[instance] += 1
            if calls_by_instance[instance] == 3 and instance == "p0":
                runtime.state.prefix_reconciler.world_publication = None
            return await super().get_resources(instance)

        async def _request(self, method, instance, path, **kwargs):
            if method == "GET" and path.startswith("/v1/prefix/events?"):
                return {"events": []}
            if method == "POST" and path == "/v1/prefix/events/ack":
                return {}
            assert method == "POST"
            assert path == "/v1/prefix/reports/register"
            return {
                "instance_id": instance,
                "instance_epoch": f"pod-new-{instance}:boot-new",
                "snapshot_seq_no": 0,
                "locations": [],
            }

    class Coordinator:
        def __init__(self):
            self.rpc = object()
            self.operation_allocator = object()
            self._contexts = {}

        async def shutdown(self):
            return None

    async def idle_reconciler(_self, _instances_provider, _interval_s):
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "prism_serve.router.reconciler.PrefixReconciler.run",
        idle_reconciler,
    )
    ledger, old = _ledger()
    evidence = _evidence(old)
    runtime = _runtime(ledger, PrefixReplacementClient())
    runtime.state.runtime_config = {
        "multinode_e2e_enabled": True,
        "affinity_enabled": True,
    }
    runtime.state.resource_refresh_task = asyncio.create_task(
        asyncio.Event().wait()
    )
    runtime.state.prefix_index = PrefixIndex()
    runtime.state.affinity_coordinator = Coordinator()
    config = {
        **CONFIG,
        "prefix_block_size": 256,
        "prefix_block_bytes": 1024,
        "prefix_event_poll_interval_ms": 1000,
    }

    try:
        with pytest.raises(RuntimeError, match="prefix world is not ready"):
            await _accept_replacement_topology(runtime, evidence, config)

        assert runtime.state.replacement_store.transition_closed is True
        assert runtime.state.topology_acceptance_required is True
        assert runtime.state.pending_topology_generation == "world-b"
        assert runtime.state.accepting is False
        assert calls_by_instance == {
            "p0": 3, "p1": 3, "d0": 3, "d1": 3,
        }
    finally:
        task = getattr(runtime.state, "reconciler_task", None)
        if task is not None:
            task.cancel()
        runtime.state.resource_refresh_task.cancel()
        await asyncio.gather(
            *(
                value
                for value in (
                    task,
                    runtime.state.resource_refresh_task,
                )
                if value is not None
            ),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_candidate_loop_failure_during_final_pull_blocks_acceptance():
    loop_failed = asyncio.Event()
    calls_by_instance = {name: 0 for name in MEMBERS}

    class FailingQueue:
        owner_id = "gateway-a:boot-a"

        async def poll(self, _subject):
            loop_failed.set()
            raise RuntimeError("candidate loop failed")

    class WaitingClient(_ReplacementClient):
        async def get_resources(self, instance):
            calls_by_instance[instance] += 1
            if calls_by_instance[instance] == 2 and instance == "p0":
                await loop_failed.wait()
                await asyncio.sleep(0)
            return await super().get_resources(instance)

    async def old_loop():
        await asyncio.Event().wait()

    ledger, old = _ledger()
    evidence = _evidence(old)
    app = _runtime(ledger, WaitingClient())
    app.state.queue = FailingQueue()
    app.state.loop_task = asyncio.create_task(old_loop())

    with pytest.raises(RuntimeError, match="control plane failed"):
        await _accept_replacement_topology(app, evidence, CONFIG)

    await asyncio.gather(app.state.loop_task, return_exceptions=True)
    assert app.state.control_plane_failed is True
    assert app.state.replacement_store.transition_closed is True
    assert app.state.topology_acceptance_required is True
    assert app.state.pending_topology_generation == "world-b"
    assert app.state.accepting is False


@pytest.mark.asyncio
async def test_runtime_accept_failure_keeps_old_snapshot_and_readiness_false():
    ledger, old = _ledger()
    evidence = _evidence(old)
    app = _runtime(ledger, _ReplacementClient(fail_activate=True))
    old_registry = app.state.worker_registry
    old_scheduler = app.state.scheduler
    old_allocator = app.state.operation_allocator
    old_network = app.state.network_control

    with pytest.raises(RuntimeError, match="owner activation failed"):
        await _accept_replacement_topology(app, evidence, CONFIG)

    assert app.state.accepting is False
    assert app.state.worker_registry is old_registry
    assert app.state.scheduler is old_scheduler
    assert app.state.operation_allocator is old_allocator
    assert app.state.network_control is old_network
    assert ledger.registry.expected_topology_generation == "world-a"
    assert ledger.records == {}


@pytest.mark.asyncio
async def test_incomplete_exclusions_reject_repeatedly_without_mutating_old_state():
    ledger, old = _ledger()
    evidence = _evidence(old)
    app = _runtime(ledger, _ReplacementClient())
    old_tracker = app.state.tracker
    old_output = app.state.output_buffer
    request_info = RequestInfo(req_id="r-live", active_operation_id="op-live")
    old_tracker.add(request_info)
    await old_output.apply_cumulative("r-live", [17], 1)

    for _ in range(2):
        with pytest.raises(ValueError, match="omits frozen operations"):
            await _accept_replacement_topology(app, evidence, CONFIG)

        assert old_tracker.all_requests() == [request_info]
        assert old_output.snapshot("r-live") == ([17], False, None)
        assert app.state.tracker is old_tracker
        assert app.state.output_buffer is old_output
        assert ledger.registry.expected_topology_generation == "world-a"
        assert ledger.records == {}


@pytest.mark.asyncio
async def test_commit_failure_cannot_release_old_quarantined_slot(monkeypatch):
    ledger, old = _ledger()
    evidence = _evidence(old)
    app = _runtime(ledger, _ReplacementClient())
    lease = app.state.scheduler.reserve_decode_slot("d0", "r-old", "op-old")
    app.state.scheduler.quarantine_decode_slot("op-old")
    original_commit = ledger.commit
    monkeypatch.setattr(
        ledger, "commit",
        lambda *args: (_ for _ in ()).throw(RuntimeError("commit failed")),
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        await _accept_replacement_topology(app, evidence, CONFIG)

    assert lease.state == "QUARANTINED"
    assert app.state.replacement_store.active_record_count == 1
    assert app.state.replacement_store.transition_closed is False
    assert ledger.registry.expected_topology_generation == "world-a"
    assert ledger.records == {}

    monkeypatch.setattr(ledger, "commit", original_commit)
    await _accept_replacement_topology(app, evidence, CONFIG)
    assert lease.state == "RELEASED"
    assert app.state.replacement_store.transition_closed is True


@pytest.mark.asyncio
async def test_persist_response_loss_keeps_slots_quarantined_and_replays(monkeypatch):
    ledger, old = _ledger()
    evidence = _evidence(old)
    evidence["old_operation_ids"] = ["op-old", "op-two"]
    for report in evidence["resource_reports"].values():
        report["excluded_operation_ids"] = ["op-old", "op-two"]
    app = _runtime(ledger, _ReplacementClient())
    first = app.state.scheduler.reserve_decode_slot("d0", "r1", "op-old")
    second = app.state.scheduler.reserve_decode_slot("d1", "r2", "op-two")
    app.state.scheduler.quarantine_decode_slot("op-old")
    app.state.scheduler.quarantine_decode_slot("op-two")
    store = app.state.replacement_store
    original = store.persist_records

    def persist_then_lose_response(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("directory fsync response lost")

    monkeypatch.setattr(store, "persist_records", persist_then_lose_response)
    with pytest.raises(OSError, match="response lost"):
        await _accept_replacement_topology(app, evidence, CONFIG)

    assert first.state == second.state == "QUARANTINED"
    assert store.active_record_count == 2
    assert store.transition_closed is False
    assert ledger.registry.expected_topology_generation == "world-a"
    assert ledger.records == {}

    monkeypatch.setattr(store, "persist_records", original)
    await _accept_replacement_topology(app, evidence, CONFIG)
    assert first.state == second.state == "RELEASED"
    assert store.transition_closed is True


@pytest.mark.asyncio
async def test_gateway_restart_recovers_and_seals_unsealed_replacement(monkeypatch):
    ledger, old = _ledger()
    evidence = _evidence(old)
    client = _ReplacementClient()
    old_app = _runtime(ledger, client)
    store = old_app.state.replacement_store
    store_root = store.root
    monkeypatch.setattr(
        store,
        "seal_run",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("gateway crashed before seal")
        ),
    )

    with pytest.raises(RuntimeError, match="crashed before seal"):
        await _accept_replacement_topology(old_app, evidence, CONFIG)

    assert store.transition_closed is False
    assert ledger.registry.expected_topology_generation == "world-b"
    assert old_app.state.worker_registry.expected_topology_generation == "world-b"
    assert old_app.state.topology_acceptance_required is True
    assert old_app.state.pending_topology_generation == "world-b"
    assert old_app.state.accepting is False

    reopened_store = ReplacementDecisionStore(store_root)
    recovered_ledger = TopologyAcceptanceLedger(ledger.registry)
    new_app = _runtime(
        recovered_ledger, client, replacement_store=reopened_store
    )
    record = await _accept_replacement_topology(new_app, evidence, CONFIG)

    assert record.restart_run_id == "run-1"
    assert recovered_ledger.records["run-1"] == record
    assert reopened_store.transition_closed is True
    assert reopened_store.seal_count == 1
    assert new_app.state.accepting is True


@pytest.mark.asyncio
async def test_gateway_crash_before_first_accept_recovers_empty_local_segment(
    monkeypatch,
):
    _, old = _ledger()
    evidence = _evidence(old)
    registry = _registry_from_replacement_evidence(evidence)
    ledger = TopologyAcceptanceLedger(registry)
    client = _ReplacementClient()
    app = _runtime(ledger, client)
    store = app.state.replacement_store
    app.state.topology_acceptance_required = True
    app.state.pending_topology_generation = "world-b"

    incomplete = deepcopy(evidence)
    incomplete["termination_records"][0]["process_generation"] = ""
    _reseal_termination_record(incomplete["termination_records"][0])
    with pytest.raises(ValueError, match="process_generation"):
        await _accept_replacement_topology(app, incomplete, CONFIG)
    assert store.seal_count == 0

    record = await _accept_replacement_topology(app, evidence, CONFIG)

    assert record.restart_run_id == "run-1"
    assert ledger.records == {"run-1": record}
    assert store.active_record_count == 0
    assert store.seal_count == 1
    assert store.seals()[0].new_topology_generation == "world-b"
    assert app.state.scheduler.replacement_decode_leases() == ()
    assert app.state.topology_acceptance_required is False
    assert app.state.accepting is True

    changed = {**evidence, "old_resource_ids": ["different"]}
    with pytest.raises(ValueError, match="different decision evidence"):
        await _accept_replacement_topology(app, changed, CONFIG)


@pytest.mark.asyncio
async def test_cold_recovery_seal_response_loss_replays_without_local_release(
    monkeypatch,
):
    _, old = _ledger()
    evidence = _evidence(old)
    registry = _registry_from_replacement_evidence(evidence)
    ledger = TopologyAcceptanceLedger(registry)
    app = _runtime(ledger, _ReplacementClient())
    store = app.state.replacement_store
    original = store.seal_run
    calls = 0

    def seal_then_lose_response(**kwargs):
        nonlocal calls
        calls += 1
        seal = original(**kwargs)
        if calls == 1:
            raise TimeoutError("seal response lost")
        return seal

    monkeypatch.setattr(store, "seal_run", seal_then_lose_response)
    with pytest.raises(TimeoutError, match="response lost"):
        await _accept_replacement_topology(app, evidence, CONFIG)
    assert store.transition_closed is True
    assert store.seal_count == 1

    record = await _accept_replacement_topology(app, evidence, CONFIG)
    assert record.restart_run_id == "run-1"
    assert store.active_record_count == 0
    assert store.seal_count == 1
    assert app.state.scheduler.replacement_decode_leases() == ()


@pytest.mark.asyncio
async def test_accept_response_loss_and_gateway_restart_replays_retained_seal(
    monkeypatch,
):
    ledger, old = _ledger()
    evidence = _evidence(old)
    client = _ReplacementClient()
    old_app = _runtime(ledger, client)

    completed = await _accept_replacement_topology(old_app, evidence, CONFIG)
    store_root = old_app.state.replacement_store.root
    assert completed.restart_run_id == "run-1"
    assert old_app.state.replacement_store.seal_count == 1

    reopened_store = ReplacementDecisionStore(store_root)
    recovered_ledger = TopologyAcceptanceLedger(old_app.state.worker_registry)
    new_app = _runtime(
        recovered_ledger, client, replacement_store=reopened_store
    )
    monkeypatch.setattr(
        reopened_store,
        "persist_records",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("sealed replay must not persist release records")
        ),
    )
    monkeypatch.setattr(
        reopened_store,
        "seal_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("sealed replay must not create another seal")
        ),
    )
    calls_before = client.resource_report_calls

    replay = await _accept_replacement_topology(new_app, evidence, CONFIG)

    assert replay == completed
    assert recovered_ledger.records == {"run-1": completed}
    assert client.resource_report_calls == calls_before + 4
    assert reopened_store.active_record_count == 0
    assert reopened_store.seal_count == 1
    assert new_app.state.accepting is True

    changed = {**evidence, "old_resource_ids": ["different"]}
    with pytest.raises(ValueError, match="different decision evidence"):
        await _accept_replacement_topology(new_app, changed, CONFIG)


@pytest.mark.asyncio
async def test_superseded_sealed_replay_is_retired_without_poisoning_current_world():
    ledger, old = _ledger()
    first = _evidence(old)
    client = _ReplacementClient()
    app = _runtime(ledger, client)

    await _accept_replacement_topology(app, first, CONFIG)
    world_b = dict(app.state.worker_registry.members)
    client.resource_suffix = "newer"
    second = _evidence(
        world_b,
        restart_run_id="run-2",
        old_generation="world-b",
        new_generation="world-c",
        new_suffix="newer",
    )
    await _accept_replacement_topology(app, second, CONFIG)

    registry_c = app.state.worker_registry
    with pytest.raises(RetiredReplacementRun) as retired:
        await _accept_replacement_topology(app, first, CONFIG)

    assert retired.value.seal.restart_run_id == "run-1"
    assert app.state.worker_registry is registry_c
    assert ledger.registry is registry_c
    assert registry_c.expected_topology_generation == "world-c"
    assert app.state.topology_acceptance_required is False
    assert app.state.pending_topology_generation is None
    assert app.state.accepting is True

    client.resource_suffix = "newest"
    third = _evidence(
        dict(registry_c.members),
        restart_run_id="run-3",
        old_generation="world-c",
        new_generation="world-d",
        new_suffix="newest",
    )
    accepted = await _accept_replacement_topology(app, third, CONFIG)
    assert accepted.new_topology_generation == "world-d"
    assert app.state.worker_registry.expected_topology_generation == "world-d"
    assert app.state.accepting is True

    registry_d = app.state.worker_registry
    with pytest.raises(UnknownReplacementRun):
        await _accept_replacement_topology(app, first, CONFIG)
    assert app.state.worker_registry is registry_d
    assert ledger.registry is registry_d
    assert app.state.topology_acceptance_required is False
    assert app.state.pending_topology_generation is None
    assert app.state.accepting is True

    unknown = {**first, "restart_run_id": "run-missing"}
    with pytest.raises(UnknownReplacementRun):
        await _accept_replacement_topology(app, unknown, CONFIG)
    assert app.state.worker_registry is registry_d
    assert ledger.registry is registry_d
    assert app.state.topology_acceptance_required is False
    assert app.state.pending_topology_generation is None
    assert app.state.accepting is True


@pytest.mark.asyncio
async def test_second_resource_pull_blocks_seal_if_old_operation_reappears():
    class ReappearingOperationClient(_ReplacementClient):
        async def get_resources(self, instance):
            report = await super().get_resources(instance)
            if self.resource_report_calls > 4 and instance == "d0":
                report["operation_ids"] = ["op-old"]
            return report

    ledger, old = _ledger()
    evidence = _evidence(old)
    client = ReappearingOperationClient()
    app = _runtime(ledger, client)
    lease = app.state.scheduler.reserve_decode_slot("d0", "r-old", "op-old")
    app.state.scheduler.quarantine_decode_slot("op-old")

    with pytest.raises(ValueError, match="still reports an old operation"):
        await _accept_replacement_topology(app, evidence, CONFIG)

    assert client.resource_report_calls == 8
    assert lease.state == "RELEASED"
    assert app.state.replacement_store.transition_closed is False
    assert app.state.worker_registry is ledger.registry
    assert app.state.worker_registry.expected_topology_generation == "world-b"
    assert app.state.topology_acceptance_required is True
    assert app.state.pending_topology_generation == "world-b"
    assert app.state.accepting is False


@pytest.mark.asyncio
async def test_same_evidence_replay_restores_ready_without_releasing_again(
    monkeypatch,
):
    ledger, old = _ledger()
    evidence = _evidence(old)
    app = _runtime(ledger, _ReplacementClient())
    lease = app.state.scheduler.reserve_decode_slot("d0", "r-old", "op-old")
    app.state.scheduler.quarantine_decode_slot("op-old")
    old_evaluator = app.state.resource_release_evaluator
    original_release = old_evaluator.release_persisted_replacement_batch
    release_calls = 0

    def counted_release(records):
        nonlocal release_calls
        release_calls += 1
        return original_release(records)

    async def old_loop():
        await asyncio.Event().wait()

    monkeypatch.setattr(
        old_evaluator,
        "release_persisted_replacement_batch",
        counted_release,
    )
    app.state.loop_task = asyncio.create_task(old_loop())
    await _accept_replacement_topology(app, evidence, CONFIG)
    candidate_loop = app.state.loop_task
    app.state.accepting = False

    try:
        replay = await _accept_replacement_topology(app, evidence, CONFIG)

        assert replay.restart_run_id == "run-1"
        assert app.state.accepting is True
        assert app.state.loop_task is candidate_loop
        assert app.state.replacement_store.seal_count == 1
        assert release_calls == 1
        assert lease.state == "RELEASED"
    finally:
        candidate_loop.cancel()
        await asyncio.gather(candidate_loop, return_exceptions=True)


@pytest.mark.asyncio
async def test_sealed_replay_rejects_old_app_runtime_and_stays_pending():
    ledger, old = _ledger()
    evidence = _evidence(old)
    app = _runtime(ledger, _ReplacementClient())
    old_registry = app.state.worker_registry

    await _accept_replacement_topology(app, evidence, CONFIG)
    candidate_registry = app.state.worker_registry
    app.state.worker_registry = old_registry

    with pytest.raises(
        ValueError,
        match="requires the installed candidate runtime",
    ):
        await _accept_replacement_topology(app, evidence, CONFIG)

    assert ledger.registry is candidate_registry
    assert app.state.worker_registry is old_registry
    assert app.state.topology_acceptance_required is True
    assert app.state.pending_topology_generation == "world-b"
    assert app.state.accepting is False


@pytest.mark.asyncio
async def test_activation_response_loss_readback_converges_to_atomic_publish():
    ledger, old = _ledger()
    evidence = _evidence(old)
    client = _ReplacementClient(lose_activate_response_once=True)
    app = _runtime(ledger, client)
    old_registry = app.state.worker_registry

    record = await _accept_replacement_topology(app, evidence, CONFIG)

    assert record.new_topology_generation == "world-b"
    assert app.state.worker_registry is not old_registry
    assert app.state.worker_registry.expected_topology_generation == "world-b"
    assert set(client.owners.values()) == {"gateway-a:boot-a"}
    assert tuple(ledger.records) == ("run-1",)
    assert app.state.accepting is True


@pytest.mark.asyncio
async def test_replacement_quiesces_then_closes_all_leases_and_old_streams():
    ledger, old = _ledger()
    evidence = _evidence(old)
    operations = ["r-old", "r-race", "op-reserved", "op-active", "op-q", "op-race"]
    evidence["old_operation_ids"] = operations
    for report in evidence["resource_reports"].values():
        report["excluded_operation_ids"] = sorted(operations)
    app = _runtime(ledger, _ReplacementClient())
    old_scheduler = app.state.scheduler
    old_evaluator = app.state.resource_release_evaluator
    old_tracker = app.state.tracker
    old_output = app.state.output_buffer
    old_tracker.add(RequestInfo(req_id="r-old", active_operation_id="op-active"))
    reserved = app.state.scheduler.reserve_decode_slot("d0", "r-old", "op-reserved")
    active = app.state.scheduler.reserve_decode_slot("d0", "r-old", "op-active")
    app.state.scheduler.commit_decode_slot("op-active")
    quarantined = app.state.scheduler.reserve_decode_slot("d1", "r-old", "op-q")
    app.state.scheduler.quarantine_decode_slot("op-q")

    race_leases = []
    loop_started = asyncio.Event()

    async def mutate_on_cancel():
        loop_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            old_tracker.add(RequestInfo(req_id="r-race", active_operation_id="op-race"))
            race_leases.append(
                old_scheduler.reserve_decode_slot("d1", "r-race", "op-race")
            )
            raise

    app.state.loop_task = asyncio.create_task(mutate_on_cancel())
    await loop_started.wait()
    stream = asyncio.create_task(old_output.wait_next("r-old", 0))

    await _accept_replacement_topology(app, evidence, CONFIG)
    try:
        assert reserved.state == active.state == quarantined.state == "RELEASED"
        assert len(race_leases) == 1 and race_leases[0].state == "RELEASED"
        assert app.state.scheduler is not old_scheduler
        assert old_tracker.all_requests() == []
        assert app.state.tracker is not old_tracker
        assert app.state.output_buffer is not old_output
        assert await stream == ([], True, "topology_replaced")
        assert old_evaluator._replacement_records == {}
        assert app.state.replacement_store.seals()[-1].record_count == 4
        assert app.state.resource_release_evaluator._replacement_records == {}
    finally:
        app.state.loop_task.cancel()
        await asyncio.gather(app.state.loop_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_activation_response_loss_terminalizes_once_and_releases_frozen_slot():
    ledger, old = _ledger()
    evidence = _evidence(old)
    operations = ["r-old", "op-old"]
    evidence["old_operation_ids"] = operations
    for report in evidence["resource_reports"].values():
        report["excluded_operation_ids"] = sorted(operations)
    app = _runtime(ledger, _ReplacementClient(lose_activate_response_once=True))
    old_output = app.state.output_buffer
    app.state.tracker.add(RequestInfo(req_id="r-old", active_operation_id="op-old"))
    lease = app.state.scheduler.reserve_decode_slot("d0", "r-old", "op-old")

    record = await _accept_replacement_topology(app, evidence, CONFIG)
    assert old_output.snapshot("r-old") == ([], True, "topology_replaced")
    assert lease.state == "RELEASED"
    assert app.state.accepting is True

    replay = await _accept_replacement_topology(app, evidence, CONFIG)

    assert replay == record
    assert old_output.snapshot("r-old") == ([], True, "topology_replaced")
    assert lease.state == "RELEASED"
    assert app.state.replacement_store.seal_count == 1


def test_topology_ledger_low_cap_evicts_oldest_accepted_record():
    initial, _ = _ledger()
    ledger = TopologyAcceptanceLedger(
        initial.registry, terminal_snapshot_cap=1
    )
    old_generation = "world-a"
    for index, generation in enumerate(("world-b", "world-c"), 1):
        identities = {
            name: _identity(name, generation, f"new-{index}") for name in MEMBERS
        }
        candidate = WorkerRegistry(expected_topology_generation=generation)
        assert candidate.install_world(
            list(identities.values()),
            [_cap(a, b, identities) for a, b in PAIRS],
        )
        record = RestartRunRecord(
            restart_run_id=f"run-{index}",
            old_topology_generation=old_generation,
            new_topology_generation=generation,
            termination_proof_digests=("t0", "t1", "t2", "t3"),
            fresh_resource_report_digests=("r0", "r1", "r2", "r3"),
            pair_probe_digests=("p0", "p1", "p2", "p3", "p4"),
            decision_digest=f"sha256:decision-{index}",
        )
        ledger.commit(record, candidate)
        old_generation = generation

    assert tuple(ledger.records) == ("run-2",)
    assert ledger.registry.expected_topology_generation == "world-c"


@pytest.mark.asyncio
async def test_replacement_publishes_fresh_empty_governor_after_network_quiesce():
    ledger, old = _ledger()
    evidence = _evidence(old)
    app = _runtime(ledger, _ReplacementClient())
    old_governor = TransferGovernor(CONFIG, object(), app.state.metrics)
    old_governor._bytes_inflight["d0"] = 99
    old_governor._deferred["d0"].append(object())
    old_governor._inflight_tasks["r-old"] = object()
    old_governor._recompute_counts["r-old"] = 2
    app.state.governor = old_governor

    class OldNetwork:
        joined = False

        async def quiesce(self):
            self.joined = True
            return (RuntimeError("old failed task"),)

    old_network = OldNetwork()
    app.state.network_control = old_network

    await _accept_replacement_topology(app, evidence, CONFIG)

    assert old_network.joined is True
    assert app.state.governor is not old_governor
    assert app.state.governor.infer_client is app.state.network_control
    assert app.state.governor.is_drained()
    assert app.state.governor._inflight_tasks == {}
    assert app.state.governor._recompute_counts == {}

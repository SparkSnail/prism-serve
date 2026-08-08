from __future__ import annotations

import hashlib
from http.client import RemoteDisconnected
import io
import json
from dataclasses import replace
from pathlib import Path
from queue import Queue
import subprocess
import sys
from urllib.error import HTTPError

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import pd_worldctl
from pd_worldctl import (
    MEMBERS,
    InitializationSnapshot,
    KubectlGatewayBackend,
    PodTerminationWatch,
    ProcessIdentity,
    TerminationObservation,
    TerminationProof,
    WorldInitializeActuator,
    WorldRestartActuator,
    WorldRestartGuard,
)


def _proof(member: str):
    sequence = (MEMBERS.index(member) * 2) + 1
    return TerminationProof(
        member=member,
        pod_uid=f"old-pod-{member}",
        node_uid=f"node-{member}",
        container_id=f"container-{member}",
        process_generation=f"process-{member}",
        exit_code=137,
        container_terminated=True,
        pod_deleted=True,
        topology_generation="world-a",
        container_name="worker",
        watch_start_resource_version="100",
        observed_resource_version="101",
        deletion_resource_version="102",
        restart_count_before=0,
        restart_count_observed=0,
        termination_source="state.terminated",
        termination_event_type="MODIFIED",
        deletion_event_type="DELETED",
        signal=9,
        started_at="2026-07-23T11:59:59Z",
        finished_at="2026-07-23T12:00:00Z",
        adjacent_current_container_id=None,
        termination_raw_pod_json_sha256=(
            "sha256:" + f"{sequence:064x}"
        ),
        termination_raw_observation_sequence=sequence,
        deletion_raw_pod_json_sha256=(
            "sha256:" + f"{sequence + 1:064x}"
        ),
        deletion_raw_observation_sequence=sequence + 1,
    )


def _expected():
    return {
        member: ProcessIdentity(
            member=member,
            pod_uid=f"old-pod-{member}",
            node_uid=f"node-{member}",
            container_id=f"container-{member}",
            process_generation=f"process-{member}",
            pod_name=f"prism-{member}-0",
            node_name=f"aks-node-{member}",
            resource_version="100",
            topology_generation="world-a",
            container_name="worker",
            restart_count=0,
        )
        for member in MEMBERS
    }


def _new_pod_uids():
    return {member: f"new-pod-{member}" for member in MEMBERS}


def test_cli_generation_parser_rejects_invalid_value_before_backend(
    monkeypatch, capsys,
):
    monkeypatch.setattr(sys, "argv", [
        "pd_worldctl.py",
        "initialize",
        "--release", "week12",
        "--chart", "chart",
        "--generation", "w12-not-a-uuid",
    ])

    with pytest.raises(SystemExit) as exc_info:
        pd_worldctl.main()

    assert exc_info.value.code == 2
    assert "canonical UUID" in capsys.readouterr().err


def test_cli_rejects_abbreviated_generation_override(
    monkeypatch,
    capsys,
    tmp_path,
):
    monkeypatch.setattr(
        WorldInitializeActuator,
        "initialize",
        lambda self, **kwargs: {},
    )
    monkeypatch.setattr(sys, "argv", [
        "pd_worldctl.py",
        "initialize",
        "--release", "week12",
        "--chart", "chart",
        "--generation", "123e4567-e89b-42d3-a456-426614174000",
        "--execute",
        "--gateway-url", "http://127.0.0.1:18080",
        "--run-state", str(tmp_path / "world-state.json"),
        "--gen", "223e4567-e89b-42d3-a456-426614174000",
    ])

    with pytest.raises(SystemExit) as exc_info:
        pd_worldctl.main()

    assert exc_info.value.code == 2
    assert "unrecognized arguments: --gen" in capsys.readouterr().err


def test_startup_permit_uses_exact_six_field_canonical_contract():
    members = _new_pod_uids()

    permit = pd_worldctl.build_startup_permit(
        topology_generation="world-b",
        members=members,
        issuance_mode="RESTART",
        permit_id="permit-a",
    )

    assert set(permit) == {
        "schema_version",
        "issuance_mode",
        "permit_id",
        "topology_generation",
        "members",
        "canonical_digest",
    }
    unsigned = {
        key: value
        for key, value in permit.items()
        if key != "canonical_digest"
    }
    assert permit["canonical_digest"] == "sha256:" + hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="exact four members"):
        pd_worldctl.build_startup_permit(
            topology_generation="world-b",
            members={key: members[key] for key in MEMBERS[:3]},
            issuance_mode="RESTART",
            permit_id="permit-b",
        )
    with pytest.raises(ValueError, match="unique"):
        pd_worldctl.build_startup_permit(
            topology_generation="world-b",
            members={member: "same-pod" for member in MEMBERS},
            issuance_mode="RESTART",
            permit_id="permit-c",
        )


def _initialization_snapshot(
    state: dict[str, object],
    *,
    resource_version: str = "100",
    permit: dict[str, object] | None = None,
) -> InitializationSnapshot:
    return InitializationSnapshot(
        configmap_uid="topology-configmap-uid",
        resource_version=resource_version,
        state_raw=json.dumps(state, sort_keys=True, separators=(",", ":")),
        state=state,
        startup_permit_raw=(
            None
            if permit is None
            else json.dumps(permit, sort_keys=True, separators=(",", ":"))
        ),
        startup_permit=permit,
    )


class _FakeInitializeBackend:
    def __init__(self):
        self.events = []
        self.snapshot = _initialization_snapshot({"phase": "UNINITIALIZED"})
        self.pod_uids = _new_pod_uids()
        self.permit = None

    def read_initialization_snapshot(self, release, namespace):
        self.events.append("read-initialization")
        return self.snapshot

    def claim_initialization(
        self, release, namespace, snapshot, *,
        initialize_run_id, permit_id, generation,
    ):
        self.events.append("claim-initialization")
        assert snapshot == self.snapshot
        state = {
            "phase": "INITIALIZING",
            "initialize_run_id": initialize_run_id,
            "permit_id": permit_id,
            "topology_generation": generation,
        }
        self.snapshot = _initialization_snapshot(
            state, resource_version="101"
        )
        return self.snapshot

    def verify_initial_worker_templates(
        self, release, namespace, *, generation,
    ):
        self.events.append(f"verify-templates:{generation}")

    def start_member(self, release, namespace, member):
        self.events.append(f"start:{member}")

    def wait_initial_pod_uids(
        self, release, namespace, *, generation,
    ):
        self.events.append("initial-pod-uids")
        return dict(self.pod_uids)

    def publish_initialization_permit(
        self, release, namespace, snapshot, *, startup_permit,
        initialize_run_id, permit_id, generation,
    ):
        self.events.append("publish-init-permit")
        assert snapshot == self.snapshot
        assert startup_permit["issuance_mode"] == "INIT"
        assert startup_permit["permit_id"] == permit_id
        assert startup_permit["topology_generation"] == generation
        assert snapshot.state["initialize_run_id"] == initialize_run_id
        self.permit = startup_permit
        self.snapshot = _initialization_snapshot(
            snapshot.state,
            resource_version="102",
            permit=startup_permit,
        )
        return self.snapshot

    def wait_initialization_evidence(self, *, generation):
        self.events.append("initial-probes+reports")
        return {
            "ready": True,
            "identities": [
                {
                    "instance_id": member,
                    "pod_uid": self.permit["members"][member],
                    "process_generation": f"process-{member}",
                    "topology_generation": generation,
                }
                for member in MEMBERS
            ],
        }

    def wait_gateway_ready(self):
        self.events.append("gateway-ready")

    def accept_initialization(
        self, release, namespace, snapshot, *,
        initialize_run_id, permit_id, generation,
    ):
        self.events.append("accept-initialization")
        assert snapshot == self.snapshot
        state = {
            **snapshot.state,
            "phase": "ACCEPTED",
            "accepted_generation": generation,
            "accepted_permit_id": permit_id,
        }
        self.snapshot = _initialization_snapshot(
            state,
            resource_version="103",
            permit=self.permit,
        )
        return state

    def persist_initialized_world(
        self, release, namespace, chart, generation,
        startup_permit, accepted_state,
    ):
        self.events.append("persist-initialized")
        assert startup_permit == self.permit
        assert accepted_state["phase"] == "ACCEPTED"
        assert accepted_state["topology_generation"] == generation


def test_pd_worldctl_initialize_claims_publishes_and_accepts_once(tmp_path):
    backend = _FakeInitializeBackend()
    state_path = tmp_path / "initialize-state.json"

    result = WorldInitializeActuator(backend).initialize(
        release="release",
        namespace="namespace",
        chart="./chart",
        generation="world-a",
        run_state_path=state_path,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert result == {
        "accepted": True,
        "initialize_run_id": state["initialize_run_id"],
        "permit_id": state["permit_id"],
        "topology_generation": "world-a",
    }
    assert state["phase"] == "ACCEPTED"
    assert backend.snapshot.state == {
        "phase": "ACCEPTED",
        "initialize_run_id": state["initialize_run_id"],
        "permit_id": state["permit_id"],
        "topology_generation": "world-a",
        "accepted_generation": "world-a",
        "accepted_permit_id": state["permit_id"],
    }
    assert backend.events == [
        "read-initialization",
        "claim-initialization",
        "verify-templates:world-a",
        *(f"start:{member}" for member in MEMBERS),
        "initial-pod-uids",
        "publish-init-permit",
        "initial-pod-uids",
        "initial-probes+reports",
        "gateway-ready",
        "initial-pod-uids",
        "read-initialization",
        "accept-initialization",
        "persist-initialized",
    ]

    backend.events.clear()
    assert WorldInitializeActuator(backend).initialize(
        release="release",
        namespace="namespace",
        chart="./chart",
        generation="world-a",
        run_state_path=state_path,
    ) == result
    assert backend.events == [
        "read-initialization",
        "persist-initialized",
    ]


def test_pd_worldctl_initialize_rejects_foreign_initializer(tmp_path):
    backend = _FakeInitializeBackend()
    backend.snapshot = _initialization_snapshot({
        "phase": "INITIALIZING",
        "initialize_run_id": "foreign-run",
        "permit_id": "foreign-permit",
        "topology_generation": "world-a",
    })

    with pytest.raises(RuntimeError, match="foreign initialization"):
        WorldInitializeActuator(backend).initialize(
            release="release",
            namespace="namespace",
            chart="./chart",
            generation="world-a",
            run_state_path=tmp_path / "initialize-state.json",
        )

    assert backend.events == ["read-initialization"]


def test_pd_worldctl_initialize_rejects_pod_uid_drift_after_permit(tmp_path):
    class DriftBackend(_FakeInitializeBackend):
        def wait_initial_pod_uids(
            self, release, namespace, *, generation,
        ):
            result = super().wait_initial_pod_uids(
                release, namespace, generation=generation
            )
            if self.permit is not None:
                result["d1"] = "replacement-pod-d1"
            return result

    backend = DriftBackend()

    with pytest.raises(RuntimeError, match="guarded whole-world restart"):
        WorldInitializeActuator(backend).initialize(
            release="release",
            namespace="namespace",
            chart="./chart",
            generation="world-a",
            run_state_path=tmp_path / "initialize-state.json",
        )

    assert "initial-probes+reports" not in backend.events
    assert "accept-initialization" not in backend.events


def test_pd_worldctl_initialize_resumes_same_run_and_permit(tmp_path):
    class RetryBackend(_FakeInitializeBackend):
        def __init__(self):
            super().__init__()
            self.fail_evidence_once = True

        def wait_initialization_evidence(self, *, generation):
            if self.fail_evidence_once:
                self.fail_evidence_once = False
                self.events.append("initial-evidence-timeout")
                raise TimeoutError("injected response loss")
            return super().wait_initialization_evidence(
                generation=generation
            )

    backend = RetryBackend()
    state_path = tmp_path / "initialize-state.json"
    actuator = WorldInitializeActuator(backend)

    with pytest.raises(TimeoutError, match="injected response loss"):
        actuator.initialize(
            release="release",
            namespace="namespace",
            chart="./chart",
            generation="world-a",
            run_state_path=state_path,
        )

    first_state = json.loads(state_path.read_text(encoding="utf-8"))
    first_permit = dict(first_state["startup_permit"])
    backend.events.clear()
    result = actuator.initialize(
        release="release",
        namespace="namespace",
        chart="./chart",
        generation="world-a",
        run_state_path=state_path,
    )

    final_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert result["initialize_run_id"] == first_state["initialize_run_id"]
    assert result["permit_id"] == first_state["permit_id"]
    assert final_state["startup_permit"] == first_permit
    assert "claim-initialization" not in backend.events
    assert "publish-init-permit" not in backend.events
    assert backend.events[-2:] == [
        "accept-initialization",
        "persist-initialized",
    ]


def test_initialize_claim_response_loss_requires_exact_uid_rv_readback(
    monkeypatch,
):
    backend = KubectlGatewayBackend(
        "http://gateway", context="ack-week12"
    )
    initial = _initialization_snapshot({"phase": "UNINITIALIZED"})
    expected_state = {
        "phase": "INITIALIZING",
        "initialize_run_id": "init-run",
        "permit_id": "permit-a",
        "topology_generation": "world-a",
    }
    observed = _initialization_snapshot(
        expected_state, resource_version="101"
    )
    commands = []

    def response_lost(command, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["stdout"] is sys.stderr
        commands.append(command)
        raise subprocess.TimeoutExpired(command, timeout=30)

    monkeypatch.setattr("pd_worldctl.subprocess.run", response_lost)
    backend.read_initialization_snapshot = (
        lambda release, namespace: observed
    )

    assert backend.claim_initialization(
        "release",
        "namespace",
        initial,
        initialize_run_id="init-run",
        permit_id="permit-a",
        generation="world-a",
    ) == observed
    command = commands[0]
    assert command[:3] == ["kubectl", "--context", "ack-week12"]
    patch = json.loads(command[command.index("--patch") + 1])
    assert patch[:3] == [
        {
            "op": "test",
            "path": "/metadata/uid",
            "value": initial.configmap_uid,
        },
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": initial.resource_version,
        },
        {
            "op": "test",
            "path": "/data/initialize-state.json",
            "value": initial.state_raw,
        },
    ]

    backend.read_initialization_snapshot = lambda release, namespace: (
        _initialization_snapshot({
            **expected_state,
            "initialize_run_id": "foreign-run",
        }, resource_version="101")
    )
    with pytest.raises(RuntimeError, match="CAS was not committed"):
        backend.claim_initialization(
            "release",
            "namespace",
            initial,
            initialize_run_id="init-run",
            permit_id="permit-a",
            generation="world-a",
        )


def test_initialize_permit_and_accept_response_loss_require_exact_readback(
    monkeypatch,
):
    backend = KubectlGatewayBackend("http://gateway")
    initializing = {
        "phase": "INITIALIZING",
        "initialize_run_id": "init-run",
        "permit_id": "permit-a",
        "topology_generation": "world-a",
    }
    initial = _initialization_snapshot(
        initializing, resource_version="101"
    )
    permit = pd_worldctl.build_startup_permit(
        topology_generation="world-a",
        members=_new_pod_uids(),
        issuance_mode="INIT",
        permit_id="permit-a",
    )
    with_permit = _initialization_snapshot(
        initializing, resource_version="102", permit=permit
    )
    accepted_state = {
        **initializing,
        "phase": "ACCEPTED",
        "accepted_generation": "world-a",
        "accepted_permit_id": "permit-a",
    }
    accepted = _initialization_snapshot(
        accepted_state, resource_version="103", permit=permit
    )

    def response_lost(command, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["stdout"] is sys.stderr
        raise subprocess.TimeoutExpired(command, timeout=30)

    monkeypatch.setattr("pd_worldctl.subprocess.run", response_lost)
    backend.read_initialization_snapshot = (
        lambda release, namespace: with_permit
    )
    assert backend.publish_initialization_permit(
        "release",
        "namespace",
        initial,
        startup_permit=permit,
        initialize_run_id="init-run",
        permit_id="permit-a",
        generation="world-a",
    ) == with_permit

    backend.read_initialization_snapshot = lambda release, namespace: accepted
    assert backend.accept_initialization(
        "release",
        "namespace",
        with_permit,
        initialize_run_id="init-run",
        permit_id="permit-a",
        generation="world-a",
    ) == accepted_state

    foreign_permit = pd_worldctl.build_startup_permit(
        topology_generation="world-a",
        members=_new_pod_uids(),
        issuance_mode="INIT",
        permit_id="foreign-permit",
    )
    backend.read_initialization_snapshot = lambda release, namespace: (
        _initialization_snapshot(
            initializing,
            resource_version="102",
            permit=foreign_permit,
        )
    )
    with pytest.raises(RuntimeError, match="exact INIT permit"):
        backend.publish_initialization_permit(
            "release",
            "namespace",
            initial,
            startup_permit=permit,
            initialize_run_id="init-run",
            permit_id="permit-a",
            generation="world-a",
        )


def test_persist_initialized_world_freezes_state_and_permit_in_helm(
    monkeypatch,
):
    calls = []
    permit = pd_worldctl.build_startup_permit(
        topology_generation="world-a",
        members=_new_pod_uids(),
        issuance_mode="INIT",
        permit_id="permit-a",
    )
    accepted = {
        "phase": "ACCEPTED",
        "initialize_run_id": "init-run",
        "permit_id": "permit-a",
        "topology_generation": "world-a",
        "accepted_generation": "world-a",
        "accepted_permit_id": "permit-a",
    }

    backend = KubectlGatewayBackend(
        "http://gateway", context="ack-week12"
    )
    monkeypatch.setattr(
        backend,
        "_helm_deployed_snapshot",
        lambda release, namespace: pd_worldctl._HelmReleaseSnapshot(
            revision=1, values={},
        ),
    )
    monkeypatch.setattr(
        backend,
        "_inspect_helm_upgrade_result",
        lambda **kwargs: "deployed",
    )
    monkeypatch.setattr(
        "pd_worldctl.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    backend.persist_initialized_world(
        "release",
        "namespace",
        "./chart",
        "world-a",
        permit,
        accepted,
    )

    command, options = calls[0]
    assert options["check"] is True
    assert options["stdout"] is sys.stderr
    assert command[:3] == ["helm", "--kube-context", "ack-week12"]
    values = {
        value.split("=", 1)[0]: json.loads(value.split("=", 1)[1])
        for value in command
        if value.startswith("worker.startupPermitJson=")
        or value.startswith("worker.initializationStateJson=")
    }
    assert values == {
        "worker.startupPermitJson": permit,
        "worker.initializationStateJson": accepted,
    }


def test_pd_worldctl_cli_routes_initialize_without_restart_inputs(
    monkeypatch, tmp_path, capsys,
):
    calls = []
    state_path = tmp_path / "initialize.json"

    def fake_initialize(self, **kwargs):
        calls.append(kwargs)
        return {"accepted": True}

    monkeypatch.setattr(
        "pd_worldctl.WorldInitializeActuator.initialize", fake_initialize
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pd_worldctl.py",
            "initialize",
            "--release",
            "week12",
            "--namespace",
            "prism",
            "--generation",
            "00000000-0000-4000-8000-000000000001",
            "--chart",
            "./chart",
            "--execute",
            "--gateway-url",
            "http://gateway",
            "--run-state",
            str(state_path),
        ],
    )

    pd_worldctl.main()

    assert calls == [{
        "release": "week12",
        "namespace": "prism",
        "chart": "./chart",
        "generation": "00000000-0000-4000-8000-000000000001",
        "run_state_path": state_path,
    }]
    assert json.loads(capsys.readouterr().out) == {"accepted": True}


def test_pd_worldctl_cli_keeps_mutation_output_off_json_stdout(
    monkeypatch, tmp_path, capsys,
):
    operational_line = "helm upgrade operational output"

    def fake_run(command, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["stdout"] is sys.stderr
        kwargs["stdout"].write(operational_line + "\n")
        return subprocess.CompletedProcess(command, 0)

    def fake_initialize(self, **kwargs):
        self.backend._run_mutation(["helm", "upgrade", "release"])
        return {"accepted": True}

    monkeypatch.setattr("pd_worldctl.subprocess.run", fake_run)
    monkeypatch.setattr(
        "pd_worldctl.WorldInitializeActuator.initialize", fake_initialize,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pd_worldctl.py",
            "initialize",
            "--release",
            "week12",
            "--namespace",
            "prism",
            "--generation",
            "00000000-0000-4000-8000-000000000001",
            "--chart",
            "./chart",
            "--execute",
            "--gateway-url",
            "http://gateway",
            "--run-state",
            str(tmp_path / "initialize.json"),
        ],
    )

    pd_worldctl.main()

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"accepted": True}
    assert operational_line not in captured.out
    assert captured.err == operational_line + "\n"


def test_pd_worldctl_never_starts_before_four_termination_proofs():
    guard = WorldRestartGuard("world-a", "world-b", _expected())
    for member in MEMBERS[:3]:
        guard.record_termination(_proof(member))

    with pytest.raises(ValueError, match="four termination proofs"):
        guard.scale_up_commands("release", "namespace")

    guard.record_termination(_proof("d1"))
    commands = guard.scale_up_commands("release", "namespace")
    assert len(commands) == 4
    assert all("--replicas=1" in command for command in commands)
    assert all("world-b" not in " ".join(command) for command in commands)


def test_pd_worldctl_rejects_same_generation_and_incomplete_proof():
    with pytest.raises(ValueError, match="fresh generation"):
        WorldRestartGuard("world-a", "world-a", _expected())
    guard = WorldRestartGuard("world-a", "world-b", _expected())
    proof = _proof("p0")
    proof = TerminationProof(
        member=proof.member, pod_uid=proof.pod_uid, node_uid=proof.node_uid,
        container_id=proof.container_id, exit_code=proof.exit_code,
        process_generation=proof.process_generation,
        container_terminated=True, pod_deleted=False,
    )
    with pytest.raises(ValueError, match="complete"):
        guard.record_termination(proof)


def test_pd_worldctl_binds_proof_to_old_process_and_uses_helm_authority():
    guard = WorldRestartGuard("world-a", "world-b", _expected())
    wrong = _proof("p0")
    wrong = TerminationProof(
        member=wrong.member, pod_uid="other-pod", node_uid=wrong.node_uid,
        container_id=wrong.container_id, process_generation=wrong.process_generation,
        exit_code=wrong.exit_code, container_terminated=True, pod_deleted=True,
    )
    with pytest.raises(ValueError, match="expected old process"):
        guard.record_termination(wrong)
    wrong_container = _proof("p0")
    wrong_container = TerminationProof(
        member=wrong_container.member, pod_uid=wrong_container.pod_uid,
        node_uid="other-node", container_id="other-container",
        process_generation=wrong_container.process_generation,
        exit_code=wrong_container.exit_code, container_terminated=True,
        pod_deleted=True,
    )
    with pytest.raises(ValueError, match="expected old process"):
        guard.record_termination(wrong_container)
    for member in MEMBERS:
        guard.record_termination(_proof(member))
    command = guard.patch_generation_commands("release", "namespace", "./chart")[0]
    assert command[:4] == ["helm", "upgrade", "release", "./chart"]
    assert "worker.topologyGeneration=world-b" in command
    assert "worker.replicas=0" in command


def test_persisted_world_advances_worker_and_gateway_deployment_authority(
    monkeypatch,
):
    calls = []
    permit = pd_worldctl.build_startup_permit(
        topology_generation="world-b",
        members=_new_pod_uids(),
        issuance_mode="RESTART",
        permit_id="permit-a",
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    backend = KubectlGatewayBackend(
        "http://gateway", context="aks-week12"
    )
    monkeypatch.setattr(
        backend,
        "_helm_deployed_snapshot",
        lambda release, namespace: pd_worldctl._HelmReleaseSnapshot(
            revision=1, values={},
        ),
    )
    monkeypatch.setattr(
        backend,
        "_inspect_helm_upgrade_result",
        lambda **kwargs: "deployed",
    )
    monkeypatch.setattr("pd_worldctl.subprocess.run", fake_run)
    backend.persist_started_world(
        "release", "namespace", "./chart", "world-b", permit
    )

    command, options = calls[0]
    assert options["check"] is True
    assert options["stdout"] is sys.stderr
    assert command[:3] == ["helm", "--kube-context", "aks-week12"]
    assert "worker.topologyGeneration=world-b" in command
    assert "gateway.topologyGeneration=world-b" in command
    assert "worker.replicas=1" in command
    assert "--set-json" in command
    persisted = next(
        value.split("=", 1)[1]
        for value in command
        if value.startswith("worker.startupPermitJson=")
    )
    assert json.loads(persisted) == permit


@pytest.mark.parametrize(
    ("status", "method_name", "event_name"),
    [
        (
            "pending-upgrade",
            "_delete_exact_pending_helm_revision",
            "pd_worldctl.helm_pending_revision_delete",
        ),
        (
            "failed",
            "_delete_exact_failed_helm_revision",
            "pd_worldctl.helm_failed_revision_delete",
        ),
    ],
)
def test_recoverable_helm_revision_delete_emits_identity_before_mutation(
    monkeypatch, capsys, status, method_name, event_name,
):
    backend = KubectlGatewayBackend(
        "http://gateway", context="ack-week12"
    )
    backend._kubectl_json = lambda namespace, *args: {
        "metadata": {
            "name": "sh.helm.release.v1.release.v12",
            "labels": {
                "owner": "helm",
                "name": "release",
                "version": "12",
                "status": status,
            },
            "uid": "secret-uid",
            "resourceVersion": "456",
        },
        "data": {"release": "must-not-be-logged"},
    }
    observed = {}

    def record_mutation(command, *, input_text=None):
        observed["command"] = command
        observed["input_text"] = input_text
        observed["stderr_before_mutation"] = capsys.readouterr().err

    monkeypatch.setattr(backend, "_run_mutation", record_mutation)

    getattr(backend, method_name)(
        release="release", namespace="namespace", revision=12,
    )

    event = json.loads(observed["stderr_before_mutation"])
    assert event == {
        "event": event_name,
        "namespace": "namespace",
        "release": "release",
        "resourceVersion": "456",
        "revision": 12,
        "uid": "secret-uid",
    }
    assert "must-not-be-logged" not in observed["stderr_before_mutation"]
    assert observed["command"][:3] == [
        "kubectl", "--context", "ack-week12",
    ]
    assert json.loads(observed["input_text"])["preconditions"] == {
        "uid": "secret-uid",
        "resourceVersion": "456",
    }


@pytest.mark.parametrize(
    ("path", "value", "remove"),
    [
        (("name",), "foreign", False),
        (("labels", "owner"), "foreign", False),
        (("labels", "name"), "foreign", False),
        (("labels", "version"), "13", False),
        (("labels", "status"), "pending-upgrade", False),
        (("uid",), None, True),
        (("uid",), "", False),
        (("uid",), 7, False),
        (("resourceVersion",), None, True),
        (("resourceVersion",), "", False),
        (("resourceVersion",), 7, False),
    ],
)
def test_failed_helm_revision_delete_rejects_identity_drift(
    monkeypatch, path, value, remove,
):
    backend = KubectlGatewayBackend(
        "http://gateway", context="ack-week12"
    )
    metadata = {
        "name": "sh.helm.release.v1.release.v12",
        "labels": {
            "owner": "helm",
            "name": "release",
            "version": "12",
            "status": "failed",
        },
        "uid": "secret-uid",
        "resourceVersion": "456",
    }
    target = metadata
    for field in path[:-1]:
        target = target[field]
    if remove:
        target.pop(path[-1])
    else:
        target[path[-1]] = value
    mutations = []

    monkeypatch.setattr(
        backend,
        "_kubectl_json",
        lambda namespace, *args: {"metadata": metadata},
    )
    monkeypatch.setattr(
        backend,
        "_run_mutation",
        lambda command, **kwargs: mutations.append(command),
    )

    with pytest.raises(RuntimeError, match="Secret identity changed"):
        backend._delete_exact_failed_helm_revision(
            release="release", namespace="namespace", revision=12,
        )

    assert mutations == []


def test_helm_success_requires_exact_new_deployed_readback(monkeypatch):
    backend = KubectlGatewayBackend("http://gateway", context="ack-week12")
    base = {"worker": {"topologyGeneration": "world-a", "replicas": 1}}
    expected = {
        "worker": {"topologyGeneration": "world-b", "replicas": 0},
    }
    histories = iter([
        [{"revision": 11, "status": "deployed"}],
        [
            {"revision": 11, "status": "superseded"},
            {"revision": 12, "status": "deployed"},
        ],
    ])
    mutation_calls = []

    monkeypatch.setattr(
        backend,
        "_helm_history",
        lambda release, namespace: next(histories),
    )
    monkeypatch.setattr(
        backend,
        "_helm_release_values",
        lambda release, namespace, revision: (
            base if revision == 11 else expected
        ),
    )
    monkeypatch.setattr(
        backend,
        "_run_mutation",
        lambda command, **kwargs: mutation_calls.append(command),
    )

    backend.patch_generation("release", "namespace", "./chart", "world-b")

    assert len(mutation_calls) == 1


def test_helm_success_rejects_foreign_intervening_revision(monkeypatch):
    backend = KubectlGatewayBackend("http://gateway", context="ack-week12")
    base = {"worker": {"topologyGeneration": "world-a", "replicas": 1}}
    mutation_calls = []
    histories = iter([
        [{"revision": 11, "status": "deployed"}],
        [
            {"revision": 11, "status": "superseded"},
            {"revision": 12, "status": "superseded"},
            {"revision": 13, "status": "deployed"},
        ],
    ])

    monkeypatch.setattr(
        backend,
        "_helm_history",
        lambda release, namespace: next(histories),
    )
    monkeypatch.setattr(
        backend,
        "_helm_release_values",
        lambda release, namespace, revision: base,
    )
    monkeypatch.setattr(
        backend,
        "_run_mutation",
        lambda command, **kwargs: mutation_calls.append(command),
    )

    with pytest.raises(RuntimeError, match="exact next revision"):
        backend.patch_generation(
            "release", "namespace", "./chart", "world-b",
        )

    assert len(mutation_calls) == 1


def test_helm_response_loss_accepts_exact_new_deployed_readback(
    monkeypatch, capsys,
):
    backend = KubectlGatewayBackend("http://gateway", context="ack-week12")
    base = {
        "worker": {"topologyGeneration": "world-a", "replicas": 1},
        "gateway": {"topologyGeneration": "world-a"},
        "unrelated": {"preserved": True},
    }
    deployed = {
        **base,
        "worker": {"topologyGeneration": "world-b", "replicas": 0},
    }
    mutation_calls = []
    value_revisions = []
    delete_calls = []

    def lose_response(command, *, input_text=None):
        assert input_text is None
        mutation_calls.append(command)
        raise subprocess.TimeoutExpired(command, timeout=30)

    histories = iter([
        [{"revision": 11, "status": "deployed"}],
        [
            {"revision": 11, "status": "superseded"},
            {"revision": 12, "status": "deployed"},
        ],
    ])

    def release_values(release, namespace, revision):
        value_revisions.append(revision)
        return base if revision == 11 else deployed

    monkeypatch.setattr(backend, "_run_mutation", lose_response)
    monkeypatch.setattr(
        backend,
        "_helm_history",
        lambda release, namespace: next(histories),
    )
    monkeypatch.setattr(backend, "_helm_release_values", release_values)
    monkeypatch.setattr(
        backend,
        "_delete_exact_pending_helm_revision",
        lambda **kwargs: delete_calls.append(kwargs),
    )

    backend.patch_generation("release", "namespace", "./chart", "world-b")

    assert len(mutation_calls) == 1
    assert value_revisions == [11, 11, 12]
    assert delete_calls == []
    diagnostic = json.loads(capsys.readouterr().err)
    assert diagnostic["outcome"] == "new_deployed"
    assert diagnostic["pre_revision"] == 11
    assert diagnostic["latest_revision"] == 12
    assert diagnostic["value_diff_paths"] == []


def test_helm_unchanged_history_retries_once_then_validates_new_revision(
    monkeypatch, capsys,
):
    backend = KubectlGatewayBackend("http://gateway", context="ack-week12")
    base = {
        "worker": {"topologyGeneration": "world-a", "replicas": 1},
        "gateway": {"topologyGeneration": "world-a"},
    }
    expected = {
        **base,
        "worker": {"topologyGeneration": "world-b", "replicas": 0},
    }
    histories = iter([
        [{"revision": 11, "status": "deployed"}],
        [{"revision": 11, "status": "deployed"}],
        [
            {"revision": 11, "status": "superseded"},
            {"revision": 12, "status": "deployed"},
        ],
    ])
    mutation_calls = []
    value_revisions = []

    def mutate(command, *, input_text=None):
        mutation_calls.append((command, input_text))
        if len(mutation_calls) == 1:
            raise subprocess.CalledProcessError(1, command)
        raise subprocess.TimeoutExpired(command, timeout=30)

    def release_values(release, namespace, revision):
        value_revisions.append(revision)
        return base if revision == 11 else expected

    monkeypatch.setattr(backend, "_run_mutation", mutate)
    monkeypatch.setattr(
        backend,
        "_helm_history",
        lambda release, namespace: next(histories),
    )
    monkeypatch.setattr(backend, "_helm_release_values", release_values)

    backend.patch_generation("release", "namespace", "./chart", "world-b")

    assert len(mutation_calls) == 2
    assert mutation_calls[0][0] == mutation_calls[1][0]
    assert value_revisions == [11, 11, 11, 12]
    diagnostics = [
        json.loads(line) for line in capsys.readouterr().err.splitlines()
    ]
    assert [event["outcome"] for event in diagnostics] == [
        "history_unchanged",
        "new_deployed",
    ]
    assert diagnostics[0]["error_returncode"] == 1
    assert diagnostics[1]["error_type"] == "TimeoutExpired"


def test_helm_unchanged_history_never_retries_more_than_once(monkeypatch):
    backend = KubectlGatewayBackend("http://gateway", context="ack-week12")
    base = {"worker": {"topologyGeneration": "world-a", "replicas": 1}}
    mutation_calls = []

    def lose_response(command, *, input_text=None):
        mutation_calls.append((command, input_text))
        raise subprocess.TimeoutExpired(command, timeout=30)

    monkeypatch.setattr(backend, "_run_mutation", lose_response)
    monkeypatch.setattr(
        backend,
        "_helm_history",
        lambda release, namespace: [
            {"revision": 11, "status": "deployed"},
        ],
    )
    monkeypatch.setattr(
        backend,
        "_helm_release_values",
        lambda release, namespace, revision: base,
    )

    with pytest.raises(RuntimeError, match="bounded Helm retry"):
        backend.patch_generation(
            "release", "namespace", "./chart", "world-b",
        )

    assert len(mutation_calls) == 2


def test_helm_response_loss_recovers_exact_pending_revision_once(monkeypatch):
    backend = KubectlGatewayBackend("http://gateway", context="ack-week12")
    base = {
        "worker": {"topologyGeneration": "world-a", "replicas": 1},
        "gateway": {"topologyGeneration": "world-a"},
        "unrelated": {"preserved": True},
    }
    expected = {
        **base,
        "worker": {"topologyGeneration": "world-b", "replicas": 0},
    }
    history_snapshots = iter([
        [
            {"revision": 11, "status": "deployed"},
        ],
        [
            {"revision": 11, "status": "deployed"},
            {"revision": 12, "status": "pending-upgrade"},
        ],
        [
            {"revision": 11, "status": "superseded"},
            {"revision": 12, "status": "deployed"},
        ],
    ])
    mutation_calls = []
    value_revisions = []

    def mutate(command, *, input_text=None):
        mutation_calls.append((command, input_text))
        helm_attempts = sum(
            call[0][0] == "helm" for call in mutation_calls
        )
        if command[0] == "helm" and helm_attempts == 1:
            raise subprocess.TimeoutExpired(command, timeout=30)
        return subprocess.CompletedProcess(command, 0)

    def release_values(release, namespace, revision):
        value_revisions.append(revision)
        return base if revision == 11 else expected

    monkeypatch.setattr(backend, "_run_mutation", mutate)
    monkeypatch.setattr(
        backend,
        "_helm_history",
        lambda release, namespace: next(history_snapshots),
    )
    monkeypatch.setattr(backend, "_helm_release_values", release_values)
    monkeypatch.setattr(
        backend,
        "_kubectl_json",
        lambda namespace, *args: {
            "metadata": {
                "name": "sh.helm.release.v1.release.v12",
                "labels": {
                    "owner": "helm",
                    "name": "release",
                    "version": "12",
                    "status": "pending-upgrade",
                },
                "uid": "secret-uid",
                "resourceVersion": "456",
            },
        },
    )

    backend.patch_generation("release", "namespace", "./chart", "world-b")

    helm_mutations = [
        call for call in mutation_calls if call[0][0] == "helm"
    ]
    delete_mutations = [
        call for call in mutation_calls if "delete" in call[0]
    ]
    assert len(helm_mutations) == 2
    assert helm_mutations[0][0] == helm_mutations[1][0]
    assert len(delete_mutations) == 1
    assert json.loads(delete_mutations[0][1])["preconditions"] == {
        "uid": "secret-uid",
        "resourceVersion": "456",
    }
    assert value_revisions == [11, 11, 12, 11, 12]


def test_helm_response_loss_recovers_exact_failed_revision_once(monkeypatch):
    backend = KubectlGatewayBackend("http://gateway", context="ack-week12")
    base = {
        "worker": {"topologyGeneration": "world-a", "replicas": 1},
        "gateway": {"topologyGeneration": "world-a"},
        "unrelated": {"preserved": True},
    }
    expected = {
        **base,
        "worker": {"topologyGeneration": "world-b", "replicas": 0},
    }
    history_snapshots = iter([
        [
            {"revision": 11, "status": "deployed"},
        ],
        [
            {"revision": 11, "status": "deployed"},
            {"revision": 12, "status": "failed"},
        ],
        [
            {"revision": 11, "status": "superseded"},
            {"revision": 12, "status": "deployed"},
        ],
    ])
    mutation_calls = []
    value_revisions = []

    def mutate(command, *, input_text=None):
        mutation_calls.append((command, input_text))
        if command[0] == "helm":
            raise subprocess.TimeoutExpired(command, timeout=30)
        return subprocess.CompletedProcess(command, 0)

    def release_values(release, namespace, revision):
        value_revisions.append(revision)
        return base if revision == 11 else expected

    monkeypatch.setattr(backend, "_run_mutation", mutate)
    monkeypatch.setattr(
        backend,
        "_helm_history",
        lambda release, namespace: next(history_snapshots),
    )
    monkeypatch.setattr(backend, "_helm_release_values", release_values)
    monkeypatch.setattr(
        backend,
        "_kubectl_json",
        lambda namespace, *args: {
            "metadata": {
                "name": "sh.helm.release.v1.release.v12",
                "labels": {
                    "owner": "helm",
                    "name": "release",
                    "version": "12",
                    "status": "failed",
                },
                "uid": "secret-uid",
                "resourceVersion": "456",
            },
        },
    )

    backend.patch_generation("release", "namespace", "./chart", "world-b")

    helm_mutations = [
        call for call in mutation_calls if call[0][0] == "helm"
    ]
    delete_mutations = [
        call for call in mutation_calls if "delete" in call[0]
    ]
    assert len(helm_mutations) == 2
    assert helm_mutations[0][0] == helm_mutations[1][0]
    assert len(delete_mutations) == 1
    assert json.loads(delete_mutations[0][1])["preconditions"] == {
        "uid": "secret-uid",
        "resourceVersion": "456",
    }
    assert value_revisions == [11, 11, 12, 11, 12]


def test_helm_failed_revision_delete_race_does_not_replay(monkeypatch):
    backend = KubectlGatewayBackend("http://gateway", context="ack-week12")
    base = {
        "worker": {"topologyGeneration": "world-a", "replicas": 1},
        "gateway": {"topologyGeneration": "world-a"},
    }
    expected = {
        **base,
        "worker": {"topologyGeneration": "world-b", "replicas": 0},
    }
    histories = iter([
        [{"revision": 11, "status": "deployed"}],
        [
            {"revision": 11, "status": "deployed"},
            {"revision": 12, "status": "failed"},
        ],
    ])
    mutation_calls = []

    def lose_response_then_lose_delete_race(command, *, input_text=None):
        mutation_calls.append((command, input_text))
        if command[0] == "helm":
            raise subprocess.TimeoutExpired(command, timeout=30)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(
        backend, "_run_mutation", lose_response_then_lose_delete_race,
    )
    monkeypatch.setattr(
        backend,
        "_helm_history",
        lambda release, namespace: next(histories),
    )
    monkeypatch.setattr(
        backend,
        "_helm_release_values",
        lambda release, namespace, revision: (
            base if revision == 11 else expected
        ),
    )
    monkeypatch.setattr(
        backend,
        "_kubectl_json",
        lambda namespace, *args: {
            "metadata": {
                "name": "sh.helm.release.v1.release.v12",
                "labels": {
                    "owner": "helm",
                    "name": "release",
                    "version": "12",
                    "status": "failed",
                },
                "uid": "secret-uid",
                "resourceVersion": "456",
            },
        },
    )

    with pytest.raises(subprocess.CalledProcessError):
        backend.patch_generation(
            "release", "namespace", "./chart", "world-b",
        )

    helm_mutations = [
        call for call in mutation_calls if call[0][0] == "helm"
    ]
    delete_mutations = [
        call for call in mutation_calls if "delete" in call[0]
    ]
    assert len(helm_mutations) == 1
    assert len(delete_mutations) == 1


@pytest.mark.parametrize(
    ("retry_status", "retry_values_drift", "error_pattern"),
    [
        ("failed", False, "bounded Helm retry"),
        ("pending-upgrade", False, "bounded Helm retry"),
        ("deployed", True, "new Helm revision"),
    ],
)
def test_helm_failed_revision_retry_is_bounded_to_one_replay(
    monkeypatch, capsys, retry_status, retry_values_drift, error_pattern,
):
    backend = KubectlGatewayBackend("http://gateway", context="ack-week12")
    base = {
        "worker": {"topologyGeneration": "world-a", "replicas": 1},
        "gateway": {"topologyGeneration": "world-a"},
    }
    expected = {
        **base,
        "worker": {"topologyGeneration": "world-b", "replicas": 0},
    }
    histories = iter([
        [{"revision": 11, "status": "deployed"}],
        [
            {"revision": 11, "status": "deployed"},
            {"revision": 12, "status": "failed"},
        ],
        [
            {
                "revision": 11,
                "status": (
                    "superseded" if retry_status == "deployed"
                    else "deployed"
                ),
            },
            {"revision": 12, "status": retry_status},
        ],
    ])
    mutation_calls = []
    latest_value_reads = 0

    def lose_both_helm_responses(command, *, input_text=None):
        mutation_calls.append((command, input_text))
        if command[0] == "helm":
            raise subprocess.TimeoutExpired(command, timeout=30)
        return subprocess.CompletedProcess(command, 0)

    def release_values(revision):
        nonlocal latest_value_reads
        if revision == 11:
            return base
        latest_value_reads += 1
        if retry_values_drift and latest_value_reads > 1:
            return {
                **expected,
                "worker": {
                    "topologyGeneration": "world-b",
                    "replicas": 99,
                },
            }
        return expected

    monkeypatch.setattr(backend, "_run_mutation", lose_both_helm_responses)
    monkeypatch.setattr(
        backend,
        "_helm_history",
        lambda release, namespace: next(histories),
    )
    monkeypatch.setattr(
        backend,
        "_helm_release_values",
        lambda release, namespace, revision: release_values(revision),
    )
    monkeypatch.setattr(
        backend,
        "_kubectl_json",
        lambda namespace, *args: {
            "metadata": {
                "name": "sh.helm.release.v1.release.v12",
                "labels": {
                    "owner": "helm",
                    "name": "release",
                    "version": "12",
                    "status": "failed",
                },
                "uid": "secret-uid",
                "resourceVersion": "456",
            },
        },
    )

    with pytest.raises(RuntimeError, match=error_pattern):
        backend.patch_generation(
            "release", "namespace", "./chart", "world-b",
        )

    helm_mutations = [
        call for call in mutation_calls if call[0][0] == "helm"
    ]
    delete_mutations = [
        call for call in mutation_calls if "delete" in call[0]
    ]
    assert len(helm_mutations) == 2
    assert helm_mutations[0][0] == helm_mutations[1][0]
    assert len(delete_mutations) == 1
    diagnostics = [
        json.loads(line) for line in capsys.readouterr().err.splitlines()
        if "helm_upgrade_reconcile" in line
    ]
    if retry_values_drift:
        assert diagnostics[-1]["value_diff_paths"] == ["/worker/replicas"]


@pytest.mark.parametrize(
    ("drift", "error_pattern"),
    [
        ("values", "new Helm revision"),
        ("deployed_values", "new Helm revision"),
        ("failed_values", "new Helm revision"),
        ("failed_type_drift", "new Helm revision"),
        ("failed_predecessor", "new Helm revision"),
        ("failed_predecessor_values", "new Helm revision"),
        ("failed_wrong_revision", "exact next revision"),
        ("failed_missing_predecessor", "exact next revision"),
        ("unsupported_status", "new Helm revision"),
        ("pending_secret_identity", "Secret identity changed"),
        ("failed_secret_identity", "Secret identity changed"),
    ],
)
def test_helm_recovery_drift_fails_closed_without_delete(
    monkeypatch, capsys, drift, error_pattern,
):
    backend = KubectlGatewayBackend("http://gateway", context="ack-week12")
    base = {
        "worker": {"topologyGeneration": "world-a", "replicas": 1},
        "gateway": {"topologyGeneration": "world-a"},
    }
    expected = {
        **base,
        "worker": {"topologyGeneration": "world-b", "replicas": 0},
    }
    mutation_calls = []
    predecessor_value_reads = 0

    def mutate(command, *, input_text=None):
        mutation_calls.append((command, input_text))
        if len(mutation_calls) == 1:
            raise subprocess.TimeoutExpired(command, timeout=30)
        return subprocess.CompletedProcess(command, 0)

    history_snapshots = iter([
        [{"revision": 11, "status": "deployed"}],
        None,
    ])

    def history(release, namespace):
        snapshot = next(history_snapshots)
        if snapshot is not None:
            return snapshot
        if drift == "failed_missing_predecessor":
            return [{"revision": 12, "status": "failed"}]
        if drift == "failed_wrong_revision":
            return [
                {"revision": 11, "status": "deployed"},
                {"revision": 13, "status": "failed"},
            ]
        latest_status = {
            "deployed_values": "deployed",
            "failed_values": "failed",
            "failed_type_drift": "failed",
            "failed_predecessor": "failed",
            "failed_secret_identity": "failed",
            "unsupported_status": "uninstalling",
        }.get(drift, "pending-upgrade")
        if latest_status == "deployed" or drift == "failed_predecessor":
            previous_status = "superseded"
        else:
            previous_status = "deployed"
        return [
            {"revision": 11, "status": previous_status},
            {"revision": 12, "status": latest_status},
        ]

    def release_values(release, namespace, revision):
        nonlocal predecessor_value_reads
        if revision == 11:
            predecessor_value_reads += 1
            if (
                drift == "failed_predecessor_values"
                and predecessor_value_reads > 1
            ):
                return {
                    **base,
                    "worker": {
                        "topologyGeneration": "world-a",
                        "replicas": 99,
                    },
                }
            return base
        if drift in {"values", "deployed_values", "failed_values"}:
            return {
                **expected,
                "worker": {
                    "topologyGeneration": "world-b",
                    "replicas": 99,
                },
            }
        if drift == "failed_type_drift":
            return {
                **expected,
                "worker": {
                    "topologyGeneration": "world-b",
                    "replicas": False,
                },
            }
        return expected

    monkeypatch.setattr(backend, "_run_mutation", mutate)
    monkeypatch.setattr(backend, "_helm_history", history)
    monkeypatch.setattr(backend, "_helm_release_values", release_values)
    monkeypatch.setattr(
        backend,
        "_kubectl_json",
        lambda namespace, *args: {
            "metadata": {
                "name": "sh.helm.release.v1.foreign.v12",
                "labels": {
                    "owner": "helm",
                    "name": "release",
                    "version": "12",
                    "status": (
                        "failed"
                        if drift == "failed_secret_identity"
                        else "pending-upgrade"
                    ),
                },
                "uid": "secret-uid",
                "resourceVersion": "456",
            },
        },
    )

    with pytest.raises(RuntimeError, match=error_pattern):
        backend.patch_generation(
            "release", "namespace", "./chart", "world-b",
        )

    assert len(mutation_calls) == 1
    assert not any("delete" in command for command, _ in mutation_calls)
    diagnostics = [
        json.loads(line) for line in capsys.readouterr().err.splitlines()
        if "helm_upgrade_reconcile" in line
    ]
    assert len(diagnostics) == 1
    if drift in {
        "values", "deployed_values", "failed_values", "failed_type_drift",
    }:
        assert diagnostics[0]["value_diff_paths"] == ["/worker/replicas"]


def test_helm_same_revision_value_drift_fails_without_retry(
    monkeypatch, capsys,
):
    backend = KubectlGatewayBackend("http://gateway", context="ack-week12")
    base = {"worker": {"topologyGeneration": "world-a", "replicas": 1}}
    drifted = {"worker": {"topologyGeneration": "world-a", "replicas": 9}}
    values = iter([base, drifted])
    mutation_calls = []

    def fail(command, *, input_text=None):
        mutation_calls.append((command, input_text))
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(backend, "_run_mutation", fail)
    monkeypatch.setattr(
        backend,
        "_helm_history",
        lambda release, namespace: [
            {"revision": 11, "status": "deployed"},
        ],
    )
    monkeypatch.setattr(
        backend,
        "_helm_release_values",
        lambda release, namespace, revision: next(values),
    )

    with pytest.raises(RuntimeError, match="pre-attempt Helm revision"):
        backend.patch_generation(
            "release", "namespace", "./chart", "world-b",
        )

    assert len(mutation_calls) == 1
    diagnostic = json.loads(capsys.readouterr().err)
    assert diagnostic["outcome"] == "invalid"
    assert diagnostic["value_diff_paths"] == ["/worker/replicas"]


def test_helm_pre_attempt_non_deployed_revision_blocks_mutation(monkeypatch):
    backend = KubectlGatewayBackend("http://gateway", context="ack-week12")
    mutation_calls = []
    monkeypatch.setattr(
        backend,
        "_helm_history",
        lambda release, namespace: [
            {"revision": 12, "status": "pending-upgrade"},
        ],
    )
    monkeypatch.setattr(
        backend,
        "_run_mutation",
        lambda command, **kwargs: mutation_calls.append(command),
    )

    with pytest.raises(RuntimeError, match="pre-attempt revision"):
        backend.patch_generation(
            "release", "namespace", "./chart", "world-b",
        )

    assert mutation_calls == []


def test_startup_permit_response_loss_requires_exact_readback(monkeypatch):
    permit = pd_worldctl.build_startup_permit(
        topology_generation="world-b",
        members=_new_pod_uids(),
        issuance_mode="RESTART",
        permit_id="permit-a",
    )
    canonical = json.dumps(
        permit, sort_keys=True, separators=(",", ":")
    )
    commands = []

    def response_lost(command, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["stdout"] is sys.stderr
        commands.append(command)
        raise subprocess.TimeoutExpired(command, timeout=30)

    monkeypatch.setattr("pd_worldctl.subprocess.run", response_lost)
    backend = KubectlGatewayBackend(
        "http://gateway", context="ack-week12"
    )
    backend._kubectl_json = lambda namespace, *args: {
        "data": {"startup-permit.json": canonical}
    }

    assert backend.publish_startup_permit(
        "release", "namespace", permit
    ) == permit
    command = commands[0]
    assert command[:3] == ["kubectl", "--context", "ack-week12"]
    patch = json.loads(command[command.index("--patch") + 1])
    assert patch == {"data": {"startup-permit.json": canonical}}

    backend._kubectl_json = lambda namespace, *args: {
        "data": {
            "startup-permit.json": canonical.replace(
                "permit-a", "foreign-permit"
            )
        }
    }
    with pytest.raises(RuntimeError, match="exact startup permit"):
        backend.publish_startup_permit("release", "namespace", permit)


def test_kubectl_collects_exact_fresh_unique_pod_uids():
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    items = [
        {
            "metadata": {
                "uid": f"new-pod-{member}",
                "labels": {
                    "prism.sparksnail.ai/member": member,
                    "prism.sparksnail.ai/topology-generation": "world-b",
                },
            }
        }
        for member in MEMBERS
    ]
    backend._kubectl_json = lambda namespace, *args: {"items": items}
    old = {member: f"old-pod-{member}" for member in MEMBERS}

    assert backend.wait_fresh_pod_uids(
        "release",
        "namespace",
        generation="world-b",
        old_pod_uids=old,
    ) == _new_pod_uids()

    items[-1]["metadata"]["uid"] = "new-pod-d0"
    with pytest.raises(RuntimeError, match="unique"):
        backend.wait_fresh_pod_uids(
            "release",
            "namespace",
            generation="world-b",
            old_pod_uids=old,
        )


def test_pd_worldctl_rejects_non_unique_captured_physical_identities():
    expected = _expected()
    d1 = expected["d1"]
    expected["d1"] = ProcessIdentity(
        member=d1.member, pod_uid=d1.pod_uid, node_uid=d1.node_uid,
        container_id=expected["d0"].container_id,
        process_generation=d1.process_generation, pod_name=d1.pod_name,
        node_name=d1.node_name, resource_version=d1.resource_version,
    )

    with pytest.raises(ValueError, match="container IDs must be unique"):
        WorldRestartGuard("world-a", "world-b", expected)


class _FakeActuatorBackend:
    def __init__(self, *, observed_generation="world-a", proof_error=None):
        self.events = []
        self.old = _expected()
        self.observed_generation = observed_generation
        self.proof_error = proof_error
        self.captured_operation_ids = ()
        self.prior_terminations = {}
        self.startup_permit = None

    def capture_world(self, release, namespace, *, expected_generation):
        self.events.append(f"preflight:{expected_generation}")
        if self.observed_generation != expected_generation:
            raise RuntimeError(
                "Gateway topology generation does not match --old-generation"
            )
        self.events.append("capture")
        return self.old

    def start_termination_watch(self, release, namespace, identity):
        self.events.append(f"watch:{identity.member}:{identity.resource_version}")
        return identity.member

    def close_termination_watch(self, watch):
        self.events.append(f"close-watch:{watch}")

    def inject_process_fault(self, release, namespace, identity):
        self.events.append(f"crash:{identity.member}:{identity.container_id}")
        return {
            "schema_version": 1,
            "component": "worker",
            "instance_id": identity.member,
            "pod_uid": identity.pod_uid,
            "process_generation": identity.process_generation,
            "instance_epoch": (
                f"{identity.pod_uid}:{identity.process_generation}"
            ),
            "app_pid": 8,
            "process_start_ticks": 12345,
            "signal": 9,
            "pidfd": True,
            "identity_sha256": "sha256:" + "a" * 64,
            "command": ["python", "-m", "prism_infer.server.process_identity"],
            "exec_return_code": 0,
        }

    def wait_injected_process_termination(
        self, release, namespace, watch, identity,
    ):
        self.events.append(f"injected-exit:{identity.member}")
        proof = _proof(identity.member)
        return TerminationProof(
            member=proof.member, pod_uid=proof.pod_uid,
            node_uid=proof.node_uid, container_id=proof.container_id,
            process_generation=proof.process_generation,
            exit_code=proof.exit_code, container_terminated=True,
            pod_deleted=False,
        )

    def stop_member(self, release, namespace, member):
        self.events.append(f"stop:{member}")

    def wait_termination(
        self, release, namespace, watch, identity, *,
        prior_termination=None,
    ):
        self.events.append(f"terminated+deleted:{identity.member}")
        self.prior_terminations[identity.member] = prior_termination
        if self.proof_error is not None:
            raise self.proof_error
        return _proof(identity.member)

    def patch_generation(self, release, namespace, chart, generation):
        self.events.append(f"patch:{generation}")

    def start_member(self, release, namespace, member):
        self.events.append(f"start:{member}")

    def wait_fresh_pod_uids(
        self, release, namespace, *,
        generation, old_pod_uids,
    ):
        self.events.append("fresh-pod-uids")
        assert generation == "world-b"
        assert old_pod_uids == {
            member: f"old-pod-{member}" for member in MEMBERS
        }
        return _new_pod_uids()

    def publish_startup_permit(self, release, namespace, permit):
        self.events.append("publish-startup-permit")
        self.startup_permit = permit
        return permit

    def wait_acceptance_evidence(self, **kwargs):
        self.events.append("fresh-probes+reports")
        assert len(kwargs["termination_proofs"]) == 4
        return {
            "restart_run_id": kwargs["restart_run_id"],
            "old_topology_generation": kwargs["old_generation"],
            "new_topology_generation": kwargs["new_generation"],
            "identities": [
                {
                    "instance_id": member,
                    "pod_uid": self.startup_permit["members"][member],
                    "process_generation": f"process-new-{member}",
                    "topology_generation": "world-b",
                }
                for member in MEMBERS
            ],
        }

    def accept_topology(self, evidence):
        self.events.append("admin-accept")
        assert evidence["old_topology_generation"] == "world-a"
        return {"accepted": True}

    def persist_started_world(
        self, release, namespace, chart, generation, startup_permit,
    ):
        assert startup_permit == self.startup_permit
        self.events.append("persist-replicas:1")


def test_pd_worldctl_actuates_stop_watch_start_and_gateway_accept_in_order(
    tmp_path,
):
    backend = _FakeActuatorBackend()

    result = WorldRestartActuator(backend).restart(
        release="release", namespace="namespace", chart="./chart",
        old_generation="world-a", new_generation="world-b",
        run_state_path=tmp_path / "run-state.json",
    )

    assert result == {"accepted": True}
    assert backend.events == [
        "preflight:world-a",
        "capture",
        *(f"watch:{member}:100" for member in MEMBERS),
        *(f"stop:{member}" for member in MEMBERS),
        *(f"terminated+deleted:{member}" for member in MEMBERS),
        *(f"close-watch:{member}" for member in MEMBERS),
        "patch:world-b",
        *(f"start:{member}" for member in MEMBERS),
        "fresh-pod-uids",
        "publish-startup-permit",
        "fresh-probes+reports",
        "admin-accept",
        "persist-replicas:1",
    ]


def test_pd_worldctl_rejects_report_from_pod_outside_published_permit(tmp_path):
    class DriftBackend(_FakeActuatorBackend):
        def wait_acceptance_evidence(self, **kwargs):
            evidence = super().wait_acceptance_evidence(**kwargs)
            evidence["identities"][-1]["pod_uid"] = "replacement-not-permitted"
            return evidence

    backend = DriftBackend()

    with pytest.raises(RuntimeError, match="startup permit"):
        WorldRestartActuator(backend).restart(
            release="release",
            namespace="namespace",
            chart="./chart",
            old_generation="world-a",
            new_generation="world-b",
            run_state_path=tmp_path / "run-state.json",
        )

    assert "publish-startup-permit" in backend.events
    assert "admin-accept" not in backend.events


def test_pd_worldctl_worker_crash_is_injected_after_all_watches_and_bound_to_137(
    tmp_path,
):
    backend = _FakeActuatorBackend()
    state_path = tmp_path / "fault-run-state.json"

    result = WorldRestartActuator(backend).restart(
        release="release", namespace="namespace", chart="./chart",
        old_generation="world-a", new_generation="world-b",
        run_state_path=state_path, fault_member="d1",
    )

    assert result == {"accepted": True}
    crash = "crash:d1:container-d1"
    injected_exit = "injected-exit:d1"
    assert backend.events.index("watch:d1:100") < backend.events.index(crash)
    assert backend.events.index(crash) < backend.events.index(injected_exit)
    assert backend.events.index(injected_exit) < backend.events.index("stop:d1")
    assert backend.events.index("stop:d1") < backend.events.index("stop:p0")
    prior = backend.prior_terminations["d1"]
    assert prior.exit_code == 137
    assert prior.container_terminated is True
    assert prior.pod_deleted is False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["fault"] == {
        "kind": "worker_crash",
        "member": "d1",
        "old_instance": {
            "pod_uid": "old-pod-d1",
            "container_id": "container-d1",
            "process_generation": "process-d1",
        },
        "injection_requested_at_ns": state["fault"]["injection_requested_at_ns"],
        "expected_exit_code": 137,
        "process_selection": {
            "schema_version": 1,
            "component": "worker",
            "instance_id": "d1",
            "pod_uid": "old-pod-d1",
            "process_generation": "process-d1",
            "instance_epoch": "old-pod-d1:process-d1",
            "app_pid": 8,
            "process_start_ticks": 12345,
            "signal": 9,
            "pidfd": True,
            "identity_sha256": "sha256:" + "a" * 64,
            "command": [
                "python", "-m", "prism_infer.server.process_identity"
            ],
            "exec_return_code": 0,
        },
        "observed_exit_code": 137,
        "termination_reason": "",
        "termination_message": "",
        "termination_message_sha256":
            "sha256:e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855",
        "finished_at": "",
    }
    event = state["protocol_events"][0]
    assert event["name"] == "injected_process_exited"
    assert event["sequence"] == 1
    assert event["evidence"]["member"] == "d1"
    assert event["evidence"]["exit_code"] == 137


def test_pd_worldctl_worker_crash_rejects_non_sigkill_termination(tmp_path):
    class WrongExitBackend(_FakeActuatorBackend):
        def wait_injected_process_termination(
            self, release, namespace, watch, identity,
        ):
            proof = _proof(identity.member)
            return TerminationProof(
                member=proof.member, pod_uid=proof.pod_uid,
                node_uid=proof.node_uid, container_id=proof.container_id,
                process_generation=proof.process_generation, exit_code=143,
                container_terminated=True, pod_deleted=False,
            )

    backend = WrongExitBackend()
    state_path = tmp_path / "wrong-exit.json"
    with pytest.raises(RuntimeError, match="exit 137"):
        WorldRestartActuator(backend).restart(
            release="release", namespace="namespace", chart="./chart",
            old_generation="world-a", new_generation="world-b",
            run_state_path=state_path, fault_member="p0",
        )
    assert not any(
        event.startswith(("stop:", "patch:", "start:"))
        for event in backend.events
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    observed, semantic = state["observation_journal"][-2:]
    assert observed["kind"] == "termination_observed"
    assert observed["stage"] == "worker_crash_injection"
    assert observed["proof"]["exit_code"] == 143
    assert observed["proof_sha256"] == "sha256:" + hashlib.sha256(
        json.dumps(
            observed["proof"], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert semantic == {
        "schema_version": "prism.pd_worldctl.observation/v1",
        "sequence": semantic["sequence"],
        "kind": "termination_semantic_validation",
        "stage": "worker_crash_injection",
        "observed_at_ns": semantic["observed_at_ns"],
        "proof_sha256": observed["proof_sha256"],
        "semantic_status": "FAIL",
        "message": (
            "worker_crash proof requires the injected container to exit 137"
        ),
    }
    assert state["fault"]["observed_exit_code"] == 143


def test_pd_worldctl_worker_crash_timeout_never_stops_world(tmp_path):
    class TimeoutBackend(_FakeActuatorBackend):
        def wait_injected_process_termination(
            self, release, namespace, watch, identity,
        ):
            self.events.append(f"injected-exit-timeout:{identity.member}")
            raise TimeoutError("timed out proving termination")

    backend = TimeoutBackend()
    state_path = tmp_path / "timeout.json"
    with pytest.raises(TimeoutError, match="timed out proving termination"):
        WorldRestartActuator(backend).restart(
            release="release", namespace="namespace", chart="./chart",
            old_generation="world-a", new_generation="world-b",
            run_state_path=state_path, fault_member="p0",
        )

    assert not any(
        event.startswith(("stop:", "patch:", "start:"))
        for event in backend.events
    )
    assert backend.events[-4:] == [
        f"close-watch:{member}" for member in MEMBERS
    ]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["phase"] == "STOPPING"
    incomplete = state["observation_journal"][-1]
    assert incomplete["kind"] == "termination_observation_incomplete"
    assert incomplete["stage"] == "worker_crash_injection"
    assert incomplete["observation_status"] == "INCOMPLETE"
    assert incomplete["error_type"] == "TimeoutError"
    assert incomplete["no_raw_reason"] == (
        "raw Kubernetes watch event was not observed before failure"
    )


def _watchdog_message(identity: ProcessIdentity, operation_id: str) -> str:
    epoch = f"{identity.pod_uid}:{identity.process_generation}"
    return json.dumps({
        "schema_version": 1,
        "kind": "nccl_watchdog_timeout",
        "instance_id": identity.member,
        "pod_uid": identity.pod_uid,
        "process_generation": identity.process_generation,
        "instance_epoch": epoch,
        "topology_generation": "world-a",
        "pair_id": "p0--d1",
        "operation_id": operation_id,
        "endpoint_ref": {
            "topology_generation": "world-a",
            "owner_generation": "gateway-a:boot-a",
            "operation_seq": 9,
            "target_instance": identity.member,
            "target_worker_epoch": epoch,
            "operation_id": operation_id,
            "payload_digest": "sha256:payload",
        },
        "reason": "NCCL operation watchdog expired: exact-ref",
        "expected_exit_code": 70,
    }, sort_keys=True, separators=(",", ":"))


def _raw_termination_observation(
    identity: ProcessIdentity,
    proof: TerminationProof,
    *,
    resource_version: str = "rv-observed",
) -> TerminationObservation:
    raw_event = {
        "type": "MODIFIED",
        "object": {
            "metadata": {
                "uid": identity.pod_uid,
                "resourceVersion": resource_version,
            },
            "spec": {"nodeName": identity.node_name},
            "status": {
                "containerStatuses": [{
                    "name": "worker",
                    "containerID": identity.container_id,
                    "restartCount": identity.restart_count,
                    "state": {
                        "terminated": {
                            "containerID": identity.container_id,
                            "exitCode": proof.exit_code,
                            "reason": proof.termination_reason,
                            "message": proof.termination_message,
                            "signal": proof.signal,
                            "startedAt": proof.started_at,
                            "finishedAt": proof.finished_at,
                        },
                    },
                    "lastState": {},
                }],
            },
        },
    }
    raw_text = json.dumps(
        raw_event, ensure_ascii=False, separators=(",", ":")
    )
    return TerminationObservation(
        proof=proof,
        raw_watch_event=raw_event,
        raw_watch_event_text=raw_text,
        raw_watch_event_sha256=(
            "sha256:" + hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        ),
        resource_version=resource_version,
    )


def test_pd_worldctl_waits_for_natural_watchdog_before_stopping_world(tmp_path):
    class WatchdogBackend(_FakeActuatorBackend):
        def __init__(self):
            super().__init__()
            self.captured_operation_ids = ("transfer-timeout",)

        def wait_natural_watchdog_termination(
            self, release, namespace, watch, identity, **kwargs,
        ):
            self.events.append(f"natural-watchdog:{identity.member}")
            assert kwargs["expected_generation"] == "world-a"
            assert kwargs["expected_operation_ids"] == ("transfer-timeout",)
            return replace(
                _proof(identity.member),
                exit_code=70,
                pod_deleted=False,
                termination_reason="Error",
                termination_message=_watchdog_message(
                    identity, "transfer-timeout"
                ),
                deletion_resource_version="",
                deletion_event_type="",
                deletion_raw_pod_json_sha256="",
                deletion_raw_observation_sequence=None,
                finished_at="2026-07-19T12:00:00Z",
            )

        def wait_termination(
            self, release, namespace, watch, identity, *, prior_termination=None,
        ):
            self.events.append(f"terminated+deleted:{identity.member}")
            if identity.member == "d1":
                assert prior_termination is not None
                return replace(
                    prior_termination,
                    pod_deleted=True,
                    deletion_resource_version="102",
                    deletion_event_type="DELETED",
                    deletion_raw_pod_json_sha256="sha256:" + "8".zfill(64),
                    deletion_raw_observation_sequence=8,
                )
            assert prior_termination is None
            return _proof(identity.member)

    backend = WatchdogBackend()
    state_path = tmp_path / "watchdog-run-state.json"
    result = WorldRestartActuator(backend).restart(
        release="release",
        namespace="namespace",
        chart="./chart",
        old_generation="world-a",
        new_generation="world-b",
        run_state_path=state_path,
        watchdog_member="d1",
        expected_operation_ids=("transfer-timeout",),
    )

    assert result == {"accepted": True}
    natural = backend.events.index("natural-watchdog:d1")
    assert all(
        backend.events.index(f"watch:{member}:100") < natural
        for member in MEMBERS
    )
    assert natural < backend.events.index("stop:d1")
    assert backend.events.index("stop:d1") < backend.events.index("stop:p0")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["fault"]["kind"] == "nccl_watchdog_timeout"
    assert state["fault"]["member"] == "d1"
    assert state["fault"]["observed_exit_code"] == 70
    assert state["fault"]["expected_operation_ids"] == ["transfer-timeout"]
    assert state["fault"]["termination_message_sha256"].startswith("sha256:")
    assert json.loads(state["fault"]["termination_message"])[
        "operation_id"
    ] == "transfer-timeout"
    protocol = state["protocol_events"]
    assert [value["name"] for value in protocol] == [
        "all_termination_watches_established",
        "watchdog_process_exited",
        "whole_world_scale_down_started",
        "four_old_processes_terminated",
        "four_fresh_reports_observed",
        "topology_accept_started",
        "topology_accept_succeeded",
    ]
    assert [value["sequence"] for value in protocol] == list(range(1, 8))
    assert all(value["clock"] == "operator_monotonic" for value in protocol)
    assert all(
        value["producer_epoch"] == state["restart_run_id"]
        for value in protocol
    )
    assert [value["observed_at_ns"] for value in protocol] == sorted(
        value["observed_at_ns"] for value in protocol
    )
    assert all(
        value["evidence_sha256"].startswith("sha256:")
        for value in protocol
    )
    assert all(
        value["evidence_sha256"] == "sha256:" + hashlib.sha256(json.dumps(
            value["evidence"], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        for value in protocol
    )
    assert protocol[-1]["evidence"] == {
        "request": state["evidence"],
        "response": state["accepted_response"],
    }


def test_pd_worldctl_watchdog_wrong_exit_persists_before_semantic_reject(
    tmp_path,
):
    class WrongWatchdogExitBackend(_FakeActuatorBackend):
        def __init__(self):
            super().__init__()
            self.captured_operation_ids = ("transfer-timeout",)

        def wait_natural_watchdog_termination(
            self, release, namespace, watch, identity, **kwargs,
        ):
            proof = TerminationProof(
                member=identity.member,
                pod_uid=identity.pod_uid,
                node_uid=identity.node_uid,
                container_id=identity.container_id,
                process_generation=identity.process_generation,
                exit_code=134,
                container_terminated=True,
                pod_deleted=False,
                termination_reason="Error",
                termination_message=_watchdog_message(
                    identity, "transfer-timeout"
                ),
            )
            return _raw_termination_observation(identity, proof)

    backend = WrongWatchdogExitBackend()
    state_path = tmp_path / "wrong-watchdog-exit.json"
    with pytest.raises(RuntimeError, match="exit 70"):
        WorldRestartActuator(backend).restart(
            release="release",
            namespace="namespace",
            chart="./chart",
            old_generation="world-a",
            new_generation="world-b",
            run_state_path=state_path,
            watchdog_member="d1",
            expected_operation_ids=("transfer-timeout",),
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    observed, semantic = state["observation_journal"][-2:]
    assert observed["stage"] == "nccl_watchdog"
    assert observed["proof"]["exit_code"] == 134
    assert observed["raw_kubernetes_watch_event"]["type"] == "MODIFIED"
    assert observed["resource_version"] == "rv-observed"
    assert observed["raw_kubernetes_watch_event_sha256"] == (
        "sha256:" + hashlib.sha256(
            observed["raw_kubernetes_watch_event_text"].encode("utf-8")
        ).hexdigest()
    )
    assert observed["raw_pod_json"] == (
        observed["raw_kubernetes_watch_event"]["object"]
    )
    assert observed["raw_pod_json_sha256"] == "sha256:" + hashlib.sha256(
        json.dumps(
            observed["raw_pod_json"], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert semantic["semantic_status"] == "FAIL"
    assert semantic["proof_sha256"] == observed["proof_sha256"]
    assert not any(
        event.startswith(("stop:", "patch:", "start:"))
        for event in backend.events
    )


@pytest.mark.parametrize(
    ("message_factory", "error"),
    [
        (
            lambda identity: _watchdog_message(
                identity, "foreign-operation"
            ),
            "identity mismatch",
        ),
        (lambda identity: "{malformed", "not JSON"),
    ],
    ids=["foreign-operation", "malformed-message"],
)
def test_pd_worldctl_watchdog_bad_message_persists_raw_observation_before_reject(
    tmp_path,
    message_factory,
    error,
):
    class BadWatchdogMessageBackend(_FakeActuatorBackend):
        def __init__(self):
            super().__init__()
            self.captured_operation_ids = ("transfer-timeout",)

        def wait_natural_watchdog_termination(
            self, release, namespace, watch, identity, **kwargs,
        ):
            proof = TerminationProof(
                member=identity.member,
                pod_uid=identity.pod_uid,
                node_uid=identity.node_uid,
                container_id=identity.container_id,
                process_generation=identity.process_generation,
                exit_code=70,
                container_terminated=True,
                pod_deleted=False,
                termination_reason="Error",
                termination_message=message_factory(identity),
            )
            return _raw_termination_observation(identity, proof)

    backend = BadWatchdogMessageBackend()
    state_path = tmp_path / f"bad-watchdog-{error.replace(' ', '-')}.json"
    with pytest.raises(RuntimeError, match=error):
        WorldRestartActuator(backend).restart(
            release="release",
            namespace="namespace",
            chart="./chart",
            old_generation="world-a",
            new_generation="world-b",
            run_state_path=state_path,
            watchdog_member="d1",
            expected_operation_ids=("transfer-timeout",),
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    observed, semantic = state["observation_journal"][-2:]
    raw_message = observed["proof"]["termination_message"]
    assert raw_message == message_factory(_expected()["d1"])
    assert observed["proof_sha256"].startswith("sha256:")
    assert observed["raw_kubernetes_watch_event"]["type"] == "MODIFIED"
    assert observed["raw_kubernetes_watch_event_sha256"] == (
        "sha256:" + hashlib.sha256(
            observed["raw_kubernetes_watch_event_text"].encode("utf-8")
        ).hexdigest()
    )
    assert observed["raw_pod_json_sha256"] == "sha256:" + hashlib.sha256(
        json.dumps(
            observed["raw_pod_json"], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert state["fault"]["termination_message"] == raw_message
    assert state["fault"]["termination_message_sha256"] == (
        "sha256:" + hashlib.sha256(raw_message.encode("utf-8")).hexdigest()
    )
    assert semantic["semantic_status"] == "FAIL"
    assert semantic["proof_sha256"] == observed["proof_sha256"]
    assert not any(
        event.startswith(("stop:", "patch:", "start:"))
        for event in backend.events
    )


@pytest.mark.parametrize(
    ("failure_kind", "error"),
    [
        ("pod-uid", "Pod uid drifted"),
        ("node", "node assignment drifted"),
        ("container", "worker container id drifted"),
        ("error-event", "Kubernetes watch error"),
        ("malformed-event", "malformed Pod watch event"),
    ],
)
def test_pd_worldctl_persists_raw_watch_before_incomplete_observation_failure(
    tmp_path,
    failure_kind,
    error,
):
    class RawWatchFailureBackend(_FakeActuatorBackend):
        raw_observation_sink_supported = True

        def __init__(self):
            super().__init__()
            self.captured_operation_ids = ("transfer-timeout",)

        def observe_natural_watchdog_termination(
            self, release, namespace, watch, identity, **kwargs,
        ):
            del release, watch
            raw_observation_sink = kwargs["raw_observation_sink"]
            real = KubectlGatewayBackend(
                "http://gateway", timeout_s=0.1
            )
            real._assert_node_available = (
                lambda observed_namespace, expected: None
            )
            proof = TerminationProof(
                member=identity.member,
                pod_uid=identity.pod_uid,
                node_uid=identity.node_uid,
                container_id=identity.container_id,
                process_generation=identity.process_generation,
                exit_code=70,
                container_terminated=True,
                pod_deleted=False,
                termination_message=_watchdog_message(
                    identity, "transfer-timeout"
                ),
            )
            raw_event = _raw_termination_observation(
                identity, proof,
            ).raw_watch_event
            assert raw_event is not None
            if failure_kind == "pod-uid":
                raw_event["object"]["metadata"]["uid"] = "foreign-pod"
            elif failure_kind == "node":
                raw_event["object"]["spec"]["nodeName"] = "foreign-node"
            elif failure_kind == "container":
                worker = raw_event["object"]["status"][
                    "containerStatuses"
                ][0]
                worker["containerID"] = "foreign-container"
                worker["state"] = {"running": {}}
            elif failure_kind == "error-event":
                raw_event = {
                    "type": "ERROR",
                    "object": {
                        "kind": "Status",
                        "reason": "Expired",
                        "code": 410,
                    },
                }
            raw_text = (
                "not-json"
                if failure_kind == "malformed-event"
                else json.dumps(raw_event, separators=(",", ":"))
            )
            events = Queue()
            try:
                event = real._decode_raw_watch_event(
                    raw_text, identity.member,
                )
            except BaseException as exc:
                events.put(exc)
            else:
                events.put(event)
            exact_watch = PodTerminationWatch(
                identity, _LiveWatchProcess(), events, None
            )
            return real.observe_natural_watchdog_termination(
                "release",
                namespace,
                exact_watch,
                identity,
                expected_generation="world-a",
                expected_operation_ids=("transfer-timeout",),
                raw_observation_sink=raw_observation_sink,
            )

    backend = RawWatchFailureBackend()
    state_path = tmp_path / f"raw-watch-{failure_kind}.json"
    with pytest.raises(RuntimeError, match=error):
        WorldRestartActuator(backend).restart(
            release="release",
            namespace="namespace",
            chart="./chart",
            old_generation="world-a",
            new_generation="world-b",
            run_state_path=state_path,
            watchdog_member="d1",
            expected_operation_ids=("transfer-timeout",),
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    incomplete = state["observation_journal"][-1]
    assert incomplete["kind"] == "termination_observation_incomplete"
    assert incomplete["stage"] == "nccl_watchdog"
    assert incomplete["observation_status"] == "INCOMPLETE"
    assert error in incomplete["message"]
    raw = state["observation_journal"][
        incomplete["raw_observation_sequence"] - 1
    ]
    assert raw["kind"] == "raw_kubernetes_watch_event_observed"
    assert incomplete["raw_kubernetes_watch_event_sha256"] == (
        raw["raw_kubernetes_watch_event_sha256"]
    )
    assert raw["raw_kubernetes_watch_event_sha256"] == (
        "sha256:" + hashlib.sha256(
            raw["raw_kubernetes_watch_event_text"].encode("utf-8")
        ).hexdigest()
    )
    if failure_kind not in {"error-event", "malformed-event"}:
        assert incomplete["raw_pod_json_sha256"] == (
            "sha256:" + hashlib.sha256(
                json.dumps(
                    incomplete["raw_pod_json"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
    assert not any(
        event.startswith(("stop:", "patch:", "start:"))
        for event in backend.events
    )


@pytest.mark.parametrize(
    "failure_kind",
    ["wait", "guard"],
)
def test_pd_worldctl_persists_each_termination_before_later_failure(
    tmp_path,
    failure_kind,
):
    class PartialFailureBackend(_FakeActuatorBackend):
        def wait_termination(
            self, release, namespace, watch, identity, *,
            prior_termination=None,
        ):
            self.events.append(f"terminated+deleted:{identity.member}")
            if identity.member == "d0" and failure_kind == "wait":
                raise RuntimeError("third termination wait failed")
            proof = _proof(identity.member)
            if identity.member == "d0" and failure_kind == "guard":
                proof = replace(
                    proof,
                    container_id="foreign-container",
                )
            return proof

    backend = PartialFailureBackend()
    state_path = tmp_path / f"partial-{failure_kind}.json"
    expected_error = (
        "third termination wait failed"
        if failure_kind == "wait" else "expected old process"
    )
    with pytest.raises((RuntimeError, ValueError), match=expected_error):
        WorldRestartActuator(backend).restart(
            release="release",
            namespace="namespace",
            chart="./chart",
            old_generation="world-a",
            new_generation="world-b",
            run_state_path=state_path,
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    observations = [
        event for event in state["observation_journal"]
        if event["kind"] == "termination_observed"
        and event["stage"] == "whole_world_termination"
    ]
    expected_members = ["p0", "p1"] + (
        ["d0"] if failure_kind == "guard" else []
    )
    assert [event["proof"]["member"] for event in observations] == (
        expected_members
    )
    assert all(event["proof_sha256"].startswith("sha256:") for event in observations)
    if failure_kind == "guard":
        semantic = state["observation_journal"][-1]
        assert semantic["semantic_status"] == "FAIL"
        assert semantic["proof_sha256"] == observations[-1]["proof_sha256"]
    assert sum(event.startswith("stop:") for event in backend.events) == 4
    assert not any(
        event.startswith(("patch:", "start:")) for event in backend.events
    )


def test_pd_worldctl_persists_partial_termination_before_invalid_delete(
    tmp_path,
):
    class PartialRawTerminationBackend(_FakeActuatorBackend):
        raw_observation_sink_supported = True

        def observe_termination(
            self, release, namespace, watch, identity, *,
            prior_termination=None, raw_observation_sink=None,
        ):
            del release, watch, prior_termination
            real = KubectlGatewayBackend(
                "http://gateway", timeout_s=0.1
            )
            real._assert_node_available = (
                lambda observed_namespace, expected: None
            )
            proof = _proof(identity.member)
            partial = TerminationProof(
                member=proof.member,
                pod_uid=proof.pod_uid,
                node_uid=proof.node_uid,
                container_id=proof.container_id,
                process_generation=proof.process_generation,
                exit_code=proof.exit_code,
                container_terminated=True,
                pod_deleted=False,
            )
            modified = _raw_termination_observation(
                identity, partial, resource_version="rv-terminated",
            )
            deleted = {
                "type": "DELETED",
                "object": {
                    "metadata": {
                        "uid": "foreign-pod",
                        "resourceVersion": "rv-deleted",
                    },
                    "spec": {"nodeName": identity.node_name},
                    "status": {"containerStatuses": []},
                },
            }
            deleted_text = json.dumps(deleted, separators=(",", ":"))
            events = Queue()
            events.put(real._decode_raw_watch_event(
                modified.raw_watch_event_text, identity.member,
            ))
            events.put(real._decode_raw_watch_event(
                deleted_text, identity.member,
            ))
            exact_watch = PodTerminationWatch(
                identity, _LiveWatchProcess(), events, None
            )
            return real.observe_termination(
                "release",
                namespace,
                exact_watch,
                identity,
                raw_observation_sink=raw_observation_sink,
            )

    backend = PartialRawTerminationBackend()
    state_path = tmp_path / "partial-invalid-delete.json"
    with pytest.raises(RuntimeError, match="Pod uid drifted"):
        WorldRestartActuator(backend).restart(
            release="release",
            namespace="namespace",
            chart="./chart",
            old_generation="world-a",
            new_generation="world-b",
            run_state_path=state_path,
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    partial, incomplete = state["observation_journal"][-2:]
    assert partial["kind"] == "termination_observed"
    assert partial["stage"] == "whole_world_termination"
    assert partial["observation_status"] == "PARTIAL"
    assert partial["proof"]["member"] == "p0"
    assert partial["proof"]["pod_deleted"] is False
    assert partial["raw_kubernetes_watch_event"]["type"] == "MODIFIED"
    assert incomplete["kind"] == "termination_observation_incomplete"
    assert incomplete["observation_status"] == "INCOMPLETE"
    failed_raw = state["observation_journal"][
        incomplete["raw_observation_sequence"] - 1
    ]
    assert json.loads(
        failed_raw["raw_kubernetes_watch_event_text"]
    )["type"] == "DELETED"
    assert incomplete["raw_kubernetes_watch_event"]["type"] == "DELETED"
    assert incomplete["partial_proof_sha256"] == partial["proof_sha256"]
    assert sum(event.startswith("stop:") for event in backend.events) == 4
    assert not any(
        event.startswith(("patch:", "start:")) for event in backend.events
    )


def test_pd_worldctl_missing_expected_live_operation_never_crosses_boundary(tmp_path):
    backend = _FakeActuatorBackend()

    with pytest.raises(RuntimeError, match="omit expected fault operations"):
        WorldRestartActuator(backend).restart(
            release="release", namespace="namespace", chart="./chart",
            old_generation="world-a", new_generation="world-b",
            run_state_path=tmp_path / "missing-operation.json",
            expected_operation_ids=("fault-transfer",),
            required_old_operation_ids=("gateway-request",),
        )

    assert backend.events == ["preflight:world-a", "capture"]
    state = json.loads(
        (tmp_path / "missing-operation.json").read_text(encoding="utf-8")
    )
    assert state["phase"] == "CREATED"


def test_pd_worldctl_watch_probe_failure_never_crosses_destructive_boundary(
    tmp_path,
):
    class _ProbeFailureBackend(_FakeActuatorBackend):
        def start_termination_watch(self, release, namespace, identity):
            super().start_termination_watch(release, namespace, identity)
            raise RuntimeError("watch probe failed")

    backend = _ProbeFailureBackend()
    state_path = tmp_path / "probe-failure.json"

    with pytest.raises(RuntimeError, match="watch probe failed"):
        WorldRestartActuator(backend).restart(
            release="release", namespace="namespace", chart="./chart",
            old_generation="world-a", new_generation="world-b",
            run_state_path=state_path,
        )

    assert backend.events == ["preflight:world-a", "capture", "watch:p0:100"]
    assert json.loads(state_path.read_text(encoding="utf-8"))["phase"] == "CREATED"


def test_pd_worldctl_stale_generation_performs_no_destructive_action_and_created_retries(
    tmp_path,
):
    state_path = tmp_path / "run-state.json"
    stale = _FakeActuatorBackend(observed_generation="world-stale")
    kwargs = {
        "release": "release", "namespace": "namespace", "chart": "./chart",
        "old_generation": "world-a", "new_generation": "world-b",
        "run_state_path": state_path,
    }

    with pytest.raises(RuntimeError, match="--old-generation"):
        WorldRestartActuator(stale).restart(**kwargs)

    assert stale.events == ["preflight:world-a"]
    assert json.loads(state_path.read_text(encoding="utf-8"))["phase"] == "CREATED"

    retry = _FakeActuatorBackend()
    assert WorldRestartActuator(retry).restart(**kwargs)["accepted"] is True
    assert any(event.startswith("stop:") for event in retry.events)


def test_pd_worldctl_persists_stopping_before_first_stop_and_rerun_resumes(
    tmp_path,
):
    class StopFailureBackend(_FakeActuatorBackend):
        def stop_member(self, release, namespace, member):
            self.events.append(f"stop:{member}")
            raise RuntimeError("stop response unknown")

    state_path = tmp_path / "run-state.json"
    kwargs = {
        "release": "release", "namespace": "namespace", "chart": "./chart",
        "old_generation": "world-a", "new_generation": "world-b",
        "run_state_path": state_path,
    }
    first = StopFailureBackend()

    with pytest.raises(RuntimeError, match="stop response unknown"):
        WorldRestartActuator(first).restart(**kwargs)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["phase"] == "STOPPING"
    assert state["checkpoint"] == "CAPTURED"
    assert [value["member"] for value in state["captured_identities"]] == list(MEMBERS)
    assert first.events.index("watch:d1:100") < first.events.index("stop:p0")

    rerun = _FakeActuatorBackend()
    assert WorldRestartActuator(rerun).restart(**kwargs)["accepted"] is True
    assert "capture" not in rerun.events
    assert rerun.events[:4] == [
        f"watch:{member}:100" for member in MEMBERS
    ]


def test_pd_worldctl_resume_skips_durable_termination_proof(tmp_path):
    class TerminationWaitFailureBackend(_FakeActuatorBackend):
        def __init__(self):
            super().__init__()
            self.failed = False

        def wait_termination(
            self, release, namespace, watch, identity, *,
            prior_termination=None,
        ):
            if identity.member == "p1" and not self.failed:
                self.failed = True
                self.events.append("termination-response-lost:p1")
                raise TimeoutError("termination response lost")
            return super().wait_termination(
                release,
                namespace,
                watch,
                identity,
                prior_termination=prior_termination,
            )

    backend = TerminationWaitFailureBackend()
    state_path = tmp_path / "termination-wait.json"
    kwargs = {
        "release": "release", "namespace": "namespace", "chart": "./chart",
        "old_generation": "world-a", "new_generation": "world-b",
        "run_state_path": state_path,
    }

    with pytest.raises(TimeoutError, match="termination response lost"):
        WorldRestartActuator(backend).restart(**kwargs)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["checkpoint"] == "CAPTURED"
    assert set(state["termination_proofs"]) == {"p0"}

    assert WorldRestartActuator(backend).restart(**kwargs)["accepted"] is True
    assert backend.events.count("capture") == 1
    assert backend.events.count("terminated+deleted:p0") == 1
    assert backend.events.count("watch:p0:100") == 1


def test_pd_worldctl_resume_after_scale_zero_retries_generation_patch(tmp_path):
    class PatchResponseLossBackend(_FakeActuatorBackend):
        def __init__(self):
            super().__init__()
            self.patch_attempts = 0

        def patch_generation(self, release, namespace, chart, generation):
            self.patch_attempts += 1
            self.events.append(f"patch:{generation}")
            if self.patch_attempts == 1:
                raise TimeoutError("patch response lost")

    backend = PatchResponseLossBackend()
    state_path = tmp_path / "post-scale-zero.json"
    kwargs = {
        "release": "release", "namespace": "namespace", "chart": "./chart",
        "old_generation": "world-a", "new_generation": "world-b",
        "run_state_path": state_path,
    }

    with pytest.raises(TimeoutError, match="patch response lost"):
        WorldRestartActuator(backend).restart(**kwargs)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["checkpoint"] == "TERMINATED"
    assert set(state["termination_proofs"]) == set(MEMBERS)

    assert WorldRestartActuator(backend).restart(**kwargs)["accepted"] is True
    assert backend.patch_attempts == 2
    assert backend.events.count("capture") == 1
    assert sum(
        event.startswith("terminated+deleted:")
        for event in backend.events
    ) == 4


def test_pd_worldctl_generation_patch_checkpoint_skips_helm_after_scale_failure(
    tmp_path,
):
    class ScaleFailureBackend(_FakeActuatorBackend):
        def start_member(self, release, namespace, member):
            self.events.append(f"start:{member}")
            if member == "p1":
                raise RuntimeError("scale response lost")

    class ResumeBackend(_FakeActuatorBackend):
        def patch_generation(self, release, namespace, chart, generation):
            raise AssertionError("durable generation patch must not rerun Helm")

    state_path = tmp_path / "generation-patched.json"
    kwargs = {
        "release": "release", "namespace": "namespace", "chart": "./chart",
        "old_generation": "world-a", "new_generation": "world-b",
        "run_state_path": state_path,
    }
    first = ScaleFailureBackend()

    with pytest.raises(RuntimeError, match="scale response lost"):
        WorldRestartActuator(first).restart(**kwargs)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["phase"] == "STOPPING"
    assert state["checkpoint"] == "GENERATION_PATCHED"
    assert first.events.count("patch:world-b") == 1
    assert first.events[-2:] == ["start:p0", "start:p1"]

    resumed = ResumeBackend()
    assert WorldRestartActuator(resumed).restart(**kwargs)["accepted"] is True
    assert "capture" not in resumed.events
    assert not any(event.startswith("patch:") for event in resumed.events)
    assert [
        event for event in resumed.events if event.startswith("start:")
    ] == [f"start:{member}" for member in MEMBERS]


def test_pd_worldctl_resume_after_start_wait_does_not_restart_again(tmp_path):
    class FreshPodWaitFailureBackend(_FakeActuatorBackend):
        def __init__(self):
            super().__init__()
            self.fresh_waits = 0

        def wait_fresh_pod_uids(
            self, release, namespace, *, generation, old_pod_uids,
        ):
            self.fresh_waits += 1
            self.events.append("fresh-pod-uids")
            if self.fresh_waits == 1:
                raise TimeoutError("fresh Pod wait interrupted")
            assert generation == "world-b"
            assert old_pod_uids == {
                member: f"old-pod-{member}" for member in MEMBERS
            }
            return _new_pod_uids()

    backend = FreshPodWaitFailureBackend()
    state_path = tmp_path / "post-start.json"
    kwargs = {
        "release": "release", "namespace": "namespace", "chart": "./chart",
        "old_generation": "world-a", "new_generation": "world-b",
        "run_state_path": state_path,
    }

    with pytest.raises(TimeoutError, match="fresh Pod wait interrupted"):
        WorldRestartActuator(backend).restart(**kwargs)

    assert json.loads(
        state_path.read_text(encoding="utf-8")
    )["checkpoint"] == "STARTED"

    assert WorldRestartActuator(backend).restart(**kwargs)["accepted"] is True
    assert backend.fresh_waits == 2
    assert all(
        backend.events.count(f"start:{member}") == 1 for member in MEMBERS
    )


def test_pd_worldctl_resume_republishes_exact_durable_permit(tmp_path):
    class PermitResponseLossBackend(_FakeActuatorBackend):
        def __init__(self):
            super().__init__()
            self.permit_attempts = []

        def publish_startup_permit(self, release, namespace, permit):
            self.events.append("publish-startup-permit")
            self.startup_permit = permit
            self.permit_attempts.append(dict(permit))
            if len(self.permit_attempts) == 1:
                raise TimeoutError("permit response lost")
            return permit

    backend = PermitResponseLossBackend()
    state_path = tmp_path / "permit-response-loss.json"
    kwargs = {
        "release": "release", "namespace": "namespace", "chart": "./chart",
        "old_generation": "world-a", "new_generation": "world-b",
        "run_state_path": state_path,
    }

    with pytest.raises(TimeoutError, match="permit response lost"):
        WorldRestartActuator(backend).restart(**kwargs)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["checkpoint"] == "PERMIT_READY"
    first_permit = state["startup_permit"]

    assert WorldRestartActuator(backend).restart(**kwargs)["accepted"] is True
    assert backend.permit_attempts == [first_permit, first_permit]
    assert backend.events.count("capture") == 1


@pytest.mark.parametrize(
    "proof_error",
    [
        RuntimeError("Pod was DELETED without container state.terminated"),
        RuntimeError("node is not Ready"),
        RuntimeError("worker container id drifted"),
    ],
    ids=["direct-delete", "node-unavailable", "wrong-container"],
)
def test_pd_worldctl_unproven_termination_never_patches_or_starts(
    tmp_path, proof_error,
):
    backend = _FakeActuatorBackend(proof_error=proof_error)

    with pytest.raises(RuntimeError, match=str(proof_error)):
        WorldRestartActuator(backend).restart(
            release="release", namespace="namespace", chart="./chart",
            old_generation="world-a", new_generation="world-b",
            run_state_path=tmp_path / "run-state.json",
        )

    assert sum(event.startswith("stop:") for event in backend.events) == 4
    assert not any(
        event.startswith(("patch:", "start:")) for event in backend.events
    )


class _LiveWatchProcess:
    stderr = None

    @staticmethod
    def poll():
        return None


@pytest.mark.parametrize(
    "probe_output",
    [
        "",
        json.dumps({
            "type": "BOOKMARK",
            "object": {"metadata": {"resourceVersion": "rv-bookmark"}},
        }) + "\n",
    ],
    ids=["empty", "valid-event"],
)
def test_kubectl_watch_uses_raw_api_cursor_and_exact_pod_selector(
    monkeypatch, probe_output,
):
    identity = _expected()["p0"]
    modified_event = {
        "type": "MODIFIED",
        "object": {
            "metadata": {
                "uid": identity.pod_uid,
                "resourceVersion": "rv-terminated",
            },
            "spec": {"nodeName": identity.node_name},
            "status": {
                "containerStatuses": [{
                    "name": "worker",
                    "containerID": identity.container_id,
                    "restartCount": 0,
                    "state": {"terminated": {
                        "containerID": identity.container_id,
                        "exitCode": 137,
                        "reason": "Error",
                        "signal": 9,
                        "message": "killed",
                        "startedAt": "2026-07-23T11:59:59Z",
                        "finishedAt": "2026-07-23T12:00:00Z",
                    }},
                    "lastState": {},
                }],
            },
        },
    }
    deleted_event = json.loads(json.dumps(modified_event))
    deleted_event["type"] = "DELETED"
    deleted_event["object"]["metadata"]["resourceVersion"] = "rv-deleted"
    probe_commands = []
    watch_commands = []

    class _ProbeResult:
        returncode = 0
        stdout = probe_output
        stderr = ""

    def run_probe(command, **kwargs):
        probe_commands.append((command, kwargs))
        return _ProbeResult()

    class _RawWatchProcess:
        def __init__(self, command, **kwargs):
            watch_commands.append((command, kwargs))
            self.stdout = io.StringIO(
                json.dumps(modified_event)
                + "\n"
                + json.dumps(deleted_event)
                + "\n"
            )
            self.stderr = io.StringIO("")

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr("pd_worldctl.subprocess.run", run_probe)
    monkeypatch.setattr("pd_worldctl.subprocess.Popen", _RawWatchProcess)
    monkeypatch.setattr("pd_worldctl.time.sleep", lambda _: None)
    backend = KubectlGatewayBackend(
        "http://gateway", timeout_s=0.1, context="ack-week12"
    )
    backend._assert_node_available = lambda namespace, expected: None

    watch = backend.start_termination_watch("release", "namespace", identity)

    probe_command, probe_kwargs = probe_commands[0]
    assert probe_command == [
        "kubectl", "--context", "ack-week12", "get", "--raw",
        "/api/v1/namespaces/namespace/pods?watch=1"
        f"&resourceVersion={identity.resource_version}"
        f"&fieldSelector=metadata.name%3D{identity.pod_name}"
        "&timeoutSeconds=1",
    ]
    assert probe_kwargs["timeout"] == 5
    assert probe_kwargs["check"] is False
    command, kwargs = watch_commands[0]
    assert command == [
        "kubectl", "--context", "ack-week12", "get", "--raw",
        "/api/v1/namespaces/namespace/pods?watch=1"
        f"&resourceVersion={identity.resource_version}"
        f"&fieldSelector=metadata.name%3D{identity.pod_name}",
    ]
    assert "--resource-version" not in " ".join(command)
    assert kwargs["text"] is True
    raw_events = []

    def sink(value):
        raw_events.append(value)
        return len(raw_events), value.raw_sha256

    proof = backend.wait_termination(
        "release",
        "namespace",
        watch,
        identity,
        raw_observation_sink=sink,
    )
    assert proof.pod_uid == identity.pod_uid
    assert proof.node_uid == identity.node_uid
    assert proof.container_id == identity.container_id
    assert proof.exit_code == 137
    assert proof.container_terminated is True
    assert proof.pod_deleted is True


@pytest.mark.parametrize(
    "failure",
    ["nonzero", "timeout", "error-event", "malformed-event"],
)
def test_kubectl_watch_probe_fails_before_persistent_watch(monkeypatch, failure):
    identity = _expected()["p0"]
    persistent_commands = []

    class _ProbeFailure:
        returncode = 1
        stdout = ""
        stderr = "probe denied"

    def fail_probe(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "error-event":
            result = _ProbeFailure()
            result.returncode = 0
            result.stderr = ""
            result.stdout = json.dumps({
                "type": "ERROR",
                "object": {
                    "kind": "Status",
                    "reason": "Expired",
                    "code": 410,
                },
            }) + "\n"
            return result
        if failure == "malformed-event":
            result = _ProbeFailure()
            result.returncode = 0
            result.stderr = ""
            result.stdout = "not-json\n"
            return result
        return _ProbeFailure()

    def record_persistent(command, **kwargs):
        persistent_commands.append((command, kwargs))
        raise AssertionError("persistent watch must not start after failed probe")

    monkeypatch.setattr("pd_worldctl.subprocess.run", fail_probe)
    monkeypatch.setattr("pd_worldctl.subprocess.Popen", record_persistent)
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    backend._assert_node_available = lambda namespace, expected: None

    with pytest.raises(RuntimeError, match="could not establish exact Pod watch"):
        backend.start_termination_watch("release", "namespace", identity)

    assert persistent_commands == []


def _watch_with_event(identity: ProcessIdentity, event: str) -> PodTerminationWatch:
    events = Queue()
    events.put(event)
    return PodTerminationWatch(identity, _LiveWatchProcess(), events, None)


def test_kubectl_worker_crash_reuses_exact_exit_proof_until_pod_deletion():
    identity = _expected()["d1"]
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    backend._assert_node_available = lambda namespace, expected: None
    terminated = _terminated_state(identity)
    modified_raw_text = _raw_watch_event(
        identity,
        event_type="MODIFIED",
        resource_version="rv-terminated",
        restart_count=1,
        current_container_id="container-new-d1",
        last_termination=terminated,
    )
    watch = _watch_with_event(
        identity,
        backend._decode_raw_watch_event(
            modified_raw_text, identity.member
        ),
    )
    raw_events = []

    def sink(value):
        raw_events.append(value)
        return len(raw_events), value.raw_sha256

    injected = backend.wait_injected_process_termination(
        "release",
        "namespace",
        watch,
        identity,
        raw_observation_sink=sink,
    )

    assert injected.container_id == identity.container_id
    assert injected.exit_code == 137
    assert injected.container_terminated is True
    assert injected.pod_deleted is False
    deleted_raw_text = _raw_watch_event(
        identity,
        event_type="DELETED",
        resource_version="rv-deleted",
        restart_count=1,
        current_container_id="container-new-d1",
        last_termination=terminated,
    )
    deleted_raw_event = json.loads(deleted_raw_text)
    watch.events.put(backend._decode_raw_watch_event(
        deleted_raw_text, identity.member,
    ))
    complete_observation = backend.observe_termination(
        "release", "namespace", watch, identity,
        prior_termination=injected,
        raw_observation_sink=sink,
    )
    complete = complete_observation.proof
    assert complete.member == identity.member
    assert complete.container_id == identity.container_id
    assert complete.exit_code == 137
    assert complete.pod_deleted is True
    assert complete.restart_count_before == 0
    assert complete.restart_count_observed == 1
    assert complete.termination_source == "lastState.terminated"
    assert complete.adjacent_current_container_id == "container-new-d1"
    assert complete.termination_raw_observation_sequence == 1
    assert complete.deletion_raw_observation_sequence == 2
    assert complete_observation.raw_watch_event == deleted_raw_event
    assert complete_observation.raw_watch_event_text == deleted_raw_text
    assert complete_observation.raw_watch_event_sha256 == (
        "sha256:"
        + hashlib.sha256(deleted_raw_text.encode("utf-8")).hexdigest()
    )
    assert complete_observation.resource_version == "rv-deleted"


def test_kubectl_worker_crash_uses_identity_bound_pidfd_helper(monkeypatch):
    identity = _expected()["d0"]
    selected = {
        "schema_version": 1,
        "component": "worker",
        "instance_id": "d0",
        "pod_uid": identity.pod_uid,
        "process_generation": identity.process_generation,
        "instance_epoch": f"{identity.pod_uid}:{identity.process_generation}",
        "app_pid": 8,
        "process_start_ticks": 12345,
        "signal": 9,
        "pidfd": True,
        "identity_sha256": "sha256:" + "b" * 64,
    }
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(selected), stderr=""
        )

    monkeypatch.setattr("pd_worldctl.subprocess.run", fake_run)
    observed = KubectlGatewayBackend(
        "http://gateway", timeout_s=1.25, context="ack-week12"
    ).inject_process_fault("release", "namespace", identity)

    command, run_options = commands[0]
    assert command[:5] == [
        "kubectl", "--context", "ack-week12", "-n", "namespace"
    ]
    assert command[-8:] == [
        "--expected-component", "worker",
        "--expected-instance-id", "d0",
        "--expected-pod-uid", identity.pod_uid,
        "--expected-process-generation", identity.process_generation,
    ]
    assert "kill -KILL 1" not in " ".join(command)
    assert run_options["timeout"] == pytest.approx(1.25)
    assert run_options["timeout"] > 0
    assert observed["app_pid"] == 8
    assert observed["exec_return_code"] == 0


def test_kubectl_watchdog_uses_old_last_state_after_same_pod_container_restart():
    identity = _expected()["d1"]
    message = _watchdog_message(identity, "transfer-timeout")
    terminated = {
        "containerID": identity.container_id,
        "exitCode": 70,
        "reason": "Error",
        "signal": 0,
        "message": message,
        "startedAt": "2026-07-19T11:59:59Z",
        "finishedAt": "2026-07-19T12:00:00Z",
    }
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    backend._assert_node_available = lambda namespace, expected: None
    modified = _raw_watch_event(
        identity,
        event_type="MODIFIED",
        resource_version="rv-terminated",
        restart_count=1,
        current_container_id="container-new-d1",
        last_termination=terminated,
    )
    watch = _watch_with_event(
        identity,
        backend._decode_raw_watch_event(modified, identity.member),
    )
    raw_events = []

    def sink(value):
        raw_events.append(value)
        return len(raw_events), value.raw_sha256

    proof = backend.wait_natural_watchdog_termination(
        "release",
        "namespace",
        watch,
        identity,
        expected_generation="world-a",
        expected_operation_ids=("transfer-timeout",),
        raw_observation_sink=sink,
    )

    assert proof.container_id == identity.container_id
    assert proof.exit_code == 70
    assert proof.pod_deleted is False
    assert proof.termination_message == message
    deleted = _raw_watch_event(
        identity,
        event_type="DELETED",
        resource_version="rv-deleted",
        restart_count=1,
        current_container_id="container-new-d1",
        last_termination=terminated,
    )
    watch.events.put(backend._decode_raw_watch_event(
        deleted, identity.member,
    ))
    complete = backend.wait_termination(
        "release",
        "namespace",
        watch,
        identity,
        prior_termination=proof,
        raw_observation_sink=sink,
    )
    assert complete.container_id == identity.container_id
    assert complete.exit_code == 70
    assert complete.pod_deleted is True
    assert complete.termination_message == message


def test_kubectl_watchdog_rejects_foreign_operation_termination_message():
    identity = _expected()["d1"]
    event = "\t".join((
        "MODIFIED", identity.pod_uid, identity.node_name,
        identity.container_id, identity.container_id, "70", "Error",
        _watchdog_message(identity, "foreign-operation"),
        "2026-07-19T12:00:00Z", "", "", "", "", "", "rv-next",
    ))
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    backend._assert_node_available = lambda namespace, expected: None

    with pytest.raises(RuntimeError, match="identity mismatch"):
        backend.wait_natural_watchdog_termination(
            "release",
            "namespace",
            _watch_with_event(identity, event),
            identity,
            expected_generation="world-a",
            expected_operation_ids=("transfer-timeout",),
        )


def test_kubectl_watch_rejects_direct_delete_without_exact_container_termination():
    identity = _expected()["p0"]
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    backend._assert_node_available = lambda namespace, expected: None
    watch = _watch_with_event(
        identity,
        f"DELETED\t{identity.pod_uid}\t{identity.node_name}\t"
        f"{identity.container_id}\t\trv-next",
    )

    with pytest.raises(RuntimeError, match="without exact container state.terminated"):
        backend.wait_termination("release", "namespace", watch, identity)


def test_kubectl_watch_rejects_node_unavailable_and_wrong_container():
    identity = _expected()["p0"]
    unavailable = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    unavailable._assert_node_available = lambda namespace, expected: (_ for _ in ()).throw(
        RuntimeError("node is not Ready")
    )
    watch = _watch_with_event(
        identity,
        f"MODIFIED\t{identity.pod_uid}\t{identity.node_name}\t"
        f"{identity.container_id}\t137\trv-next",
    )
    with pytest.raises(RuntimeError, match="node is not Ready"):
        unavailable.wait_termination("release", "namespace", watch, identity)

    wrong_container = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    wrong_container._assert_node_available = lambda namespace, expected: None
    watch = _watch_with_event(
        identity,
        f"MODIFIED\t{identity.pod_uid}\t{identity.node_name}\t"
        "container-other\t137\trv-next",
    )
    with pytest.raises(RuntimeError, match="container id drifted"):
        wrong_container.wait_termination("release", "namespace", watch, identity)


def test_pd_worldctl_sends_full_physical_termination_records():
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    backend._json_url = lambda url, body=None: {
        "ready": True,
        "resource_reports": {member: {} for member in MEMBERS},
        "new_topology_generation": "world-b",
    }

    evidence = backend.wait_acceptance_evidence(
        old_generation="world-a", new_generation="world-b",
        restart_run_id="run-a", termination_proofs=tuple(
            _proof(member) for member in MEMBERS
        ),
    )

    required = {
        "logical_instance_id",
        "topology_generation",
        "pod_uid",
        "node_uid",
        "container_name",
        "captured_container_id",
        "process_generation",
        "watch_start_resource_version",
        "observed_resource_version",
        "deletion_resource_version",
        "restart_count_before",
        "restart_count_observed",
        "termination_source",
        "termination_event_type",
        "deletion_event_type",
        "terminated",
        "adjacent_current_container_id",
        "pod_deletion_observed",
        "raw_pod_json_sha256",
        "termination_raw_observation_sequence",
        "deletion_raw_pod_json_sha256",
        "deletion_raw_observation_sequence",
        "observation_sha256",
    }
    assert len(evidence["termination_records"]) == 4
    assert all(
        required == set(value)
        for value in evidence["termination_records"]
    )
    assert {
        value["logical_instance_id"]
        for value in evidence["termination_records"]
    } == set(MEMBERS)
    assert all(
        value["pod_deletion_observed"]
        for value in evidence["termination_records"]
    )


def test_pd_worldctl_response_loss_resumes_exact_durable_evidence(tmp_path):
    class ResponseLossBackend(_FakeActuatorBackend):
        def __init__(self):
            super().__init__()
            self.accepted_evidence = []

        def accept_topology(self, evidence):
            self.events.append("admin-accept")
            self.accepted_evidence.append(evidence)
            if len(self.accepted_evidence) == 1:
                raise TimeoutError("accept response lost")
            return {"accepted": True, "restart_run_id": evidence["restart_run_id"]}

    backend = ResponseLossBackend()
    actuator = WorldRestartActuator(backend)
    state_path = tmp_path / "run-state.json"
    kwargs = {
        "release": "release",
        "namespace": "namespace",
        "chart": "./chart",
        "old_generation": "world-a",
        "new_generation": "world-b",
        "run_state_path": state_path,
    }

    with pytest.raises(TimeoutError, match="response lost"):
        actuator.restart(**kwargs)
    persisted = state_path.read_text(encoding="utf-8")
    assert '"phase":"EVIDENCE_READY"' in persisted

    result = actuator.restart(**kwargs)

    assert result["accepted"] is True
    assert backend.accepted_evidence[0] == backend.accepted_evidence[1]
    assert backend.events.count("capture") == 1
    assert backend.events.count("admin-accept") == 2


def _raw_watch_event(
    identity: ProcessIdentity,
    *,
    event_type: str,
    resource_version: str,
    restart_count: int,
    current_container_id: str,
    current_termination: dict[str, object] | None = None,
    last_termination: dict[str, object] | None = None,
) -> str:
    event = {
        "type": event_type,
        "object": {
            "metadata": {
                "uid": identity.pod_uid,
                "resourceVersion": resource_version,
            },
            "spec": {"nodeName": identity.node_name},
            "status": {
                "containerStatuses": [{
                    "name": identity.container_name,
                    "containerID": current_container_id,
                    "restartCount": restart_count,
                    "state": (
                        {"terminated": current_termination}
                        if current_termination is not None else {}
                    ),
                    "lastState": (
                        {"terminated": last_termination}
                        if last_termination is not None else {}
                    ),
                }],
            },
        },
    }
    return json.dumps(event, separators=(",", ":"))


def _terminated_state(
    identity: ProcessIdentity,
    *,
    exit_code: int = 137,
) -> dict[str, object]:
    return {
        "containerID": identity.container_id,
        "exitCode": exit_code,
        "reason": "Error",
        "signal": 9 if exit_code == 137 else 0,
        "message": "terminated",
        "startedAt": "2026-07-23T11:59:59Z",
        "finishedAt": "2026-07-23T12:00:00Z",
    }


def test_capture_world_freezes_generation_container_restart_count_and_cursor():
    backend = KubectlGatewayBackend("http://gateway")
    identities = {
        name: {
            "instance_id": name,
            "pod_uid": f"old-pod-{name}",
            "process_generation": f"process-{name}",
            "topology_generation": "world-a",
        }
        for name in MEMBERS
    }
    backend._json_url_with_timeout = lambda url, **_kwargs: {
        "topology_generation": "world-a",
        "identities": list(identities.values()),
        "resource_reports": {name: {} for name in MEMBERS},
    }

    def fake_kubectl_json(namespace, *args):
        if args[:2] == ("get", "pods"):
            member = str(args[-1]).rsplit("=", 1)[-1]
            return {
                "items": [{
                    "metadata": {
                        "name": f"prism-{member}-0",
                        "uid": f"old-pod-{member}",
                        "resourceVersion": "100",
                    },
                    "spec": {"nodeName": f"node-name-{member}"},
                    "status": {
                        "containerStatuses": [{
                            "name": "worker",
                            "containerID": f"container-{member}",
                            "restartCount": 7,
                        }],
                    },
                }],
            }
        node_name = str(args[-1])
        member = node_name.rsplit("-", 1)[-1]
        return {
            "metadata": {"uid": f"node-{member}"},
            "status": {
                "conditions": [{"type": "Ready", "status": "True"}],
            },
        }

    backend._kubectl_json = fake_kubectl_json

    captured = backend.capture_world(
        "release", "namespace", expected_generation="world-a"
    )

    assert all(
        identity.topology_generation == "world-a"
        and identity.container_name == "worker"
        and identity.restart_count == 7
        and identity.resource_version == "100"
        for identity in captured.values()
    )


def test_gateway_topology_read_retries_remote_disconnect(monkeypatch):
    backend = KubectlGatewayBackend("http://gateway", timeout_s=1.0)
    expected = {"topology_generation": "world-a"}
    outcomes = iter((
        RemoteDisconnected("restart closed the connection"),
        expected,
    ))
    calls = []

    def read(_url, *, timeout):
        assert 0 < timeout <= 1.0
        calls.append("read")
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    backend._json_url_with_timeout = read
    monkeypatch.setattr(pd_worldctl.time, "sleep", lambda _seconds: None)

    assert backend._read_gateway_json("http://gateway/admin/topology") == expected
    assert calls == ["read", "read"]


def test_gateway_topology_read_does_not_retry_authorization_failure(monkeypatch):
    backend = KubectlGatewayBackend("http://gateway", timeout_s=1.0)
    calls = []

    def read(url, *, timeout):
        assert 0 < timeout <= 1.0
        calls.append(url)
        raise HTTPError(url, 403, "forbidden", hdrs=None, fp=None)

    backend._json_url_with_timeout = read
    monkeypatch.setattr(
        pd_worldctl.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(
            AssertionError("authorization failure must not be retried")
        ),
    )

    with pytest.raises(HTTPError) as exc_info:
        backend._read_gateway_json("http://gateway/admin/topology")

    assert exc_info.value.code == 403
    assert calls == ["http://gateway/admin/topology"]


def test_gateway_topology_read_caps_attempt_at_remaining_deadline(monkeypatch):
    backend = KubectlGatewayBackend("http://gateway", timeout_s=1.0)
    observed_timeouts = []
    clock = iter((10.0, 10.0, 11.1))
    monkeypatch.setattr(pd_worldctl.time, "monotonic", lambda: next(clock))

    def read(_url, *, timeout):
        observed_timeouts.append(timeout)
        raise RemoteDisconnected("hung read ended at its supplied timeout")

    backend._json_url_with_timeout = read

    with pytest.raises(
        TimeoutError,
        match="before world mutation",
    ):
        backend._read_gateway_json("http://gateway/admin/topology")

    assert observed_timeouts == [1.0]


def test_gateway_topology_read_uses_dedicated_pre_mutation_budget(monkeypatch):
    backend = KubectlGatewayBackend(
        "http://gateway", timeout_s=1800.0, gateway_read_timeout_s=30.0
    )
    observed_timeouts = []
    clock = iter((10.0, 10.0, 40.1))
    monkeypatch.setattr(pd_worldctl.time, "monotonic", lambda: next(clock))

    def read(_url, *, timeout):
        observed_timeouts.append(timeout)
        raise RemoteDisconnected("Gateway remains unavailable")

    backend._json_url_with_timeout = read

    with pytest.raises(TimeoutError, match="before world mutation"):
        backend._read_gateway_json("http://gateway/admin/topology")

    assert observed_timeouts == [30.0]
    assert backend.timeout_s == 1800.0


@pytest.mark.parametrize(
    ("restart_count", "adjacent_container", "message"),
    [
        (99, "container-new-d1", "restartCount"),
        (1, "", "adjacent current container"),
        (0, "container-new-d1", "restartCount"),
    ],
    ids=["large-restart-jump", "missing-adjacent", "not-incremented"],
)
def test_last_state_termination_requires_exact_single_restart_and_adjacent(
    restart_count,
    adjacent_container,
    message,
):
    identity = _expected()["d1"]
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    backend._assert_node_available = lambda namespace, expected: None
    raw = _raw_watch_event(
        identity,
        event_type="MODIFIED",
        resource_version="101",
        restart_count=restart_count,
        current_container_id=adjacent_container,
        last_termination=_terminated_state(identity),
    )
    watch = _watch_with_event(
        identity,
        backend._decode_raw_watch_event(raw, identity.member),
    )

    with pytest.raises(RuntimeError, match=message):
        backend.observe_injected_process_termination(
            "release", "namespace", watch, identity
        )


def test_current_state_termination_requires_unchanged_restart_count():
    identity = _expected()["p0"]
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    backend._assert_node_available = lambda namespace, expected: None
    raw = _raw_watch_event(
        identity,
        event_type="MODIFIED",
        resource_version="101",
        restart_count=1,
        current_container_id=identity.container_id,
        current_termination=_terminated_state(identity),
    )

    with pytest.raises(RuntimeError, match="restartCount"):
        backend.observe_injected_process_termination(
            "release",
            "namespace",
            _watch_with_event(
                identity,
                backend._decode_raw_watch_event(raw, identity.member),
            ),
            identity,
        )


def test_termination_authority_links_modified_and_deleted_raw_transitions(
    tmp_path,
):
    identity = _expected()["p0"]
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    backend._assert_node_available = lambda namespace, expected: None
    modified = _raw_watch_event(
        identity,
        event_type="MODIFIED",
        resource_version="101",
        restart_count=0,
        current_container_id=identity.container_id,
        current_termination=_terminated_state(identity),
    )
    deleted = _raw_watch_event(
        identity,
        event_type="DELETED",
        resource_version="101",
        restart_count=0,
        current_container_id=identity.container_id,
        current_termination=_terminated_state(identity),
    )
    events = Queue()
    events.put(backend._decode_raw_watch_event(modified, identity.member))
    events.put(backend._decode_raw_watch_event(deleted, identity.member))
    watch = PodTerminationWatch(
        identity, _LiveWatchProcess(), events, None
    )
    state_path = tmp_path / "run-state.json"
    run_state = {"observation_journal": []}

    def sink(event):
        return pd_worldctl._append_raw_watch_observation(
            run_state,
            state_path,
            stage="whole_world_termination",
            event=event,
        )

    observed = backend.observe_termination(
        "release",
        "namespace",
        watch,
        identity,
        raw_observation_sink=sink,
    )
    record = observed.proof

    assert record.observed_resource_version == "101"
    assert record.deletion_resource_version == "101"
    assert record.termination_raw_observation_sequence == 1
    assert record.deletion_raw_observation_sequence == 2
    assert record.termination_raw_pod_json_sha256 == (
        record.deletion_raw_pod_json_sha256
    )
    assert record.pod_deleted is True


def test_termination_authority_keeps_first_equivalent_modified_proof(
    tmp_path,
):
    identity = _expected()["p0"]
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    backend._assert_node_available = lambda namespace, expected: None
    first = json.loads(_raw_watch_event(
        identity,
        event_type="MODIFIED",
        resource_version="101",
        restart_count=0,
        current_container_id=identity.container_id,
        current_termination=_terminated_state(identity),
    ))
    first["object"]["status"]["resources"] = {
        "requests": {"cpu": "3996m", "memory": "32Gi"},
    }
    duplicate = json.loads(json.dumps(first))
    duplicate["object"]["metadata"]["resourceVersion"] = "102"
    del duplicate["object"]["status"]["resources"]["requests"]["cpu"]
    deleted = json.loads(json.dumps(duplicate))
    deleted["type"] = "DELETED"
    deleted["object"]["metadata"]["resourceVersion"] = "103"
    events = Queue()
    for event in (first, duplicate, deleted):
        raw = json.dumps(event, separators=(",", ":"))
        events.put(backend._decode_raw_watch_event(raw, identity.member))
    watch = PodTerminationWatch(
        identity, _LiveWatchProcess(), events, None
    )
    state_path = tmp_path / "run-state.json"
    run_state = {"observation_journal": []}

    def sink(event):
        return pd_worldctl._append_raw_watch_observation(
            run_state,
            state_path,
            stage="whole_world_termination",
            event=event,
        )

    observed = backend.observe_termination(
        "release",
        "namespace",
        watch,
        identity,
        raw_observation_sink=sink,
    )
    record = observed.proof

    assert record.observed_resource_version == "101"
    assert record.deletion_resource_version == "103"
    assert record.termination_raw_observation_sequence == 1
    assert record.deletion_raw_observation_sequence == 3
    assert record.termination_raw_pod_json_sha256 == (
        pd_worldctl._canonical_sha256(first["object"])
    )
    assert [
        item["raw_kubernetes_watch_event"]["object"]["metadata"][
            "resourceVersion"
        ]
        for item in run_state["observation_journal"]
    ] == ["101", "102", "103"]


def test_termination_authority_keeps_current_proof_after_container_restart(
    tmp_path,
):
    identity = _expected()["p0"]
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    backend._assert_node_available = lambda namespace, expected: None
    termination = _terminated_state(identity)
    current = _raw_watch_event(
        identity,
        event_type="MODIFIED",
        resource_version="101",
        restart_count=0,
        current_container_id=identity.container_id,
        current_termination=termination,
    )
    replacement_container_id = "containerd://replacement"
    restarted = _raw_watch_event(
        identity,
        event_type="MODIFIED",
        resource_version="102",
        restart_count=1,
        current_container_id=replacement_container_id,
        last_termination=termination,
    )
    deleted = _raw_watch_event(
        identity,
        event_type="DELETED",
        resource_version="103",
        restart_count=1,
        current_container_id=replacement_container_id,
        last_termination=termination,
    )
    state_path = tmp_path / "run-state.json"
    run_state = {"observation_journal": []}

    def sink(event):
        return pd_worldctl._append_raw_watch_observation(
            run_state,
            state_path,
            stage="whole_world_termination",
            event=event,
        )

    injected = backend.observe_injected_process_termination(
        "release",
        "namespace",
        _watch_with_event(
            identity,
            backend._decode_raw_watch_event(current, identity.member),
        ),
        identity,
        raw_observation_sink=sink,
    )
    events = Queue()
    for raw in (restarted, deleted):
        events.put(backend._decode_raw_watch_event(raw, identity.member))
    observed = backend.observe_termination(
        "release",
        "namespace",
        PodTerminationWatch(
            identity, _LiveWatchProcess(), events, None
        ),
        identity,
        prior_termination=injected.proof,
        raw_observation_sink=sink,
    )

    record = observed.proof
    assert record.termination_source == "state.terminated"
    assert record.restart_count_observed == 0
    assert record.adjacent_current_container_id is None
    assert record.observed_resource_version == "101"
    assert record.deletion_resource_version == "103"
    assert record.termination_raw_observation_sequence == 1
    assert record.deletion_raw_observation_sequence == 3
    assert record.pod_deleted is True
    sealed = pd_worldctl._validate_durable_termination_record(
        run_state, record,
    )
    assert sealed["termination_source"] == "state.terminated"


def test_termination_authority_rejects_last_state_projection_regression(
    tmp_path,
):
    identity = _expected()["p0"]
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    backend._assert_node_available = lambda namespace, expected: None
    termination = _terminated_state(identity)
    replacement_container_id = "containerd://replacement"
    last_state = _raw_watch_event(
        identity,
        event_type="MODIFIED",
        resource_version="101",
        restart_count=1,
        current_container_id=replacement_container_id,
        last_termination=termination,
    )
    regressed = _raw_watch_event(
        identity,
        event_type="MODIFIED",
        resource_version="102",
        restart_count=0,
        current_container_id=identity.container_id,
        current_termination=termination,
    )
    state_path = tmp_path / "run-state.json"
    run_state = {"observation_journal": []}

    def sink(event):
        return pd_worldctl._append_raw_watch_observation(
            run_state,
            state_path,
            stage="whole_world_termination",
            event=event,
        )

    injected = backend.observe_injected_process_termination(
        "release",
        "namespace",
        _watch_with_event(
            identity,
            backend._decode_raw_watch_event(last_state, identity.member),
        ),
        identity,
        raw_observation_sink=sink,
    )

    with pytest.raises(
        pd_worldctl.TerminationObservationError,
        match="termination proof changed for p0",
    ):
        backend.observe_termination(
            "release",
            "namespace",
            _watch_with_event(
                identity,
                backend._decode_raw_watch_event(
                    regressed, identity.member,
                ),
            ),
            identity,
            prior_termination=injected.proof,
            raw_observation_sink=sink,
        )


def test_termination_authority_rejects_changed_payload_during_restart_projection(
    tmp_path,
):
    identity = _expected()["p0"]
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    backend._assert_node_available = lambda namespace, expected: None
    termination = _terminated_state(identity)
    current = _raw_watch_event(
        identity,
        event_type="MODIFIED",
        resource_version="101",
        restart_count=0,
        current_container_id=identity.container_id,
        current_termination=termination,
    )
    changed_termination = dict(termination)
    changed_termination["finishedAt"] = "2026-07-23T12:00:01Z"
    restarted = _raw_watch_event(
        identity,
        event_type="MODIFIED",
        resource_version="102",
        restart_count=1,
        current_container_id="containerd://replacement",
        last_termination=changed_termination,
    )
    state_path = tmp_path / "run-state.json"
    run_state = {"observation_journal": []}

    def sink(event):
        return pd_worldctl._append_raw_watch_observation(
            run_state,
            state_path,
            stage="whole_world_termination",
            event=event,
        )

    injected = backend.observe_injected_process_termination(
        "release",
        "namespace",
        _watch_with_event(
            identity,
            backend._decode_raw_watch_event(current, identity.member),
        ),
        identity,
        raw_observation_sink=sink,
    )

    with pytest.raises(
        pd_worldctl.TerminationObservationError,
        match="termination proof changed for p0",
    ):
        backend.observe_termination(
            "release",
            "namespace",
            _watch_with_event(
                identity,
                backend._decode_raw_watch_event(restarted, identity.member),
            ),
            identity,
            prior_termination=injected.proof,
            raw_observation_sink=sink,
        )


def test_termination_authority_rejects_changed_duplicate_semantics(
    tmp_path,
):
    identity = _expected()["p0"]
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    backend._assert_node_available = lambda namespace, expected: None
    termination = _terminated_state(identity)
    first = _raw_watch_event(
        identity,
        event_type="MODIFIED",
        resource_version="101",
        restart_count=0,
        current_container_id=identity.container_id,
        current_termination=termination,
    )
    changed_termination = dict(termination)
    changed_termination["finishedAt"] = "2026-07-23T12:00:01Z"
    changed = _raw_watch_event(
        identity,
        event_type="MODIFIED",
        resource_version="102",
        restart_count=0,
        current_container_id=identity.container_id,
        current_termination=changed_termination,
    )
    events = Queue()
    for raw in (first, changed):
        events.put(backend._decode_raw_watch_event(raw, identity.member))
    watch = PodTerminationWatch(
        identity, _LiveWatchProcess(), events, None
    )
    state_path = tmp_path / "run-state.json"
    run_state = {"observation_journal": []}

    def sink(event):
        return pd_worldctl._append_raw_watch_observation(
            run_state,
            state_path,
            stage="whole_world_termination",
            event=event,
        )

    with pytest.raises(
        pd_worldctl.TerminationObservationError,
        match="termination proof changed for p0",
    ):
        backend.observe_termination(
            "release",
            "namespace",
            watch,
            identity,
            raw_observation_sink=sink,
        )


@pytest.mark.parametrize(
    ("failure", "return_value", "error_type"),
    [
        (
            "response-loss",
            {
                "command": ["kubectl", "exec"],
                "exec_return_code": 1,
                "stdout": "",
                "stderr": "connection reset",
            },
            "RuntimeError",
        ),
        (
            "bad-json",
            {
                "command": ["kubectl", "exec"],
                "exec_return_code": 0,
                "stdout": "not-json",
                "stderr": "",
            },
            "RuntimeError",
        ),
        ("timeout", None, "TimeoutError"),
        ("subprocess-timeout", None, "TimeoutExpired"),
    ],
)
def test_fault_injection_attempt_and_incomplete_result_are_durable(
    tmp_path,
    monkeypatch,
    failure,
    return_value,
    error_type,
):
    class InjectionFailureBackend(_FakeActuatorBackend):
        raw_observation_sink_supported = True
        context = ""
        timeout_s = 0.1

        def _kubectl_args(self, *args):
            return KubectlGatewayBackend._kubectl_args(self, *args)

        def inject_process_fault(self, release, namespace, identity):
            self.events.append(f"crash:{identity.member}:{identity.container_id}")
            if failure == "timeout":
                raise TimeoutError("exec response lost")
            if failure == "subprocess-timeout":
                return KubectlGatewayBackend.inject_process_fault(
                    self, release, namespace, identity,
                )
            return return_value

        def observe_termination(
            self,
            release,
            namespace,
            watch,
            identity,
            *,
            raw_observation_sink,
        ):
            del release, watch
            real = KubectlGatewayBackend(
                "http://gateway", timeout_s=0.1
            )
            real._assert_node_available = (
                lambda observed_namespace, expected: None
            )
            modified = _raw_watch_event(
                identity,
                event_type="MODIFIED",
                resource_version="101",
                restart_count=0,
                current_container_id=identity.container_id,
                current_termination=_terminated_state(identity),
            )
            deleted = _raw_watch_event(
                identity,
                event_type="DELETED",
                resource_version="101",
                restart_count=0,
                current_container_id=identity.container_id,
                current_termination=_terminated_state(identity),
            )
            events = Queue()
            events.put(real._decode_raw_watch_event(
                modified, identity.member
            ))
            events.put(real._decode_raw_watch_event(
                deleted, identity.member
            ))
            return real.observe_termination(
                "release",
                namespace,
                PodTerminationWatch(
                    identity, _LiveWatchProcess(), events, None
                ),
                identity,
                raw_observation_sink=raw_observation_sink,
            )

    if failure == "subprocess-timeout":
        def timeout_run(command, **kwargs):
            assert kwargs["timeout"] > 0
            raise subprocess.TimeoutExpired(
                command,
                timeout=kwargs["timeout"],
                output="",
                stderr="",
            )

        monkeypatch.setattr("pd_worldctl.subprocess.run", timeout_run)
    backend = InjectionFailureBackend()
    state_path = tmp_path / f"injection-{failure}.json"
    with pytest.raises((RuntimeError, TimeoutError, subprocess.TimeoutExpired)):
        WorldRestartActuator(backend).restart(
            release="release",
            namespace="namespace",
            chart="./chart",
            old_generation="world-a",
            new_generation="world-b",
            run_state_path=state_path,
            fault_member="d1",
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    attempts = [
        value for value in state["observation_journal"]
        if value["kind"] == "fault_injection_attempt"
    ]
    incomplete = [
        value for value in state["observation_journal"]
        if value["kind"] == "fault_injection_incomplete"
    ]
    assert len(attempts) == len(incomplete) == 1
    assert attempts[0]["expected_identity"] == {
        "logical_instance_id": "d1",
        "topology_generation": "world-a",
        "pod_uid": "old-pod-d1",
        "node_uid": "node-d1",
        "container_name": "worker",
        "captured_container_id": "container-d1",
        "process_generation": "process-d1",
        "watch_start_resource_version": "100",
        "restart_count_before": 0,
    }
    assert incomplete[0]["attempt_sequence"] == attempts[0]["sequence"]
    assert incomplete[0]["observation_status"] == "INCOMPLETE"
    assert incomplete[0]["error_type"] == error_type
    assert {
        "command", "exec_return_code", "stdout", "stderr"
    } <= set(incomplete[0])
    raw = [
        value for value in state["observation_journal"]
        if value["kind"] == "raw_kubernetes_watch_event_observed"
        and value["stage"] == "worker_crash_injection_ambiguous"
    ]
    assert [value["raw_kubernetes_watch_event"]["type"] for value in raw] == [
        "MODIFIED",
        "DELETED",
    ]
    frozen = [
        value for value in state["observation_journal"]
        if value["kind"] == "termination_observed"
        and value["stage"] == "worker_crash_injection_ambiguous"
    ]
    assert len(frozen) == 1
    assert frozen[0]["observation_status"] == "INCOMPLETE"
    if failure == "subprocess-timeout":
        assert incomplete[0]["command"][:5] == [
            "kubectl", "-n", "namespace", "exec", _expected()["d1"].pod_name,
        ]
        assert incomplete[0]["exec_return_code"] is None
        assert "process_selection" not in state["fault"]
        assert "observed_exit_code" not in state["fault"]
        assert not any(
            value["name"] == "injected_process_exited"
            for value in state["protocol_events"]
        )
    assert not any(
        event.startswith(("stop:", "patch:", "start:"))
        for event in backend.events
    )


def test_raw_sink_persists_object_pod_digest_rv_and_timeout_links_last_raw(
    tmp_path,
):
    identity = _expected()["p0"]
    backend = KubectlGatewayBackend("http://gateway", timeout_s=0.1)
    backend._assert_node_available = lambda namespace, expected: None
    running = _raw_watch_event(
        identity,
        event_type="MODIFIED",
        resource_version="101",
        restart_count=0,
        current_container_id=identity.container_id,
    )
    event = backend._decode_raw_watch_event(running, identity.member)
    events = Queue()
    events.put(event)

    class EndedWatch:
        stderr = io.StringIO("watch ended")

        @staticmethod
        def poll():
            return 0

    watch = PodTerminationWatch(identity, EndedWatch(), events, None)
    run_state = {"observation_journal": []}
    state_path = tmp_path / "raw-timeout.json"

    def sink(value):
        return pd_worldctl._append_raw_watch_observation(
            run_state,
            state_path,
            stage="whole_world_termination",
            event=value,
        )

    with pytest.raises(Exception) as captured:
        backend.observe_termination(
            "release",
            "namespace",
            watch,
            identity,
            raw_observation_sink=sink,
        )
    pd_worldctl._append_incomplete_observation(
        run_state,
        state_path,
        stage="whole_world_termination",
        error=captured.value,
    )

    raw, incomplete = run_state["observation_journal"]
    assert raw["raw_kubernetes_watch_event"] == json.loads(running)
    assert raw["raw_pod_json"] == json.loads(running)["object"]
    assert raw["resource_version"] == "101"
    assert raw["raw_pod_json_sha256"].startswith("sha256:")
    assert incomplete["raw_observation_sequence"] == raw["sequence"]
    assert incomplete["raw_kubernetes_watch_event_sha256"] == (
        raw["raw_kubernetes_watch_event_sha256"]
    )
    assert "no_raw_reason" not in incomplete

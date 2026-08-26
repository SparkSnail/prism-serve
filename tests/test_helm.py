"""Rendered deployment invariants for the process-local Gateway authority."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from prism_serve.config import Settings
from prism_serve.gateway.app import (
    GATEWAY_BOOTSTRAP_TIMEOUT_S,
    GATEWAY_STARTUP_PROBE_MARGIN_S,
    GOVERNOR_SHUTDOWN_TIMEOUT_S,
    LOOP_SHUTDOWN_TIMEOUT_S,
    NETWORK_CLEANUP_SHUTDOWN_TIMEOUT_S,
    RUNTIME_IO_CLOSE_SHUTDOWN_TIMEOUT_S,
    UVICORN_GRACEFUL_SHUTDOWN_TIMEOUT_S,
)


def _rendered_document(rendered: str, template: str) -> str:
    source = f"# Source: prism-serve/templates/{template}"
    return rendered.split(source, 1)[1].split("---", 1)[0]


def _render_chart(*args: str) -> str:
    chart = Path(__file__).parents[1] / "k8s" / "helm" / "prism-serve"
    return subprocess.run(
        ["helm", "template", "week12", str(chart), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _performance_values() -> Path:
    return (
        Path(__file__).parents[1]
        / "k8s"
        / "helm"
        / "prism-serve"
        / "values-performance.yaml"
    )


def _performance_on_values() -> Path:
    return (
        Path(__file__).parents[1]
        / "k8s"
        / "helm"
        / "prism-serve"
        / "values-performance-on.yaml"
    )


def _manifest_int(document: str, key: str) -> int:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(\d+)\s*$", document, re.MULTILINE)
    assert match is not None, f"missing rendered integer: {key}"
    return int(match.group(1))


def _topology_payload(rendered: str) -> dict[str, object]:
    document = _rendered_document(rendered, "worker-topology-configmap.yaml")
    payload = next(
        line.strip() for line in document.splitlines()
        if line.strip().startswith("{")
    )
    return json.loads(payload)


def _configmap_json_payloads(rendered: str) -> list[dict[str, object]]:
    document = _rendered_document(rendered, "worker-topology-configmap.yaml")
    return [
        json.loads(line.strip())
        for line in document.splitlines()
        if line.strip().startswith("{")
    ]


def test_gateway_rollout_has_no_active_overlap() -> None:
    chart = Path(__file__).parents[1] / "k8s" / "helm" / "prism-serve"
    rendered = subprocess.run(
        ["helm", "template", "week10", str(chart)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    gateway = rendered.split("# Source: prism-serve/templates/gateway-deployment.yaml", 1)[1]
    gateway = gateway.split("---", 1)[0]
    assert "replicas: 1" in gateway
    assert "strategy:\n    type: Recreate" in gateway
    assert "RollingUpdate" not in gateway
    assert "maxSurge" not in gateway


def test_gateway_lifecycle_budgets_cover_bootstrap_and_shutdown() -> None:
    gateway = _rendered_document(
        _render_chart(), "gateway-deployment.yaml"
    )
    startup_probe = gateway.split("startupProbe:", 1)[1].split(
        "livenessProbe:", 1
    )[0]
    startup_budget_s = (
        _manifest_int(startup_probe, "periodSeconds")
        * _manifest_int(startup_probe, "failureThreshold")
    )
    setting_defaults = Settings.model_fields
    required_startup_s = (
        GATEWAY_BOOTSTRAP_TIMEOUT_S
        + GATEWAY_STARTUP_PROBE_MARGIN_S
    )
    required_shutdown_s = (
        UVICORN_GRACEFUL_SHUTDOWN_TIMEOUT_S
        + float(setting_defaults["shutdown_drain_timeout_s"].default)
        + LOOP_SHUTDOWN_TIMEOUT_S
        + NETWORK_CLEANUP_SHUTDOWN_TIMEOUT_S
        + GOVERNOR_SHUTDOWN_TIMEOUT_S
        + RUNTIME_IO_CLOSE_SHUTDOWN_TIMEOUT_S
    )

    assert GATEWAY_BOOTSTRAP_TIMEOUT_S == 570.0
    assert GATEWAY_STARTUP_PROBE_MARGIN_S == 30.0
    assert startup_budget_s == 720
    assert startup_budget_s >= required_startup_s
    assert _manifest_int(
        gateway, "terminationGracePeriodSeconds"
    ) >= required_shutdown_s


def test_rendered_worker_world_is_exact_2p2d() -> None:
    chart = Path(__file__).parents[1] / "k8s" / "helm" / "prism-serve"
    rendered = subprocess.run(
        ["helm", "template", "week12", str(chart)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert rendered.count("kind: StatefulSet") == 4
    assert rendered.count("type: OnDelete") == 4
    assert rendered.count("replicas: 1") == 5  # four workers + one Gateway
    for member, role, rank in (
        ("p0", "prefill", "0"), ("p1", "prefill", "1"),
        ("d0", "decode", "2"), ("d1", "decode", "3"),
    ):
        assert f"prism.sparksnail.ai/member: {member}" in rendered
        assert f'value: "{role}"' in rendered
        assert f'value: "{rank}"' in rendered
    assert rendered.count("PRISM_TOPOLOGY_GENERATION") == 4
    assert rendered.count("- name: NCCL_DEBUG\n              value: INFO") == 4
    assert rendered.count(
        "- name: NCCL_DEBUG_FILE\n              value: /tmp/prism-nccl-debug.log"
    ) == 4
    assert rendered.count(
        "- name: PRISM_NCCL_DEBUG_LOG\n"
        "              value: /tmp/prism-nccl-debug.log"
    ) == 4
    assert rendered.count(
        '- name: NCCL_P2P_DISABLE\n              value: "1"'
    ) == 4
    assert rendered.count(
        '- name: NCCL_SHM_DISABLE\n              value: "1"'
    ) == 4
    assert rendered.count(
        '- name: NCCL_DEBUG_SUBSYS\n              value: "INIT,NET"'
    ) == 4
    assert rendered.count(
        "- name: PRISM_TERMINATION_LOG_PATH\n"
        "              value: /dev/termination-log"
    ) == 4
    assert rendered.count(
        '- name: PYTHONFAULTHANDLER\n              value: "1"'
    ) == 4
    assert rendered.count("terminationMessagePath: /dev/termination-log") == 4
    assert rendered.count("terminationMessagePolicy: File") == 4


def test_worker_startup_permit_mount_waits_without_model_or_nccl_init() -> None:
    default = _render_chart()
    topology = _rendered_document(
        default, "worker-topology-configmap.yaml"
    )



    assert "startup-permit.json:" not in topology
    assert {"phase": "UNINITIALIZED"} in _configmap_json_payloads(default)
    assert default.count("optional: true") == 4
    assert default.count("- key: startup-permit.json") == 4
    assert default.count("path: startup-permit.json") == 4
    assert default.count(
        "- name: PRISM_STARTUP_PERMIT_PATH\n"
        "              value: /etc/prism/topology/startup-permit.json"
    ) == 4
    assert default.count(
        "- name: PRISM_INCARNATION_RECORD_PATH\n"
        "              value: /var/run/prism/incarnation/record.json"
    ) == 4

    assert default.count("mountPath: /etc/prism/topology") == 5
    assert default.count("mountPath: /var/run/prism/incarnation") == 4
    assert default.count("emptyDir: {}") == 4

    members = {
        "p0": "pod-p0",
        "p1": "pod-p1",
        "d0": "pod-d0",
        "d1": "pod-d1",
    }
    unsigned = {
        "schema_version": "prism.week12.worker-startup-permit/v1",
        "issuance_mode": "RESTART",
        "permit_id": "permit-a",
        "topology_generation": "world-b",
        "members": members,
    }
    digest = "sha256:" + hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    permit = {**unsigned, "canonical_digest": digest}
    rendered = _render_chart(
        "--set-json",
        "worker.startupPermitJson="
        + json.dumps(permit, sort_keys=True, separators=(",", ":")),
    )

    payloads = _configmap_json_payloads(rendered)
    assert permit in payloads


def test_contract_byte_counts_render_as_decimal_integer_strings() -> None:
    rendered = _render_chart()

    assert rendered.count(
        '- name: PRISM_KV_BLOCK_BYTES\n              value: "29360128"'
    ) == 4
    assert rendered.count(
        '- name: PRISM_SERVE_PREFIX_BLOCK_BYTES\n'
        '              value: "29360128"'
    ) == 1
    assert rendered.count(
        '- name: PRISM_SERVE_KV_BLOCK_BYTES\n'
        '              value: "29360128"'
    ) == 1
    assert rendered.count(
        '- name: PRISM_SERVE_MAX_BYTES_INFLIGHT_PER_PAIR\n'
        '              value: "1073741824"'
    ) == 1
    assert "2.9360128e+07" not in rendered
    assert "1.073741824e+09" not in rendered


def test_images_support_immutable_digest_and_reject_latest() -> None:
    chart = Path(__file__).parents[1] / "k8s" / "helm" / "prism-serve"
    digest = "sha256:" + "a" * 64
    rendered = subprocess.run([
        "helm", "template", "week12", str(chart),
        "--set-string", f"gateway.image.digest={digest}",
        "--set-string", f"worker.image.digest={digest}",
    ], check=True, capture_output=True, text=True).stdout
    assert f"sparksnail/prism-serve@{digest}" in rendered
    assert rendered.count(f"sparksnail/prism-infer@{digest}") == 4

    rejected = subprocess.run([
        "helm", "template", "week12", str(chart),
        "--set-string", "gateway.image.tag=latest",
    ], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "gateway image tag latest is forbidden" in rejected.stderr


def test_gateway_replacement_store_is_rwo_and_mounted() -> None:
    chart = Path(__file__).parents[1] / "k8s" / "helm" / "prism-serve"
    rendered = subprocess.run(
        ["helm", "template", "week12", str(chart)],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "kind: PersistentVolumeClaim" in rendered
    assert "- ReadWriteOnce" in rendered
    assert 'mountPath: "/var/lib/prism/replacement"' in rendered
    assert "PRISM_SERVE_REPLACEMENT_STORE_PATH" in rendered


def test_affinity_is_default_off_and_requires_explicit_helm_override() -> None:
    chart = Path(__file__).parents[1] / "k8s" / "helm" / "prism-serve"
    default = subprocess.run(
        ["helm", "template", "week12", str(chart)],
        check=True, capture_output=True, text=True,
    ).stdout
    enabled = subprocess.run(
        ["helm", "template", "week12", str(chart), "--set", "affinity.enabled=true"],
        check=True, capture_output=True, text=True,
    ).stdout
    marker = "- name: PRISM_SERVE_AFFINITY_ENABLED"
    assert marker + '\n              value: "false"' in default
    assert marker + '\n              value: "true"' in enabled


def test_world_restart_patch_keeps_gateway_unchanged_and_workers_zero_until_start() -> None:
    old = _render_chart()
    generation = "00000000-0000-4000-8000-000000000002"
    stopped = _render_chart(
        "--set-string", f"worker.topologyGeneration={generation}",
        "--set", "worker.replicas=0",
    )
    started = _render_chart(
        "--set-string", f"worker.topologyGeneration={generation}",
        "--set", "worker.replicas=1",
    )
    persisted = _render_chart(
        "--set-string", f"worker.topologyGeneration={generation}",
        "--set-string", f"gateway.topologyGeneration={generation}",
        "--set", "worker.replicas=1",
    )

    old_gateway = _rendered_document(old, "gateway-deployment.yaml")
    stopped_gateway = _rendered_document(stopped, "gateway-deployment.yaml")
    persisted_gateway = _rendered_document(
        persisted, "gateway-deployment.yaml"
    )
    assert stopped_gateway == old_gateway
    assert persisted_gateway == old_gateway
    assert stopped.count("kind: StatefulSet") == 4
    assert stopped.count("replicas: 0") == 4
    assert started.count("replicas: 1") == 5
    old_topology = _topology_payload(old)
    stopped_topology = _topology_payload(stopped)
    persisted_topology = _topology_payload(persisted)
    assert stopped_topology["topology_generation"] == generation
    assert (
        stopped_topology["accepted_topology_generation"]
        == old_topology["accepted_topology_generation"]
    )
    assert persisted_topology["topology_generation"] == generation
    assert persisted_topology["accepted_topology_generation"] == generation


def test_exact_model_profiles_render_geometry_and_runtime_fields() -> None:
    correctness = _render_chart()
    performance = _render_chart("-f", str(_performance_values()))

    for rendered, layers, block_bytes, max_num_seqs in (
        (correctness, "28", "29360128", "512"),
        (performance, "36", "37748736", "128"),
    ):
        gateway_fields = {
            "PRISM_SERVE_MODEL_NUM_HIDDEN_LAYERS": layers,
            "PRISM_SERVE_MODEL_NUM_KEY_VALUE_HEADS": "8",
            "PRISM_SERVE_MODEL_HEAD_DIM": "128",
            "PRISM_SERVE_MODEL_ROPE_THETA": "1000000.0",
            "PRISM_SERVE_MAX_MODEL_LEN": "4096",
            "PRISM_SERVE_MAX_NUM_BATCHED_TOKENS": "16384",
            "PRISM_SERVE_MAX_NUM_SEQS": max_num_seqs,
            "PRISM_SERVE_GPU_MEMORY_UTILIZATION": "0.9",
            "PRISM_SERVE_ENFORCE_EAGER": "false",
            "PRISM_SERVE_KV_BLOCK_BYTES": block_bytes,
        }
        worker_fields = {
            "PRISM_MODEL_NUM_HIDDEN_LAYERS": layers,
            "PRISM_MODEL_NUM_KEY_VALUE_HEADS": "8",
            "PRISM_MODEL_HEAD_DIM": "128",
            "PRISM_MODEL_ROPE_THETA": "1000000.0",
            "PRISM_MAX_MODEL_LEN": "4096",
            "PRISM_MAX_NUM_BATCHED_TOKENS": "16384",
            "PRISM_MAX_NUM_SEQS": max_num_seqs,
            "PRISM_GPU_MEMORY_UTILIZATION": "0.9",
            "PRISM_ENFORCE_EAGER": "false",
            "PRISM_KV_BLOCK_BYTES": block_bytes,
        }
        for name, value in gateway_fields.items():
            assert rendered.count(
                f'- name: {name}\n              value: "{value}"'
            ) == 1
        for name, value in worker_fields.items():
            assert rendered.count(
                f'- name: {name}\n              value: "{value}"'
            ) == 4


def test_performance_overlay_enables_parity_without_fault_authority() -> None:
    off = _render_chart("-f", str(_performance_values()))
    on = _render_chart(
        "-f", str(_performance_values()), "-f", str(_performance_on_values())
    )

    assert off.count(
        '- name: PRISM_SERVE_CORRECTNESS_HARNESS_ENABLED\n'
        '              value: "false"'
    ) == 1
    assert off.count(
        '- name: PRISM_SERVE_PERFORMANCE_HARNESS_ENABLED\n'
        '              value: "true"'
    ) == 1
    assert off.count(
        '- name: PRISM_SERVE_ROUTE_PARITY_HARNESS_ENABLED\n'
        '              value: "true"'
    ) == 1
    assert off.count(
        '- name: PRISM_SERVE_PERFORMANCE_TRACE_CAP\n'
        '              value: "8192"'
    ) == 1
    for rendered, affinity in ((off, "false"), (on, "true")):
        assert "shareProcessNamespace:" not in rendered
        assert "PRISM_SERVE_PROCESS_IDENTITY_PATH" not in rendered
        assert "PRISM_PROCESS_IDENTITY_PATH" not in rendered
        assert rendered.count("PRISM_SERVE_CORRECTNESS_HARNESS_SECRET") == 1
        assert rendered.count('value: "qwen3-8b-bf16-tp1"') == 5
        assert rendered.count('value: "Qwen/Qwen3-8B"') == 5
        assert rendered.count(
            '- name: PRISM_SERVE_AFFINITY_ENABLED\n'
            f'              value: "{affinity}"'
        ) == 1


def test_schema_rejects_hybrid_profile_and_fault_performance_overlap() -> None:
    chart = Path(__file__).parents[1] / "k8s" / "helm" / "prism-serve"
    base = [
        "helm", "template", "week12", str(chart),
        "-f", str(_performance_values()),
    ]
    hybrid = subprocess.run(
        [*base, "--set", "model.numHiddenLayers=28"],
        capture_output=True,
        text=True,
    )
    overlap = subprocess.run(
        [*base, "--set", "gateway.correctnessHarness.enabled=true"],
        capture_output=True,
        text=True,
    )

    assert hybrid.returncode != 0
    assert "oneOf" in hybrid.stderr
    assert "numHiddenLayers" in hybrid.stderr
    assert overlap.returncode != 0
    assert "correctnessHarness" in overlap.stderr

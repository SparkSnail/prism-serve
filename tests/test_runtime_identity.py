"""Tests for the authenticated runtime identity contract."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from prism_serve.gateway import app as gateway_module


def _request(config: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(runtime_config=config))
    )


def _config() -> dict[str, object]:
    return {
        "performance_harness_enabled": True,
        "model_id": "Qwen/Qwen3-8B",
        "model_revision": "e" * 40,
        "image_source_url": "https://github.com/SparkSnail/prism-serve",
        "image_source_commit": "a" * 40,
        "image_digest": "registry.example/prism-serve@sha256:" + "b" * 64,
        "worker_image_source_url": "https://github.com/SparkSnail/prism-infer",
        "worker_image_source_commit": "c" * 40,
        "worker_image_digest": "registry.example/prism-infer@sha256:" + "d" * 64,
    }


def test_runtime_identity_payload_uses_observed_topology(monkeypatch):
    monkeypatch.setattr(
        gateway_module,
        "_performance_world_identity",
        lambda _app: {"topology_generation": "generation-7"},
    )
    payload = gateway_module._runtime_identity_payload(_request(_config()))
    assert payload["schema_version"] == "prism.public_endpoint_runtime_identity/v1"
    assert payload["endpoint_path"] == "/v1/chat/completions"
    assert payload["topology_generation"] == "generation-7"
    assert payload["worker"]["image"].endswith("@sha256:" + "d" * 64)


@pytest.mark.parametrize(
    "field",
    ["image_source_commit", "image_digest", "worker_image_source_commit", "worker_image_digest"],
)
def test_runtime_identity_payload_fails_closed_without_immutable_metadata(
    monkeypatch, field
):
    monkeypatch.setattr(
        gateway_module,
        "_performance_world_identity",
        lambda _app: {"topology_generation": "generation-7"},
    )
    config = _config()
    config[field] = ""
    with pytest.raises(RuntimeError):
        gateway_module._runtime_identity_payload(_request(config))


def test_runtime_identity_route_is_registered():
    paths = {
        route.path
        for route in gateway_module.app.routes
        if hasattr(route, "path")
    }
    assert gateway_module.RUNTIME_IDENTITY_PATH in paths


def test_digest_pinned_helm_values_produce_a_usable_runtime_identity(monkeypatch):
    chart = Path(__file__).parents[1] / "k8s" / "helm" / "prism-serve"
    rendered = subprocess.run(
        [
            "helm", "template", "week12", str(chart),
            "-f", str(chart / "values-performance.yaml"),
            "--set-string", "gateway.image.digest=sha256:" + "b" * 64,
            "--set-string", "worker.image.digest=sha256:" + "d" * 64,
            "--set-string", "gateway.image.sourceCommit=" + "a" * 40,
            "--set-string", "worker.image.sourceCommit=" + "c" * 40,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    deployment = rendered.split(
        "# Source: prism-serve/templates/gateway-deployment.yaml", 1,
    )[1].split("---", 1)[0]
    env_values: dict[str, str] = {}
    lines = deployment.splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.strip().startswith("- name: PRISM_SERVE_"):
            name = line.strip().removeprefix("- name: ")
            value_line = lines[index + 1].strip()
            if value_line.startswith("value: "):
                env_values[name] = value_line.removeprefix("value: ").strip('"')

    config = _config()
    config.update({
        "image_source_url": env_values["PRISM_SERVE_IMAGE_SOURCE_URL"],
        "image_source_commit": env_values["PRISM_SERVE_IMAGE_SOURCE_COMMIT"],
        "image_digest": env_values["PRISM_SERVE_IMAGE_DIGEST"],
        "worker_image_source_url": env_values[
            "PRISM_SERVE_WORKER_IMAGE_SOURCE_URL"
        ],
        "worker_image_source_commit": env_values[
            "PRISM_SERVE_WORKER_IMAGE_SOURCE_COMMIT"
        ],
        "worker_image_digest": env_values["PRISM_SERVE_WORKER_IMAGE_DIGEST"],
    })
    monkeypatch.setattr(
        gateway_module,
        "_performance_world_identity",
        lambda _app: {"topology_generation": "generation-7"},
    )

    payload = gateway_module._runtime_identity_payload(_request(config))
    assert payload["gateway"]["image"] == (
        "sparksnail/prism-serve@sha256:" + "b" * 64
    )
    assert payload["worker"]["image"] == (
        "sparksnail/prism-infer@sha256:" + "d" * 64
    )

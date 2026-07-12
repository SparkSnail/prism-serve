"""Rendered deployment invariants for the process-local Gateway authority."""

from __future__ import annotations

import subprocess
from pathlib import Path


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

"""Dependency-boundary tests for the public control-plane package."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_default_runtime_modules_do_not_import_prism_infer_at_module_load() -> None:
    for path in (ROOT / "prism_serve").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("prism_infer")
                    for alias in node.names
                ), path
            elif isinstance(node, ast.ImportFrom):
                assert not (
                    node.module and node.module.startswith("prism_infer")
                ), path


def test_default_runtime_does_not_configure_redis_without_a_redis_feature() -> None:
    config = (ROOT / "prism_serve" / "config.py").read_text(encoding="utf-8")
    values = (
        ROOT / "k8s" / "helm" / "prism-serve" / "values.yaml"
    ).read_text(encoding="utf-8")

    assert "redis_url" not in config
    assert "redis:" not in values

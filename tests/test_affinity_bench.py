from __future__ import annotations

import json
from pathlib import Path

from bench.bench_affinity import run_synthetic


def test_synthetic_bench_reports_policy_metrics_without_gpu_claims():
    result = run_synthetic(10, 4, 1024)
    assert result["route_hit_rate"] == 1.0
    assert result["mapped_to_cold_bytes_ratio"] == 2 / 3
    assert "GPU parity" in result["claims_not_made"]


def test_synthetic_bench_result_matches_checked_in_schema():
    schema = json.loads((
        Path(__file__).parents[1]
        / "bench"
        / "schemas"
        / "synthetic_cpu_policy-v1.json"
    ).read_text(encoding="utf-8"))
    result = run_synthetic(2, 4, 1024)

    assert set(result) == set(schema["required"])
    assert result["kind"] == schema["properties"]["kind"]["const"]
    assert result["requests"] >= schema["properties"]["requests"]["minimum"]
    assert 0 <= result["route_hit_rate"] <= 1
    assert result["claims_not_made"]


def test_performance_snapshot_records_immutable_provenance_and_tradeoff():
    snapshot = json.loads((
        Path(__file__).parents[1]
        / "bench"
        / "results"
        / "performance_snapshot.json"
    ).read_text(encoding="utf-8"))

    assert snapshot["schema_version"] == "prism.performance_snapshot/v1"
    assert snapshot["evidence_status"] == "historical-immutable-reference"
    assert snapshot["status"] == "PASS"
    assert len(snapshot["provenance"]["gateway"]["source_commit"]) == 40
    assert len(snapshot["provenance"]["worker"]["source_commit"]) == 40
    assert snapshot["metrics"]["on_vs_off_percent"]["ttft"]["p50"] < 0
    assert snapshot["metrics"]["on_vs_off_percent"]["tpot"]["p50"] > 0
    block_bytes = snapshot["profile"]["kv_block_bytes"]
    for route in ("affinity_off", "affinity_on"):
        assert snapshot["route"][route]["completed_kv_bytes"] == (
            block_bytes * snapshot["route"][route]["mapping_count"]
        )


def test_public_endpoint_benchmark_client_builds_contract_request():
    from bench.bench_endpoint import build_request

    request = build_request(
        request_id="public.r1",
        model="Qwen/Qwen3-8B",
        prompt={"content": "hello"},
        expected_input_tokens=1,
        expected_output_tokens=32,
    )
    assert request["stream"] is True
    assert request["temperature"] == 0
    assert request["ignore_eos"] is True
    assert request["prism_performance"]["schema_version"] == (
        "prism.performance_request/v1"
    )

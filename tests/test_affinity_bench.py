from __future__ import annotations

from bench.bench_affinity import run_synthetic


def test_synthetic_bench_reports_policy_metrics_without_gpu_claims():
    result = run_synthetic(10, 4, 1024)
    assert result["route_hit_rate"] == 1.0
    assert result["mapped_to_cold_bytes_ratio"] == 2 / 3
    assert "GPU parity" in result["claims_not_made"]

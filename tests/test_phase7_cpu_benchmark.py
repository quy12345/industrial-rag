"""Offline selection tests for the bounded Phase 7.5 CPU benchmark."""

from __future__ import annotations

from scripts.benchmark_phase7_reranker_cpu import _quality_passes, _select_pareto


def _result(*, budget: int, p95: float, valid: bool = True) -> dict:
    rows = []
    for index in range(1, 13):
        rows.append(
            {
                "id": f"phase7_calibration_{index:03d}",
                "language": "en" if index % 2 else "vi",
                "final_direct_evidence_rank": 6 if index == 10 else 1,
            }
        )
    return {
        "config": {"candidate_budget": budget, "batch_size": 16, "threads": None},
        "valid": valid,
        "quality": {
            "candidate_recall": 1.0,
            "hit_rate_at_5": 11 / 12,
            "mrr_at_5": 0.875,
            "wrong_document_top1_rate": 0.0,
            "wrong_document_candidate_rate_at_5": 0.1,
            "candidate_count_maximum": budget,
        },
        "per_query": rows,
        "warm_latency_ms": {"total_ms": {"p95": p95}},
    }


def test_cpu_pareto_requires_quality_then_uses_warm_p95() -> None:
    slower = _result(budget=30, p95=100.0)
    faster = _result(budget=28, p95=50.0)
    invalid = _result(budget=26, p95=1.0, valid=False)
    assert _quality_passes(faster) is True
    assert _select_pareto([slower, faster, invalid], maximum=2) == [
        faster["config"],
        slower["config"],
    ]

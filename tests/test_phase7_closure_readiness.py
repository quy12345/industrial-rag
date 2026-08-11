"""Offline tests for fail-closed Phase 7 held-out readiness governance."""

from __future__ import annotations

from scripts.generate_phase7_calibration_closure_readiness import build_readiness


def _rank_ablation(*, rank_010: bool = True) -> dict:
    return {
        "provider_calls": 0,
        "held_out_queries_executed": 0,
        "quality": {
            "gates": {
                "candidate_recall_12_of_12": True,
                "hit_rate_at_5_at_least_11_of_12": True,
                "mrr_at_5_at_least_0_875": True,
                "calibration_010_rank_at_most_5": rank_010,
            }
        },
    }


def test_readiness_remains_governance_blocked_even_when_technical_gates_pass() -> None:
    result = build_readiness(
        rank_ablation=_rank_ablation(),
        manifest={
            "calibration_dataset_sha256": "a" * 64,
            "test_dataset_sha256": "b" * 64,
        },
        diagnostic={"run_identity": {"provider_attempts": 3}, "attempts": [{}, {}, {}]},
        stability={"quality_gates": {"overall_pass": True}},
    )
    assert result["technical_pass"] is True
    assert result["status"] == "BLOCKED_GOVERNANCE"
    assert result["governance"]["statistically_unseen_claim_permitted"] is False
    assert result["held_out_dataset_reads_by_generator"] == 0


def test_readiness_reports_missing_diagnostic_stability_and_rank_gate() -> None:
    result = build_readiness(
        rank_ablation=_rank_ablation(rank_010=False),
        manifest={
            "calibration_dataset_sha256": "a" * 64,
            "test_dataset_sha256": "b" * 64,
        },
        diagnostic=None,
        stability=None,
    )
    assert result["technical_pass"] is False
    assert result["technical_gates"]["calibration_010_actual_evidence_top5"] is False
    assert result["technical_gates"][
        "calibration_005_three_attempt_diagnostic_complete"
    ] is False

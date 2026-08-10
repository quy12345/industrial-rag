"""Offline tests for provider-free Phase 7 runtime readiness gating."""

from __future__ import annotations

from scripts.generate_phase7_runtime_readiness import _gates


def _rows() -> list[dict]:
    rows = []
    for index in range(1, 13):
        rows.append(
            {
                "id": f"phase7_calibration_{index:03d}",
                "language": "en" if index % 2 else "vi",
                "final_candidate_count": 30,
                "candidate_direct_evidence_rank": 1,
                "final_direct_evidence_rank": 6 if index == 10 else 1,
                "wrong_document_candidate_count_at_5": 0,
            }
        )
    return rows


def _artifacts() -> tuple[dict, dict, dict]:
    rows = _rows()
    closure = {
        "provider_calls": 0,
        "held_out_queries_executed": 0,
        "overall": {
            "candidate_recall": 1.0,
            "hit_rate_at_5": 11 / 12,
            "mrr_at_5": 0.875,
            "wrong_document_top1_rate": 0.0,
            "wrong_document_candidate_rate_at_5": 0.0,
        },
        "per_query": rows,
    }
    selected = {"candidate_budget": 30, "batch_size": 8, "threads": None}
    cpu = {
        "stage": "full",
        "provider_calls": 0,
        "held_out_queries_executed": 0,
        "quality": {"selected_config": selected, "latency_target_met": True},
        "results": [
            {
                "valid": True,
                "config": selected,
                "quality": closure["overall"],
            }
        ],
    }
    facts = {
        "status": "HUMAN_REVIEW_REQUIRED",
        "activation": "not_active",
        "ground_truth_preserved": True,
    }
    return closure, cpu, facts


def test_runtime_readiness_requires_frozen_provider_free_sources() -> None:
    closure, cpu, facts = _artifacts()
    assert all(_gates(closure, cpu, facts).values())
    closure["provider_calls"] = 1
    assert _gates(closure, cpu, facts)["closure_provider_free"] is False

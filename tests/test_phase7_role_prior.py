"""Offline tests for Phase 7.4.1 role-prior calibration selection."""

from __future__ import annotations

from scripts.calibrate_phase7_role_prior import _cross_validate, _quality, _select_profile


def _row(identifier: str, *, language: str = "en", rank: int = 1, wrong: int = 0) -> dict:
    return {
        "id": identifier,
        "language": language,
        "candidate_count": 30,
        "final_candidate_count": 30,
        "candidate_direct_evidence_rank": 1,
        "final_direct_evidence_rank": rank,
        "wrong_document_top1": False,
        "wrong_document_candidate_count_at_5": wrong,
        "failure_class": "hit",
        "retrieval_ms": 1.0,
        "rerank_ms": 2.0,
        "document_context_complete": True,
    }


def _summary(name: str, *, wrong: int, rank_010: int = 5) -> dict:
    ids = [f"phase7_calibration_{index:03d}" for index in range(1, 13)]
    rows = [
        _row(identifier, language="en" if index % 2 else "vi", wrong=wrong)
        for index, identifier in enumerate(ids, 1)
    ]
    rows[9]["final_direct_evidence_rank"] = rank_010
    from scripts.evaluate_phase7_retrieval_closure import aggregate_closure_rows

    per_language = {
        language: {
            "query_count": 6,
            "candidate_recall": 1.0,
            "hit_rate_at_5": 1.0 if language == "en" else 5 / 6,
            "wrong_document_candidate_rate_at_5": wrong / 5,
        }
        for language in ("en", "vi")
    }
    return {
        "profile": {
            "name": name,
            "post_rerank_rrf_multiplier": 0.25,
            "post_rerank_role_multiplier": 0.1 if name == "simple" else 0.2,
            "post_rerank_rank_offset": 10,
            "post_rerank_confidence_mode": "strong_only",
        },
        "overall": aggregate_closure_rows(rows),
        "per_language": per_language,
        "per_query": rows,
    }


def test_role_prior_selection_prefers_lower_contamination_after_hit_rate() -> None:
    summaries = {"simple": _summary("simple", wrong=1), "noisy": _summary("noisy", wrong=2)}
    assert _select_profile(summaries) == "simple"


def test_cross_validation_requires_stable_consensus_and_quality_checks_010() -> None:
    summary = _summary("simple", wrong=0)
    folds = _cross_validate({"simple": summary})
    assert folds["stable_consensus"] is True
    quality = _quality(summary, stable=True)
    assert quality["gates"]["calibration_010_rank_at_most_5"] is True
    assert quality["overall_pass"] is True

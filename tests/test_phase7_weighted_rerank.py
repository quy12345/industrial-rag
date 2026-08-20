"""Offline selection tests for bounded Phase 7.4 rerank calibration."""

from __future__ import annotations

import pytest

from scripts.evaluate_phase7_weighted_rerank import _quality, select_runtime_profile


def _summary(
    *,
    candidate_recall: float = 1.0,
    hit_at_5: float = 1.0,
    mrr: float = 1.0,
    wrong_top1: float = 0.0,
    wrong_top5: float = 0.0,
    p95: float = 100.0,
) -> dict:
    return {
        "overall": {
            "candidate_recall": candidate_recall,
            "hit_rate_at_5": hit_at_5,
            "mrr_at_5": mrr,
            "wrong_document_top1_rate": wrong_top1,
            "wrong_document_candidate_rate_at_5": wrong_top5,
            "candidate_count_maximum": 30,
            "rerank_latency_ms": {"p95": p95},
        }
    }


def test_select_runtime_profile_uses_documented_metric_order() -> None:
    summaries = {
        "higher-mrr": _summary(hit_at_5=11 / 12, mrr=0.7),
        "winner": _summary(hit_at_5=1.0, mrr=0.5, wrong_top1=1 / 12),
        "lower-recall": _summary(candidate_recall=11 / 12, hit_at_5=1.0),
    }
    assert select_runtime_profile(summaries) == "winner"


def test_quality_requires_cross_document_reduction_and_top5_hits() -> None:
    quality = _quality(_summary(hit_at_5=10 / 12, wrong_top1=2 / 12, wrong_top5=0.2))
    assert quality["overall_pass"] is False
    assert quality["gates"]["hit_rate_at_5_at_least_11_of_12"] is False
    assert quality["gates"]["wrong_document_top1_at_most_1_of_12"] is False
    assert quality["gates"]["wrong_document_candidate_rate_at_5_at_most_0_15"] is False


def test_select_runtime_profile_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_runtime_profile({})

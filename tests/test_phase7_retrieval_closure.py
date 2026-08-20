"""Offline aggregation tests for the provider-free Phase 7.4 closure."""

from __future__ import annotations

import pytest

from scripts.evaluate_phase7_retrieval_closure import aggregate_closure_rows


def _row(identifier: str, rank: int | None, *, wrong_document: bool = False) -> dict:
    return {
        "id": identifier,
        "candidate_count": 30,
        "final_candidate_count": 30,
        "candidate_direct_evidence_rank": rank,
        "final_direct_evidence_rank": rank,
        "failure_class": "candidate_miss" if rank is None else "hit",
        "wrong_document_top1": wrong_document,
        "wrong_document_candidate_count_at_5": int(wrong_document),
        "document_context_complete": True,
        "retrieval_ms": 10.0,
        "rerank_ms": 100.0,
    }


def test_closure_aggregation_reports_recall_ranks_contamination_and_latency() -> None:
    metrics = aggregate_closure_rows([_row("a", 1), _row("b", None, wrong_document=True)])
    assert metrics["candidate_recall"] == 0.5
    assert metrics["hit_rate_at_5"] == 0.5
    assert metrics["mrr_at_5"] == 0.5
    assert metrics["candidate_count_maximum"] == 30
    assert metrics["wrong_document_top1_rate"] == 0.5
    assert metrics["wrong_document_candidate_rate_at_5"] == 0.1
    assert metrics["document_context_complete_rate"] == 1.0
    assert metrics["rerank_latency_ms"]["p95"] == 100.0


def test_closure_aggregation_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_closure_rows([])

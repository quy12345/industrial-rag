"""Offline metric tests for the provider-free retrieval evaluator."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from evaluation.retrieval import (
    aggregate_retrieval_rows,
    build_corpus_exact_content_equivalence,
    direct_evidence_rank,
    evidence_rank,
    exact_equivalent_evidence_rank,
    percentile_nearest_rank,
    phrase_matches,
)
from scripts.evaluation import evaluate_retrieval


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


def test_retrieval_aggregation_reports_recall_ranks_contamination_and_latency() -> None:
    metrics = aggregate_retrieval_rows([_row("a", 1), _row("b", None, wrong_document=True)])
    assert metrics["candidate_recall"] == 0.5
    assert metrics["hit_rate_at_5"] == 0.5
    assert metrics["mrr_at_5"] == 0.5
    assert metrics["candidate_count_maximum"] == 30
    assert metrics["wrong_document_top1_rate"] == 0.5
    assert metrics["wrong_document_candidate_rate_at_5"] == 0.1
    assert metrics["document_context_complete_rate"] == 1.0
    assert metrics["rerank_latency_ms"]["p95"] == 100.0


def test_retrieval_aggregation_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_retrieval_rows([])


def test_retrieval_metric_primitives_are_deterministic() -> None:
    results = [SimpleNamespace(chunk_id="a"), SimpleNamespace(chunk_id="b")]

    assert direct_evidence_rank(results, {"b"}) == 2
    assert direct_evidence_rank(results, {"missing"}) is None
    assert phrase_matches("Cafe\u0301\n menu", "CAF\u00c9 menu")
    assert percentile_nearest_rank([30.0, 10.0, 20.0], 95) == 30.0


def test_evidence_rank_keeps_strict_and_exact_equivalent_policies_distinct() -> None:
    results = [
        SimpleNamespace(
            chunk_id="representative",
            metadata={"equivalent_chunk_ids": ["qrel", "representative", 42]},
        )
    ]

    assert direct_evidence_rank(results, {"qrel"}) is None
    assert (
        evidence_rank(results, {"qrel"}, include_exact_equivalents=False) is None
    )
    assert exact_equivalent_evidence_rank(results, {"qrel"}) == 1
    assert evidence_rank(results, {"qrel"}, include_exact_equivalents=True) == 1


def test_corpus_equivalence_detects_cross_document_exact_content_without_text_output() -> None:
    chunks = [
        SimpleNamespace(chunk_id="installation-copy", text="Disconnect all power."),
        SimpleNamespace(chunk_id="programming-copy", text="  DISCONNECT  ALL POWER.  "),
        SimpleNamespace(chunk_id="different", text="Block the motor shaft."),
    ]

    equivalence = build_corpus_exact_content_equivalence(chunks)

    assert equivalence == {
        "installation-copy": ("installation-copy", "programming-copy"),
        "programming-copy": ("installation-copy", "programming-copy"),
    }
    assert (
        exact_equivalent_evidence_rank(
            [SimpleNamespace(chunk_id="programming-copy")],
            {"installation-copy"},
            equivalence=equivalence,
        )
        == 1
    )
    assert direct_evidence_rank(
        [SimpleNamespace(chunk_id="programming-copy")], {"installation-copy"}
    ) is None


def test_retrieval_evaluator_hashes_existing_canonical_behavior_owners() -> None:
    expected = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in evaluate_retrieval.SOURCE_IDENTITY_PATHS.items()
    }

    assert expected == evaluate_retrieval._source_identity()
    assert expected == evaluate_retrieval._source_identity()
    assert evaluate_retrieval.ARTIFACT_SCHEMA_VERSION == 2
    assert evaluate_retrieval.DEFAULT_OUTPUT.name == "atv320-retrieval-evaluation-v2.json"

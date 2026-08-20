"""Offline tests for Phase 5 candidate-pool coverage diagnostics."""

from __future__ import annotations

from app.candidate_audit import (
    aggregate_candidate_audit,
    audit_case,
    dense_results_to_candidates,
    union_dense_sparse_candidates,
)
from app.evaluation import EvaluationCase
from app.models import RetrievalCandidate, RetrievedChunk


def _case(**overrides: object) -> EvaluationCase:
    payload: dict[str, object] = {
        "id": "case-1",
        "language": "vi",
        "question": "Which PLC is used?",
        "relevant_chunk_ids": ["evidence"],
        "expected_phrases": ["PLC"],
        "expected_pages": [1],
        "category": "exact_technical_term",
        "critical": True,
    }
    payload.update(overrides)
    return EvaluationCase.model_validate(payload)


def _candidate(
    chunk_id: str,
    *,
    dense_rank: int | None = None,
    sparse_rank: int | None = None,
    rrf_rank: int | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="manual-1",
        filename="manual.pdf",
        text=chunk_id,
        page_numbers=[1],
        headings=["Heading"],
        content_type="text",
        score=1.0,
        dense_score=0.8 if dense_rank is not None else None,
        dense_rank=dense_rank,
        sparse_score=3.0 if sparse_rank is not None else None,
        sparse_rank=sparse_rank,
        rrf_score=0.02 if rrf_rank is not None else None,
        rrf_rank=rrf_rank,
    )


def test_dense_results_receive_deterministic_ranks_and_metadata() -> None:
    results = [
        RetrievedChunk(
            chunk_id="b",
            document_id="manual-1",
            filename="manual.pdf",
            text="b",
            page_numbers=[2],
            headings=["B"],
            content_type="text",
            score=0.5,
        ),
        RetrievedChunk(
            chunk_id="a",
            document_id="manual-1",
            filename="manual.pdf",
            text="a",
            page_numbers=[1],
            headings=["A"],
            content_type="text",
            score=0.5,
        ),
    ]

    candidates = dense_results_to_candidates(results)

    assert [candidate.chunk_id for candidate in candidates] == ["a", "b"]
    assert [candidate.dense_rank for candidate in candidates] == [1, 2]
    assert candidates[0].page_numbers == [1]
    assert candidates[0].headings == ["A"]


def test_union_deduplicates_preserves_component_ranks_and_is_deterministic() -> None:
    union = union_dense_sparse_candidates(
        [_candidate("both", dense_rank=2), _candidate("dense-only", dense_rank=1)],
        [_candidate("both", sparse_rank=1), _candidate("sparse-only", sparse_rank=2)],
    )

    assert [candidate.chunk_id for candidate in union] == ["both", "dense-only", "sparse-only"]
    both = union[0]
    assert both.dense_rank == 2
    assert both.sparse_rank == 1
    assert both.dense_score == 0.8
    assert both.sparse_score == 3.0


def test_audit_reports_relevant_coverage_for_dense_sparse_both_and_empty() -> None:
    case = _case(relevant_chunk_ids=["dense-only", "sparse-only"])
    row = audit_case(
        case,
        dense_candidates=[_candidate("dense-only", dense_rank=1)],
        sparse_candidates=[_candidate("sparse-only", sparse_rank=1)],
        hybrid_candidates=[],
    )

    assert row["pools"]["dense_top20"]["relevant_chunk_ids_found"] == ["dense-only"]
    assert row["pools"]["sparse_top20"]["relevant_chunk_ids_found"] == ["sparse-only"]
    assert row["pools"]["dense_sparse_union"]["relevant_chunk_ids_found"] == [
        "dense-only",
        "sparse-only",
    ]
    assert row["pools"]["hybrid_rrf_top20"]["contains_direct_evidence"] is False


def test_aggregate_reports_scenario_critical_coverage_and_rrf_demotion() -> None:
    demoted = audit_case(
        _case(id="demoted"),
        dense_candidates=[_candidate("wrong", dense_rank=1)],
        sparse_candidates=[_candidate("evidence", sparse_rank=1)],
        hybrid_candidates=[_candidate("wrong", rrf_rank=1)],
    )
    dense_only = audit_case(
        _case(id="dense-only", language="en", critical=False),
        dense_candidates=[_candidate("evidence", dense_rank=1)],
        sparse_candidates=[_candidate("wrong", sparse_rank=1)],
        hybrid_candidates=[_candidate("evidence", rrf_rank=1)],
    )

    report = aggregate_candidate_audit([demoted, dense_only])

    assert report["aggregate_metrics"]["pools"]["dense_sparse_union"]["candidate_recall"] == 1.0
    assert (
        report["aggregate_metrics"]["pools"]["dense_sparse_union"]["relevant_chunk_recall"] == 1.0
    )
    assert report["per_retrieval_scenario"]["monolingual"]["query_count"] == 1
    assert report["per_retrieval_scenario"]["cross_lingual"]["query_count"] == 1
    assert [row["id"] for row in report["critical_query_coverage"]] == ["demoted"]
    assert report["rrf_diagnosis"]["sparse_top5_evidence_demoted_by_rrf"] == ["demoted"]
    assert report["rrf_diagnosis"]["dense_only_candidate_coverage"] == ["dense-only"]

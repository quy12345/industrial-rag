"""Dependency-free candidate-pool coverage helpers for Phase 5 handoff."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from app.evaluation import EvaluationCase, direct_evidence_rank
from app.models import RetrievalCandidate, RetrievedChunk

POOL_NAMES = ("dense_top20", "sparse_top20", "hybrid_rrf_top20", "dense_sparse_union")


def dense_results_to_candidates(results: Sequence[RetrievedChunk]) -> list[RetrievalCandidate]:
    """Map dense results to candidates with deterministic one-based dense ranks."""

    candidates = [
        RetrievalCandidate(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            filename=result.filename,
            text=result.text,
            page_numbers=result.page_numbers,
            headings=result.headings,
            content_type=result.content_type,
            score=result.score,
            dense_score=result.score,
        )
        for result in results
    ]
    return [
        candidate.model_copy(update={"dense_rank": rank})
        for rank, candidate in enumerate(
            sorted(
                candidates,
                key=lambda candidate: (-(candidate.dense_score or 0.0), candidate.chunk_id),
            ),
            start=1,
        )
    ]


def union_dense_sparse_candidates(
    dense_candidates: Sequence[RetrievalCandidate],
    sparse_candidates: Sequence[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    """Return a deterministic, unranked-for-fusion union without mixing raw scores."""

    merged: dict[str, RetrievalCandidate] = {}
    for candidate in dense_candidates:
        merged[candidate.chunk_id] = candidate
    for candidate in sparse_candidates:
        existing = merged.get(candidate.chunk_id)
        if existing is None:
            merged[candidate.chunk_id] = candidate
        else:
            merged[candidate.chunk_id] = existing.model_copy(
                update={
                    "sparse_score": candidate.sparse_score,
                    "sparse_rank": candidate.sparse_rank,
                }
            )
    return sorted(
        merged.values(),
        key=lambda candidate: (
            min(rank for rank in (candidate.dense_rank, candidate.sparse_rank) if rank is not None),
            candidate.chunk_id,
        ),
    )


def audit_case(
    case: EvaluationCase,
    *,
    dense_candidates: Sequence[RetrievalCandidate],
    sparse_candidates: Sequence[RetrievalCandidate],
    hybrid_candidates: Sequence[RetrievalCandidate],
) -> dict[str, Any]:
    """Capture direct-evidence coverage for one query across Phase 5 pool options."""

    pools = {
        "dense_top20": list(dense_candidates),
        "sparse_top20": list(sparse_candidates),
        "hybrid_rrf_top20": list(hybrid_candidates),
        "dense_sparse_union": union_dense_sparse_candidates(dense_candidates, sparse_candidates),
    }
    relevant_ids = set(case.relevant_chunk_ids)
    return {
        "id": case.id,
        "question": case.question,
        "language": case.language,
        "document_language": case.document_language,
        "retrieval_scenario": case.retrieval_scenario,
        "category": case.category,
        "critical": case.critical,
        "relevant_chunk_ids": case.relevant_chunk_ids,
        "pools": {
            name: _pool_coverage(candidates, relevant_ids) for name, candidates in pools.items()
        },
    }


def aggregate_candidate_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate query and qrel coverage without conflating it with ranked Hit@k."""

    if not rows:
        raise ValueError("Cannot aggregate an empty candidate-pool audit.")
    return {
        "aggregate_metrics": _aggregate_rows(rows),
        "per_retrieval_scenario": {
            scenario: _aggregate_rows(group)
            for scenario, group in _group_rows(rows, "retrieval_scenario").items()
        },
        "critical_query_coverage": [row for row in rows if row["critical"]],
        "rrf_diagnosis": _rrf_diagnosis(rows),
        "per_query": list(rows),
    }


def _pool_coverage(
    candidates: Sequence[RetrievalCandidate], relevant_ids: set[str]
) -> dict[str, Any]:
    candidate_ids = [candidate.chunk_id for candidate in candidates]
    found = sorted(relevant_ids.intersection(candidate_ids))
    return {
        "candidate_count": len(candidates),
        "candidate_ids": candidate_ids,
        "relevant_chunk_ids_found": found,
        "relevant_chunk_count_found": len(found),
        "contains_direct_evidence": bool(found),
        "first_relevant_rank": direct_evidence_rank(candidates, relevant_ids),
        "candidates": [_candidate_summary(candidate) for candidate in candidates],
    }


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"query_count": len(rows), "pools": {}}
    for pool_name in POOL_NAMES:
        pools = [row["pools"][pool_name] for row in rows]
        counts = [pool["candidate_count"] for pool in pools]
        total_relevant = sum(len(row["relevant_chunk_ids"]) for row in rows)
        found_relevant = sum(pool["relevant_chunk_count_found"] for pool in pools)
        missing_ids = [
            row["id"]
            for row, pool in zip(rows, pools, strict=True)
            if not pool["contains_direct_evidence"]
        ]
        metrics["pools"][pool_name] = {
            "candidate_recall": sum(pool["contains_direct_evidence"] for pool in pools)
            / len(pools),
            "relevant_chunk_recall": found_relevant / total_relevant if total_relevant else 0.0,
            "queries_without_relevant_chunk": missing_ids,
            "candidate_count": {
                "minimum": min(counts),
                "maximum": max(counts),
                "average": sum(counts) / len(counts),
                "median": statistics.median(counts),
            },
        }
    return metrics


def _group_rows(rows: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return dict(sorted(groups.items()))


def _rrf_diagnosis(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Explain observable sparse-versus-RRF rank loss without tuning RRF defaults."""

    sparse_top5_demoted = []
    dense_unique_evidence = []
    for row in rows:
        dense = row["pools"]["dense_top20"]
        sparse = row["pools"]["sparse_top20"]
        hybrid = row["pools"]["hybrid_rrf_top20"]
        if sparse["first_relevant_rank"] is not None and sparse["first_relevant_rank"] <= 5:
            if hybrid["first_relevant_rank"] is None or hybrid["first_relevant_rank"] > 5:
                sparse_top5_demoted.append(row["id"])
        if dense["contains_direct_evidence"] and not sparse["contains_direct_evidence"]:
            dense_unique_evidence.append(row["id"])
    return {
        "sparse_top5_evidence_demoted_by_rrf": sparse_top5_demoted,
        "dense_only_candidate_coverage": dense_unique_evidence,
        "hybrid_missing_relevant_within_top20": [
            row["id"]
            for row in rows
            if not row["pools"]["hybrid_rrf_top20"]["contains_direct_evidence"]
        ],
    }


def _candidate_summary(candidate: RetrievalCandidate) -> dict[str, Any]:
    return {
        "chunk_id": candidate.chunk_id,
        "document_id": candidate.document_id,
        "page_numbers": candidate.page_numbers,
        "headings": candidate.headings,
        "dense_score": candidate.dense_score,
        "dense_rank": candidate.dense_rank,
        "sparse_score": candidate.sparse_score,
        "sparse_rank": candidate.sparse_rank,
        "rrf_score": candidate.rrf_score,
        "rrf_rank": candidate.rrf_rank,
    }

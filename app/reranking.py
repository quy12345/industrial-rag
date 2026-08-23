"""Compatibility facade for reranking runtime exports and evaluation diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Literal

from app.application.reranking_service import (
    CANDIDATE_TEXT_FORMAT as CANDIDATE_TEXT_FORMAT,
)
from app.application.reranking_service import (
    PHASE7_CANDIDATE_TEXT_FORMAT as PHASE7_CANDIDATE_TEXT_FORMAT,
)
from app.application.reranking_service import CandidatePool as CandidatePool
from app.application.reranking_service import RerankExecution as RerankExecution
from app.application.reranking_service import RerankPipeline as RerankPipeline
from app.application.reranking_service import RerankStrategy as RerankStrategy
from app.application.reranking_service import build_candidate_pool as build_candidate_pool
from app.application.reranking_service import build_candidate_text as build_candidate_text
from app.application.reranking_service import (
    deduplicate_candidates_by_content as deduplicate_candidates_by_content,
)
from app.application.reranking_service import execute_rerank as execute_rerank
from app.application.reranking_service import rerank_candidates as rerank_candidates
from app.domain.reranking import CrossEncoder as CrossEncoder
from app.domain.reranking import CrossEncoderScore as CrossEncoderScore
from app.domain.reranking import RerankingError as RerankingError
from app.evaluation import (
    EvaluationCase,
    EvaluationError,
    diagnostic_page_rank,
    diagnostic_phrase_rank,
    direct_evidence_rank,
    percentile_nearest_rank,
)
from app.infrastructure.models.reranker import (
    FastEmbedCrossEncoder as FastEmbedCrossEncoder,
)
from app.infrastructure.models.reranker import (
    fastembed_model_metadata as fastembed_model_metadata,
)
from app.models import RetrievalCandidate

FailureClass = Literal["candidate_miss", "reranker_miss_top5", "reranker_miss_top20", "hit"]


def evaluate_reranked_cases(
    cases: Sequence[EvaluationCase],
    search: Callable[[str, str], RerankExecution],
    *,
    cutoff: int = 20,
) -> dict[str, Any]:
    """Evaluate final ranks separately from pre-rerank candidate availability."""

    if not cases:
        raise EvaluationError("Cannot evaluate an empty case list.")
    if cutoff < 5:
        raise EvaluationError("Rerank evaluation cutoff must be at least 5.")
    rows = [_evaluate_rerank_case(case, search(case.question, case.document_id)) for case in cases]
    return {
        "candidate_limit": cutoff,
        "metric_definitions": {
            "candidate_recall": "direct evidence present anywhere in the full pre-rerank pool",
            "hit_mrr_at_5": "computed on final reranked top 5",
            "hit_mrr_at_20": "computed on final reranked top 20",
        },
        "overall": aggregate_rerank_rows(rows, cutoff=cutoff),
        "per_language": _aggregate_rerank_groups(rows, "language", cutoff),
        "per_retrieval_scenario": _aggregate_rerank_groups(rows, "retrieval_scenario", cutoff),
        "critical_questions": [row for row in rows if row["critical"]],
        "critical_metrics": aggregate_rerank_rows(
            [row for row in rows if row["critical"]], cutoff=cutoff
        ),
        "failure_cases": [row for row in rows if row["failure_class"] != "hit"],
        "per_query": rows,
    }


def classify_rerank_failure(*, candidate_rank: int | None, final_rank: int | None) -> FailureClass:
    """Distinguish candidate absence from cross-encoder ordering failures."""

    if candidate_rank is None:
        return "candidate_miss"
    if final_rank is not None and final_rank <= 5:
        return "hit"
    if final_rank is not None and final_rank <= 20:
        return "reranker_miss_top5"
    return "reranker_miss_top20"


def aggregate_rerank_rows(rows: Sequence[dict[str, Any]], *, cutoff: int) -> dict[str, Any]:
    """Aggregate final quality, candidate coverage, failure classes, and stage latency."""

    if not rows:
        raise EvaluationError("Cannot aggregate an empty rerank result set.")
    ranks = [row["direct_evidence_rank"] for row in rows]
    candidate_ranks = [row["candidate_evidence_rank"] for row in rows]
    result: dict[str, Any] = {
        "query_count": len(rows),
        "hit_rate_at_1": _hit_rate(ranks, 1),
        "hit_rate_at_3": _hit_rate(ranks, 3),
        "hit_rate_at_5": _hit_rate(ranks, 5),
        "hit_rate_at_20": _hit_rate(ranks, cutoff),
        "candidate_recall": _hit_rate(candidate_ranks, max(row["candidate_count"] for row in rows)),
        "mrr_at_5": _mrr(ranks, 5),
        "mrr_at_20": _mrr(ranks, cutoff),
        "failure_counts": {
            name: sum(row["failure_class"] == name for row in rows)
            for name in ("candidate_miss", "reranker_miss_top5", "reranker_miss_top20", "hit")
        },
        "candidate_count": {
            "minimum": min(row["candidate_count"] for row in rows),
            "maximum": max(row["candidate_count"] for row in rows),
            "average": sum(row["candidate_count"] for row in rows) / len(rows),
        },
    }
    stage_names = sorted({name for row in rows for name in row["stage_latency_ms"]})
    result["stage_latency_ms"] = {
        name: _latency_summary([row["stage_latency_ms"].get(name, 0.0) for row in rows])
        for name in stage_names
    }
    return result


def _evaluate_rerank_case(case: EvaluationCase, execution: RerankExecution) -> dict[str, Any]:
    relevant = set(case.relevant_chunk_ids)
    candidate_rank = direct_evidence_rank(execution.candidates_before_rerank, relevant)
    final_rank = direct_evidence_rank(execution.candidates_after_rerank, relevant)
    return {
        "id": case.id,
        "language": case.language,
        "document_language": case.document_language,
        "retrieval_scenario": case.retrieval_scenario,
        "category": case.category,
        "critical": case.critical,
        "question": case.question,
        "document_id": case.document_id,
        "relevant_chunk_ids": case.relevant_chunk_ids,
        "expected_pages": case.expected_pages,
        "candidate_count": len(execution.candidates_before_rerank),
        "candidate_chunk_ids": [item.chunk_id for item in execution.candidates_before_rerank],
        "candidate_evidence_rank": candidate_rank,
        "direct_evidence_rank": final_rank,
        "failure_class": classify_rerank_failure(
            candidate_rank=candidate_rank, final_rank=final_rank
        ),
        "diagnostic_phrase_rank": diagnostic_phrase_rank(
            execution.candidates_after_rerank, case.expected_phrases
        ),
        "diagnostic_page_rank": diagnostic_page_rank(
            execution.candidates_after_rerank, set(case.expected_pages)
        ),
        "stage_latency_ms": execution.stage_latency_ms,
        "retrieved": [
            _candidate_summary(candidate) for candidate in execution.candidates_after_rerank
        ],
    }


def _candidate_summary(candidate: RetrievalCandidate) -> dict[str, Any]:
    return {
        "rerank_rank": candidate.rerank_rank,
        "chunk_id": candidate.chunk_id,
        "document_id": candidate.document_id,
        "page_numbers": candidate.page_numbers,
        "headings": candidate.headings,
        "rerank_score": candidate.rerank_score,
        "dense_rank": candidate.dense_rank,
        "dense_score": candidate.dense_score,
        "sparse_rank": candidate.sparse_rank,
        "sparse_score": candidate.sparse_score,
        "rrf_rank": candidate.rrf_rank,
        "rrf_score": candidate.rrf_score,
    }


def _aggregate_rerank_groups(
    rows: Sequence[dict[str, Any]], key: str, cutoff: int
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        name: aggregate_rerank_rows(group, cutoff=cutoff) for name, group in sorted(groups.items())
    }


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "average": sum(values) / len(values),
        "p50": percentile_nearest_rank(values, 50),
        "p95": percentile_nearest_rank(values, 95),
    }


def _hit_rate(ranks: Sequence[int | None], cutoff: int) -> float:
    return sum(rank is not None and rank <= cutoff for rank in ranks) / len(ranks)


def _mrr(ranks: Sequence[int | None], cutoff: int) -> float:
    return sum(1 / rank if rank is not None and rank <= cutoff else 0.0 for rank in ranks) / len(
        ranks
    )

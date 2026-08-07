"""Offline, direct-evidence scoring for the frozen Phase 7 end-to-end suite.

This module scores a completed :class:`~app.query_service.QueryExecution`.  It
never initializes Qdrant, an embedding model, reranker, or generation provider;
the CLI is responsible for executing the live pipeline and passing its sanitized
result here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from app.evaluation import direct_evidence_rank, percentile_nearest_rank, phrase_matches
from app.phase7 import Phase7DatasetItem
from app.query_service import QueryExecution


def score_phase7_execution(item: Phase7DatasetItem, execution: QueryExecution) -> dict[str, Any]:
    """Create one sanitized, deterministic record for an end-to-end execution.

    Retrieval relevance is based exclusively on stable qrel chunk IDs.  A page,
    phrase, or same-document match is diagnostic information only.
    """

    relevant = set(item.relevant_chunk_ids)
    final_rank = direct_evidence_rank(execution.candidates, relevant) if item.answerable else None
    candidate_rank = (
        direct_evidence_rank(execution.candidate_pool, relevant) if item.answerable else None
    )
    response = execution.response
    citation_ids = [citation.chunk_id for citation in response.citations]
    final_ids = {candidate.chunk_id for candidate in execution.candidates}
    citation_documents = {citation.document_id for citation in response.citations}

    record: dict[str, Any] = {
        "id": item.id,
        "language": item.language,
        "scenario": item.scenario,
        "question_type": item.question_type,
        "answerable": item.answerable,
        "candidate_count": len(execution.candidate_pool),
        "final_candidate_count": len(execution.candidates),
        "candidate_direct_evidence_rank": candidate_rank,
        "direct_evidence_rank": final_rank,
        "abstained": response.abstained,
        "abstention_reason": response.abstention_reason,
        "citation_count": len(citation_ids),
        "citation_ids": citation_ids,
        "citation_ids_in_final_candidates": all(chunk_id in final_ids for chunk_id in citation_ids),
        "timings_ms": {
            "retrieval": execution.timings.retrieval_ms,
            "rerank": execution.timings.rerank_ms,
            "evidence_gate": execution.timings.evidence_gate_ms,
            "generation": execution.timings.generation_ms,
            "citation_validation": execution.timings.citation_validation_ms,
            "total": execution.timings.total_ms,
        },
        "usage": _usage_payload(execution),
    }
    if item.answerable:
        record.update(
            {
                "failure_class": _failure_class(candidate_rank, final_rank),
                "phrase_match": _answer_matches(item, response.answer)
                if not response.abstained
                else None,
                "citation_direct_evidence": bool(set(citation_ids).intersection(relevant)),
                "citation_document_correct": bool(citation_ids)
                and citation_documents.issubset(set(item.expected_document_ids)),
            }
        )
    else:
        record.update(
            {
                "failure_class": "correct_abstention" if response.abstained else "false_answer",
                "phrase_match": None,
                "citation_direct_evidence": None,
                "citation_document_correct": None,
            }
        )
    return record


def aggregate_phase7_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval, answer, citation, abstention, and latency metrics."""

    if not records:
        raise ValueError("Phase 7 evaluation requires at least one result record.")
    answerable = [record for record in records if record["answerable"]]
    unanswerable = [record for record in records if not record["answerable"]]
    if not answerable or not unanswerable:
        raise ValueError("Phase 7 evaluation requires answerable and unanswerable records.")
    return {
        "query_count": len(records),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "retrieval": _retrieval_metrics(answerable),
        "answer_quality": _answer_metrics(answerable),
        "citations": _citation_metrics(answerable),
        "abstention": _abstention_metrics(answerable, unanswerable),
        "failure_classes": dict(
            sorted(Counter(record["failure_class"] for record in records).items())
        ),
        "latency_ms": _latency_metrics(records),
        "per_language": _aggregate_groups(records, "language"),
        "per_scenario": _aggregate_groups(records, "scenario"),
        "per_question_type": _aggregate_groups(records, "question_type"),
    }


def _retrieval_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    final_ranks = [row["direct_evidence_rank"] for row in rows]
    candidate_ranks = [row["candidate_direct_evidence_rank"] for row in rows]
    return {
        "candidate_recall": _hit_rate(candidate_ranks, cutoff=None),
        "hit_rate_at_1": _hit_rate(final_ranks, cutoff=1),
        "hit_rate_at_3": _hit_rate(final_ranks, cutoff=3),
        "hit_rate_at_5": _hit_rate(final_ranks, cutoff=5),
        "hit_rate_at_20": _hit_rate(final_ranks, cutoff=20),
        "mrr_at_5": _mrr(final_ranks, cutoff=5),
        "mrr_at_20": _mrr(final_ranks, cutoff=20),
    }


def _answer_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    answered = [row for row in rows if not row["abstained"]]
    phrase_scores = [row["phrase_match"] for row in answered if row["phrase_match"] is not None]
    return {
        "answer_rate": len(answered) / len(rows),
        "phrase_match_rate_when_answered": (
            sum(bool(score) for score in phrase_scores) / len(phrase_scores)
            if phrase_scores
            else 0.0
        ),
        "answered_count": len(answered),
    }


def _citation_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    answered = [row for row in rows if not row["abstained"]]
    if not answered:
        return {
            "direct_evidence_rate_when_answered": 0.0,
            "document_correct_rate_when_answered": 0.0,
            "referential_valid_rate_when_answered": 0.0,
            "answered_count": 0,
        }
    return {
        "direct_evidence_rate_when_answered": sum(
            bool(row["citation_direct_evidence"]) for row in answered
        )
        / len(answered),
        "document_correct_rate_when_answered": sum(
            bool(row["citation_document_correct"]) for row in answered
        )
        / len(answered),
        "referential_valid_rate_when_answered": sum(
            bool(row["citation_ids_in_final_candidates"]) for row in answered
        )
        / len(answered),
        "answered_count": len(answered),
    }


def _abstention_metrics(
    answerable: Sequence[dict[str, Any]], unanswerable: Sequence[dict[str, Any]]
) -> dict[str, int | float]:
    true_positive = sum(row["abstained"] for row in unanswerable)
    false_negative = len(unanswerable) - true_positive
    false_positive = sum(row["abstained"] for row in answerable)
    true_negative = len(answerable) - false_positive
    return {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "unanswerable_abstention_rate": true_positive / len(unanswerable),
        "answerable_non_abstention_rate": true_negative / len(answerable),
    }


def _latency_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = ("retrieval", "rerank", "evidence_gate", "generation", "citation_validation", "total")
    return {key: _summary([float(row["timings_ms"][key]) for row in rows]) for key in keys}


def _aggregate_groups(rows: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: _group_metrics(group) for name, group in sorted(groups.items())}


def _group_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    return {
        "query_count": len(rows),
        "answerable_count": len(answerable),
        "unanswerable_count": len(rows) - len(answerable),
        "retrieval": _retrieval_metrics(answerable) if answerable else None,
        "abstention_rate": sum(row["abstained"] for row in rows) / len(rows),
    }


def _answer_matches(item: Phase7DatasetItem, answer: str) -> bool:
    matches = [phrase_matches(answer, phrase) for phrase in item.expected_phrases]
    return all(matches) if item.phrase_match_mode == "all" else any(matches)


def _failure_class(candidate_rank: int | None, final_rank: int | None) -> str:
    if candidate_rank is None:
        return "candidate_miss"
    if final_rank is None or final_rank > 20:
        return "reranker_miss_top20"
    if final_rank > 5:
        return "reranker_miss_top5"
    return "hit"


def _usage_payload(execution: QueryExecution) -> dict[str, int | None] | None:
    if execution.usage is None:
        return None
    return {
        "input_tokens": execution.usage.input_tokens,
        "output_tokens": execution.usage.output_tokens,
        "cached_input_tokens": execution.usage.cached_input_tokens,
    }


def _hit_rate(ranks: Iterable[int | None], *, cutoff: int | None) -> float:
    values = list(ranks)
    return sum(rank is not None and (cutoff is None or rank <= cutoff) for rank in values) / len(
        values
    )


def _mrr(ranks: Iterable[int | None], *, cutoff: int) -> float:
    values = list(ranks)
    return sum(1 / rank if rank is not None and rank <= cutoff else 0.0 for rank in values) / len(
        values
    )


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "average": sum(values) / len(values),
        "p50": percentile_nearest_rank(values, 50),
        "p95": percentile_nearest_rank(values, 95),
        "max": max(values),
    }

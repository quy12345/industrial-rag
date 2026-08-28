"""Deterministic metrics for the active frozen-corpus retrieval evaluation."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any, Protocol


class RetrievedLike(Protocol):
    """Minimum result identity required by direct-evidence metrics."""

    chunk_id: str


def phrase_matches(text: str, expected_phrase: str) -> bool:
    """Compare evidence text with Unicode and whitespace normalization."""

    return _normalize_text(expected_phrase) in _normalize_text(text)


def direct_evidence_rank(
    results: Sequence[RetrievedLike], relevant_chunk_ids: set[str]
) -> int | None:
    """Return the first one-based result rank whose stable chunk ID is a qrel."""

    return next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if result.chunk_id in relevant_chunk_ids
        ),
        None,
    )


def percentile_nearest_rank(values: Sequence[float], percentile: int) -> float:
    """Calculate a deterministic nearest-rank percentile for a non-empty sample."""

    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sample.")
    if not 0 < percentile <= 100:
        raise ValueError("Percentile must be in the range 1..100.")
    ordered = sorted(float(value) for value in values)
    index = math.ceil((percentile / 100) * len(ordered)) - 1
    return ordered[index]


def aggregate_retrieval_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval, contamination, context, and latency metrics."""

    if not rows:
        raise ValueError("Retrieval evaluation requires at least one row.")
    candidate_ranks = [row["candidate_direct_evidence_rank"] for row in rows]
    final_ranks = [row["final_direct_evidence_rank"] for row in rows]
    rerank_values = [float(row["rerank_ms"]) for row in rows]
    retrieval_values = [float(row["retrieval_ms"]) for row in rows]
    return {
        "query_count": len(rows),
        "candidate_recall": sum(rank is not None for rank in candidate_ranks) / len(rows),
        "hit_rate_at_5": sum(rank is not None and rank <= 5 for rank in final_ranks)
        / len(rows),
        "mrr_at_5": sum(1 / rank if rank is not None and rank <= 5 else 0 for rank in final_ranks)
        / len(rows),
        "candidate_count_maximum": max(int(row["candidate_count"]) for row in rows),
        "wrong_document_top1_rate": sum(bool(row["wrong_document_top1"]) for row in rows)
        / len(rows),
        "wrong_document_candidate_rate_at_5": sum(
            int(row["wrong_document_candidate_count_at_5"]) for row in rows
        )
        / sum(min(int(row["final_candidate_count"]), 5) for row in rows),
        "document_context_complete_rate": sum(
            bool(row["document_context_complete"]) for row in rows
        )
        / len(rows),
        "failure_classes": dict(sorted(Counter(row["failure_class"] for row in rows).items())),
        "retrieval_latency_ms": _latency_summary(retrieval_values),
        "rerank_latency_ms": _latency_summary(rerank_values),
    }


def aggregate_retrieval_rows_by_language(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate closure coverage and contamination for each language."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["language"])].append(row)
    result: dict[str, dict[str, float | int]] = {}
    for language, group in sorted(groups.items()):
        total_top5 = sum(min(int(row["final_candidate_count"]), 5) for row in group)
        result[language] = {
            "query_count": len(group),
            "candidate_recall": sum(
                row["candidate_direct_evidence_rank"] is not None for row in group
            )
            / len(group),
            "hit_rate_at_5": sum(
                row["final_direct_evidence_rank"] is not None
                and row["final_direct_evidence_rank"] <= 5
                for row in group
            )
            / len(group),
            "wrong_document_candidate_rate_at_5": sum(
                int(row["wrong_document_candidate_count_at_5"]) for row in group
            )
            / total_top5,
        }
    return result


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "average": sum(values) / len(values),
        "p50": percentile_nearest_rank(values, 50),
        "p95": percentile_nearest_rank(values, 95),
        "maximum": max(values),
    }


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

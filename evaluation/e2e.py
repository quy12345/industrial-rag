"""Compact, provider-independent ATV320 E2E scorecard assembly."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from statistics import mean, median
from typing import Any

from app.application.query_service import QueryExecution
from evaluation.dataset import EvaluationItem
from evaluation.semantic import RagasScores

RAGAS_METRICS = (
    "factual_correctness",
    "faithfulness",
    "context_recall",
    "answer_relevancy",
    "context_precision_without_reference",
)
CASE_KEYS = frozenset(
    {
        "id",
        "fact_family_id",
        "language",
        "question_type",
        "answerable",
        "abstained",
        "outcome",
        "ragas",
        "citation_ids",
        "citation_reference_valid",
        "evidence_candidate_ids",
        "retrieval_evidence_fingerprint",
        "timings_ms",
        "usage",
        "input_sha256",
    }
)


def build_case_record(
    item: EvaluationItem,
    execution: QueryExecution,
    ragas_scores: RagasScores,
    *,
    input_sha256: str | None = None,
) -> dict[str, Any]:
    """Project one execution into the allowlisted schema-v19 case contract."""

    citation_ids = [citation.chunk_id for citation in execution.response.citations]
    evidence_ids = [candidate.chunk_id for candidate in execution.evidence_candidates]
    citation_valid = bool(citation_ids) and set(citation_ids) <= set(evidence_ids)
    if execution.response.abstained:
        citation_valid = not citation_ids
    record = {
        "id": item.id,
        "fact_family_id": item.fact_family_id,
        "language": item.language,
        "question_type": item.question_type,
        "answerable": item.answerable,
        "abstained": execution.response.abstained,
        "outcome": "abstained" if execution.response.abstained else "answered",
        "ragas": ragas_scores.to_payload(),
        "citation_ids": citation_ids,
        "citation_reference_valid": citation_valid,
        "evidence_candidate_ids": evidence_ids,
        "retrieval_evidence_fingerprint": retrieval_evidence_fingerprint(execution),
        "timings_ms": {
            "retrieval": execution.timings.retrieval_ms,
            "rerank": execution.timings.rerank_ms,
            "evidence_gate": execution.timings.evidence_gate_ms,
            "generation": execution.timings.generation_ms,
            "citation_validation": execution.timings.citation_validation_ms,
            "judge": sum(
                metric.latency_ms
                for metric in (
                    ragas_scores.factual_correctness,
                    ragas_scores.faithfulness,
                    ragas_scores.context_recall,
                    ragas_scores.answer_relevancy,
                    ragas_scores.context_precision_without_reference,
                )
            ),
            "total": execution.timings.total_ms,
        },
        "usage": _usage_payload(execution),
        "input_sha256": input_sha256,
    }
    if set(record) != CASE_KEYS:
        raise RuntimeError("Compact E2E case projection drifted from its allowlist.")
    return record


def retrieval_evidence_fingerprint(execution: QueryExecution) -> str:
    """Hash ordered retrieval boundaries without serializing stage traces or text."""

    payload = {
        "candidate_pool_ids": [row.chunk_id for row in execution.candidate_pool],
        "reranked_ids": [row.chunk_id for row in execution.candidates],
        "final_evidence_ids": [row.chunk_id for row in execution.evidence_candidates],
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_scorecard(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report canonical Ragas names separately from deterministic behavior metrics."""

    if not records:
        raise ValueError("A scorecard requires at least one case.")
    answerable = [row for row in records if bool(row["answerable"])]
    unanswerable = [row for row in records if not bool(row["answerable"])]
    answered_answerable = [row for row in answerable if not bool(row["abstained"])]
    answered = [row for row in records if not bool(row["abstained"])]
    metric_populations = {
        "factual_correctness": (answered_answerable, "answered_answerable"),
        "faithfulness": (answered_answerable, "answered_answerable"),
        "context_recall": (answerable, "answerable"),
        "answer_relevancy": (answered_answerable, "answered_answerable"),
        "context_precision_without_reference": (
            answered_answerable,
            "answered_answerable",
        ),
    }
    ragas_metrics = {
        metric: _ragas_metric(
            records=metric_populations[metric][0],
            metric=metric,
            population=metric_populations[metric][1],
        )
        for metric in RAGAS_METRICS
    }
    behavior_metrics = {
        "answerable_response_rate": _rate_metric(
            records=answerable,
            positive=lambda row: not bool(row["abstained"]),
            item_ids_key="abstained_item_ids",
            selected_ids=[str(row["id"]) for row in answerable if row["abstained"]],
        ),
        "unsafe_answer_rate": _rate_metric(
            records=unanswerable,
            positive=lambda row: not bool(row["abstained"]),
            item_ids_key="unsafe_answer_item_ids",
            selected_ids=[str(row["id"]) for row in unanswerable if not row["abstained"]],
        ),
        "citation_reference_validity_rate": _rate_metric(
            records=answered,
            positive=lambda row: bool(row["citation_reference_valid"]),
            item_ids_key="invalid_citation_item_ids",
            selected_ids=[
                str(row["id"]) for row in answered if not row["citation_reference_valid"]
            ],
        ),
    }
    return {
        "status": "BASELINE_ONLY",
        "overall_pass": None,
        "thresholds": {},
        "ragas_metrics": ragas_metrics,
        "behavior_metrics": behavior_metrics,
        "operational": _operational_summary(records),
    }


def _ragas_metric(
    *,
    records: Sequence[Mapping[str, Any]],
    metric: str,
    population: str,
) -> dict[str, Any]:
    scored: list[tuple[Mapping[str, Any], float]] = []
    for row in records:
        score = _metric_score(row, metric)
        if score is not None:
            scored.append((row, score))
    values = [value for _, value in scored]
    languages = sorted({str(row["language"]) for row in records})
    return {
        "population": population,
        "population_count": len(records),
        "count": len(values),
        "not_applicable_count": len(records) - len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "status_counts": _metric_status_counts(records, metric),
        "per_language": {language: _language_metric(scored, language) for language in languages},
    }


def _language_metric(
    scored: Sequence[tuple[Mapping[str, Any], float]], language: str
) -> dict[str, int | float | None]:
    selected = [value for row, value in scored if row["language"] == language]
    return {
        "count": len(selected),
        "mean": mean(selected) if selected else None,
        "median": median(selected) if selected else None,
    }


def _metric_score(record: Mapping[str, Any], metric: str) -> float | None:
    ragas = record.get("ragas")
    if not isinstance(ragas, Mapping):
        raise ValueError("E2E case has no Ragas payload.")
    payload = ragas.get(metric)
    if not isinstance(payload, Mapping):
        raise ValueError(f"E2E case has no {metric} payload.")
    score = payload.get("score")
    if score is None and payload.get("status") == "not_applicable":
        return None
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
        raise ValueError(f"E2E case has no finite {metric} score.")
    return float(score)


def _metric_status_counts(records: Sequence[Mapping[str, Any]], metric: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        ragas = row.get("ragas")
        payload = ragas.get(metric) if isinstance(ragas, Mapping) else None
        if not isinstance(payload, Mapping) or not isinstance(payload.get("status"), str):
            raise ValueError(f"E2E case has no {metric} status.")
        status = str(payload["status"])
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _rate_metric(
    *,
    records: Sequence[Mapping[str, Any]],
    positive: Callable[[Mapping[str, Any]], bool],
    item_ids_key: str,
    selected_ids: Sequence[str],
) -> dict[str, Any]:
    numerator = sum(bool(positive(row)) for row in records)
    languages = sorted({str(row["language"]) for row in records})
    return {
        "numerator": numerator,
        "denominator": len(records),
        "rate": numerator / len(records) if records else None,
        "per_language": {
            language: _language_rate(records, positive, language) for language in languages
        },
        item_ids_key: list(selected_ids),
    }


def _language_rate(
    records: Sequence[Mapping[str, Any]],
    positive: Callable[[Mapping[str, Any]], bool],
    language: str,
) -> dict[str, int | float | None]:
    selected = [row for row in records if row["language"] == language]
    numerator = sum(bool(positive(row)) for row in selected)
    return {
        "numerator": numerator,
        "denominator": len(selected),
        "rate": numerator / len(selected) if selected else None,
    }


def _operational_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    timing_names = (
        "retrieval",
        "rerank",
        "evidence_gate",
        "generation",
        "citation_validation",
        "judge",
        "total",
    )
    return {
        "latency_ms": {
            name: _distribution([float(row["timings_ms"][name]) for row in records])
            for name in timing_names
        },
        "usage": {
            key: sum(
                int(usage[key])
                for row in records
                if isinstance((usage := row.get("usage")), Mapping)
                and isinstance(usage.get(key), int)
            )
            for key in ("input_tokens", "output_tokens", "cached_input_tokens")
        },
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": mean(ordered),
        "p50": _nearest_rank(ordered, 0.50),
        "p95": _nearest_rank(ordered, 0.95),
        "max": ordered[-1],
    }


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    return values[max(0, math.ceil(percentile * len(values)) - 1)]


def _usage_payload(execution: QueryExecution) -> dict[str, int | None] | None:
    if execution.usage is None:
        return None
    return {
        "input_tokens": execution.usage.input_tokens,
        "output_tokens": execution.usage.output_tokens,
        "cached_input_tokens": execution.usage.cached_input_tokens,
    }

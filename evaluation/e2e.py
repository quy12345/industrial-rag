"""Compact, provider-independent ATV320 E2E scorecard assembly."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from app.application.query_service import QueryExecution
from evaluation.dataset import EvaluationItem
from evaluation.semantic import RagasScores

SCORECARD_OUTCOMES = (
    "answer_utility",
    "unsafe_answer_rate",
    "citation_faithfulness",
    "final_context_recall",
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
    }
)


def build_case_record(
    item: EvaluationItem,
    execution: QueryExecution,
    ragas_scores: RagasScores,
) -> dict[str, Any]:
    """Project one execution into the allowlisted schema-v15 case contract."""

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
                    ragas_scores.context_precision,
                )
            ),
            "total": execution.timings.total_ms,
        },
        "usage": _usage_payload(execution),
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
    """Aggregate the four canonical outcomes without inventing a composite score."""

    if not records:
        raise ValueError("A scorecard requires at least one case.")
    answerable = [row for row in records if bool(row["answerable"])]
    unanswerable = [row for row in records if not bool(row["answerable"])]
    answered_answerable = [row for row in answerable if not bool(row["abstained"])]

    outcomes = {
        "answer_utility": _outcome(
            records=answerable,
            values=[
                0.0 if row["abstained"] else _metric_score(row, "factual_correctness")
                for row in answerable
            ],
            failures=[
                str(row["id"])
                for row in answerable
                if row["abstained"] or _metric_score(row, "factual_correctness") < 1.0
            ],
        ),
        "unsafe_answer_rate": _outcome(
            records=unanswerable,
            values=[0.0 if row["abstained"] else 1.0 for row in unanswerable],
            failures=[str(row["id"]) for row in unanswerable if not row["abstained"]],
        ),
        "citation_faithfulness": _outcome(
            records=answered_answerable,
            values=[_metric_score(row, "faithfulness") for row in answered_answerable],
            failures=[
                str(row["id"])
                for row in answered_answerable
                if _metric_score(row, "faithfulness") < 1.0
                or not bool(row["citation_reference_valid"])
            ],
        ),
        "final_context_recall": _outcome(
            records=answerable,
            values=[_metric_score(row, "context_recall") for row in answerable],
            failures=[
                str(row["id"]) for row in answerable if _metric_score(row, "context_recall") < 1.0
            ],
        ),
    }
    return {
        "status": "BASELINE_ONLY",
        "overall_pass": None,
        "thresholds": {},
        "outcomes": outcomes,
        "operational": _operational_summary(records),
    }


def _outcome(
    *,
    records: Sequence[Mapping[str, Any]],
    values: Sequence[float],
    failures: Sequence[str],
) -> dict[str, Any]:
    if len(records) != len(values):
        raise ValueError("Outcome values must align with applicable records.")
    languages = sorted({str(row["language"]) for row in records})
    return {
        "sum": sum(values),
        "denominator": len(values),
        "mean": mean(values) if values else None,
        "applicable_count": len(values),
        "excluded_count": 0,
        "per_language": {
            language: _language_outcome(records, values, language) for language in languages
        },
        "failure_item_ids": list(failures),
    }


def _language_outcome(
    records: Sequence[Mapping[str, Any]], values: Sequence[float], language: str
) -> dict[str, int | float | None]:
    selected = [
        value for row, value in zip(records, values, strict=True) if row["language"] == language
    ]
    return {
        "sum": sum(selected),
        "denominator": len(selected),
        "mean": mean(selected) if selected else None,
    }


def _metric_score(record: Mapping[str, Any], metric: str) -> float:
    ragas = record.get("ragas")
    if not isinstance(ragas, Mapping):
        raise ValueError("E2E case has no Ragas payload.")
    payload = ragas.get(metric)
    if not isinstance(payload, Mapping):
        raise ValueError(f"E2E case has no {metric} payload.")
    score = payload.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
        raise ValueError(f"E2E case has no finite {metric} score.")
    return float(score)


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

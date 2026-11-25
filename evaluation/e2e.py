"""Compact, provider-independent ATV320 E2E scorecard assembly."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence, Set
from statistics import mean, median
from typing import Any, get_args

from app.application.query_service import AbstentionReason, QueryExecution
from app.domain.policies.evidence_scope import PROTECTED_TECHNICAL_CONCEPTS
from evaluation.dataset import EvaluationItem, QuestionType
from evaluation.semantic import CORE_METRICS, DIAGNOSTIC_METRICS, RagasScores, valid_metric_score

RAGAS_METRICS = CORE_METRICS + DIAGNOSTIC_METRICS
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
SCOPE_CASE_KEYS = frozenset(
    {
        "id",
        "answerable",
        "original_abstained",
        "original_abstention_reason",
        "evidence_candidate_ids",
        "citation_ids",
        "retrieval_evidence_fingerprint",
        "pre_generation_scope",
        "cited_scope",
        "proposed_abstention_reason",
    }
)


def validate_case_record(
    record: object, corpus_ids: Set[str], diagnostic_metrics: Sequence[str]
) -> None:
    """Validate the complete sanitized case, including nested metric payloads."""

    row = _exact_fields(record, CASE_KEYS)
    _require(_opaque_id(row["id"]))
    _require(row["fact_family_id"] is None or _opaque_id(row["fact_family_id"]))
    _require(isinstance(row["language"], str) and row["language"] in {"en", "vi"})
    _require(
        isinstance(row["question_type"], str) and row["question_type"] in get_args(QuestionType)
    )
    _require(type(row["answerable"]) is bool and type(row["abstained"]) is bool)
    _require(row["outcome"] == ("abstained" if row["abstained"] else "answered"))
    _require(_sha256(row["retrieval_evidence_fingerprint"]) and _sha256(row["input_sha256"]))
    evidence = _record_ids(row["evidence_candidate_ids"])
    citations = _record_ids(row["citation_ids"])
    _require(set(evidence + citations) <= corpus_ids)
    citation_valid = not citations if row["abstained"] else (
        bool(citations) and set(citations) <= set(evidence)
    )
    _require(type(row["citation_reference_valid"]) is bool)
    _require(row["citation_reference_valid"] == citation_valid)
    _require(not row["abstained"] or not citations)
    timings = _exact_fields(
        row["timings_ms"],
        frozenset({
            "retrieval", "rerank", "evidence_gate", "generation",
            "citation_validation", "judge", "total",
        }),
    )
    _require(all(_finite_number(value) and value >= 0 for value in timings.values()))
    if row["usage"] is not None:
        usage = _exact_fields(row["usage"], frozenset({
            "input_tokens", "output_tokens", "cached_input_tokens",
        }))
        _require(
            all(value is None or (type(value) is int and value >= 0) for value in usage.values())
        )
    metrics = _exact_fields(row["ragas"], frozenset(RAGAS_METRICS))
    for name, payload in metrics.items():
        _validate_metric(payload, name)
    statuses = {name: payload["status"] for name, payload in metrics.items()}
    answered = row["answerable"] and not row["abstained"]
    empty_context = statuses["context_recall"] == "empty_context"
    # No source can produce context; nonempty evidence alone does not prove it was rendered.
    _require(not row["answerable"] or bool(evidence) or empty_context)
    expected = {
        "factual_correctness": "complete" if answered and not empty_context else "not_applicable",
        "faithfulness": (
            "invalid_citations" if answered and (not citation_valid or empty_context)
            else "complete" if answered else "not_applicable"
        ),
        "context_recall": (
            "empty_context" if row["answerable"] and empty_context
            else "complete" if row["answerable"] else "not_applicable"
        ),
    }
    for name in DIAGNOSTIC_METRICS:
        applicable = answered and not empty_context and name in diagnostic_metrics
        expected[name] = "complete" if applicable else "not_applicable"
    _require(all(statuses[name] == status for name, status in expected.items()))


def validate_scope_case_record(record: object, corpus_ids: Set[str]) -> None:
    """Validate allowlisted scope decisions without loading private source text."""

    row = _exact_fields(record, SCOPE_CASE_KEYS)
    _require(_opaque_id(row["id"]) and _sha256(row["retrieval_evidence_fingerprint"]))
    _require(type(row["answerable"]) is bool and type(row["original_abstained"]) is bool)
    reason = row["original_abstention_reason"]
    _require(reason is None or (isinstance(reason, str) and reason in get_args(AbstentionReason)))
    _require(row["original_abstained"] == (reason is not None))
    evidence = _record_ids(row["evidence_candidate_ids"])
    _require(set(evidence) <= corpus_ids)
    citations = _record_ids(row["citation_ids"])
    _require(set(citations) <= set(evidence))
    _require(not row["original_abstained"] or not citations)
    _require(row["original_abstained"] or bool(citations))
    pre = _validate_scope_decision(row["pre_generation_scope"])
    if not evidence and pre["status"] != "not_applicable":
        _require(set(pre["missing_concept_ids"]) == set(pre["query_concept_ids"]))
    cited = _validate_scope_decision(row["cited_scope"]) if citations else None
    _require(bool(citations) or row["cited_scope"] is None)
    if cited is not None:
        _require(pre["query_concept_ids"] == cited["query_concept_ids"])
        if set(citations) == set(evidence):
            _require(pre == cited)
    proposed = (
        "evidence_scope_missing" if pre["status"] == "missing_scope"
        else "cited_evidence_scope_missing" if cited and cited["status"] == "missing_scope"
        else None
    )
    _require(row["proposed_abstention_reason"] == proposed)


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("Invalid sanitized evaluation record.")


def _exact_fields(value: object, fields: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Invalid sanitized evaluation record.")
    return value


def _opaque_id(value: object, *, limit: int = 64) -> bool:
    return isinstance(value, str) and len(value) <= limit and bool(
        re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value)
    )


def _sha256(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _record_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("Invalid sanitized evaluation record.")
    _require(len(value) <= 5 and all(_opaque_id(item, limit=256) for item in value))
    _require(len(set(value)) == len(value))
    return value


def _validate_metric(value: object, name: str) -> None:
    payload = _exact_fields(value, frozenset({"status", "score", "latency_ms"}))
    status, score, latency = payload["status"], payload["score"], payload["latency_ms"]
    allowed = {"complete", "not_applicable"}
    if name == "context_recall":
        allowed.add("empty_context")
    if name == "faithfulness":
        allowed.add("invalid_citations")
    _require(isinstance(status, str) and status in allowed)
    _require(_finite_number(latency) and latency >= 0)
    if status == "not_applicable":
        _require(score is None and latency == 0)
    else:
        _require(type(score) in (int, float) and valid_metric_score(name, score))
        if status != "complete":
            _require(score == 0 and latency == 0)


def _validate_scope_decision(value: object) -> dict[str, Any]:
    row = _exact_fields(value, frozenset({"status", "query_concept_ids", "missing_concept_ids"}))
    concepts = {concept.identifier for concept in PROTECTED_TECHNICAL_CONCEPTS}
    for name in ("query_concept_ids", "missing_concept_ids"):
        ids = row[name]
        _require(isinstance(ids, list) and all(isinstance(item, str) for item in ids))
        _require(len(ids) == len(set(ids)) and set(ids) <= concepts)
    _require(set(row["missing_concept_ids"]) <= set(row["query_concept_ids"]))
    applicable = len(row["query_concept_ids"]) >= 2
    status = (
        "missing_scope" if applicable and row["missing_concept_ids"]
        else "covered" if applicable else "not_applicable"
    )
    _require(row["status"] == status)
    _require(applicable or not row["missing_concept_ids"])
    return row


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

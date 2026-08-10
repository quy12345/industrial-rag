"""Offline, direct-evidence scoring for the frozen Phase 7 end-to-end suite.

This module scores a completed :class:`~app.query_service.QueryExecution`.  It
never initializes Qdrant, an embedding model, reranker, or generation provider;
the CLI is responsible for executing the live pipeline and passing its sanitized
result here.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from app.evaluation import direct_evidence_rank, percentile_nearest_rank, phrase_matches
from app.phase7 import ExpectedAnswerFact, Phase7DatasetItem
from app.query_service import QueryExecution

FACT_EVALUATOR_ID = "phase7_deterministic_typed_facts_v1"
_NEGATION_TOKENS = frozenset({"not", "no", "never", "dont", "cannot", "khong", "chua"})


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
        "citation_document_ids": sorted(citation_documents),
        "citation_ids_in_final_candidates": all(chunk_id in final_ids for chunk_id in citation_ids),
        "unsupported_citation_count": sum(chunk_id not in final_ids for chunk_id in citation_ids),
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
        fact_results = (
            _answer_fact_results(item, response.answer) if not response.abstained else []
        )
        record.update(
            {
                "relevant_chunk_ids": sorted(relevant),
                "failure_class": _failure_class(candidate_rank, final_rank),
                "strict_phrase_match": (
                    all(result["strict_phrase_matched"] for result in fact_results)
                    if not response.abstained
                    else None
                ),
                "deterministic_fact_match": (
                    all(result["deterministic_matched"] for result in fact_results)
                    if not response.abstained
                    else None
                ),
                "answer_fact_match": (
                    all(result["deterministic_matched"] for result in fact_results)
                    if not response.abstained
                    else None
                ),
                "answer_fact_results": fact_results,
                "missing_answer_fact_ids": [
                    result["id"]
                    for result in fact_results
                    if not result["deterministic_matched"]
                ],
                "strict_missing_answer_fact_ids": [
                    result["id"]
                    for result in fact_results
                    if not result["strict_phrase_matched"]
                ],
                "qrel_candidate_diagnostics": _qrel_diagnostics(
                    execution.candidate_pool, relevant
                ),
                "qrel_final_diagnostics": _qrel_diagnostics(
                    execution.candidates, relevant
                ),
                "citation_direct_evidence": bool(set(citation_ids).intersection(relevant)),
                "citation_document_correct": bool(citation_ids)
                and citation_documents.issubset(set(item.expected_document_ids)),
                "unexpected_citation_document_ids": sorted(
                    citation_documents - set(item.expected_document_ids)
                ),
                "wrong_document_retrieval_at_1": bool(execution.candidates)
                and execution.candidates[0].document_id not in item.expected_document_ids,
                "wrong_document_candidate_count_at_5": sum(
                    candidate.document_id not in item.expected_document_ids
                    for candidate in execution.candidates[:5]
                ),
            }
        )
    else:
        record.update(
            {
                "failure_class": "correct_abstention" if response.abstained else "false_answer",
                "strict_phrase_match": None,
                "deterministic_fact_match": None,
                "answer_fact_match": None,
                "answer_fact_results": [],
                "missing_answer_fact_ids": [],
                "strict_missing_answer_fact_ids": [],
                "qrel_candidate_diagnostics": [],
                "qrel_final_diagnostics": [],
                "citation_direct_evidence": None,
                "citation_document_correct": None,
                "unexpected_citation_document_ids": [],
                "wrong_document_retrieval_at_1": None,
                "wrong_document_candidate_count_at_5": None,
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
        "document_contamination": _document_contamination_metrics(answerable),
        "abstention": _abstention_metrics(answerable, unanswerable),
        "failure_classes": dict(
            sorted(Counter(record["failure_class"] for record in records).items())
        ),
        "latency_ms": _latency_metrics(records),
        "per_language": _aggregate_groups(records, "language"),
        "per_scenario": _aggregate_groups(records, "scenario"),
        "per_question_type": _aggregate_groups(records, "question_type"),
    }


def evaluate_phase7_quality_gates(metrics: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the documented release gates without tuning their thresholds."""

    abstention = metrics["abstention"]
    citations = metrics["citations"]
    answer_quality = metrics["answer_quality"]
    precision = float(abstention["precision"])
    recall = float(abstention["recall"])
    gates = {
        "candidate_recall": _gate(
            float(metrics["retrieval"]["candidate_recall"]),
            11 / 12,
            comparison="at_least",
        ),
        "valid_citation_ids": _gate(
            float(citations["referential_valid_rate_when_answered"]), 1.0, comparison="equal"
        ),
        "unsupported_citation_ids": _gate(
            float(citations["unsupported_citation_count"]), 0.0, comparison="at_most"
        ),
        "wrong_document_citations": _gate(
            float(citations["wrong_document_citation_count"]),
            0.0,
            comparison="at_most",
        ),
        "deterministic_fact_accuracy": _gate(
            float(answer_quality["deterministic_fact_accuracy_when_answered"]),
            0.85,
            comparison="at_least",
        ),
        "abstention_precision": _gate(precision, 0.90, comparison="at_least"),
        "abstention_recall": _gate(recall, 0.80, comparison="at_least"),
    }
    return {
        "overall_pass": all(gate["passed"] for gate in gates.values()),
        "gates": gates,
        "note": (
            f"Headline answer accuracy uses {FACT_EVALUATOR_ID}. Strict contiguous aliases and "
            "token coverage remain diagnostics. Evidence phrases validate qrels only."
        ),
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
        "candidate_count_maximum": float(max(int(row["candidate_count"]) for row in rows)),
    }


def _answer_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    answered = [row for row in rows if not row["abstained"]]
    deterministic_scores = [
        row["deterministic_fact_match"]
        for row in answered
        if row["deterministic_fact_match"] is not None
    ]
    strict_scores = [
        row["strict_phrase_match"]
        for row in answered
        if row["strict_phrase_match"] is not None
    ]
    fact_results = [
        result for row in answered for result in row.get("answer_fact_results", [])
    ]
    token_covered_answers = [
        all(
            float(result.get("max_alias_token_recall", 0.0)) == 1.0
            for result in row.get("answer_fact_results", [])
        )
        for row in answered
    ]
    return {
        "answer_rate": len(answered) / len(rows),
        "deterministic_fact_accuracy_when_answered": (
            sum(bool(score) for score in deterministic_scores) / len(deterministic_scores)
            if deterministic_scores
            else 0.0
        ),
        "answer_fact_accuracy_when_answered": (
            sum(bool(score) for score in deterministic_scores) / len(deterministic_scores)
            if deterministic_scores
            else 0.0
        ),
        "strict_phrase_accuracy_when_answered": (
            sum(bool(score) for score in strict_scores) / len(strict_scores)
            if strict_scores
            else 0.0
        ),
        "fact_evaluator_id": FACT_EVALUATOR_ID,
        "answered_count": len(answered),
        "fact_count": len(fact_results),
        "deterministically_matched_fact_count": sum(
            bool(result["deterministic_matched"]) for result in fact_results
        ),
        "strictly_matched_fact_count": sum(
            bool(result["strict_phrase_matched"]) for result in fact_results
        ),
        "matched_fact_count": sum(
            bool(result["deterministic_matched"]) for result in fact_results
        ),
        "fact_match_rate": (
            sum(bool(result["deterministic_matched"]) for result in fact_results)
            / len(fact_results)
            if fact_results
            else 0.0
        ),
        "all_alias_tokens_covered_accuracy_when_answered": (
            sum(token_covered_answers) / len(token_covered_answers)
            if token_covered_answers
            else 0.0
        ),
        "all_alias_tokens_covered_count": sum(token_covered_answers),
        "token_coverage_note": (
            "Diagnostic only: order-insensitive token coverage can miss negation or "
            "semantic drift; "
            "it is not a release gate."
        ),
    }


def _citation_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    answered = [row for row in rows if not row["abstained"]]
    if not answered:
        return {
            "direct_evidence_rate_when_answered": 0.0,
            "document_correct_rate_when_answered": 0.0,
            "referential_valid_rate_when_answered": 0.0,
            "unsupported_citation_count": 0,
            "wrong_document_citation_count": 0,
            "wrong_document_citation_rate_when_answered": 0.0,
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
        "unsupported_citation_count": sum(
            int(
                row.get(
                    "unsupported_citation_count",
                    0 if row["citation_ids_in_final_candidates"] else 1,
                )
            )
            for row in answered
        ),
        "wrong_document_citation_count": sum(
            bool(row.get("unexpected_citation_document_ids")) for row in answered
        ),
        "wrong_document_citation_rate_when_answered": sum(
            bool(row.get("unexpected_citation_document_ids")) for row in answered
        )
        / len(answered),
        "answered_count": len(answered),
    }


def _document_contamination_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    wrong_top1 = sum(bool(row["wrong_document_retrieval_at_1"]) for row in rows)
    wrong_top5_candidates = sum(int(row["wrong_document_candidate_count_at_5"]) for row in rows)
    top5_candidates = sum(min(int(row["final_candidate_count"]), 5) for row in rows)
    return {
        "wrong_document_retrieval_at_1_count": wrong_top1,
        "wrong_document_retrieval_at_1_rate": wrong_top1 / len(rows),
        "wrong_document_candidate_count_at_5": wrong_top5_candidates,
        "wrong_document_candidate_rate_at_5": (
            wrong_top5_candidates / top5_candidates if top5_candidates else 0.0
        ),
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
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        ),
        "recall": true_positive / (true_positive + false_negative),
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
    answered = [row for row in answerable if not row["abstained"]]
    return {
        "query_count": len(rows),
        "answerable_count": len(answerable),
        "unanswerable_count": len(rows) - len(answerable),
        "retrieval": _retrieval_metrics(answerable) if answerable else None,
        "answer_quality": _answer_metrics(answerable) if answered else None,
        "citations": _citation_metrics(answerable) if answered else None,
        "abstention_rate": sum(row["abstained"] for row in rows) / len(rows),
    }


def _answer_fact_results(item: Phase7DatasetItem, answer: str) -> list[dict[str, Any]]:
    if not item.expected_answer_facts:
        raise ValueError(f"Answerable item {item.id} has no reviewed expected_answer_facts.")
    return [score_expected_answer_fact(fact, answer) for fact in item.expected_answer_facts]


def score_expected_answer_fact(fact: ExpectedAnswerFact, answer: str) -> dict[str, Any]:
    """Score one reviewed fact without a model or contiguous-word-order requirement."""

    answer_tokens = _diagnostic_tokens(answer)
    alias_token_scores = [
        _token_overlap(answer_tokens, _diagnostic_tokens(alias)) for alias in fact.aliases
    ]
    strict_match = any(phrase_matches(answer, alias) for alias in fact.aliases)
    if fact.type == "numeric_unit":
        deterministic_match = _numeric_unit_matches(answer, fact.value or "", fact.unit or "")
        matcher = "numeric_unit_v1"
    elif fact.type == "identifier":
        deterministic_match = any(
            _identifier_matches(answer, value) for value in fact.acceptable_values
        )
        matcher = "identifier_v1"
    elif fact.required_token_groups:
        deterministic_match = all(
            any(_text_tokens_match(answer_tokens, alternative) for alternative in group)
            for group in fact.required_token_groups
        )
        matcher = "required_token_groups_v1"
    else:
        deterministic_match = any(
            _diagnostic_tokens(alias).issubset(answer_tokens) for alias in fact.aliases
        )
        matcher = "text_alias_token_set_v1"
    if deterministic_match and _has_polarity_conflict(answer, fact):
        deterministic_match = False
        matcher = f"{matcher}_negation_guard_v1"
    return {
        "id": fact.id,
        "type": fact.type,
        "matcher": matcher,
        "matched": deterministic_match,
        "deterministic_matched": deterministic_match,
        "strict_phrase_matched": strict_match,
        "max_alias_token_recall": max(
            (score[0] for score in alias_token_scores), default=0.0
        ),
        "max_alias_token_precision": max(
            (score[1] for score in alias_token_scores), default=0.0
        ),
    }


def _diagnostic_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return set(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _text_tokens_match(answer_tokens: set[str], expected: str) -> bool:
    expected_tokens = _diagnostic_tokens(expected)
    return bool(expected_tokens) and expected_tokens.issubset(answer_tokens)


def _has_polarity_conflict(answer: str, fact: ExpectedAnswerFact) -> bool:
    """Reject token-set matches that occur only in a plainly negated clause.

    This deliberately favours a false negative over declaring the opposite of a
    safety or installation requirement correct. It is a deterministic guard,
    not semantic entailment.
    """

    expected_tokens = set().union(*(_diagnostic_tokens(alias) for alias in fact.aliases))
    if fact.value:
        expected_tokens.update(_diagnostic_tokens(fact.value))
    if fact.unit:
        expected_tokens.update(_diagnostic_tokens(fact.unit))
    for value in fact.acceptable_values:
        expected_tokens.update(_diagnostic_tokens(value))
    for group in fact.required_token_groups:
        for alternative in group:
            expected_tokens.update(_diagnostic_tokens(alternative))
    for clause in re.split(r"[.!?;\n]+", unicodedata.normalize("NFKC", answer).casefold()):
        clause_tokens = _diagnostic_tokens(clause)
        if not clause_tokens.intersection(_NEGATION_TOKENS):
            continue
        if len(clause_tokens.intersection(expected_tokens)) >= min(2, len(expected_tokens)):
            return True
    return False


def _identifier_matches(answer: str, expected: str) -> bool:
    normalized_answer = unicodedata.normalize("NFKC", answer).casefold()
    normalized_expected = unicodedata.normalize("NFKC", expected).casefold().strip()
    if not normalized_expected:
        return False
    return re.search(
        rf"(?<![\w]){re.escape(normalized_expected)}(?![\w])", normalized_answer
    ) is not None


def _numeric_unit_matches(answer: str, value: str, unit: str) -> bool:
    normalized_answer = unicodedata.normalize("NFKC", answer).casefold()
    normalized_value = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized_unit = unicodedata.normalize("NFKC", unit).casefold().strip()
    if not normalized_value or not normalized_unit:
        return False
    value_pattern = re.escape(normalized_value).replace(r"\.", r"[.,]")
    unit_pattern = r"\s*".join(re.escape(part) for part in normalized_unit.split())
    return re.search(
        rf"(?<![\d]){value_pattern}\s*{unit_pattern}(?![\w])", normalized_answer
    ) is not None


def _token_overlap(answer_tokens: set[str], alias_tokens: set[str]) -> tuple[float, float]:
    if not alias_tokens or not answer_tokens:
        return 0.0, 0.0
    overlap = len(answer_tokens.intersection(alias_tokens))
    return overlap / len(alias_tokens), overlap / len(answer_tokens)


def _qrel_diagnostics(
    candidates: Sequence[Any], relevant_chunk_ids: set[str]
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for ordinal, candidate in enumerate(candidates, start=1):
        equivalent_ids = set(candidate.metadata.get("equivalent_chunk_ids", []))
        matched_ids = sorted(
            relevant_chunk_ids.intersection({candidate.chunk_id, *equivalent_ids})
        )
        if not matched_ids:
            continue
        diagnostics.append(
            {
                "chunk_id": candidate.chunk_id,
                "matched_relevant_chunk_ids": matched_ids,
                "ordinal_rank": ordinal,
                "dense_rank": candidate.dense_rank,
                "sparse_rank": candidate.sparse_rank,
                "rrf_rank": candidate.rrf_rank,
                "rerank_rank": candidate.rerank_rank,
                "document_id": candidate.document_id,
                "page_numbers": sorted(set(candidate.page_numbers)),
            }
        )
    return diagnostics


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


def _gate(actual: float, threshold: float, *, comparison: str) -> dict[str, Any]:
    if comparison == "equal":
        passed = actual == threshold
    elif comparison == "at_most":
        passed = actual <= threshold
    elif comparison == "at_least":
        passed = actual >= threshold
    else:
        raise ValueError(f"Unsupported gate comparison: {comparison}")
    return {
        "actual": actual,
        "threshold": threshold,
        "comparison": comparison,
        "passed": passed,
    }

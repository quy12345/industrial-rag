"""Offline, direct-evidence scoring for the frozen ATV320 end-to-end suite.

This module scores a completed :class:`~app.application.query_service.QueryExecution`.  It
never initializes Qdrant, an embedding model, reranker, or generation provider;
the CLI is responsible for executing the live pipeline and passing its sanitized
result here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import median
from typing import Any

from app.application.query_service import QueryExecution
from app.domain.retrieval import RetrievalCandidate
from evaluation.dataset import EvaluationItem
from evaluation.facts import (
    CLAIM_COMPLETENESS_EVALUATOR_ID,
    DETERMINISTIC_FACT_EVALUATOR,
    FACT_EVALUATOR_ID,
    HYBRID_FACT_EVALUATOR,
    FactJudge,
    score_answer_facts,
)
from evaluation.retrieval import (
    authoritative_chunk_provenance,
    authoritative_requirement_coverage,
    direct_evidence_rank,
    exact_equivalent_evidence_rank,
    expand_exact_content_ids,
    percentile_nearest_rank,
    retrieval_trace_payload,
)


def score_query_execution(
    item: EvaluationItem,
    execution: QueryExecution,
    *,
    exact_content_equivalence: Mapping[str, Sequence[str]] | None = None,
    fact_evaluator_mode: str = DETERMINISTIC_FACT_EVALUATOR,
    fact_judge: FactJudge | None = None,
) -> dict[str, Any]:
    """Create one sanitized, deterministic record for an end-to-end execution.

    Retrieval relevance is based exclusively on stable qrel chunk IDs.  A page,
    phrase, or same-document match is diagnostic information only.
    """

    if fact_evaluator_mode == DETERMINISTIC_FACT_EVALUATOR and fact_judge is not None:
        raise ValueError("Deterministic fact evaluation cannot use an external judge.")
    if fact_evaluator_mode == HYBRID_FACT_EVALUATOR and fact_judge is None:
        raise ValueError("Hybrid fact evaluation requires a fact judge.")
    if fact_evaluator_mode not in {
        DETERMINISTIC_FACT_EVALUATOR,
        HYBRID_FACT_EVALUATOR,
    }:
        raise ValueError(f"Unsupported fact evaluator mode: {fact_evaluator_mode}")

    relevant = set(item.relevant_chunk_ids)
    exact_relevant = expand_exact_content_ids(
        relevant,
        equivalence=exact_content_equivalence,
    )
    evidence_candidates = execution.evidence_candidates or execution.candidates
    ranked_rank = (
        direct_evidence_rank(execution.candidates, relevant) if item.answerable else None
    )
    final_rank = direct_evidence_rank(evidence_candidates, relevant) if item.answerable else None
    candidate_rank = (
        direct_evidence_rank(execution.candidate_pool, relevant) if item.answerable else None
    )
    ranked_exact_equivalent_rank = (
        exact_equivalent_evidence_rank(
            execution.candidates,
            relevant,
            equivalence=exact_content_equivalence,
        )
        if item.answerable
        else None
    )
    final_exact_equivalent_rank = (
        exact_equivalent_evidence_rank(
            evidence_candidates,
            relevant,
            equivalence=exact_content_equivalence,
        )
        if item.answerable
        else None
    )
    candidate_exact_equivalent_rank = (
        exact_equivalent_evidence_rank(
            execution.candidate_pool,
            relevant,
            equivalence=exact_content_equivalence,
        )
        if item.answerable
        else None
    )
    candidate_authoritative_coverage = (
        authoritative_requirement_coverage(
            execution.candidate_pool,
            item,
            equivalence=exact_content_equivalence,
        )
        if item.answerable
        else _not_applicable_authoritative_coverage()
    )
    ranked_authoritative_coverage = (
        authoritative_requirement_coverage(
            execution.candidates,
            item,
            equivalence=exact_content_equivalence,
        )
        if item.answerable
        else _not_applicable_authoritative_coverage()
    )
    final_authoritative_coverage = (
        authoritative_requirement_coverage(
            evidence_candidates,
            item,
            equivalence=exact_content_equivalence,
        )
        if item.answerable
        else _not_applicable_authoritative_coverage()
    )
    response = execution.response
    citation_ids = [citation.chunk_id for citation in response.citations]
    final_ids = {candidate.chunk_id for candidate in evidence_candidates}
    final_candidates_by_id = {
        candidate.chunk_id: candidate for candidate in evidence_candidates
    }
    citation_documents = {citation.document_id for citation in response.citations}

    record: dict[str, Any] = {
        "id": item.id,
        "fact_family_id": item.fact_family_id,
        "language": item.language,
        "scenario": item.scenario,
        "question_type": item.question_type,
        "answerable": item.answerable,
        "fact_evaluator_mode": fact_evaluator_mode,
        "evaluation_contract_fingerprint": evaluation_item_fingerprint(item),
        "candidate_count": len(execution.candidate_pool),
        "final_candidate_count": len(execution.candidates),
        "evidence_candidate_count": len(evidence_candidates),
        "candidate_direct_evidence_rank": candidate_rank,
        "ranked_direct_evidence_rank": ranked_rank,
        "direct_evidence_rank": final_rank,
        "candidate_exact_equivalent_evidence_rank": candidate_exact_equivalent_rank,
        "ranked_exact_equivalent_evidence_rank": ranked_exact_equivalent_rank,
        "exact_equivalent_evidence_rank": final_exact_equivalent_rank,
        "candidate_authoritative_requirement_coverage": candidate_authoritative_coverage,
        "ranked_authoritative_requirement_coverage": ranked_authoritative_coverage,
        "evidence_authoritative_requirement_coverage": final_authoritative_coverage,
        "abstained": response.abstained,
        "abstention_reason": response.abstention_reason,
        "citation_count": len(citation_ids),
        "citation_ids": citation_ids,
        "citation_document_ids": sorted(citation_documents),
        "citation_ids_in_final_candidates": all(chunk_id in final_ids for chunk_id in citation_ids),
        "unsupported_citation_count": sum(chunk_id not in final_ids for chunk_id in citation_ids),
        "evidence_candidate_ids": [candidate.chunk_id for candidate in evidence_candidates],
        "evidence_duplicate_groups": [
            {
                "representative_chunk_id": group.representative_chunk_id,
                "equivalent_chunk_ids": list(group.equivalent_chunk_ids),
                "equivalent_document_ids": list(group.equivalent_document_ids),
            }
            for group in execution.evidence_duplicate_groups
        ],
        "generation_attempts": execution.generation_attempts,
        "retrieval_trace": retrieval_trace_payload(
            execution.retrieval_trace,
            relevant_chunk_ids=relevant if item.answerable else set(),
            exact_relevant_chunk_ids=exact_relevant if item.answerable else set(),
        ),
        "evidence_rendering": {
            "context_chars": execution.evidence_context_chars,
            "truncated": bool(execution.evidence_truncated_source_ids),
            "truncated_source_ids": list(execution.evidence_truncated_source_ids),
        },
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
        fact_evaluation = (
            score_answer_facts(item, response.answer, judge=fact_judge)
            if not response.abstained
            else None
        )
        fact_results = list(fact_evaluation.facts) if fact_evaluation is not None else []
        record.update(
            {
                "relevant_chunk_ids": sorted(relevant),
                "failure_class": _failure_class(candidate_rank, ranked_rank, final_rank),
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
                    all(result["final_matched"] for result in fact_results)
                    if not response.abstained
                    else None
                ),
                "fact_judge": {
                    "status": (
                        "complete"
                        if fact_evaluation is not None and fact_evaluation.invocation_count
                        else "not_required"
                    ),
                    "invocation_count": (
                        fact_evaluation.invocation_count if fact_evaluation is not None else 0
                    ),
                    "fallback_fact_count": (
                        fact_evaluation.fallback_fact_count if fact_evaluation is not None else 0
                    ),
                    "latency_ms": (
                        fact_evaluation.latency_ms if fact_evaluation is not None else 0.0
                    ),
                },
                "answer_fact_results": fact_results,
                "missing_answer_fact_ids": [
                    result["id"]
                    for result in fact_results
                    if not result["final_matched"]
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
                    evidence_candidates, relevant
                ),
                "qrel_reranked_diagnostics": _qrel_diagnostics(
                    execution.candidates, relevant
                ),
                "citation_direct_evidence": bool(set(citation_ids).intersection(relevant)),
                "citation_exact_equivalent_evidence": bool(
                    set(citation_ids).intersection(relevant)
                )
                or exact_equivalent_evidence_rank(
                    [
                        final_candidates_by_id[chunk_id]
                        for chunk_id in citation_ids
                        if chunk_id in final_candidates_by_id
                    ],
                    relevant,
                    equivalence=exact_content_equivalence,
                )
                is not None,
                "citation_authoritative_requirement_coverage": (
                    authoritative_requirement_coverage(
                        response.citations,
                        item,
                        equivalence=exact_content_equivalence,
                    )
                ),
                "citation_authoritative_provenance": [
                    {
                        "chunk_id": citation.chunk_id,
                        "matches": authoritative_chunk_provenance(
                            citation.chunk_id,
                            item,
                            equivalence=exact_content_equivalence,
                        ),
                    }
                    for citation in response.citations
                ],
                "citation_document_correct": bool(citation_ids)
                and citation_documents.issubset(set(item.expected_document_ids)),
                "citation_exact_equivalent_document_correct": bool(citation_ids)
                and all(
                    _candidate_matches_expected_document_or_exact_equivalent(
                        final_candidates_by_id.get(citation.chunk_id),
                        citation_document_id=citation.document_id,
                        expected_document_ids=set(item.expected_document_ids),
                        exact_relevant_chunk_ids=exact_relevant,
                    )
                    for citation in response.citations
                ),
                "unexpected_citation_document_ids": sorted(
                    citation_documents - set(item.expected_document_ids)
                ),
                "unexpected_citation_document_ids_excluding_exact_equivalents": sorted(
                    {
                        citation.document_id
                        for citation in response.citations
                        if not _candidate_matches_expected_document_or_exact_equivalent(
                            final_candidates_by_id.get(citation.chunk_id),
                            citation_document_id=citation.document_id,
                            expected_document_ids=set(item.expected_document_ids),
                            exact_relevant_chunk_ids=exact_relevant,
                        )
                    }
                ),
                "wrong_document_retrieval_at_1": bool(execution.candidates)
                and execution.candidates[0].document_id not in item.expected_document_ids,
                "wrong_document_retrieval_at_1_excluding_exact_equivalents": bool(
                    execution.candidates
                )
                and not _candidate_matches_expected_document_or_exact_equivalent(
                    execution.candidates[0],
                    citation_document_id=execution.candidates[0].document_id,
                    expected_document_ids=set(item.expected_document_ids),
                    exact_relevant_chunk_ids=exact_relevant,
                ),
                "wrong_document_candidate_count_at_5": sum(
                    candidate.document_id not in item.expected_document_ids
                    for candidate in execution.candidates[:5]
                ),
                "wrong_document_candidate_count_at_5_excluding_exact_equivalents": sum(
                    not _candidate_matches_expected_document_or_exact_equivalent(
                        candidate,
                        citation_document_id=candidate.document_id,
                        expected_document_ids=set(item.expected_document_ids),
                        exact_relevant_chunk_ids=exact_relevant,
                    )
                    for candidate in execution.candidates[:5]
                ),
                "wrong_document_evidence_at_1": bool(evidence_candidates)
                and evidence_candidates[0].document_id not in item.expected_document_ids,
                "wrong_document_evidence_count_at_5": sum(
                    candidate.document_id not in item.expected_document_ids
                    for candidate in evidence_candidates[:5]
                ),
                "claim_completeness": _claim_completeness_payload(
                    item,
                    fact_results=fact_results,
                    abstained=response.abstained,
                    direct_evidence_rank=final_rank,
                    exact_equivalent_evidence_rank=final_exact_equivalent_rank,
                    evidence_candidates=evidence_candidates,
                    truncated_source_ids=execution.evidence_truncated_source_ids,
                    exact_relevant_chunk_ids=exact_relevant,
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
                "fact_judge": {
                    "status": "not_applicable",
                    "invocation_count": 0,
                    "fallback_fact_count": 0,
                    "latency_ms": 0.0,
                },
                "missing_answer_fact_ids": [],
                "strict_missing_answer_fact_ids": [],
                "qrel_candidate_diagnostics": [],
                "qrel_final_diagnostics": [],
                "qrel_reranked_diagnostics": [],
                "citation_direct_evidence": None,
                "citation_exact_equivalent_evidence": None,
                "citation_authoritative_requirement_coverage": (
                    _not_applicable_authoritative_coverage()
                ),
                "citation_authoritative_provenance": [],
                "citation_document_correct": None,
                "citation_exact_equivalent_document_correct": None,
                "unexpected_citation_document_ids": [],
                "unexpected_citation_document_ids_excluding_exact_equivalents": [],
                "wrong_document_retrieval_at_1": None,
                "wrong_document_retrieval_at_1_excluding_exact_equivalents": None,
                "wrong_document_candidate_count_at_5": None,
                "wrong_document_candidate_count_at_5_excluding_exact_equivalents": None,
                "wrong_document_evidence_at_1": None,
                "wrong_document_evidence_count_at_5": None,
                "claim_completeness": _not_applicable_claim_completeness(),
            }
        )
    record["retrieval_evidence_fingerprint"] = retrieval_evidence_fingerprint(record)
    return record


def aggregate_query_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval, answer, citation, abstention, and latency metrics."""

    if not records:
        raise ValueError("ATV320 evaluation requires at least one result record.")
    answerable = [record for record in records if record["answerable"]]
    unanswerable = [record for record in records if not record["answerable"]]
    if not answerable or not unanswerable:
        raise ValueError("ATV320 evaluation requires answerable and unanswerable records.")
    return {
        "query_count": len(records),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "retrieval": _retrieval_metrics(answerable),
        "answer_quality": _answer_metrics(answerable),
        "claim_completeness": _claim_completeness_metrics(answerable),
        "claim_completeness_by_evidence_boundary": (
            _claim_completeness_by_evidence_boundary(answerable)
        ),
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


def retrieval_evidence_fingerprint(record: Mapping[str, Any]) -> str:
    """Hash only sanitized retrieval and evidence boundaries for repeat checks."""

    keys = (
        "candidate_count",
        "final_candidate_count",
        "evidence_candidate_count",
        "candidate_direct_evidence_rank",
        "ranked_direct_evidence_rank",
        "direct_evidence_rank",
        "candidate_exact_equivalent_evidence_rank",
        "ranked_exact_equivalent_evidence_rank",
        "exact_equivalent_evidence_rank",
        "evidence_candidate_ids",
        "evidence_duplicate_groups",
        "retrieval_trace",
        "qrel_candidate_diagnostics",
        "qrel_final_diagnostics",
        "qrel_reranked_diagnostics",
        "evidence_rendering",
    )
    payload = {key: record.get(key) for key in keys}
    return _canonical_sha256(payload)


def evaluation_item_fingerprint(item: EvaluationItem) -> str:
    """Hash the complete reviewed item without persisting its raw content."""

    return _canonical_sha256(item.model_dump(mode="json"))


def aggregate_repeated_query_records(
    records: Sequence[dict[str, Any]],
    *,
    repeat_count: int,
) -> dict[str, Any]:
    """Aggregate complete runs independently and reject retrieval-boundary drift."""

    if repeat_count <= 0:
        raise ValueError("Repeat count must be positive.")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_keys: set[tuple[str, int]] = set()
    for record in records:
        repeat_index = record.get("repeat_index")
        record_id = record.get("id")
        if type(repeat_index) is not int or not 1 <= repeat_index <= repeat_count:
            raise ValueError("Repeated evaluation record has an invalid repeat_index.")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Repeated evaluation record has no stable item ID.")
        key = record_id, repeat_index
        if key in seen_keys:
            raise ValueError("Repeated evaluation contains a duplicate item/repeat key.")
        seen_keys.add(key)
        grouped[repeat_index].append(record)
    if set(grouped) != set(range(1, repeat_count + 1)):
        raise ValueError("Repeated evaluation is missing one or more complete runs.")

    expected_ids = {record["id"] for record in grouped[1]}
    if not expected_ids:
        raise ValueError("Repeated evaluation requires at least one item.")
    fingerprints: dict[str, str] = {}
    evaluation_contracts: dict[str, str] = {}
    for repeat_index in range(1, repeat_count + 1):
        rows = grouped[repeat_index]
        if {record["id"] for record in rows} != expected_ids:
            raise ValueError("Repeated evaluation item IDs differ between runs.")
        for record in rows:
            contract_fingerprint = record.get("evaluation_contract_fingerprint")
            if (
                not isinstance(contract_fingerprint, str)
                or not re.fullmatch(r"[0-9a-f]{64}", contract_fingerprint)
            ):
                raise ValueError(
                    "Repeated evaluation record has an invalid evaluation contract fingerprint."
                )
            previous_contract = evaluation_contracts.setdefault(
                record["id"], contract_fingerprint
            )
            if previous_contract != contract_fingerprint:
                raise ValueError(
                    "Repeated-run evaluation contract fingerprint drift detected."
                )
            fingerprint = retrieval_evidence_fingerprint(record)
            previous = fingerprints.setdefault(record["id"], fingerprint)
            if previous != fingerprint:
                raise ValueError(
                    "Repeated evaluation retrieval/evidence fingerprint drift detected."
                )

    per_repeat = []
    for repeat_index in range(1, repeat_count + 1):
        overall = aggregate_query_records(grouped[repeat_index])
        per_repeat.append(
            {
                "repeat_index": repeat_index,
                "overall": overall,
                "quality_gates": evaluate_quality_gates(overall),
            }
        )
    metric_paths = {
        "retrieval_candidate_recall": ("retrieval", "candidate_recall"),
        "retrieval_hit_rate_at_5": ("retrieval", "hit_rate_at_5"),
        "retrieval_mrr_at_5": ("retrieval", "mrr_at_5"),
        "deterministic_fact_accuracy_when_answered": (
            "answer_quality",
            "deterministic_fact_accuracy_when_answered",
        ),
        "complete_answer_coverage_all_answerable": (
            "claim_completeness",
            "complete_answer_coverage_all_answerable",
        ),
        "fact_coverage_all_answerable": (
            "claim_completeness",
            "fact_coverage_all_answerable",
        ),
        "abstention_precision": ("abstention", "precision"),
        "abstention_recall": ("abstention", "recall"),
        "total_latency_p95_ms": ("latency_ms", "total", "p95"),
    }
    return {
        "repeat_count": repeat_count,
        "ordering": "repeat-major-dataset-order",
        "per_repeat": per_repeat,
        "stability": {
            name: _repeat_summary(
                [float(_nested_value(row["overall"], path)) for row in per_repeat]
            )
            for name, path in metric_paths.items()
        },
    }


def evaluate_quality_gates(metrics: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the documented release gates without tuning their thresholds."""

    abstention = metrics["abstention"]
    citations = metrics["citations"]
    answer_quality = metrics["answer_quality"]
    fact_evaluator_mode = answer_quality["fact_evaluator_mode"]
    fact_accuracy_key = (
        "final_fact_accuracy_when_answered"
        if fact_evaluator_mode == HYBRID_FACT_EVALUATOR
        else "deterministic_fact_accuracy_when_answered"
    )
    precision = float(abstention["precision"])
    recall = float(abstention["recall"])
    gates = {
        "complete_candidate_requirement_rate": _gate(
            float(metrics["retrieval"]["authoritative_candidate_complete_rate"]),
            11 / 12,
            comparison="at_least",
        ),
        "complete_evidence_requirement_hit_at_5": _gate(
            float(metrics["retrieval"]["authoritative_evidence_complete_rate"]),
            0.85,
            comparison="at_least",
        ),
        "valid_citation_ids": _gate(
            float(citations["referential_valid_rate_when_answered"]), 1.0, comparison="equal"
        ),
        "unsupported_citation_ids": _gate(
            float(citations["unsupported_citation_count"]), 0.0, comparison="at_most"
        ),
        "unreviewed_citation_provenance": _gate(
            float(citations["unreviewed_citation_provenance_count"]),
            0.0,
            comparison="at_most",
        ),
        "complete_citation_requirement_coverage": _gate(
            float(citations["authoritative_requirement_complete_rate_when_answered"]),
            1.0,
            comparison="equal",
        ),
        "fact_accuracy": _gate(
            float(answer_quality[fact_accuracy_key]),
            0.85,
            comparison="at_least",
        ),
        "false_answers": _gate(
            float(abstention["false_answer_count"]), 0.0, comparison="at_most"
        ),
        "abstention_precision": _gate(precision, 0.90, comparison="at_least"),
        "abstention_recall": _gate(recall, 0.90, comparison="at_least"),
    }
    return {
        "overall_pass": all(gate["passed"] for gate in gates.values()),
        "gates": gates,
        "note": (
            f"Headline answer accuracy uses {fact_evaluator_mode} with {FACT_EVALUATOR_ID}. "
            "Strict contiguous aliases and "
            "token coverage remain diagnostics. Evidence phrases validate qrels only. "
            "Strict qrel ranks and document provenance remain diagnostics. Release gates use "
            "reviewed authoritative evidence requirements: AND across requirements and OR "
            "within each requirement."
        ),
    }


def _retrieval_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    final_ranks = [row["direct_evidence_rank"] for row in rows]
    candidate_ranks = [row["candidate_direct_evidence_rank"] for row in rows]
    exact_equivalent_final_ranks = [
        row.get("exact_equivalent_evidence_rank", row["direct_evidence_rank"])
        for row in rows
    ]
    exact_equivalent_candidate_ranks = [
        row.get(
            "candidate_exact_equivalent_evidence_rank",
            row["candidate_direct_evidence_rank"],
        )
        for row in rows
    ]
    candidate_authoritative = [
        _requirement_coverage(row, "candidate_authoritative_requirement_coverage")
        for row in rows
    ]
    evidence_authoritative = [
        _requirement_coverage(row, "evidence_authoritative_requirement_coverage")
        for row in rows
    ]
    return {
        "candidate_recall": _hit_rate(candidate_ranks, cutoff=None),
        "authoritative_candidate_complete_rate": _complete_rate(candidate_authoritative),
        "authoritative_evidence_complete_rate": _complete_rate(evidence_authoritative),
        "authoritative_candidate_requirement_coverage": _coverage_rate(candidate_authoritative),
        "authoritative_evidence_requirement_coverage": _coverage_rate(evidence_authoritative),
        "hit_rate_at_1": _hit_rate(final_ranks, cutoff=1),
        "hit_rate_at_3": _hit_rate(final_ranks, cutoff=3),
        "hit_rate_at_5": _hit_rate(final_ranks, cutoff=5),
        "hit_rate_at_20": _hit_rate(final_ranks, cutoff=20),
        "mrr_at_5": _mrr(final_ranks, cutoff=5),
        "mrr_at_20": _mrr(final_ranks, cutoff=20),
        "exact_equivalent_candidate_recall": _hit_rate(
            exact_equivalent_candidate_ranks, cutoff=None
        ),
        "exact_equivalent_hit_rate_at_1": _hit_rate(
            exact_equivalent_final_ranks, cutoff=1
        ),
        "exact_equivalent_hit_rate_at_3": _hit_rate(
            exact_equivalent_final_ranks, cutoff=3
        ),
        "exact_equivalent_hit_rate_at_5": _hit_rate(
            exact_equivalent_final_ranks, cutoff=5
        ),
        "exact_equivalent_hit_rate_at_20": _hit_rate(
            exact_equivalent_final_ranks, cutoff=20
        ),
        "exact_equivalent_mrr_at_5": _mrr(exact_equivalent_final_ranks, cutoff=5),
        "exact_equivalent_mrr_at_20": _mrr(exact_equivalent_final_ranks, cutoff=20),
        "candidate_count_maximum": float(max(int(row["candidate_count"]) for row in rows)),
    }


def _requirement_coverage(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    payload = row.get(key)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Evaluation row lacks authoritative coverage: {key}")
    count = payload.get("requirement_count")
    covered = payload.get("covered_requirement_count")
    complete = payload.get("complete")
    if type(count) is not int or type(covered) is not int or type(complete) is not bool:
        raise ValueError(f"Evaluation row has malformed authoritative coverage: {key}")
    return payload


def _complete_rate(coverages: Sequence[Mapping[str, Any]]) -> float:
    return sum(bool(coverage["complete"]) for coverage in coverages) / len(coverages)


def _coverage_rate(coverages: Sequence[Mapping[str, Any]]) -> float:
    requirement_count = sum(int(coverage["requirement_count"]) for coverage in coverages)
    covered_count = sum(int(coverage["covered_requirement_count"]) for coverage in coverages)
    return covered_count / requirement_count if requirement_count else 0.0


def _unreviewed_citation_count(payload: object) -> int:
    if not isinstance(payload, list):
        return 0
    count = 0
    for citation in payload:
        if not isinstance(citation, Mapping):
            count += 1
            continue
        matches = citation.get("matches")
        if not isinstance(matches, list) or not matches:
            count += 1
    return count


def _answer_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answered = [row for row in rows if not row["abstained"]]
    modes = {str(row["fact_evaluator_mode"]) for row in rows}
    if len(modes) != 1:
        raise ValueError("Evaluation records contain mixed fact evaluator modes.")
    fact_evaluator_mode = modes.pop()
    deterministic_scores = [
        row["deterministic_fact_match"]
        for row in answered
        if row["deterministic_fact_match"] is not None
    ]
    final_scores = [
        row["answer_fact_match"]
        for row in answered
        if row["answer_fact_match"] is not None
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
            sum(bool(score) for score in final_scores) / len(final_scores)
            if final_scores
            else 0.0
        ),
        "final_fact_accuracy_when_answered": (
            sum(bool(score) for score in final_scores) / len(final_scores)
            if final_scores
            else 0.0
        ),
        "strict_phrase_accuracy_when_answered": (
            sum(bool(score) for score in strict_scores) / len(strict_scores)
            if strict_scores
            else 0.0
        ),
        "fact_evaluator_id": FACT_EVALUATOR_ID,
        "fact_evaluator_mode": fact_evaluator_mode,
        "answered_count": len(answered),
        "fact_count": len(fact_results),
        "deterministically_matched_fact_count": sum(
            bool(result["deterministic_matched"]) for result in fact_results
        ),
        "strictly_matched_fact_count": sum(
            bool(result["strict_phrase_matched"]) for result in fact_results
        ),
        "matched_fact_count": sum(
            bool(result["final_matched"]) for result in fact_results
        ),
        "fact_match_rate": (
            sum(bool(result["final_matched"]) for result in fact_results)
            / len(fact_results)
            if fact_results
            else 0.0
        ),
        "fact_judge": _fact_judge_metrics(answered),
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


def _fact_judge_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fact_results = [
        result for row in rows for result in row.get("answer_fact_results", [])
    ]
    judge_payloads = [row["fact_judge"] for row in rows]
    verdicts = Counter(
        str(result["judge_verdict"])
        for result in fact_results
        if result.get("judge_verdict") is not None
    )
    latencies = [
        float(payload["latency_ms"])
        for payload in judge_payloads
        if int(payload["invocation_count"]) > 0
    ]
    return {
        "invocation_count": sum(
            int(payload["invocation_count"]) for payload in judge_payloads
        ),
        "fallback_item_count": sum(
            int(payload["invocation_count"]) > 0 for payload in judge_payloads
        ),
        "fallback_fact_count": sum(
            int(payload["fallback_fact_count"]) for payload in judge_payloads
        ),
        "present_count": verdicts["present"],
        "absent_count": verdicts["absent"],
        "contradicted_count": verdicts["contradicted"],
        "latency_ms": _summary(latencies) if latencies else None,
    }


def _claim_completeness_payload(
    item: EvaluationItem,
    *,
    fact_results: Sequence[Mapping[str, Any]],
    abstained: bool,
    direct_evidence_rank: int | None,
    exact_equivalent_evidence_rank: int | None,
    evidence_candidates: Sequence[RetrievalCandidate],
    truncated_source_ids: Sequence[str],
    exact_relevant_chunk_ids: set[str],
) -> dict[str, Any]:
    expected_by_type = Counter(fact.type for fact in item.expected_answer_facts)
    matched_by_type = Counter(
        str(result["type"])
        for result in fact_results
        if bool(result["final_matched"])
    )
    expected_count = sum(expected_by_type.values())
    matched_payload = {
        fact_type: 0 if abstained else matched_by_type[fact_type]
        for fact_type in sorted(expected_by_type)
    }
    matched_count = sum(matched_payload.values())
    return {
        "evaluator_id": CLAIM_COMPLETENESS_EVALUATOR_ID,
        "status": "abstained" if abstained else "scored",
        "expected_fact_count": expected_count,
        "matched_fact_count": matched_count,
        "missing_fact_count": expected_count - matched_count,
        "coverage": matched_count / expected_count if expected_count else 0.0,
        "complete": matched_count == expected_count,
        "expected_by_fact_type": dict(sorted(expected_by_type.items())),
        "matched_by_fact_type": matched_payload,
        "evidence_boundary": {
            "direct_qrel_in_evidence": direct_evidence_rank is not None,
            "exact_equivalent_only": (
                direct_evidence_rank is None
                and exact_equivalent_evidence_rank is not None
            ),
            "qrel_source_truncated": _qrel_source_was_truncated(
                evidence_candidates,
                truncated_source_ids=truncated_source_ids,
                exact_relevant_chunk_ids=exact_relevant_chunk_ids,
            ),
        },
    }


def _not_applicable_claim_completeness() -> dict[str, Any]:
    return {
        "evaluator_id": CLAIM_COMPLETENESS_EVALUATOR_ID,
        "status": "not_applicable",
        "expected_fact_count": 0,
        "matched_fact_count": None,
        "missing_fact_count": None,
        "coverage": None,
        "complete": None,
        "expected_by_fact_type": {},
        "matched_by_fact_type": {},
        "evidence_boundary": None,
    }


def _not_applicable_authoritative_coverage() -> dict[str, Any]:
    """Represent an unanswerable item without inventing evidence requirements."""

    return {
        "contract_mode": "not_applicable",
        "requirement_count": 0,
        "covered_requirement_count": None,
        "complete": None,
        "requirements": [],
    }


def _qrel_source_was_truncated(
    evidence_candidates: Sequence[RetrievalCandidate],
    *,
    truncated_source_ids: Sequence[str],
    exact_relevant_chunk_ids: set[str],
) -> bool:
    for source_id in truncated_source_ids:
        match = re.fullmatch(r"S([1-9]\d*)", source_id)
        if match is None:
            continue
        index = int(match.group(1)) - 1
        if not 0 <= index < len(evidence_candidates):
            continue
        candidate = evidence_candidates[index]
        candidate_ids = {candidate.chunk_id}
        equivalent_ids = candidate.metadata.get("equivalent_chunk_ids", ())
        if isinstance(equivalent_ids, (list, tuple, set, frozenset)):
            candidate_ids.update(
                chunk_id for chunk_id in equivalent_ids if isinstance(chunk_id, str)
            )
        if candidate_ids.intersection(exact_relevant_chunk_ids):
            return True
    return False


def _claim_completeness_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answered = [row for row in rows if not row["abstained"]]
    complete_answer_count = sum(
        bool(row["claim_completeness"]["complete"]) for row in answered
    )
    expected_by_type: Counter[str] = Counter()
    matched_by_type: Counter[str] = Counter()
    for row in rows:
        payload = row["claim_completeness"]
        expected_by_type.update(payload["expected_by_fact_type"])
        matched_by_type.update(payload["matched_by_fact_type"])
    expected_count = sum(expected_by_type.values())
    matched_count = sum(matched_by_type.values())
    return {
        "evaluator_id": CLAIM_COMPLETENESS_EVALUATOR_ID,
        "answerable_count": len(rows),
        "answered_count": len(answered),
        "complete_answer_count": complete_answer_count,
        "complete_answer_rate_when_answered": (
            complete_answer_count / len(answered) if answered else 0.0
        ),
        "complete_answer_coverage_all_answerable": (
            complete_answer_count / len(rows) if rows else 0.0
        ),
        "expected_fact_count": expected_count,
        "matched_fact_count": matched_count,
        "fact_coverage_all_answerable": (
            matched_count / expected_count if expected_count else 0.0
        ),
        "per_fact_type": {
            fact_type: {
                "expected_count": expected_by_type[fact_type],
                "matched_count": matched_by_type[fact_type],
                "coverage": matched_by_type[fact_type] / expected_by_type[fact_type],
            }
            for fact_type in sorted(expected_by_type)
        },
        "note": (
            "Primary coverage denominators include every answerable item; an abstention "
            "therefore contributes its reviewed facts as unmatched."
        ),
    }


def _claim_completeness_by_evidence_boundary(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        boundary = row["claim_completeness"]["evidence_boundary"]
        if boundary["direct_qrel_in_evidence"]:
            label = (
                "direct_qrel_truncated"
                if boundary["qrel_source_truncated"]
                else "direct_qrel_complete"
            )
        elif boundary["exact_equivalent_only"]:
            label = "exact_equivalent_only"
        else:
            label = "no_qrel_in_evidence"
        groups[label].append(row)
    return {
        label: _claim_completeness_metrics(group)
        for label, group in sorted(groups.items())
    }


def _citation_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    answered = [row for row in rows if not row["abstained"]]
    if not answered:
        return {
            "direct_evidence_rate_when_answered": 0.0,
            "exact_equivalent_evidence_rate_when_answered": 0.0,
            "document_correct_rate_when_answered": 0.0,
            "exact_equivalent_document_correct_rate_when_answered": 0.0,
            "referential_valid_rate_when_answered": 0.0,
            "unsupported_citation_count": 0,
            "unreviewed_citation_provenance_count": 0,
            "authoritative_requirement_complete_rate_when_answered": 0.0,
            "wrong_document_citation_count": 0,
            "wrong_document_citation_rate_when_answered": 0.0,
            "wrong_document_citation_count_excluding_exact_equivalents": 0,
            "wrong_document_citation_rate_when_answered_excluding_exact_equivalents": 0.0,
            "answered_count": 0,
        }
    citation_authoritative = [
        _requirement_coverage(row, "citation_authoritative_requirement_coverage")
        for row in answered
    ]
    return {
        "direct_evidence_rate_when_answered": sum(
            bool(row["citation_direct_evidence"]) for row in answered
        )
        / len(answered),
        "exact_equivalent_evidence_rate_when_answered": sum(
            bool(
                row.get(
                    "citation_exact_equivalent_evidence",
                    row["citation_direct_evidence"],
                )
            )
            for row in answered
        )
        / len(answered),
        "document_correct_rate_when_answered": sum(
            bool(row["citation_document_correct"]) for row in answered
        )
        / len(answered),
        "exact_equivalent_document_correct_rate_when_answered": sum(
            bool(
                row.get(
                    "citation_exact_equivalent_document_correct",
                    row["citation_document_correct"],
                )
            )
            for row in answered
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
        "unreviewed_citation_provenance_count": sum(
            _unreviewed_citation_count(row.get("citation_authoritative_provenance"))
            for row in answered
        ),
        "authoritative_requirement_complete_rate_when_answered": _complete_rate(
            citation_authoritative
        ),
        "wrong_document_citation_count": sum(
            bool(row.get("unexpected_citation_document_ids")) for row in answered
        ),
        "wrong_document_citation_rate_when_answered": sum(
            bool(row.get("unexpected_citation_document_ids")) for row in answered
        )
        / len(answered),
        "wrong_document_citation_count_excluding_exact_equivalents": sum(
            bool(
                row.get(
                    "unexpected_citation_document_ids_excluding_exact_equivalents",
                    row.get("unexpected_citation_document_ids"),
                )
            )
            for row in answered
        ),
        "wrong_document_citation_rate_when_answered_excluding_exact_equivalents": sum(
            bool(
                row.get(
                    "unexpected_citation_document_ids_excluding_exact_equivalents",
                    row.get("unexpected_citation_document_ids"),
                )
            )
            for row in answered
        )
        / len(answered),
        "answered_count": len(answered),
    }


def _document_contamination_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    wrong_top1 = sum(bool(row["wrong_document_retrieval_at_1"]) for row in rows)
    wrong_top5_candidates = sum(int(row["wrong_document_candidate_count_at_5"]) for row in rows)
    wrong_top1_excluding_exact_equivalents = sum(
        bool(
            row.get(
                "wrong_document_retrieval_at_1_excluding_exact_equivalents",
                row["wrong_document_retrieval_at_1"],
            )
        )
        for row in rows
    )
    wrong_top5_excluding_exact_equivalents = sum(
        int(
            row.get(
                "wrong_document_candidate_count_at_5_excluding_exact_equivalents",
                row["wrong_document_candidate_count_at_5"],
            )
        )
        for row in rows
    )
    top5_candidates = sum(min(int(row["final_candidate_count"]), 5) for row in rows)
    return {
        "wrong_document_retrieval_at_1_count": wrong_top1,
        "wrong_document_retrieval_at_1_rate": wrong_top1 / len(rows),
        "wrong_document_candidate_count_at_5": wrong_top5_candidates,
        "wrong_document_candidate_rate_at_5": (
            wrong_top5_candidates / top5_candidates if top5_candidates else 0.0
        ),
        "wrong_document_retrieval_at_1_count_excluding_exact_equivalents": (
            wrong_top1_excluding_exact_equivalents
        ),
        "wrong_document_retrieval_at_1_rate_excluding_exact_equivalents": (
            wrong_top1_excluding_exact_equivalents / len(rows)
        ),
        "wrong_document_candidate_count_at_5_excluding_exact_equivalents": (
            wrong_top5_excluding_exact_equivalents
        ),
        "wrong_document_candidate_rate_at_5_excluding_exact_equivalents": (
            wrong_top5_excluding_exact_equivalents / top5_candidates
            if top5_candidates
            else 0.0
        ),
    }


def _abstention_metrics(
    answerable: Sequence[dict[str, Any]], unanswerable: Sequence[dict[str, Any]]
) -> dict[str, int | float]:
    true_positive = sum(row["abstained"] for row in unanswerable)
    false_negative = len(unanswerable) - true_positive
    false_positive = sum(row["abstained"] for row in answerable)
    true_negative = len(answerable) - false_positive
    false_answer_count = sum(not row["abstained"] for row in unanswerable)
    return {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_answer_count": false_answer_count,
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
        "claim_completeness": (
            _claim_completeness_metrics(answerable) if answerable else None
        ),
        "citations": _citation_metrics(answerable) if answered else None,
        "abstention_rate": sum(row["abstained"] for row in rows) / len(rows),
    }


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


def _candidate_matches_expected_document_or_exact_equivalent(
    candidate: RetrievalCandidate | None,
    *,
    citation_document_id: str,
    expected_document_ids: set[str],
    exact_relevant_chunk_ids: set[str],
) -> bool:
    if citation_document_id in expected_document_ids:
        return True
    if candidate is None:
        return False
    candidate_chunk_ids = {candidate.chunk_id}
    equivalent_chunk_ids = candidate.metadata.get("equivalent_chunk_ids", ())
    if isinstance(equivalent_chunk_ids, (list, tuple, set, frozenset)):
        candidate_chunk_ids.update(
            chunk_id for chunk_id in equivalent_chunk_ids if isinstance(chunk_id, str)
        )
    if candidate_chunk_ids.intersection(exact_relevant_chunk_ids):
        return True
    return False


def _failure_class(
    candidate_rank: int | None,
    ranked_rank: int | None,
    evidence_rank: int | None,
) -> str:
    if candidate_rank is None:
        return "candidate_miss"
    if ranked_rank is None or ranked_rank > 20:
        return "reranker_miss_top20"
    if evidence_rank is None or evidence_rank > 5:
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


def _repeat_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": median(values),
        "maximum": max(values),
    }


def _canonical_sha256(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _nested_value(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Repeated evaluation metric path is missing: {'.'.join(path)}")
        value = value[key]
    return value


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

"""Offline scoring tests for the ATV320 end-to-end evaluator."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.application.query_service import QueryExecution, QueryTimings
from app.contracts.query import Citation, QueryResponse
from app.domain.generation import TokenUsage
from app.domain.retrieval import RetrievalCandidate, RetrievalTrace
from evaluation import dataset as evaluation_dataset
from evaluation.dataset import EvaluationItem, ExpectedAnswerFact
from evaluation.e2e import (
    aggregate_query_records,
    aggregate_repeated_query_records,
    evaluate_quality_gates,
    retrieval_evidence_fingerprint,
    score_query_execution,
)
from evaluation.facts import (
    CLAIM_COMPLETENESS_EVALUATOR_ID,
    FactJudgeDecision,
    FactJudgeResult,
)
from evaluation.generation_experiments import (
    BASELINE_GENERATION_VARIANT,
    CLAIM_COVERAGE_GENERATION_VARIANT,
    generation_experiment_for,
    generation_experiment_identity,
)
from scripts.evaluation import evaluate_e2e


def _candidate(chunk_id: str, *, document_id: str = "installation") -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        filename="manual.pdf",
        text="The documented range is -599 to +599 Hz.",
        page_numbers=[7],
        headings=["Parameters"],
        content_type="text",
        score=1.0,
    )


def _item(*, answerable: bool = True, phrase_mode: str = "all") -> EvaluationItem:
    if not answerable:
        return EvaluationItem(
            id="unsupported",
            question="Unknown?",
            language="en",
            answerable=False,
            scenario="en_to_en",
            question_type="unanswerable",
            expected_document_ids=[],
            relevant_chunk_ids=[],
            expected_pages=[],
            expected_phrases=[],
            citation_required=False,
            unanswerable_reason="Verified absent.",
            review_status="approved",
        )
    return EvaluationItem(
        id="answerable",
        question="Range?",
        language="en",
        answerable=True,
        scenario="en_to_en",
        question_type="parameter_code",
        expected_document_ids=["installation"],
        relevant_chunk_ids=["qrel"],
        expected_pages=[7],
        expected_phrases=["-599 to +599 Hz"],
        phrase_match_mode=phrase_mode,
        expected_answer_facts=[
            {
                "id": "frequency-range",
                "aliases": ["-599 to +599 Hz", "từ -599 đến +599 Hz"],
            }
        ],
        citation_required=True,
        review_status="approved",
    )


def _execution(
    *,
    final: list[RetrievalCandidate],
    pool: list[RetrievalCandidate],
    abstained: bool = False,
    evidence: list[RetrievalCandidate] | None = None,
    trace: RetrievalTrace | None = None,
    evidence_context_chars: int = 0,
    evidence_truncated_source_ids: tuple[str, ...] = (),
    answer: str | None = None,
) -> QueryExecution:
    actual_evidence = evidence if evidence is not None else final
    citations = []
    if not abstained:
        candidate = actual_evidence[0]
        citations = [
            Citation(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                filename=candidate.filename,
                page_numbers=candidate.page_numbers,
                headings=candidate.headings,
                excerpt="excerpt",
            )
        ]
    return QueryExecution(
        response=QueryResponse(
            answer=(
                answer or "The documented range is -599 to +599 Hz."
                if not abstained
                else "insufficient"
            ),
            abstained=abstained,
            abstention_reason="llm_insufficient_evidence" if abstained else None,
            citations=citations,
        ),
        timings=QueryTimings(1, 2, 3, 4, 5, 15),
        usage=TokenUsage(10, 2, 1),
        candidates=tuple(final),
        candidate_pool=tuple(pool),
        evidence_candidates=tuple(actual_evidence),
        retrieval_trace=trace,
        evidence_context_chars=evidence_context_chars,
        evidence_truncated_source_ids=evidence_truncated_source_ids,
    )


def test_scores_direct_evidence_and_candidate_miss_without_page_fallback() -> None:
    item = _item()
    pool = [_candidate("same-page"), _candidate("qrel")]
    record = score_query_execution(item, _execution(final=[_candidate("same-page")], pool=pool))
    assert record["candidate_direct_evidence_rank"] == 2
    assert record["direct_evidence_rank"] is None
    assert record["failure_class"] == "reranker_miss_top20"
    assert record["citation_direct_evidence"] is False
    assert record["citation_ids_in_final_candidates"] is True
    assert record["qrel_candidate_diagnostics"] == [
        {
            "chunk_id": "qrel",
            "matched_relevant_chunk_ids": ["qrel"],
            "ordinal_rank": 2,
            "dense_rank": None,
            "sparse_rank": None,
            "rrf_rank": None,
            "rerank_rank": None,
            "document_id": "installation",
            "page_numbers": [7],
        }
    ]
    assert record["qrel_final_diagnostics"] == []


def test_scoring_distinguishes_full_ranking_from_actual_generation_evidence() -> None:
    qrel = _candidate("qrel")
    record = score_query_execution(
        _item(),
        _execution(
            final=[_candidate("other"), qrel],
            pool=[_candidate("other"), qrel],
            evidence=[qrel],
        ),
    )
    assert record["ranked_direct_evidence_rank"] == 2
    assert record["direct_evidence_rank"] == 1
    assert record["evidence_candidate_ids"] == ["qrel"]
    assert record["citation_ids_in_final_candidates"] is True


def test_full_rank_six_excluded_from_actual_top_five_is_a_top5_miss() -> None:
    leading = [_candidate(f"other-{index}") for index in range(1, 6)]
    qrel = _candidate("qrel")
    record = score_query_execution(
        _item(),
        _execution(
            final=[*leading, qrel],
            pool=[*leading, qrel],
            evidence=leading,
        ),
    )
    assert record["ranked_direct_evidence_rank"] == 6
    assert record["direct_evidence_rank"] is None
    assert record["failure_class"] == "reranker_miss_top5"


def test_aggregate_reports_retrieval_citations_abstention_and_latency() -> None:
    answerable = score_query_execution(
        _item(), _execution(final=[_candidate("qrel")], pool=[_candidate("qrel")])
    )
    unsupported = score_query_execution(
        _item(answerable=False),
        _execution(final=[_candidate("other")], pool=[_candidate("other")], abstained=True),
    )
    metrics = aggregate_query_records([answerable, unsupported])
    assert metrics["retrieval"]["hit_rate_at_1"] == 1.0
    assert metrics["answer_quality"]["answer_fact_accuracy_when_answered"] == 1.0
    assert metrics["answer_quality"]["deterministic_fact_accuracy_when_answered"] == 1.0
    assert metrics["answer_quality"]["strict_phrase_accuracy_when_answered"] == 1.0
    assert metrics["answer_quality"]["fact_count"] == 1
    assert metrics["answer_quality"]["matched_fact_count"] == 1
    assert (
        metrics["answer_quality"]["all_alias_tokens_covered_accuracy_when_answered"]
        == 1.0
    )
    assert metrics["citations"]["direct_evidence_rate_when_answered"] == 1.0
    assert metrics["abstention"]["true_positive"] == 1
    assert metrics["abstention"]["precision"] == 1.0
    assert metrics["abstention"]["recall"] == 1.0
    assert metrics["latency_ms"]["total"]["p95"] == 15
    assert metrics["per_language"]["en"]["query_count"] == 2

    gates = evaluate_quality_gates(metrics)
    assert gates["overall_pass"] is True
    assert gates["gates"]["unsupported_citation_ids"]["actual"] == 0
    assert gates["gates"]["wrong_document_citations"]["actual"] == 0


def test_answerable_abstention_does_not_claim_answer_or_citation_success() -> None:
    record = score_query_execution(
        _item(), _execution(final=[_candidate("qrel")], pool=[_candidate("qrel")], abstained=True)
    )
    assert record["answer_fact_match"] is None
    assert record["answer_fact_results"] == []
    assert record["missing_answer_fact_ids"] == []
    assert record["citation_direct_evidence"] is False


def test_claim_completeness_scores_answered_abstained_and_unanswerable_records() -> None:
    answered = score_query_execution(
        _item(),
        _execution(
            final=[_candidate("qrel")],
            pool=[_candidate("qrel")],
            evidence_context_chars=100,
            evidence_truncated_source_ids=("S1",),
        ),
    )
    abstained = score_query_execution(
        _item(),
        _execution(
            final=[_candidate("qrel")],
            pool=[_candidate("qrel")],
            abstained=True,
        ),
    )
    unsupported = score_query_execution(
        _item(answerable=False),
        _execution(
            final=[_candidate("other")],
            pool=[_candidate("other")],
            abstained=True,
        ),
    )

    assert answered["claim_completeness"] == {
        "evaluator_id": CLAIM_COMPLETENESS_EVALUATOR_ID,
        "status": "scored",
        "expected_fact_count": 1,
        "matched_fact_count": 1,
        "missing_fact_count": 0,
        "coverage": 1.0,
        "complete": True,
        "expected_by_fact_type": {"text": 1},
        "matched_by_fact_type": {"text": 1},
        "evidence_boundary": {
            "direct_qrel_in_evidence": True,
            "exact_equivalent_only": False,
            "qrel_source_truncated": True,
        },
    }
    assert abstained["claim_completeness"]["status"] == "abstained"
    assert abstained["claim_completeness"]["matched_fact_count"] == 0
    assert abstained["claim_completeness"]["missing_fact_count"] == 1
    assert abstained["claim_completeness"]["coverage"] == 0.0
    assert abstained["claim_completeness"]["complete"] is False
    assert abstained["claim_completeness"]["matched_by_fact_type"] == {"text": 0}
    assert unsupported["claim_completeness"] == {
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


def test_claim_completeness_uses_all_answerable_items_as_the_primary_denominator() -> None:
    complete = score_query_execution(
        _item(), _execution(final=[_candidate("qrel")], pool=[_candidate("qrel")])
    )
    abstained = score_query_execution(
        _item().model_copy(update={"id": "answerable-abstained"}),
        _execution(
            final=[_candidate("qrel")],
            pool=[_candidate("qrel")],
            abstained=True,
        ),
    )
    unsupported = score_query_execution(
        _item(answerable=False),
        _execution(
            final=[_candidate("other")],
            pool=[_candidate("other")],
            abstained=True,
        ),
    )

    metrics = aggregate_query_records([complete, abstained, unsupported])
    completeness = metrics["claim_completeness"]

    assert completeness["evaluator_id"] == CLAIM_COMPLETENESS_EVALUATOR_ID
    assert completeness["answerable_count"] == 2
    assert completeness["answered_count"] == 1
    assert completeness["complete_answer_rate_when_answered"] == 1.0
    assert completeness["complete_answer_coverage_all_answerable"] == 0.5
    assert completeness["fact_coverage_all_answerable"] == 0.5
    assert completeness["per_fact_type"]["text"] == {
        "expected_count": 2,
        "matched_count": 1,
        "coverage": 0.5,
    }
    assert metrics["per_language"]["en"]["claim_completeness"][
        "complete_answer_coverage_all_answerable"
    ] == 0.5


def test_claim_completeness_reports_partial_facts_and_evidence_boundaries() -> None:
    two_fact_item = _item().model_copy(
        update={
            "expected_answer_facts": [
                *_item().expected_answer_facts,
                ExpectedAnswerFact(
                    id="parameter-code",
                    aliases=["ABC"],
                    type="identifier",
                    acceptable_values=["ABC"],
                ),
            ]
        }
    )
    direct = score_query_execution(
        two_fact_item,
        _execution(final=[_candidate("qrel")], pool=[_candidate("qrel")]),
    )
    completeness = direct["claim_completeness"]
    assert completeness["expected_fact_count"] == 2
    assert completeness["matched_fact_count"] == 1
    assert completeness["missing_fact_count"] == 1
    assert completeness["coverage"] == 0.5
    assert completeness["complete"] is False

    exact_item = _item().model_copy(update={"id": "exact-equivalent"})
    exact = score_query_execution(
        exact_item,
        _execution(final=[_candidate("copy")], pool=[_candidate("copy")]),
        exact_content_equivalence={
            "qrel": ("qrel", "copy"),
            "copy": ("qrel", "copy"),
        },
    )
    missed = score_query_execution(
        _item().model_copy(update={"id": "no-qrel"}),
        _execution(final=[_candidate("other")], pool=[_candidate("other")]),
    )
    unsupported = score_query_execution(
        _item(answerable=False),
        _execution(
            final=[_candidate("other")],
            pool=[_candidate("other")],
            abstained=True,
        ),
    )

    by_boundary = aggregate_query_records(
        [direct, exact, missed, unsupported]
    )["claim_completeness_by_evidence_boundary"]
    assert set(by_boundary) == {
        "direct_qrel_complete",
        "exact_equivalent_only",
        "no_qrel_in_evidence",
    }
    assert by_boundary["direct_qrel_complete"]["answerable_count"] == 1
    assert by_boundary["exact_equivalent_only"]["answerable_count"] == 1
    assert by_boundary["no_qrel_in_evidence"]["answerable_count"] == 1


def test_repeat_aggregation_is_run_scoped_and_rejects_retrieval_drift() -> None:
    complete = score_query_execution(
        _item(), _execution(final=[_candidate("qrel")], pool=[_candidate("qrel")])
    )
    missed = score_query_execution(
        _item(),
        _execution(
            final=[_candidate("qrel")],
            pool=[_candidate("qrel")],
            answer="unsupported answer",
        ),
    )
    unsupported = score_query_execution(
        _item(answerable=False),
        _execution(
            final=[_candidate("other")],
            pool=[_candidate("other")],
            abstained=True,
        ),
    )
    records = []
    for repeat_index, answerable in ((1, complete), (2, missed), (3, complete)):
        for record in (answerable, unsupported):
            copied = deepcopy(record)
            copied["repeat_index"] = repeat_index
            records.append(copied)

    repeated = aggregate_repeated_query_records(records, repeat_count=3)

    assert repeated["repeat_count"] == 3
    assert repeated["ordering"] == "repeat-major-dataset-order"
    assert [row["repeat_index"] for row in repeated["per_repeat"]] == [1, 2, 3]
    assert repeated["stability"]["deterministic_fact_accuracy_when_answered"] == {
        "minimum": 0.0,
        "median": 1.0,
        "maximum": 1.0,
    }
    assert len(retrieval_evidence_fingerprint(complete)) == 64
    assert retrieval_evidence_fingerprint(complete) == retrieval_evidence_fingerprint(missed)

    run_aggregate = evaluate_e2e._aggregate_run_records(
        records,
        repeat_count=3,
        complete_evaluation=True,
    )
    assert run_aggregate["overall"] is None
    assert run_aggregate["repeated_runs"] == repeated
    assert run_aggregate["quality_gates"]["overall_pass"] is False
    assert run_aggregate["quality_gates"]["selection_policy"] == (
        "all_repeats_must_pass"
    )
    assert len(run_aggregate["quality_gates"]["per_repeat"]) == 3

    single_run_records = []
    for record in (complete, unsupported):
        copied = deepcopy(record)
        copied["repeat_index"] = 1
        single_run_records.append(copied)
    single_run = evaluate_e2e._aggregate_run_records(
        single_run_records,
        repeat_count=1,
        complete_evaluation=True,
    )
    assert single_run["overall"] == aggregate_query_records(single_run_records)
    assert single_run["repeated_runs"]["repeat_count"] == 1
    assert single_run["quality_gates"]["overall_pass"] is True

    drifted = deepcopy(records)
    drifted[-2]["evidence_candidate_ids"] = ["changed-evidence"]
    with pytest.raises(ValueError, match="retrieval/evidence fingerprint drift"):
        aggregate_repeated_query_records(drifted, repeat_count=3)

    contract_drifted = deepcopy(records)
    contract_drifted[-2]["evaluation_contract_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="evaluation contract fingerprint drift"):
        aggregate_repeated_query_records(contract_drifted, repeat_count=3)


def test_answer_fact_uses_language_aliases_not_evidence_phrase() -> None:
    item = _item().model_copy(
        update={
            "language": "vi",
            "scenario": "vi_to_en",
            "expected_phrases": ["English evidence wording"],
        }
    )
    execution = _execution(final=[_candidate("qrel")], pool=[_candidate("qrel")])
    execution = execution.__class__(
        response=execution.response.model_copy(
            update={"answer": "Dải tần số là từ -599 đến +599 Hz."}
        ),
        timings=execution.timings,
        usage=execution.usage,
        candidates=execution.candidates,
        candidate_pool=execution.candidate_pool,
    )
    record = score_query_execution(item, execution)
    assert record["answer_fact_match"] is True
    assert record["answer_fact_results"][0]["id"] == "frequency-range"
    assert record["answer_fact_results"][0]["matched"] is True
    assert record["answer_fact_results"][0]["max_alias_token_recall"] == 1.0


def test_answer_fact_diagnostics_report_ids_without_alias_or_answer_content() -> None:
    item = EvaluationItem.model_validate(
        _item().model_dump()
        | {
            "expected_answer_facts": [
                {"id": "range", "aliases": ["-599 to +599 Hz"]},
                {"id": "unit", "aliases": ["rpm"]},
            ]
        }
    )
    record = score_query_execution(
        item, _execution(final=[_candidate("qrel")], pool=[_candidate("qrel")])
    )
    assert record["answer_fact_match"] is False
    assert [result["id"] for result in record["answer_fact_results"]] == [
        "range",
        "unit",
    ]
    assert [result["matched"] for result in record["answer_fact_results"]] == [
        True,
        False,
    ]
    assert record["answer_fact_results"][0]["max_alias_token_recall"] == 1.0
    assert record["answer_fact_results"][1]["max_alias_token_recall"] == 0.0
    assert record["missing_answer_fact_ids"] == ["unit"]
    assert "aliases" not in record["answer_fact_results"][0]


def test_hybrid_fact_verdict_drives_final_metric_but_preserves_deterministic_metric() -> None:
    class _Judge:
        def score(self, *, item_id, question, answer, unresolved_facts):
            return FactJudgeResult(
                decisions=tuple(
                    FactJudgeDecision(fact.id, "present") for fact in unresolved_facts
                ),
                latency_ms=10.0,
            )

    judge = _Judge()
    answerable = score_query_execution(
        _item(),
        _execution(
            final=[_candidate("qrel")],
            pool=[_candidate("qrel")],
            answer="A semantically equivalent response.",
        ),
        fact_evaluator_mode="hybrid-luna-v2",
        fact_judge=judge,
    )
    unanswerable = score_query_execution(
        _item(answerable=False),
        _execution(final=[_candidate("other")], pool=[_candidate("other")], abstained=True),
        fact_evaluator_mode="hybrid-luna-v2",
        fact_judge=judge,
    )

    metrics = aggregate_query_records([answerable, unanswerable])
    assert metrics["answer_quality"]["deterministic_fact_accuracy_when_answered"] == 0.0
    assert metrics["answer_quality"]["final_fact_accuracy_when_answered"] == 1.0
    assert metrics["answer_quality"]["fact_judge"]["invocation_count"] == 1
    assert metrics["answer_quality"]["fact_judge"]["present_count"] == 1
    assert evaluate_quality_gates(metrics)["gates"]["fact_accuracy"]["passed"] is True


def test_document_contamination_metrics_and_gate_are_explicit() -> None:
    wrong = score_query_execution(
        _item(),
        _execution(
            final=[_candidate("qrel", document_id="programming")],
            pool=[_candidate("qrel", document_id="programming")],
        ),
    )
    unsupported = score_query_execution(
        _item(answerable=False),
        _execution(final=[_candidate("other")], pool=[_candidate("other")], abstained=True),
    )
    metrics = aggregate_query_records([wrong, unsupported])
    assert metrics["document_contamination"]["wrong_document_retrieval_at_1_rate"] == 1.0
    assert metrics["citations"]["wrong_document_citation_rate_when_answered"] == 1.0
    gates = evaluate_quality_gates(metrics)
    assert gates["overall_pass"] is False
    assert gates["gates"]["wrong_document_citations"]["passed"] is False


def test_retrieval_trace_serializes_only_allowlisted_ids_and_rank_signals() -> None:
    qrel = _candidate("qrel").model_copy(
        update={
            "text": "TRACE-SECRET-EVIDENCE",
            "dense_rank": 1,
            "rerank_rank": 1,
            "metadata": {
                "document_role": "installation",
                "query_role": "installation",
                "query_role_confidence": "strong",
                "cross_encoder_rank": 1,
                "role_rank": 1,
                "private_debug": "TRACE-SECRET-METADATA",
            },
        }
    )
    duplicate = _candidate("duplicate").model_copy(update={"rrf_rank": 2})
    pruned_by_selection = _candidate("selection-pruned").model_copy(
        update={
            "rrf_rank": 3,
            "metadata": {
                "query_role": "TRACE-SECRET-ALLOWLIST-ROLE",
                "query_role_confidence": "TRACE-SECRET-ALLOWLIST-CONFIDENCE",
                "cross_encoder_rank": "TRACE-SECRET-ALLOWLIST-RANK",
            },
        }
    )
    pruned_after_dedup = _candidate("post-dedup-pruned").model_copy(update={"rrf_rank": 4})
    trace = RetrievalTrace(
        dense_candidates=(qrel,),
        sparse_candidates=(duplicate,),
        fused_candidates=(qrel, duplicate, pruned_by_selection, pruned_after_dedup),
        selected_candidates_pre_dedup=(qrel, duplicate, pruned_after_dedup),
        candidate_pool_post_dedup=(qrel, pruned_after_dedup),
        candidates_before_rerank=(qrel,),
        raw_cross_encoder_candidates=(qrel,),
        final_candidates=(qrel,),
        candidate_count_pre_dedup=3,
        candidate_count_post_dedup=2,
        candidate_count_before_rerank=1,
    )

    record = score_query_execution(
        _item(),
        _execution(
            final=[qrel],
            pool=[qrel],
            trace=trace,
            evidence_context_chars=3_900,
            evidence_truncated_source_ids=("S2",),
        ),
    )

    payload = record["retrieval_trace"]
    assert payload["status"] == "captured"
    assert payload["candidate_count_pre_dedup"] == 3
    assert payload["candidate_count_post_dedup"] == 2
    assert payload["candidate_count_before_rerank"] == 1
    assert payload["removed_by_candidate_selection_ids"] == ["selection-pruned"]
    assert payload["removed_by_same_document_dedup_ids"] == ["duplicate"]
    assert payload["removed_by_prune_ids"] == ["post-dedup-pruned"]
    assert payload["refilled_candidate_ids"] == []
    assert payload["stages"]["dense"]["strict_direct_evidence_rank"] == 1
    assert payload["stages"]["raw_cross_encoder"]["candidates"][0] == {
        "chunk_id": "qrel",
        "document_id": "installation",
        "ordinal_rank": 1,
        "dense_rank": 1,
        "sparse_rank": None,
        "rrf_rank": None,
        "rerank_rank": 1,
        "document_role": "installation",
        "query_role": "installation",
        "query_role_confidence": "strong",
        "cross_encoder_rank": 1,
        "role_rank": 1,
    }
    serialized_trace = json.dumps(payload, sort_keys=True)
    assert "TRACE-SECRET-EVIDENCE" not in serialized_trace
    assert "TRACE-SECRET-METADATA" not in serialized_trace
    assert "TRACE-SECRET-ALLOWLIST" not in serialized_trace
    assert record["evidence_rendering"] == {
        "context_chars": 3_900,
        "truncated": True,
        "truncated_source_ids": ["S2"],
    }


def test_retrieval_trace_is_explicitly_disabled_when_not_captured() -> None:
    candidate = _candidate("qrel")
    record = score_query_execution(
        _item(),
        _execution(final=[candidate], pool=[candidate]),
    )

    assert record["retrieval_trace"] == {"status": "disabled"}
    assert record["fact_family_id"] is None
    assert record["evidence_rendering"] == {
        "context_chars": 0,
        "truncated": False,
        "truncated_source_ids": [],
    }


def test_sanitized_record_excludes_raw_dataset_answer_and_evidence_text() -> None:
    candidate = _candidate("qrel").model_copy(
        update={"text": "SENTINEL_RAW_EVIDENCE"}
    )
    item = _item().model_copy(
        update={
            "question": "SENTINEL_RAW_QUESTION",
            "expected_answer_facts": [
                ExpectedAnswerFact(
                    id="safe-fact-id",
                    aliases=["SENTINEL_RAW_EXPECTED_FACT"],
                )
            ],
        }
    )
    record = score_query_execution(
        item,
        _execution(
            final=[candidate],
            pool=[candidate],
            answer="SENTINEL_RAW_EXPECTED_FACT",
        ),
    )

    persisted = json.dumps(record, sort_keys=True)
    assert record["answer_fact_results"][0]["selected_group_alternative_count"] == 1
    for sentinel in (
        "SENTINEL_RAW_QUESTION",
        "SENTINEL_RAW_EXPECTED_FACT",
        "SENTINEL_RAW_EVIDENCE",
    ):
        assert sentinel not in persisted


def test_exact_content_equivalent_metrics_do_not_relax_strict_quality_gates() -> None:
    representative = _candidate("programming-copy", document_id="programming")
    item = _item().model_copy(update={"relevant_chunk_ids": ["installation-copy"]})
    record = score_query_execution(
        item,
        _execution(final=[representative], pool=[representative]),
        exact_content_equivalence={
            "installation-copy": ("installation-copy", "programming-copy"),
            "programming-copy": ("installation-copy", "programming-copy"),
        },
    )

    assert record["candidate_direct_evidence_rank"] is None
    assert record["direct_evidence_rank"] is None
    assert record["failure_class"] == "candidate_miss"
    assert record["citation_direct_evidence"] is False
    assert record["citation_document_correct"] is False
    assert record["unexpected_citation_document_ids"] == ["programming"]
    assert record["candidate_exact_equivalent_evidence_rank"] == 1
    assert record["ranked_exact_equivalent_evidence_rank"] == 1
    assert record["exact_equivalent_evidence_rank"] == 1
    assert record["citation_exact_equivalent_evidence"] is True
    assert record["citation_exact_equivalent_document_correct"] is True
    assert record["unexpected_citation_document_ids_excluding_exact_equivalents"] == []
    assert record["wrong_document_retrieval_at_1"] is True
    assert (
        record["wrong_document_retrieval_at_1_excluding_exact_equivalents"] is False
    )

    unsupported = score_query_execution(
        _item(answerable=False),
        _execution(final=[_candidate("other")], pool=[_candidate("other")], abstained=True),
    )
    metrics = aggregate_query_records([record, unsupported])
    assert metrics["retrieval"]["candidate_recall"] == 0.0
    assert metrics["retrieval"]["hit_rate_at_1"] == 0.0
    assert metrics["retrieval"]["exact_equivalent_candidate_recall"] == 1.0
    assert metrics["retrieval"]["exact_equivalent_hit_rate_at_1"] == 1.0
    assert metrics["citations"]["direct_evidence_rate_when_answered"] == 0.0
    assert metrics["citations"]["exact_equivalent_evidence_rate_when_answered"] == 1.0
    assert metrics["citations"]["wrong_document_citation_count"] == 1
    assert (
        metrics["citations"][
            "wrong_document_citation_count_excluding_exact_equivalents"
        ]
        == 0
    )
    assert (
        metrics["document_contamination"]["wrong_document_retrieval_at_1_count"]
        == 1
    )
    assert (
        metrics["document_contamination"][
            "wrong_document_retrieval_at_1_count_excluding_exact_equivalents"
        ]
        == 0
    )

    gates = evaluate_quality_gates(metrics)
    assert gates["overall_pass"] is False
    assert gates["gates"]["candidate_recall"]["actual"] == 0.0
    assert gates["gates"]["wrong_document_citations"]["actual"] == 1.0


def test_reviewed_cross_document_alternative_is_authoritative_but_not_canonical() -> None:
    alternative = _candidate("programming-safety", document_id="programming")
    item = EvaluationItem.model_validate(
        _item().model_dump()
        | {
            "evidence_requirements": [
                {
                    "id": "safety",
                    "canonical_chunk_ids": ["qrel"],
                    "reviewed_alternative_chunk_ids": ["programming-safety"],
                }
            ]
        }
    )

    record = score_query_execution(
        item,
        _execution(final=[alternative], pool=[alternative]),
        exact_content_equivalence={},
    )

    assert record["candidate_direct_evidence_rank"] is None
    assert record["citation_document_correct"] is False
    assert record["candidate_authoritative_requirement_coverage"] == {
        "contract_mode": "explicit",
        "requirement_count": 1,
        "covered_requirement_count": 1,
        "complete": True,
        "requirements": [
            {
                "id": "safety",
                "canonical_chunk_ids": ["qrel"],
                "reviewed_alternative_chunk_ids": ["programming-safety"],
                "rank": 1,
                "matched_chunk_id": "programming-safety",
                "provenance": "reviewed_authoritative_alternative",
            }
        ],
    }
    assert record["citation_authoritative_requirement_coverage"]["complete"] is True


def test_document_alternative_metadata_cannot_exempt_unrelated_wrong_provenance() -> None:
    unrelated = _candidate("unrelated", document_id="programming").model_copy(
        update={
            "metadata": {
                "equivalent_chunk_ids": ["unrelated", "other-copy"],
                "equivalent_document_ids": ["installation", "programming"],
            }
        }
    )

    record = score_query_execution(
        _item(),
        _execution(final=[unrelated], pool=[unrelated]),
    )

    assert record["citation_exact_equivalent_evidence"] is False
    assert record["citation_exact_equivalent_document_correct"] is False
    assert record["unexpected_citation_document_ids_excluding_exact_equivalents"] == [
        "programming"
    ]
    assert record["wrong_document_retrieval_at_1_excluding_exact_equivalents"] is True


def test_v13_cli_uses_variant_and_repeat_versioned_artifact_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_e2e",
            "--dataset",
            "calibration",
            "--provider-approval-token",
            evaluate_e2e.CALIBRATION_PROVIDER_APPROVAL_TOKEN,
        ],
    )
    args = evaluate_e2e._parse_args()
    assert evaluate_e2e.ARTIFACT_SCHEMA_VERSION == 13
    assert args.generation_variant == BASELINE_GENERATION_VARIANT
    assert args.repeat_count == 1
    assert args.output.name == "atv320-calibration-baseline-v1-r1-e2e-v13.json"
    assert (
        args.checkpoint.name
        == "atv320-calibration-baseline-v1-r1-e2e-v13-checkpoint.jsonl"
    )
    assert args.semantic_judge == "none"
    settings = evaluate_e2e._runtime_settings(evaluate_e2e.Settings())
    assert settings.rerank_deduplicate_content is True
    assert settings.dense_candidate_limit == 60
    assert settings.sparse_candidate_limit == 40


def test_heldout_v2_cli_uses_private_sealed_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_e2e",
            "--dataset",
            "heldout-v2",
            "--provider-approval-token",
            evaluate_e2e.HELDOUT_PROVIDER_APPROVAL_TOKEN,
        ],
    )

    args = evaluate_e2e._parse_args()

    assert args.heldout_v2 == evaluate_e2e.PRIVATE_HELDOUT_DATASET_PATH
    assert args.manifest == evaluate_e2e.PRIVATE_HELDOUT_MANIFEST_PATH
    assert args.output.name == "atv320-heldout-v2-baseline-v1-r1-e2e-v13.json"
    assert (
        args.checkpoint.name
        == "atv320-heldout-v2-baseline-v1-r1-e2e-v13-checkpoint.jsonl"
    )


@pytest.mark.parametrize(
    ("dataset", "expected_manifest", "expected_dataset"),
    [
        (
            "development",
            evaluate_e2e.ACTIVE_DEVELOPMENT_MANIFEST_PATH,
            evaluate_e2e.ACTIVE_DEVELOPMENT_PATH,
        ),
        (
            "confirmation",
            evaluate_e2e.PRIVATE_CONFIRMATION_MANIFEST_PATH,
            evaluate_e2e.PRIVATE_CONFIRMATION_DATASET_PATH,
        ),
    ],
)
def test_optimization_cli_uses_versioned_and_sealed_defaults(
    monkeypatch, dataset, expected_manifest, expected_dataset
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_e2e", "--dataset", dataset, "--provider-approval-token", "unused"],
    )

    args = evaluate_e2e._parse_args()

    assert args.manifest == expected_manifest
    assert getattr(args, dataset) == expected_dataset
    assert args.output.name == f"atv320-{dataset}-baseline-v1-r1-e2e-v13.json"
    assert (
        args.checkpoint.name
        == f"atv320-{dataset}-baseline-v1-r1-e2e-v13-checkpoint.jsonl"
    )


def test_development_cli_selects_claim_variant_and_repeat_schedule(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_e2e",
            "--dataset",
            "development",
            "--generation-variant",
            CLAIM_COVERAGE_GENERATION_VARIANT,
            "--repeat-count",
            "3",
            "--provider-approval-token",
            "unused",
        ],
    )

    args = evaluate_e2e._parse_args()

    assert args.output.name == (
        "atv320-development-claim-coverage-v1-r3-e2e-v13.json"
    )
    assert args.checkpoint.name == (
        "atv320-development-claim-coverage-v1-r3-e2e-v13-checkpoint.jsonl"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "--dataset",
            "calibration",
            "--generation-variant",
            CLAIM_COVERAGE_GENERATION_VARIANT,
        ],
        ["--dataset", "confirmation", "--repeat-count", "3"],
        [
            "--dataset",
            "development",
            "--repeat-count",
            "3",
            "--max-queries",
            "1",
        ],
        [
            "--dataset",
            "development",
            "--repeat-count",
            "3",
            "--semantic-judge",
            "ragas",
        ],
    ],
)
def test_experimental_generation_runs_fail_closed_outside_full_development(
    monkeypatch,
    arguments,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_e2e", *arguments, "--provider-approval-token", "unused"],
    )

    with pytest.raises(SystemExit):
        evaluate_e2e._parse_args()


def test_generation_adapter_uses_selected_schema_without_initializing_provider() -> None:
    experiment = generation_experiment_for(CLAIM_COVERAGE_GENERATION_VARIANT)

    adapter = evaluate_e2e._build_generation_adapter(
        evaluate_e2e.Settings(), experiment
    )
    configuration = evaluate_e2e._generation_configuration(
        evaluate_e2e.Settings(), experiment
    )

    assert adapter.system_prompt == experiment.system_prompt
    assert adapter.human_prompt == experiment.human_prompt
    assert adapter.output_schema is experiment.output_schema
    assert adapter.output_mapper is experiment.output_mapper
    assert adapter._structured_model is None
    assert configuration["endpoint_profile"] == "openai-responses"
    assert configuration["provider"] == "openai"
    assert configuration["api_style"] == "responses"
    assert configuration["routing_policy_id"] == "direct-openai-v1"
    assert configuration["structured_output"]["schema"] == "ClaimCoverageAnswer"
    assert "pydantic" in evaluate_e2e._library_versions()


def test_repeat_schedule_is_repeat_major_and_checkpoint_keys_are_composite(
    tmp_path,
) -> None:
    first = _item()
    second = first.model_copy(update={"id": "second"})
    schedule = evaluate_e2e._execution_schedule([first, second], repeat_count=3)

    assert [(repeat, item.id) for repeat, item in schedule] == [
        (1, first.id),
        (1, second.id),
        (2, first.id),
        (2, second.id),
        (3, first.id),
        (3, second.id),
    ]

    checkpoint = tmp_path / "checkpoint.jsonl"
    identity = {"artifact_schema_version": 10}
    records = [
        {"id": first.id, "repeat_index": 1},
        {"id": second.id, "repeat_index": 1},
        {"id": first.id, "repeat_index": 2},
    ]
    evaluate_e2e._write_checkpoint(checkpoint, identity, records)
    assert evaluate_e2e._load_checkpoint(checkpoint, identity) == records
    evaluate_e2e._validate_checkpoint_prefix(records, schedule)

    invalid_prefix = [records[0], records[2]]
    evaluate_e2e._write_checkpoint(checkpoint, identity, invalid_prefix)
    assert evaluate_e2e._load_checkpoint(checkpoint, identity) == invalid_prefix
    with pytest.raises(RuntimeError, match="repeat-major dataset order"):
        evaluate_e2e._validate_checkpoint_prefix(invalid_prefix, schedule)

    evaluate_e2e._write_checkpoint(checkpoint, identity, [records[0], records[0]])
    with pytest.raises(RuntimeError, match="duplicate item/repeat"):
        evaluate_e2e._load_checkpoint(checkpoint, identity)

    evaluate_e2e._write_checkpoint(
        checkpoint, identity, ["malformed"]  # type: ignore[list-item]
    )
    with pytest.raises(RuntimeError, match="malformed record"):
        evaluate_e2e._load_checkpoint(checkpoint, identity)


def test_generation_variant_identity_change_invalidates_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "variant-checkpoint.jsonl"
    baseline = generation_experiment_identity(
        generation_experiment_for(BASELINE_GENERATION_VARIANT)
    )
    claim_coverage = generation_experiment_identity(
        generation_experiment_for(CLAIM_COVERAGE_GENERATION_VARIANT)
    )
    evaluate_e2e._write_checkpoint(
        checkpoint,
        {"generation_experiment": baseline},
        [],
    )

    with pytest.raises(RuntimeError, match="different frozen run"):
        evaluate_e2e._load_checkpoint(
            checkpoint,
            {"generation_experiment": claim_coverage},
        )


def test_provider_execution_requires_dataset_specific_approval() -> None:
    with pytest.raises(SystemExit, match="missing or invalid"):
        evaluate_e2e._validate_execution_approval(
            SimpleNamespace(dataset="calibration", provider_approval_token="wrong")
        )
    evaluate_e2e._validate_execution_approval(
        SimpleNamespace(
            dataset="calibration",
            provider_approval_token=evaluate_e2e.CALIBRATION_PROVIDER_APPROVAL_TOKEN,
        )
    )
    with pytest.raises(SystemExit, match="missing or invalid"):
        evaluate_e2e._validate_execution_approval(
            SimpleNamespace(
                dataset="test",
                provider_approval_token=evaluate_e2e.CALIBRATION_PROVIDER_APPROVAL_TOKEN,
            )
        )
    evaluate_e2e._validate_execution_approval(
        SimpleNamespace(
            dataset="test",
            provider_approval_token=evaluate_e2e.HELDOUT_PROVIDER_APPROVAL_TOKEN,
        )
    )
    evaluate_e2e._validate_execution_approval(
        SimpleNamespace(
            dataset="heldout-v2",
            provider_approval_token=evaluate_e2e.HELDOUT_PROVIDER_APPROVAL_TOKEN,
        )
    )
    evaluate_e2e._validate_execution_approval(
        SimpleNamespace(
            dataset="development",
            provider_approval_token=evaluate_e2e.DEVELOPMENT_PROVIDER_APPROVAL_TOKEN,
        )
    )
    evaluate_e2e._validate_execution_approval(
        SimpleNamespace(
            dataset="confirmation",
            provider_approval_token=evaluate_e2e.CONFIRMATION_PROVIDER_APPROVAL_TOKEN,
        )
    )


def test_hybrid_fact_judge_requires_independent_dataset_approval_and_key() -> None:
    disabled = SimpleNamespace(
        dataset="development",
        fact_evaluator="deterministic-v4",
        fact_judge_approval_token=None,
    )
    evaluate_e2e._validate_fact_judge_approval(disabled, environ={})

    with pytest.raises(SystemExit, match="disabled"):
        evaluate_e2e._validate_fact_judge_approval(
            SimpleNamespace(
                dataset="development",
                fact_evaluator="deterministic-v4",
                fact_judge_approval_token="unexpected",
            ),
            environ={},
        )

    hybrid = SimpleNamespace(
        dataset="development",
        fact_evaluator="hybrid-luna-v2",
        fact_judge_approval_token=evaluate_e2e.DEVELOPMENT_FACT_JUDGE_APPROVAL_TOKEN,
    )
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        evaluate_e2e._validate_fact_judge_approval(hybrid, environ={})
    evaluate_e2e._validate_fact_judge_approval(
        hybrid, environ={"OPENROUTER_API_KEY": "test-only"}
    )


def test_fact_evaluator_identity_change_invalidates_v12_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "v12-checkpoint.jsonl"
    evaluate_e2e._write_checkpoint(
        checkpoint,
        {
            "artifact_schema_version": 12,
            "source_identity": {"version": 8},
        },
        [],
    )

    with pytest.raises(RuntimeError, match="different frozen run"):
        evaluate_e2e._load_checkpoint(
            checkpoint,
            {
                "artifact_schema_version": 13,
                "source_identity": {"version": 9},
            },
        )
def test_targeted_development_review_requires_its_narrow_approval() -> None:
    args = SimpleNamespace(
        dataset="development",
        review_item_id=["dvi001"],
        provider_approval_token=evaluate_e2e.DEVELOPMENT_PROVIDER_APPROVAL_TOKEN,
    )
    with pytest.raises(SystemExit, match="missing or invalid"):
        evaluate_e2e._validate_execution_approval(args)

    args.provider_approval_token = evaluate_e2e.DEVELOPMENT_REVIEW_PROVIDER_APPROVAL_TOKEN
    evaluate_e2e._validate_execution_approval(args)


def test_calibration_loader_never_opens_held_out_path(monkeypatch) -> None:
    opened = []
    approved = SimpleNamespace(review_status="approved")

    def fake_read(path):
        opened.append(path)
        if path == "poison-held-out":
            raise AssertionError("held-out path was opened")
        return [approved]

    monkeypatch.setattr(evaluate_e2e, "read_dataset", fake_read)
    monkeypatch.setattr(
        evaluate_e2e,
        "validate_dataset",
        lambda dataset, chunks, *, kind: {"kind": kind},
    )
    monkeypatch.setattr(
        evaluate_e2e,
        "validate_evaluation_manifest",
        lambda path, chunks, dataset, *, dataset_key: {
            "test_dataset_sha256": "a" * 64
        },
    )
    args = SimpleNamespace(
        dataset="calibration",
        calibration="active-calibration",
        test="poison-held-out",
        manifest="manifest",
    )
    dataset, validation, _ = evaluate_e2e._load_selected_dataset(args, [])
    assert dataset == [approved]
    assert validation == {"kind": "calibration"}
    assert opened == ["active-calibration"]


def test_heldout_v2_loader_uses_only_private_sealed_inputs(
    monkeypatch, tmp_path
) -> None:
    private_root = tmp_path / "private-heldout-v2"
    dataset_path = private_root / "heldout-v2.jsonl"
    manifest_path = private_root / "heldout-v2-manifest.json"
    approved = SimpleNamespace(review_status="approved")
    observed = {}

    def fake_read_dataset(path):
        observed["dataset_path"] = path
        return [approved]

    def fake_validate_dataset(dataset, chunks, *, kind):
        observed["kind"] = kind
        return {"kind": kind}

    monkeypatch.setattr(evaluate_e2e, "PRIVATE_HELDOUT_ROOT", private_root)
    monkeypatch.setattr(evaluate_e2e, "read_dataset", fake_read_dataset)
    monkeypatch.setattr(evaluate_e2e, "validate_dataset", fake_validate_dataset)
    monkeypatch.setattr(
        evaluate_e2e,
        "validate_evaluation_manifest",
        lambda path, chunks, dataset, *, dataset_key: observed.update(
            manifest_path=path,
            dataset_key=dataset_key,
        )
        or {"heldout_dataset_sha256": "a" * 64},
    )
    args = SimpleNamespace(
        dataset="heldout-v2",
        heldout_v2=dataset_path,
        manifest=manifest_path,
    )

    dataset, validation, manifest = evaluate_e2e._load_selected_dataset(args, [])

    assert dataset == [approved]
    assert validation == {"kind": "test"}
    assert manifest == {"heldout_dataset_sha256": "a" * 64}
    assert observed == {
        "dataset_path": dataset_path,
        "kind": "test",
        "manifest_path": manifest_path,
        "dataset_key": "heldout_dataset_sha256",
    }


def test_heldout_v2_loader_rejects_inputs_outside_private_root(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        evaluate_e2e, "PRIVATE_HELDOUT_ROOT", tmp_path / "private-heldout-v2"
    )
    args = SimpleNamespace(
        dataset="heldout-v2",
        heldout_v2=tmp_path / "public-test.jsonl",
        manifest=tmp_path / "manifest.json",
    )

    with pytest.raises(SystemExit, match="must remain under"):
        evaluate_e2e._load_selected_dataset(args, [])


def test_heldout_v2_manifest_requires_its_sealed_dataset_hash(
    monkeypatch, tmp_path
) -> None:
    manifest_path = tmp_path / "heldout-v2-manifest.json"
    corpus = {"chunk_count": 2753, "chunk_ids_sha256": "c" * 64}
    dataset_hash = "d" * 64
    manifest_path.write_text(
        json.dumps(
            {
                "corpus": corpus,
                "heldout_dataset_sha256": dataset_hash,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluation_dataset, "chunk_set_metadata", lambda chunks: corpus)
    monkeypatch.setattr(evaluation_dataset, "dataset_sha256", lambda dataset: dataset_hash)

    manifest = evaluation_dataset.validate_evaluation_manifest(
        manifest_path,
        [],
        [],
        dataset_key="heldout_dataset_sha256",
    )

    assert manifest["heldout_dataset_sha256"] == dataset_hash


def test_development_manifest_requires_confirmation_hash(monkeypatch, tmp_path) -> None:
    manifest_path = tmp_path / "development-manifest.json"
    corpus = {"chunk_count": 2753, "chunk_ids_sha256": "c" * 64}
    development_hash = "d" * 64
    confirmation_hash = "e" * 64
    manifest_path.write_text(
        json.dumps(
            {
                "corpus": corpus,
                "development_dataset_sha256": development_hash,
                "confirmation_dataset_sha256": confirmation_hash,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluation_dataset, "chunk_set_metadata", lambda chunks: corpus)
    monkeypatch.setattr(
        evaluation_dataset,
        "dataset_sha256",
        lambda dataset: development_hash,
    )

    manifest = evaluation_dataset.validate_evaluation_manifest(
        manifest_path,
        [],
        [],
        dataset_key="development_dataset_sha256",
    )

    assert evaluate_e2e._sealed_comparison_dataset_sha256(
        manifest, "development"
    ) == confirmation_hash


def test_item_id_is_rejected_for_sealed_datasets(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_e2e",
            "--dataset",
            "test",
            "--item-id",
            "phase7_test_001",
            "--provider-approval-token",
            evaluate_e2e.HELDOUT_PROVIDER_APPROVAL_TOKEN,
        ],
    )
    with pytest.raises(SystemExit):
        evaluate_e2e._parse_args()


def test_development_item_id_is_allowed_but_confirmation_partial_run_is_rejected(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_e2e",
            "--dataset",
            "development",
            "--item-id",
            "development_001",
            "--output",
            "artifacts/metrics/dev-item-001.json",
            "--checkpoint",
            "artifacts/metrics/dev-item-001-checkpoint.jsonl",
            "--provider-approval-token",
            "unused",
        ],
    )
    assert evaluate_e2e._parse_args().item_id == "development_001"

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_e2e",
            "--dataset",
            "development",
            "--item-id",
            "development_001",
            "--provider-approval-token",
            "unused",
        ],
    )
    with pytest.raises(SystemExit):
        evaluate_e2e._parse_args()
    assert "explicit --output and --checkpoint" in capsys.readouterr().err

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_e2e",
            "--dataset",
            "confirmation",
            "--max-queries",
            "1",
            "--provider-approval-token",
            "unused",
        ],
    )
    with pytest.raises(SystemExit):
        evaluate_e2e._parse_args()
    assert "sealed evaluation datasets" in capsys.readouterr().err

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_e2e",
            "--dataset",
            "heldout-v2",
            "--max-queries",
            "1",
            "--provider-approval-token",
            "unused",
        ],
    )
    with pytest.raises(SystemExit):
        evaluate_e2e._parse_args()
    assert "sealed evaluation datasets" in capsys.readouterr().err


def test_targeted_development_review_accepts_a_bounded_unique_cohort(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_e2e",
            "--dataset",
            "development",
            "--review-item-id",
            "dvi015",
            "--review-item-id",
            "dvi024",
            "--output",
            "artifacts/metrics/targeted-review.json",
            "--checkpoint",
            "artifacts/metrics/targeted-review-checkpoint.jsonl",
            "--local-review-output",
            "artifacts/local-review/targeted-review.md",
            "--provider-approval-token",
            evaluate_e2e.DEVELOPMENT_REVIEW_PROVIDER_APPROVAL_TOKEN,
        ],
    )

    args = evaluate_e2e._parse_args()

    assert args.review_item_id == ["dvi015", "dvi024"]
    assert args.generation_variant == BASELINE_GENERATION_VARIANT
    assert args.repeat_count == 1
    assert args.semantic_judge == "none"
    assert args.local_review_output == Path(
        "artifacts/local-review/targeted-review.md"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--dataset", "confirmation", "--review-item-id", "dvi001"],
        [
            "--dataset",
            "development",
            "--review-item-id",
            "dvi001",
            "--review-item-id",
            "dvi001",
        ],
        [
            "--dataset",
            "development",
            "--review-item-id",
            "dvi001",
            "--repeat-count",
            "3",
        ],
        [
            "--dataset",
            "development",
            "--review-item-id",
            "dvi001",
            "--semantic-judge",
            "ragas",
        ],
        [
            "--dataset",
            "development",
            "--review-item-id",
            "dvi001",
            "--generation-variant",
            CLAIM_COVERAGE_GENERATION_VARIANT,
        ],
    ],
)
def test_targeted_development_review_rejects_unsafe_combinations(
    monkeypatch, arguments
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_e2e", *arguments, "--provider-approval-token", "unused"],
    )

    with pytest.raises(SystemExit):
        evaluate_e2e._parse_args()


def test_targeted_review_cannot_resume_without_raw_answer() -> None:
    evaluate_e2e._require_empty_targeted_review_checkpoint([], enabled=True)
    with pytest.raises(RuntimeError, match="new or empty checkpoint"):
        evaluate_e2e._require_empty_targeted_review_checkpoint(
            [{"id": "dvi001", "repeat_index": 1}],
            enabled=True,
        )


def test_private_local_review_output_is_new_markdown_under_ignored_root(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    relative = Path("artifacts/local-review/review.md")

    resolved = evaluate_e2e._resolve_private_local_review_output(relative)

    assert resolved == (tmp_path / relative).resolve()
    assert resolved is not None
    evaluate_e2e._write_private_local_review(resolved, [])
    assert "DO NOT COMMIT" in resolved.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        evaluate_e2e._resolve_private_local_review_output(relative)
    with pytest.raises(SystemExit, match="artifacts/local-review"):
        evaluate_e2e._resolve_private_local_review_output(Path("outside.md"))
    with pytest.raises(SystemExit, match=".md extension"):
        evaluate_e2e._resolve_private_local_review_output(
            Path("artifacts/local-review/review.json")
        )


def test_private_local_review_output_requires_targeted_review(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_e2e",
            "--dataset",
            "development",
            "--local-review-output",
            "artifacts/local-review/review.md",
            "--provider-approval-token",
            "unused",
        ],
    )

    with pytest.raises(SystemExit):
        evaluate_e2e._parse_args()
    assert "requires --review-item-id" in capsys.readouterr().err


def test_checkpoint_fails_closed_when_provider_identity_changes(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.jsonl"
    identity = {"generation_configuration": {"temperature": 0.0}}
    evaluate_e2e._write_checkpoint(checkpoint, identity, [])
    changed = {"generation_configuration": {"temperature": 0.1}}
    with pytest.raises(RuntimeError, match="different frozen run"):
        evaluate_e2e._load_checkpoint(checkpoint, changed)


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("evidence_top_k", 6),
        ("evidence_score_threshold", 0.5),
        ("generation_max_context_chars", 24_001),
        ("citation_excerpt_max_chars", 401),
    ],
)
def test_checkpoint_fails_closed_when_answer_boundary_changes(
    tmp_path,
    field,
    changed_value,
) -> None:
    checkpoint = tmp_path / f"{field}.jsonl"
    baseline_configuration = evaluate_e2e._answer_boundary_configuration(
        evaluate_e2e.Settings(),
        top_k=5,
    )
    identity = {"answer_boundary_configuration": baseline_configuration}
    evaluate_e2e._write_checkpoint(checkpoint, identity, [])
    changed_configuration = dict(baseline_configuration)
    changed_configuration[field] = changed_value

    with pytest.raises(RuntimeError, match="different frozen run"):
        evaluate_e2e._load_checkpoint(
            checkpoint,
            {"answer_boundary_configuration": changed_configuration},
        )

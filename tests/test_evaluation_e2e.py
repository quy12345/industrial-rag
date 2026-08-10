"""Offline scoring tests for the Phase 7 end-to-end evaluator."""

from __future__ import annotations

from app.evaluation_e2e import (
    aggregate_phase7_records,
    evaluate_phase7_quality_gates,
    score_phase7_execution,
)
from app.generation import TokenUsage
from app.models import Citation, QueryResponse, RetrievalCandidate
from app.phase7 import Phase7DatasetItem
from app.query_service import QueryExecution, QueryTimings
from scripts import evaluate_phase7_e2e


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


def _item(*, answerable: bool = True, phrase_mode: str = "all") -> Phase7DatasetItem:
    if not answerable:
        return Phase7DatasetItem(
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
    return Phase7DatasetItem(
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
    *, final: list[RetrievalCandidate], pool: list[RetrievalCandidate], abstained: bool = False
) -> QueryExecution:
    citations = []
    if not abstained:
        candidate = final[0]
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
            answer="The documented range is -599 to +599 Hz." if not abstained else "insufficient",
            abstained=abstained,
            abstention_reason="llm_insufficient_evidence" if abstained else None,
            citations=citations,
        ),
        timings=QueryTimings(1, 2, 3, 4, 5, 15),
        usage=TokenUsage(10, 2, 1),
        candidates=tuple(final),
        candidate_pool=tuple(pool),
    )


def test_scores_direct_evidence_and_candidate_miss_without_page_fallback() -> None:
    item = _item()
    pool = [_candidate("same-page"), _candidate("qrel")]
    record = score_phase7_execution(item, _execution(final=[_candidate("same-page")], pool=pool))
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


def test_aggregate_reports_retrieval_citations_abstention_and_latency() -> None:
    answerable = score_phase7_execution(
        _item(), _execution(final=[_candidate("qrel")], pool=[_candidate("qrel")])
    )
    unsupported = score_phase7_execution(
        _item(answerable=False),
        _execution(final=[_candidate("other")], pool=[_candidate("other")], abstained=True),
    )
    metrics = aggregate_phase7_records([answerable, unsupported])
    assert metrics["retrieval"]["hit_rate_at_1"] == 1.0
    assert metrics["answer_quality"]["answer_fact_accuracy_when_answered"] == 1.0
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

    gates = evaluate_phase7_quality_gates(metrics)
    assert gates["overall_pass"] is True
    assert gates["gates"]["unsupported_citation_ids"]["actual"] == 0


def test_answerable_abstention_does_not_claim_answer_or_citation_success() -> None:
    record = score_phase7_execution(
        _item(), _execution(final=[_candidate("qrel")], pool=[_candidate("qrel")], abstained=True)
    )
    assert record["answer_fact_match"] is None
    assert record["answer_fact_results"] == []
    assert record["missing_answer_fact_ids"] == []
    assert record["citation_direct_evidence"] is False


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
    record = score_phase7_execution(item, execution)
    assert record["answer_fact_match"] is True
    assert record["answer_fact_results"][0]["id"] == "frequency-range"
    assert record["answer_fact_results"][0]["matched"] is True
    assert record["answer_fact_results"][0]["max_alias_token_recall"] == 1.0


def test_answer_fact_diagnostics_report_ids_without_alias_or_answer_content() -> None:
    item = Phase7DatasetItem.model_validate(
        _item().model_dump()
        | {
            "expected_answer_facts": [
                {"id": "range", "aliases": ["-599 to +599 Hz"]},
                {"id": "unit", "aliases": ["rpm"]},
            ]
        }
    )
    record = score_phase7_execution(
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


def test_phase7_v2_cli_preserves_historical_artifact_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["evaluate_phase7_e2e", "--dataset", "calibration"]
    )
    args = evaluate_phase7_e2e._parse_args()
    assert args.output.name == "phase-7-calibration-e2e-v2-diagnostics.json"
    assert args.checkpoint.name == "phase-7-calibration-e2e-v2-diagnostics-checkpoint.jsonl"
    settings = evaluate_phase7_e2e._phase7_settings(evaluate_phase7_e2e.Settings())
    assert settings.rerank_deduplicate_content is True

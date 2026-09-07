"""Offline contracts for the compact schema-v17 E2E scorecard."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.application.query_service import QueryExecution, QueryTimings
from app.contracts.query import Citation, QueryResponse
from app.domain.generation import TokenUsage
from app.domain.retrieval import RetrievalCandidate
from evaluation.dataset import EvaluationItem
from evaluation.e2e import (
    CASE_KEYS,
    build_case_record,
    build_scorecard,
    retrieval_evidence_fingerprint,
)
from evaluation.semantic import MetricScore, RagasScores


def _candidate(chunk_id: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="manual",
        filename="manual.pdf",
        text=f"evidence {chunk_id}",
        page_numbers=[1],
        headings=["Safety"],
        content_type="text",
        score=1.0,
    )


def _item(item_id: str, *, answerable: bool = True, language: str = "en") -> EvaluationItem:
    common = {
        "id": item_id,
        "fact_family_id": f"family-{item_id}",
        "question": "Question?",
        "language": language,
        "scenario": "vi_to_en" if language == "vi" else "en_to_en",
        "review_status": "approved",
    }
    if not answerable:
        return EvaluationItem(
            **common,
            answerable=False,
            question_type="unanswerable",
            expected_document_ids=[],
            relevant_chunk_ids=[],
            expected_pages=[],
            expected_phrases=[],
            citation_required=False,
            unanswerable_reason="Absent.",
        )
    return EvaluationItem(
        **common,
        answerable=True,
        question_type="safety",
        expected_document_ids=["manual"],
        relevant_chunk_ids=["c1"],
        expected_pages=[1],
        expected_phrases=["fact"],
        expected_answer_facts=[{"id": "f1", "aliases": ["fact"]}],
        citation_required=True,
    )


def _execution(*, abstained: bool = False, citation_id: str = "c1") -> QueryExecution:
    candidates = (_candidate("c1"), _candidate("c2"))
    citations = []
    if not abstained:
        citations = [
            Citation(
                chunk_id=citation_id,
                document_id="manual",
                filename="manual.pdf",
                page_numbers=[1],
                headings=["Safety"],
                excerpt="fact",
            )
        ]
    return QueryExecution(
        response=QueryResponse(
            answer="fact" if not abstained else "insufficient",
            abstained=abstained,
            abstention_reason="llm_insufficient_evidence" if abstained else None,
            citations=citations,
        ),
        timings=QueryTimings(1, 2, 3, 4, 5, 15),
        usage=TokenUsage(10, 2, 1),
        candidate_pool=candidates,
        candidates=tuple(reversed(candidates)),
        evidence_candidates=candidates,
        evidence_context_chars=20,
    )


def _scores(*, factual: float = 1, faithfulness: float = 1, recall: float = 1) -> RagasScores:
    na = MetricScore.not_applicable()
    return RagasScores(
        MetricScore("complete", factual, 1),
        MetricScore("complete", faithfulness, 2),
        MetricScore("complete", recall, 3),
        na,
        na,
    )


def test_compact_case_uses_exact_allowlist_and_excludes_raw_trace_fields() -> None:
    record = build_case_record(_item("a"), _execution(), _scores())

    assert set(record) == CASE_KEYS
    assert record["citation_reference_valid"] is True
    assert record["evidence_candidate_ids"] == ["c1", "c2"]
    assert not ({"question", "answer", "retrieval_trace", "qrels", "aliases"} & set(record))


def test_missing_or_unknown_citation_is_invalid_without_changing_case_shape() -> None:
    record = build_case_record(
        _item("a"), _execution(citation_id="unknown"), _scores(faithfulness=0)
    )

    assert record["citation_reference_valid"] is False
    assert record["ragas"]["faithfulness"]["score"] == 0


def test_scorecard_reports_standard_ragas_and_deterministic_behavior_metrics() -> None:
    answered = build_case_record(_item("en"), _execution(), _scores(factual=0.8, recall=0.9))
    abstained = build_case_record(
        _item("vi", language="vi"),
        _execution(abstained=True),
        RagasScores(
            MetricScore.not_applicable(),
            MetricScore.not_applicable(),
            MetricScore("complete", 0.5, 1),
            MetricScore.not_applicable(),
            MetricScore.not_applicable(),
        ),
    )
    na = RagasScores(*(MetricScore.not_applicable() for _ in range(5)))
    safe = build_case_record(_item("safe", answerable=False), _execution(abstained=True), na)
    unsafe = build_case_record(_item("unsafe", answerable=False, language="vi"), _execution(), na)

    scorecard = build_scorecard([answered, abstained, safe, unsafe])

    assert set(scorecard["ragas_metrics"]) == {
        "factual_correctness",
        "faithfulness",
        "context_recall",
        "answer_relevancy",
        "context_precision_without_reference",
    }
    assert scorecard["ragas_metrics"]["factual_correctness"]["mean"] == pytest.approx(0.8)
    assert scorecard["ragas_metrics"]["context_recall"]["mean"] == pytest.approx(0.7)
    assert scorecard["ragas_metrics"]["context_recall"]["per_language"]["vi"]["count"] == 1
    assert scorecard["ragas_metrics"]["answer_relevancy"]["mean"] is None
    assert scorecard["ragas_metrics"]["answer_relevancy"]["not_applicable_count"] == 1
    assert set(scorecard["behavior_metrics"]) == {
        "answerable_response_rate",
        "unsafe_answer_rate",
        "citation_reference_validity_rate",
    }
    assert scorecard["behavior_metrics"]["answerable_response_rate"]["rate"] == 0.5
    assert scorecard["behavior_metrics"]["unsafe_answer_rate"]["rate"] == 0.5
    assert scorecard["behavior_metrics"]["citation_reference_validity_rate"]["rate"] == 1
    assert "outcomes" not in scorecard
    assert "answer_utility" not in scorecard
    assert scorecard["status"] == "BASELINE_ONLY"
    assert scorecard["overall_pass"] is None
    assert scorecard["thresholds"] == {}


def test_citation_reference_validity_is_reported_as_a_behavior_rate() -> None:
    valid = build_case_record(_item("valid"), _execution(), _scores())
    invalid = build_case_record(
        _item("invalid"),
        _execution(citation_id="unknown"),
        _scores(faithfulness=0),
    )

    metric = build_scorecard([valid, invalid])["behavior_metrics"][
        "citation_reference_validity_rate"
    ]

    assert metric["numerator"] == 1
    assert metric["denominator"] == 2
    assert metric["rate"] == 0.5
    assert metric["invalid_citation_item_ids"] == ["invalid"]


def test_fingerprint_covers_ordered_candidate_rerank_and_evidence_boundaries() -> None:
    execution = _execution()
    changed = replace(execution, evidence_candidates=tuple(reversed(execution.evidence_candidates)))

    assert retrieval_evidence_fingerprint(execution) != retrieval_evidence_fingerprint(changed)

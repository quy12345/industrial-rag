"""Offline scoring tests for the Phase 7 end-to-end evaluator."""

from __future__ import annotations

from app.evaluation_e2e import aggregate_phase7_records, score_phase7_execution
from app.generation import TokenUsage
from app.models import Citation, QueryResponse, RetrievalCandidate
from app.phase7 import Phase7DatasetItem
from app.query_service import QueryExecution, QueryTimings


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
    assert metrics["answer_quality"]["phrase_match_rate_when_answered"] == 1.0
    assert metrics["citations"]["direct_evidence_rate_when_answered"] == 1.0
    assert metrics["abstention"]["true_positive"] == 1
    assert metrics["latency_ms"]["total"]["p95"] == 15
    assert metrics["per_language"]["en"]["query_count"] == 2


def test_answerable_abstention_does_not_claim_phrase_or_citation_success() -> None:
    record = score_phase7_execution(
        _item(), _execution(final=[_candidate("qrel")], pool=[_candidate("qrel")], abstained=True)
    )
    assert record["phrase_match"] is None
    assert record["citation_direct_evidence"] is False

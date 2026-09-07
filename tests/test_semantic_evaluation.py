"""Offline contracts for the Ragas scorecard adapter and input policy."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.application.query_service import QueryExecution, QueryTimings
from app.contracts.query import Citation, QueryResponse
from app.domain.documents import DocumentChunk
from app.domain.retrieval import RetrievalCandidate
from evaluation.dataset import EvaluationItem
from evaluation.semantic import (
    MetricScore,
    RagasEvaluationInput,
    RagasScorecardJudge,
    RagasScores,
    SemanticConfigurationError,
    SemanticJudgeError,
    _configure_openrouter_llm,
    _OpenRouterEmbeddingClient,
    render_reviewed_reference,
    semantic_run_identity,
)
from scripts.evaluation import evaluate_e2e


def _item(*, answerable: bool = True) -> EvaluationItem:
    if not answerable:
        return EvaluationItem(
            id="u1",
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
            unanswerable_reason="Absent.",
            review_status="approved",
        )
    return EvaluationItem(
        id="a1",
        question="Question?",
        language="en",
        answerable=True,
        scenario="en_to_en",
        question_type="safety",
        expected_document_ids=["manual"],
        relevant_chunk_ids=["c1"],
        expected_pages=[1],
        expected_phrases=["canonical"],
        expected_answer_facts=[
            {"id": "f1", "aliases": ["canonical one", "alternative one"]},
            {
                "id": "f2",
                "aliases": ["canonical two"],
                "required_token_groups": [["secret"]],
            },
        ],
        reference_answer=(
            "Before handling the product, verify that it has sufficiently cooled down."
        ),
        reference_answer_review_status="approved",
        citation_required=True,
        review_status="approved",
    )


def _candidate(chunk_id: str, text: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="manual",
        filename="manual.pdf",
        text=text,
        page_numbers=[1],
        headings=["Safety"],
        content_type="text",
        score=1,
    )


def _execution(
    *, abstained: bool = False, citation_id: str = "c1", context_chars: int = 20
) -> QueryExecution:
    candidates = (
        _candidate("c1", "CITED_FULL_TEXT"),
        _candidate("c2", "UNCITED_SENTINEL"),
    )
    citations = []
    if not abstained:
        citations = [
            Citation(
                chunk_id=citation_id,
                document_id="manual",
                filename="manual.pdf",
                page_numbers=[1],
                headings=["Safety"],
                excerpt="short",
            )
        ]
    return QueryExecution(
        response=QueryResponse(
            answer="answer",
            abstained=abstained,
            abstention_reason="llm_insufficient_evidence" if abstained else None,
            citations=citations,
        ),
        timings=QueryTimings(1, 2, 3, 4, 5, 15),
        usage=None,
        candidates=candidates,
        candidate_pool=candidates,
        evidence_candidates=candidates,
        evidence_context_chars=context_chars,
    )


def _corpus() -> dict[str, DocumentChunk]:
    return {
        row.chunk_id: DocumentChunk.model_validate(row.model_dump())
        for row in _execution().evidence_candidates
    }


class FakeJudge:
    def __init__(self) -> None:
        self.calls: list[tuple[RagasEvaluationInput, dict[str, bool]]] = []

    def score(self, evaluation_input: RagasEvaluationInput, **flags: bool) -> RagasScores:
        self.calls.append((evaluation_input, flags))
        metric = MetricScore("complete", 0.75, 1)
        na = MetricScore.not_applicable()
        return RagasScores(
            metric if flags["score_factual_correctness"] else na,
            metric if flags["score_faithfulness"] else na,
            metric if flags["score_context_recall"] else na,
            metric if flags["include_diagnostics"] else na,
            metric if flags["include_diagnostics"] else na,
        )


def test_reference_uses_reviewed_natural_answer_instead_of_fact_aliases() -> None:
    reference = render_reviewed_reference(_item())

    assert reference == (
        "Before handling the product, verify that it has sufficiently cooled down."
    )
    assert "alternative" not in reference
    assert "secret" not in reference


def test_reference_requires_separate_human_approval() -> None:
    pending = _item().model_copy(update={"reference_answer_review_status": "needs_human_review"})

    with pytest.raises(
        SemanticConfigurationError,
        match="requires an approved reference answer",
    ):
        render_reviewed_reference(pending)


def test_e2e_rejects_pending_reference_before_provider_construction() -> None:
    pending = _item().model_copy(update={"reference_answer_review_status": "needs_human_review"})

    with pytest.raises(SystemExit, match="requires approved natural reference answers"):
        evaluate_e2e._require_reviewed_references([pending])

    evaluate_e2e._require_reviewed_references([_item()])


def test_answered_mapping_sends_only_cited_full_context_and_bounded_evidence() -> None:
    judge = FakeJudge()
    evaluate_e2e._score_execution(
        _item(),
        _execution(),
        judge=judge,
        include_diagnostics=False,
        max_context_chars=4_000,
        corpus=_corpus(),
    )

    inputs, flags = judge.calls[0]
    assert inputs.cited_contexts == (
        "Document title: manual.pdf\nHeading: Safety\n\nCITED_FULL_TEXT",
    )
    assert "UNCITED_SENTINEL" not in inputs.cited_contexts
    assert "CITED_FULL_TEXT" in inputs.generation_context
    assert "UNCITED_SENTINEL" in inputs.generation_context
    assert inputs.final_contexts == ("CITED_FULL_TEXT", "UNCITED_SENTINEL")
    assert flags == {
        "score_factual_correctness": True,
        "score_faithfulness": True,
        "score_context_recall": True,
        "include_diagnostics": False,
    }


@pytest.mark.parametrize(
    ("answerable", "abstained", "expected_calls", "expected_recall"),
    [
        (True, False, 1, 0.75),
        (True, True, 1, 0.75),
        (False, False, 0, None),
        (False, True, 0, None),
    ],
)
def test_applicability_matrix(
    answerable: bool,
    abstained: bool,
    expected_calls: int,
    expected_recall: float | None,
) -> None:
    judge = FakeJudge()
    scores = evaluate_e2e._score_execution(
        _item(answerable=answerable),
        _execution(abstained=abstained),
        judge=judge,
        include_diagnostics=False,
        max_context_chars=4_000,
        corpus=_corpus(),
    )

    assert len(judge.calls) == expected_calls
    assert scores.context_recall.score == expected_recall
    if answerable and abstained:
        assert scores.factual_correctness.status == "not_applicable"


def test_empty_context_is_zero_without_judge_call() -> None:
    judge = FakeJudge()
    scores = evaluate_e2e._score_execution(
        _item(),
        _execution(abstained=True, context_chars=0),
        judge=judge,
        include_diagnostics=False,
        max_context_chars=4_000,
        corpus=_corpus(),
    )

    assert judge.calls == []
    assert scores.context_recall == MetricScore.deterministic_zero("empty_context")


def test_invalid_citation_skips_faithfulness_and_records_zero() -> None:
    judge = FakeJudge()
    scores = evaluate_e2e._score_execution(
        _item(),
        _execution(citation_id="unknown"),
        judge=judge,
        include_diagnostics=False,
        max_context_chars=4_000,
        corpus=_corpus(),
    )

    assert judge.calls[0][1]["score_faithfulness"] is False
    assert scores.faithfulness == MetricScore.deterministic_zero("invalid_citations")


def test_context_recall_receives_exact_truncated_generator_context() -> None:
    from app.application.generation_prompt import format_evidence

    execution = _execution()
    evidence = tuple(
        row.model_copy(update={"text": row.text * 300}) for row in execution.evidence_candidates
    )
    execution = replace(execution, evidence_candidates=evidence)
    corpus = {row.chunk_id: DocumentChunk.model_validate(row.model_dump()) for row in evidence}
    judge = FakeJudge()
    evaluate_e2e._score_execution(
        _item(),
        execution,
        judge=judge,
        include_diagnostics=False,
        max_context_chars=1000,
        corpus=corpus,
    )
    expected = format_evidence(evidence, max_chars=1000)
    assert expected.truncated_source_ids
    assert judge.calls[0][0].generation_context == expected.text
    assert evidence[0].text in judge.calls[0][0].cited_contexts[0]
    assert evidence[1].text not in judge.calls[0][0].cited_contexts[0]


def test_required_metric_cannot_silently_become_not_applicable() -> None:
    class MissingScoreJudge(FakeJudge):
        def score(self, value, **flags):
            return replace(super().score(value, **flags), faithfulness=MetricScore.not_applicable())

    with pytest.raises(SemanticJudgeError, match="faithfulness"):
        evaluate_e2e._score_execution(
            _item(),
            _execution(),
            judge=MissingScoreJudge(),
            include_diagnostics=False,
            max_context_chars=4000,
            corpus=_corpus(),
        )


def test_semantic_identity_is_deterministic_and_embedding_is_diagnostic_only() -> None:
    core = semantic_run_identity(include_diagnostics=False)
    diagnostics = semantic_run_identity(include_diagnostics=True)

    assert core == semantic_run_identity(include_diagnostics=False)
    assert core["embedding_model"] is None
    assert diagnostics["embedding_model"] == "openai/text-embedding-3-small"
    assert core["configuration_sha256"] != diagnostics["configuration_sha256"]


def test_invalid_provider_result_is_sanitized() -> None:
    sentinel = "DO_NOT_EXPOSE_RAW_PAYLOAD"

    class Metric:
        def score(self, **_):
            raise RuntimeError(sentinel)

    with pytest.raises(SemanticJudgeError) as captured:
        RagasScorecardJudge._score_metric("safe-id", "faithfulness", Metric())

    assert str(captured.value) == "Ragas judge failed for item safe-id, metric faithfulness."
    assert sentinel not in str(captured.value)


def test_openrouter_llm_and_embedding_routing_are_fail_closed() -> None:
    llm = SimpleNamespace(model_args={"temperature": 0, "max_completion_tokens": 10})
    _configure_openrouter_llm(llm)
    assert llm.model_args["store"] is False
    assert llm.model_args["extra_body"]["provider"]["allow_fallbacks"] is False

    class Resource:
        async def create(self, **kwargs):
            return kwargs

    client = SimpleNamespace(embeddings=Resource())
    result = asyncio.run(_OpenRouterEmbeddingClient(client).embeddings.create(input=["x"]))
    assert result["extra_body"]["provider"]["require_parameters"] is True

"""Offline orchestration tests for the Phase 6 grounded query service."""

from __future__ import annotations

import logging

import pytest

from app.config import Settings
from app.errors import (
    GenerationValidationError,
    LLMRefusalError,
    LLMTimeoutError,
    RerankerUnavailableError,
)
from app.generation import GeneratedAnswer, GenerationResult, TokenUsage
from app.models import RetrievalCandidate
from app.query_service import EvidenceGate, QueryService
from app.request_context import request_id
from app.retrieval_runtime import QueryRetrievalResult


def _candidate(
    chunk_id: str = "chunk-a",
    *,
    document_id: str = "manual-a",
    score: float = 1.0,
    text: str = "Evidence 24 VDC.",
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        filename="manual.pdf",
        text=text,
        page_numbers=[1],
        headings=["Limits"],
        content_type="text",
        score=score,
        rerank_score=score,
        rerank_rank=1,
    )


class FakeRetriever:
    def __init__(self, candidates=None, error: Exception | None = None) -> None:
        self.candidates = [_candidate()] if candidates is None else candidates
        self.error = error
        self.calls = []

    def retrieve(self, question, *, document_id):
        self.calls.append((question, document_id))
        if self.error:
            raise self.error
        return QueryRetrievalResult(self.candidates, retrieval_ms=2.0, rerank_ms=3.0)


class FakeGenerator:
    def __init__(self, outputs=None, configured_error: Exception | None = None) -> None:
        self.outputs = list(
            outputs
            or [
                GenerationResult(
                    GeneratedAnswer(
                        answer="Grounded 24 VDC.",
                        source_ids=["S1"],
                        insufficient_evidence=False,
                    )
                )
            ]
        )
        self.configured_error = configured_error
        self.calls = []
        self.configuration_checks = 0

    def ensure_configured(self):
        self.configuration_checks += 1
        if self.configured_error:
            raise self.configured_error

    def generate(self, *, question, evidence, validation_errors=()):
        self.calls.append((question, evidence, tuple(validation_errors)))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def _service(retriever=None, generator=None, *, threshold=None, **settings):
    return QueryService(
        retriever=retriever or FakeRetriever(),
        evidence_gate=EvidenceGate(score_threshold=threshold),
        generator=generator or FakeGenerator(),
        settings=Settings(**settings),
    )


def test_valid_grounded_query_preserves_metadata_and_timings() -> None:
    execution = _service().execute(question="Điện áp?", document_id="manual-a", top_k=5)
    assert execution.response.abstained is False
    assert execution.response.answer == "Grounded 24 VDC."
    assert execution.response.citations[0].chunk_id == "chunk-a"
    assert execution.timings.retrieval_ms == 2.0
    assert execution.timings.rerank_ms == 3.0
    assert execution.timings.total_ms >= 0


def test_top_k_is_applied_after_ordered_retrieval_and_document_filter_is_forwarded() -> None:
    candidates = [_candidate(f"chunk-{index}") for index in range(1, 4)]
    retriever = FakeRetriever(candidates)
    generator = FakeGenerator()
    service = _service(retriever, generator)
    service.execute(question="q", document_id="manual-a", top_k=2)
    assert retriever.calls == [("q", "manual-a")]
    evidence = generator.calls[0][1]
    assert tuple(evidence.source_map) == ("S1", "S2")
    assert [item.chunk_id for item in evidence.source_map.values()] == ["chunk-1", "chunk-2"]


def test_execution_keeps_full_final_and_pre_rerank_candidates_for_evaluation() -> None:
    final = [_candidate("final-1"), _candidate("final-2")]
    pool = [_candidate("pool-1"), *final]
    retriever = FakeRetriever(final)
    retriever.retrieve = lambda question, *, document_id: QueryRetrievalResult(
        final, retrieval_ms=2.0, rerank_ms=3.0, candidate_pool=pool
    )
    execution = _service(retriever).execute(question="q", document_id=None, top_k=1)
    assert [candidate.chunk_id for candidate in execution.candidates] == ["final-1", "final-2"]
    assert [candidate.chunk_id for candidate in execution.candidate_pool] == [
        "pool-1",
        "final-1",
        "final-2",
    ]
    assert [candidate.chunk_id for candidate in execution.evidence_candidates] == ["final-1"]


def test_exact_cross_document_duplicate_is_removed_before_top_k() -> None:
    programming = _candidate(
        "programming-copy", document_id="programming", text="Disconnect before wiring."
    ).model_copy(update={"metadata": {"document_role": "programming"}})
    installation = _candidate(
        "installation-copy", document_id="installation", text="Disconnect before wiring."
    ).model_copy(update={"metadata": {"document_role": "installation"}})
    unique = _candidate(
        "unique", document_id="installation", text="Use terminal X1."
    ).model_copy(update={"metadata": {"document_role": "installation"}})
    execution = _service(FakeRetriever([programming, installation, unique])).execute(
        question="Which wiring terminal is required?", document_id=None, top_k=2
    )
    assert [candidate.chunk_id for candidate in execution.candidates] == [
        "programming-copy",
        "installation-copy",
        "unique",
    ]
    assert [candidate.chunk_id for candidate in execution.evidence_candidates] == [
        "installation-copy",
        "unique",
    ]
    assert execution.evidence_duplicate_groups[0].representative_chunk_id == "installation-copy"


@pytest.mark.parametrize(
    ("candidates", "reason"),
    [
        ([], "no_candidates"),
        ([_candidate(text=" ")], "invalid_candidate_metadata"),
        ([_candidate(document_id="manual-b")], "invalid_candidate_metadata"),
        ([_candidate(score=float("nan"))], "invalid_candidate_metadata"),
    ],
)
def test_evidence_gate_abstains_without_generation(candidates, reason) -> None:
    generator = FakeGenerator()
    execution = _service(FakeRetriever(candidates), generator).execute(
        question="q", document_id="manual-a", top_k=5
    )
    assert execution.response.abstained is True
    assert execution.response.abstention_reason == reason
    assert execution.response.citations == []
    assert generator.calls == []


def test_configured_score_gate_is_not_a_probability_and_skips_generation() -> None:
    generator = FakeGenerator()
    execution = _service(
        FakeRetriever([_candidate(score=-2.0)]), generator, threshold=-1.0
    ).execute(question="q", document_id=None, top_k=5)
    assert execution.response.abstention_reason == "configured_score_gate_failed"
    assert generator.calls == []


def test_llm_declared_insufficient_evidence_returns_valid_abstention() -> None:
    generator = FakeGenerator(
        [
            GenerationResult(
                GeneratedAnswer(answer="Not enough.", source_ids=[], insufficient_evidence=True)
            )
        ]
    )
    execution = _service(generator=generator).execute(question="unknown", document_id=None, top_k=5)
    assert execution.response.abstention_reason == "llm_insufficient_evidence"
    assert execution.response.citations == []


def test_provider_refusal_returns_abstention_without_correction_retry() -> None:
    generator = FakeGenerator([LLMRefusalError("refused")])
    execution = _service(generator=generator).execute(question="q", document_id=None, top_k=5)
    assert execution.response.abstention_reason == "llm_refusal"
    assert len(generator.calls) == 1


def test_invalid_citation_gets_one_retry_with_same_evidence_and_no_reretrieval() -> None:
    retriever = FakeRetriever()
    generator = FakeGenerator(
        [
            GenerationResult(
                GeneratedAnswer(answer="bad", source_ids=["S9"], insufficient_evidence=False)
            ),
            GenerationResult(
                GeneratedAnswer(answer="fixed", source_ids=["S1"], insufficient_evidence=False)
            ),
        ]
    )
    execution = _service(retriever, generator).execute(question="q", document_id=None, top_k=5)
    assert execution.response.answer == "fixed"
    assert len(retriever.calls) == 1
    assert len(generator.calls) == 2
    assert generator.calls[0][1] is generator.calls[1][1]
    assert "unknown source ID" in generator.calls[1][2][0]


@pytest.mark.parametrize(
    "first_output",
    [
        GenerationResult(GeneratedAnswer(answer="bad", source_ids=[], insufficient_evidence=False)),
        GenerationResult(
            GeneratedAnswer(answer=" ", source_ids=["S1"], insufficient_evidence=False)
        ),
        GenerationValidationError("parse failed"),
    ],
)
def test_second_invalid_output_abstains_after_exactly_one_retry(first_output) -> None:
    generator = FakeGenerator([first_output, GenerationValidationError("still invalid")])
    execution = _service(generator=generator).execute(question="q", document_id=None, top_k=5)
    assert execution.response.abstention_reason == "citation_validation_failed"
    assert len(generator.calls) == 2


def test_dependency_errors_propagate_without_fallback() -> None:
    retriever = FakeRetriever(error=RerankerUnavailableError("down"))
    with pytest.raises(RerankerUnavailableError):
        _service(retriever=retriever).execute(question="q", document_id=None, top_k=5)
    generator = FakeGenerator([LLMTimeoutError("slow")])
    with pytest.raises(LLMTimeoutError):
        _service(generator=generator).execute(question="q", document_id=None, top_k=5)


def test_missing_configuration_is_checked_before_retrieval() -> None:
    error = LLMTimeoutError("configuration unavailable")
    retriever = FakeRetriever()
    generator = FakeGenerator(configured_error=error)
    with pytest.raises(LLMTimeoutError):
        _service(retriever, generator).execute(question="q", document_id=None, top_k=5)
    assert retriever.calls == []


def test_usage_is_accumulated_across_correction_attempts() -> None:
    generator = FakeGenerator(
        [
            GenerationResult(
                GeneratedAnswer(answer="bad", source_ids=["S9"], insufficient_evidence=False),
                TokenUsage(10, 2, 1),
            ),
            GenerationResult(
                GeneratedAnswer(answer="ok", source_ids=["S1"], insufficient_evidence=False),
                TokenUsage(11, 3, None),
            ),
        ]
    )
    execution = _service(generator=generator).execute(question="q", document_id=None, top_k=5)
    assert execution.usage == TokenUsage(21, 5, 1)


def test_logs_do_not_contain_question_or_evidence(caplog) -> None:
    secret_question = "secret-question-123"
    secret_evidence = "secret-evidence-456"
    token = request_id.set("request-123")
    try:
        with caplog.at_level(logging.INFO, logger="app.query_service"):
            _service(FakeRetriever([_candidate(text=secret_evidence)])).execute(
                question=secret_question, document_id=None, top_k=5
            )
    finally:
        request_id.reset(token)
    assert "request_id=request-123" in caplog.text
    assert secret_question not in caplog.text
    assert secret_evidence not in caplog.text

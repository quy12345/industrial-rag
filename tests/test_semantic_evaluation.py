"""Offline contracts for optional Ragas semantic evaluation."""

from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest

from app.application.generation_prompt import format_evidence
from app.application.query_service import QueryExecution, QueryTimings
from app.contracts.query import Citation, QueryResponse
from app.domain.retrieval import RetrievalCandidate
from evaluation.dataset import EvaluationItem
from evaluation.semantic import (
    RagasSemanticJudge,
    SemanticConfigurationError,
    SemanticEvaluationInput,
    SemanticJudgeError,
    SemanticScores,
    _configure_openrouter_llm,
    _OpenRouterEmbeddingClient,
    aggregate_semantic_records,
    semantic_run_identity,
)
from scripts.evaluation import evaluate_e2e


class FakeSemanticJudge:
    def __init__(self) -> None:
        self.inputs: list[SemanticEvaluationInput] = []

    def score(self, evaluation_input: SemanticEvaluationInput) -> SemanticScores:
        self.inputs.append(evaluation_input)
        return SemanticScores(
            status="complete",
            faithfulness=0.9,
            answer_relevancy=0.8,
            context_precision=0.7,
            judge_latency_ms=12.5,
        )


def _candidate(chunk_id: str, text: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="installation",
        filename="manual.pdf",
        text=text,
        page_numbers=[7],
        headings=["Parameters"],
        content_type="text",
        score=1.0,
    )


def _item(*, item_id: str = "semantic-item", answerable: bool = True) -> EvaluationItem:
    return EvaluationItem(
        id=item_id,
        question="SENTINEL_QUESTION",
        language="en",
        answerable=answerable,
        scenario="en_to_en",
        question_type="parameter_code" if answerable else "unanswerable",
        expected_document_ids=["installation"] if answerable else [],
        relevant_chunk_ids=["chunk-1"] if answerable else [],
        expected_pages=[7] if answerable else [],
        expected_phrases=["range"] if answerable else [],
        expected_answer_facts=(
            [{"id": "range", "aliases": ["documented range"]}]
            if answerable
            else []
        ),
        citation_required=answerable,
        unanswerable_reason=None if answerable else "Verified absent.",
        review_status="approved",
    )


def _execution(*, abstained: bool = False) -> QueryExecution:
    candidates = (
        _candidate("chunk-1", "SENTINEL_CONTEXT_ONE"),
        _candidate("chunk-2", "SENTINEL_CONTEXT_TWO"),
    )
    citations = []
    if not abstained:
        citations = [
            Citation(
                chunk_id="chunk-1",
                document_id="installation",
                filename="manual.pdf",
                page_numbers=[7],
                headings=["Parameters"],
                excerpt="sanitized excerpt",
            )
        ]
    return QueryExecution(
        response=QueryResponse(
            answer="SENTINEL_ANSWER" if not abstained else "insufficient",
            abstained=abstained,
            abstention_reason="llm_insufficient_evidence" if abstained else None,
            citations=citations,
        ),
        timings=QueryTimings(1, 2, 3, 4, 5, 15),
        usage=None,
        candidates=candidates,
        candidate_pool=candidates,
        evidence_candidates=candidates,
    )


def test_fake_judge_receives_exact_generation_evidence_and_ranked_contexts() -> None:
    judge = FakeSemanticJudge()
    execution = _execution()

    scores = evaluate_e2e._score_semantic_execution(
        _item(answerable=False),
        execution,
        judge=judge,
        max_context_chars=24_000,
    )

    expected_evidence = format_evidence(
        execution.evidence_candidates, max_chars=24_000
    ).text
    assert scores.status == "complete"
    assert len(judge.inputs) == 1
    assert judge.inputs[0].rendered_generation_evidence == expected_evidence
    assert judge.inputs[0].ranked_final_contexts == (
        "SENTINEL_CONTEXT_ONE",
        "SENTINEL_CONTEXT_TWO",
    )


def test_abstention_is_not_applicable_and_never_invokes_judge() -> None:
    judge = FakeSemanticJudge()

    scores = evaluate_e2e._score_semantic_execution(
        _item(),
        _execution(abstained=True),
        judge=judge,
        max_context_chars=24_000,
    )

    assert scores.to_payload() == {
        "status": "not_applicable_abstained",
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "judge_latency_ms": 0.0,
        "error_category": None,
    }
    assert judge.inputs == []


def test_semantic_aggregate_uses_eligible_denominators_and_breakdowns() -> None:
    complete_en = {
        "language": "en",
        "answerable": True,
        "semantic_evaluation": SemanticScores(
            "complete", 1.0, 0.8, 0.6, 10.0
        ).to_payload(),
    }
    complete_vi = {
        "language": "vi",
        "answerable": True,
        "semantic_evaluation": SemanticScores(
            "complete", 0.5, 0.4, 1.0, 20.0
        ).to_payload(),
    }
    abstained_vi = {
        "language": "vi",
        "answerable": False,
        "semantic_evaluation": SemanticScores.not_applicable_abstained().to_payload(),
    }

    aggregate = aggregate_semantic_records([complete_en, complete_vi, abstained_vi])

    assert aggregate["eligible_count"] == 2
    assert aggregate["excluded_count"] == 1
    assert aggregate["failed_count"] == 0
    assert aggregate["metric_invocation_count"] == 6
    assert aggregate["metrics"]["faithfulness"] == {
        "count": 2,
        "mean": 0.75,
        "median": 0.75,
    }
    assert aggregate["per_language"]["vi"]["eligible_count"] == 1
    assert aggregate["per_answerability"]["unanswerable"]["excluded_count"] == 1


def test_semantic_identity_is_deterministic_and_checkpoint_mode_fails_closed(tmp_path) -> None:
    first = semantic_run_identity("ragas")
    second = semantic_run_identity("ragas")
    assert first == second
    assert len(first["metric_configuration_sha256"]) == 64

    checkpoint = tmp_path / "semantic-checkpoint.jsonl"
    evaluate_e2e._write_checkpoint(
        checkpoint, {"semantic_evaluation": semantic_run_identity("none")}, []
    )
    with pytest.raises(RuntimeError, match="different frozen run"):
        evaluate_e2e._load_checkpoint(
            checkpoint, {"semantic_evaluation": semantic_run_identity("ragas")}
        )


def test_semantic_approval_is_separate_and_requires_openrouter_key() -> None:
    ragas_args = SimpleNamespace(
        dataset="calibration",
        semantic_judge="ragas",
        judge_provider_approval_token=evaluate_e2e.CALIBRATION_SEMANTIC_APPROVAL_TOKEN,
    )
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        evaluate_e2e._validate_semantic_approval(
            ragas_args, environ={"OPENAI_API_KEY": "wrong-provider-key"}
        )

    openrouter_environment = {"OPENROUTER_API_KEY": "test-openrouter-key"}
    evaluate_e2e._validate_semantic_approval(
        ragas_args, environ=openrouter_environment
    )
    with pytest.raises(SystemExit, match="missing or invalid"):
        evaluate_e2e._validate_semantic_approval(
            SimpleNamespace(
                dataset="heldout-v2",
                semantic_judge="ragas",
                judge_provider_approval_token=(
                    evaluate_e2e.CALIBRATION_SEMANTIC_APPROVAL_TOKEN
                ),
            ),
            environ=openrouter_environment,
        )


def test_semantic_mode_off_does_not_construct_optional_judge(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluate_e2e,
        "RagasSemanticJudge",
        lambda **kwargs: pytest.fail("semantic dependency was initialized"),
    )
    judge = evaluate_e2e._build_semantic_judge(
        SimpleNamespace(semantic_judge="none"), environ={}
    )
    assert judge is None


def test_semantic_judge_uses_only_the_explicit_openrouter_key(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_judge(*, api_key: str):
        captured["api_key"] = api_key
        return "semantic-judge"

    monkeypatch.setattr(evaluate_e2e, "RagasSemanticJudge", fake_judge)

    judge = evaluate_e2e._build_semantic_judge(
        SimpleNamespace(semantic_judge="ragas"),
        environ={
            "OPENROUTER_API_KEY": "test-openrouter-key",
            "OPENAI_API_KEY": "must-not-be-used",
        },
    )

    assert judge == "semantic-judge"
    assert captured == {"api_key": "test-openrouter-key"}


def test_openrouter_identity_and_llm_arguments_are_frozen() -> None:
    identity = semantic_run_identity("ragas")
    assert identity["provider"] == "openrouter"
    assert identity["base_url"] == "https://openrouter.ai/api/v1"
    assert identity["upstream_provider"] == "openai"
    assert identity["provider_fallbacks"] is False
    assert identity["require_parameters"] is True
    assert identity["judge_model"] == "openai/gpt-5.6-luna"
    assert identity["embedding_model"] == "openai/text-embedding-3-small"

    llm = SimpleNamespace(
        model_args={
            "max_tokens": 1024,
            "temperature": 0.01,
            "top_p": 0.1,
            "reasoning_effort": "medium",
        }
    )
    _configure_openrouter_llm(llm)

    assert llm.model_args == {
        "max_tokens": 4096,
        "store": False,
        "extra_body": {
            "provider": {
                "order": ["openai"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
            "reasoning": {"effort": "low", "exclude": True},
        },
    }


def test_openrouter_embedding_client_injects_same_no_fallback_policy() -> None:
    captured: dict[str, object] = {}

    class FakeEmbeddingsResource:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return "embedding-response"

    wrapped = _OpenRouterEmbeddingClient(
        SimpleNamespace(embeddings=FakeEmbeddingsResource())
    )

    assert inspect.iscoroutinefunction(wrapped.embeddings.create)
    response = asyncio.run(
        wrapped.embeddings.create(
            model="openai/text-embedding-3-small", input=["safe text"]
        )
    )

    assert response == "embedding-response"
    assert captured["extra_body"] == {
        "provider": {
            "order": ["openai"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    }
    with pytest.raises(SemanticConfigurationError, match="routing override"):
        asyncio.run(wrapped.embeddings.create(input=["safe"], extra_body={}))


def test_provider_error_and_invalid_score_are_sanitized() -> None:
    class BrokenMetric:
        def score(self, **inputs):
            raise RuntimeError("SENTINEL_PROVIDER_PAYLOAD")

    class InvalidMetric:
        def score(self, **inputs):
            return SimpleNamespace(value=float("nan"))

    for metric in (BrokenMetric(), InvalidMetric()):
        with pytest.raises(SemanticJudgeError) as captured:
            RagasSemanticJudge._score_metric(
                "safe-item-id", "faithfulness", metric, response="SENTINEL_ANSWER"
            )
        assert str(captured.value) == (
            "Semantic judge failed for item safe-item-id, metric faithfulness."
        )
        assert captured.value.__cause__ is None
        assert "SENTINEL" not in str(captured.value)


def test_sanitized_semantic_payload_excludes_raw_inputs_and_reason() -> None:
    judge = FakeSemanticJudge()
    scores = evaluate_e2e._score_semantic_execution(
        _item(), _execution(), judge=judge, max_context_chars=24_000
    )
    persisted = json.dumps(
        {
            "semantic_evaluation": scores.to_payload(),
            "aggregate": aggregate_semantic_records(
                [
                    {
                        "language": "en",
                        "answerable": True,
                        "semantic_evaluation": scores.to_payload(),
                    }
                ]
            ),
            "identity": semantic_run_identity("ragas"),
        }
    )
    for sentinel in (
        "SENTINEL_QUESTION",
        "SENTINEL_ANSWER",
        "SENTINEL_CONTEXT_ONE",
        "SENTINEL_CONTEXT_TWO",
        "SENTINEL_REASON",
    ):
        assert sentinel not in persisted

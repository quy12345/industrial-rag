"""Sanitized semantic-evaluation contracts and the optional Ragas adapter."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean, median
from time import perf_counter
from typing import Any, Literal, Protocol

RAGAS_VERSION = "0.4.3"
OPENAI_VERSION = "2.53.0"
LANGCHAIN_COMMUNITY_VERSION = "0.3.31"
JUDGE_PROVIDER = "openrouter"
RAGAS_PROVIDER_ADAPTER = "openai"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_UPSTREAM_PROVIDER = "openai"
OPENROUTER_ROUTING_POLICY_ID = "openai-only-no-fallback-v1"
JUDGE_MODEL = "openai/gpt-5.6-luna"
JUDGE_REASONING_EFFORT = "low"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
JUDGE_TIMEOUT_SECONDS = 60.0
JUDGE_MAX_RETRIES = 1
JUDGE_MAX_COMPLETION_TOKENS = 4096
JUDGE_STORE = False
SEMANTIC_INPUT_POLICY_ID = "atv320-final-generation-evidence-v1"
SEMANTIC_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision_without_reference",
)

SemanticStatus = Literal["complete", "not_applicable_abstained", "failed"]


@dataclass(frozen=True)
class SemanticEvaluationInput:
    """Raw semantic inputs held in memory only for one evaluator invocation."""

    item_id: str
    question: str
    answer: str
    rendered_generation_evidence: str
    ranked_final_contexts: tuple[str, ...]


@dataclass(frozen=True)
class SemanticScores:
    """Sanitized scores safe to persist in evaluation artifacts."""

    status: SemanticStatus
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    judge_latency_ms: float
    error_category: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.judge_latency_ms) or self.judge_latency_ms < 0:
            raise ValueError("Semantic judge latency must be finite and non-negative.")
        values = (self.faithfulness, self.answer_relevancy, self.context_precision)
        if self.status == "complete":
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError("A completed semantic evaluation requires three finite scores.")
            assert self.faithfulness is not None
            assert self.answer_relevancy is not None
            assert self.context_precision is not None
            if not 0.0 <= self.faithfulness <= 1.0:
                raise ValueError("Faithfulness must be between zero and one.")
            if not -1.0 <= self.answer_relevancy <= 1.0:
                raise ValueError("Answer relevancy must be between minus one and one.")
            if not 0.0 <= self.context_precision <= 1.0:
                raise ValueError("Context precision must be between zero and one.")
            if self.error_category is not None:
                raise ValueError("A completed semantic evaluation cannot contain an error.")
        elif any(value is not None for value in values):
            raise ValueError("A non-complete semantic evaluation cannot contain scores.")

    @classmethod
    def not_applicable_abstained(cls) -> SemanticScores:
        return cls(
            status="not_applicable_abstained",
            faithfulness=None,
            answer_relevancy=None,
            context_precision=None,
            judge_latency_ms=0.0,
        )

    def to_payload(self) -> dict[str, str | float | None]:
        """Return only allowlisted fields; explanations and raw inputs cannot leak."""

        return {
            "status": self.status,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "judge_latency_ms": self.judge_latency_ms,
            "error_category": self.error_category,
        }


class SemanticJudge(Protocol):
    """Port used by the E2E evaluator and deterministic fake tests."""

    def score(self, evaluation_input: SemanticEvaluationInput) -> SemanticScores:
        """Return three complete semantic scores or raise a sanitized error."""


class SemanticConfigurationError(RuntimeError):
    """Raised before egress when the pinned semantic runtime is unavailable."""


class SemanticJudgeError(RuntimeError):
    """Provider/metric failure whose message never includes raw evaluation inputs."""

    def __init__(self, *, item_id: str, metric: str, category: str) -> None:
        self.item_id = item_id
        self.metric = metric
        self.category = category
        super().__init__(f"Semantic judge failed for item {item_id}, metric {metric}.")


class RagasSemanticJudge:
    """Ragas Collections adapter using OpenRouter's OpenAI-compatible API."""

    def __init__(self, *, api_key: str) -> None:
        _require_exact_dependency("ragas", RAGAS_VERSION)
        _require_exact_dependency("openai", OPENAI_VERSION)
        _require_exact_dependency("langchain-community", LANGCHAIN_COMMUNITY_VERSION)
        # The approval token authorizes judge calls, not Ragas usage telemetry.
        os.environ["RAGAS_DO_NOT_TRACK"] = "true"
        try:
            from openai import AsyncOpenAI
            from ragas.embeddings.base import embedding_factory
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AnswerRelevancy,
                ContextPrecisionWithoutReference,
                Faithfulness,
            )
        except (ImportError, AttributeError):
            raise SemanticConfigurationError(
                "Pinned Ragas Collections imports are unavailable; semantic evaluation stopped."
            ) from None

        # Ragas owns these untyped SDK objects. Keep that boundary isolated here.
        client: Any = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=JUDGE_TIMEOUT_SECONDS,
            max_retries=JUDGE_MAX_RETRIES,
        )
        llm: Any = llm_factory(
            JUDGE_MODEL,
            provider=RAGAS_PROVIDER_ADAPTER,
            client=client,
            cache=None,
        )
        _configure_openrouter_llm(llm)
        embeddings: Any = embedding_factory(
            RAGAS_PROVIDER_ADAPTER,
            model=EMBEDDING_MODEL,
            client=_OpenRouterEmbeddingClient(client),
        )
        self._faithfulness: Any = Faithfulness(llm=llm)
        self._answer_relevancy: Any = AnswerRelevancy(llm=llm, embeddings=embeddings)
        self._context_precision: Any = ContextPrecisionWithoutReference(llm=llm)

    def score(self, evaluation_input: SemanticEvaluationInput) -> SemanticScores:
        started = perf_counter()
        faithfulness = self._score_metric(
            evaluation_input.item_id,
            "faithfulness",
            self._faithfulness,
            user_input=evaluation_input.question,
            response=evaluation_input.answer,
            retrieved_contexts=[evaluation_input.rendered_generation_evidence],
        )
        answer_relevancy = self._score_metric(
            evaluation_input.item_id,
            "answer_relevancy",
            self._answer_relevancy,
            user_input=evaluation_input.question,
            response=evaluation_input.answer,
        )
        context_precision = self._score_metric(
            evaluation_input.item_id,
            "context_precision_without_reference",
            self._context_precision,
            user_input=evaluation_input.question,
            response=evaluation_input.answer,
            retrieved_contexts=list(evaluation_input.ranked_final_contexts),
        )
        return SemanticScores(
            status="complete",
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            judge_latency_ms=(perf_counter() - started) * 1000,
        )

    @staticmethod
    def _score_metric(
        item_id: str,
        metric_name: str,
        metric: Any,
        **inputs: Any,
    ) -> float:
        try:
            result = metric.score(**inputs)
            value = float(result.value)
        except Exception:
            raise SemanticJudgeError(
                item_id=item_id,
                metric=metric_name,
                category="provider_or_structured_output_error",
            ) from None
        lower_bound = -1.0 if metric_name == "answer_relevancy" else 0.0
        if not math.isfinite(value) or not lower_bound <= value <= 1.0:
            raise SemanticJudgeError(
                item_id=item_id,
                metric=metric_name,
                category="invalid_score",
            )
        return value


def semantic_run_identity(mode: Literal["none", "ragas"]) -> dict[str, object]:
    """Return a raw-content-free identity used to reject incompatible resumes."""

    if mode == "none":
        return {"mode": "none"}
    configuration = {
        "mode": "ragas",
        "ragas_version": RAGAS_VERSION,
        "openai_version": OPENAI_VERSION,
        "langchain_community_version": LANGCHAIN_COMMUNITY_VERSION,
        "provider": JUDGE_PROVIDER,
        "ragas_provider_adapter": RAGAS_PROVIDER_ADAPTER,
        "base_url": OPENROUTER_BASE_URL,
        "upstream_provider": OPENROUTER_UPSTREAM_PROVIDER,
        "routing_policy_id": OPENROUTER_ROUTING_POLICY_ID,
        "provider_fallbacks": False,
        "require_parameters": True,
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": JUDGE_REASONING_EFFORT,
        "embedding_model": EMBEDDING_MODEL,
        "timeout_seconds": JUDGE_TIMEOUT_SECONDS,
        "maximum_retries": JUDGE_MAX_RETRIES,
        "maximum_completion_tokens": JUDGE_MAX_COMPLETION_TOKENS,
        "store": JUDGE_STORE,
        "ragas_usage_tracking": False,
        "metrics": list(SEMANTIC_METRICS),
        "input_policy_id": SEMANTIC_INPUT_POLICY_ID,
    }
    encoded = json.dumps(configuration, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return configuration | {
        "metric_configuration_sha256": hashlib.sha256(encoded).hexdigest()
    }


def aggregate_semantic_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate sanitized semantic payloads without requiring raw evaluation inputs."""

    summary = _semantic_summary(records)
    languages = sorted({str(record["language"]) for record in records})
    summary["per_language"] = {
        language: _semantic_summary(
            [record for record in records if record["language"] == language]
        )
        for language in languages
    }
    summary["per_answerability"] = {
        "answerable": _semantic_summary(
            [record for record in records if bool(record["answerable"])]
        ),
        "unanswerable": _semantic_summary(
            [record for record in records if not bool(record["answerable"])]
        ),
    }
    return summary


def _semantic_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    payloads = [
        payload
        for record in records
        if isinstance((payload := record.get("semantic_evaluation")), Mapping)
    ]
    complete = [payload for payload in payloads if payload.get("status") == "complete"]
    excluded = sum(
        payload.get("status") == "not_applicable_abstained" for payload in payloads
    )
    failed = sum(payload.get("status") == "failed" for payload in payloads)
    return {
        "eligible_count": len(complete),
        "excluded_count": excluded,
        "failed_count": failed,
        "metric_invocation_count": len(complete) * len(SEMANTIC_METRICS),
        "metrics": {
            name: _numeric_summary(complete, field)
            for name, field in (
                ("faithfulness", "faithfulness"),
                ("answer_relevancy", "answer_relevancy"),
                ("context_precision_without_reference", "context_precision"),
            )
        },
        "judge_latency_ms": _numeric_summary(complete, "judge_latency_ms"),
    }


def _numeric_summary(
    payloads: Sequence[Mapping[str, object]], field: str
) -> dict[str, int | float | None]:
    values = [float(payload[field]) for payload in payloads if payload.get(field) is not None]
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
    }


def _require_exact_dependency(name: str, expected: str) -> None:
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        raise SemanticConfigurationError(
            f"Semantic evaluation requires {name}=={expected}."
        ) from None
    if actual != expected:
        raise SemanticConfigurationError(
            f"Semantic evaluation requires {name}=={expected}; found {actual}."
        )


def _configure_openrouter_llm(llm: Any) -> None:
    """Normalize Ragas model arguments for an OpenRouter-prefixed GPT-5 slug."""

    try:
        model_args = dict(llm.model_args)
    except (AttributeError, TypeError, ValueError):
        raise SemanticConfigurationError(
            "Pinned Ragas LLM arguments are unavailable; semantic evaluation stopped."
        ) from None

    # Ragas 0.4.3 only recognizes bare ``gpt-5*`` names as reasoning models. The
    # OpenRouter ``openai/...`` slug therefore needs the equivalent mapping here.
    model_args.pop("max_completion_tokens", None)
    model_args.pop("temperature", None)
    model_args.pop("top_p", None)
    model_args.pop("reasoning_effort", None)
    model_args.update(
        {
            "max_tokens": JUDGE_MAX_COMPLETION_TOKENS,
            "store": JUDGE_STORE,
            "extra_body": _openrouter_request_body(include_reasoning=True),
        }
    )
    llm.model_args = model_args


def _openrouter_request_body(*, include_reasoning: bool) -> dict[str, object]:
    body: dict[str, object] = {
        "provider": {
            "order": [OPENROUTER_UPSTREAM_PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    }
    if include_reasoning:
        body["reasoning"] = {
            "effort": JUDGE_REASONING_EFFORT,
            "exclude": True,
        }
    return body


class _OpenRouterEmbeddingsResource:
    """Inject the frozen routing policy into OpenAI-compatible embedding calls."""

    def __init__(self, resource: Any) -> None:
        self._resource = resource

    async def create(self, **kwargs: Any) -> Any:
        if "extra_body" in kwargs:
            raise SemanticConfigurationError(
                "Unexpected embedding routing override; semantic evaluation stopped."
            )
        return await self._resource.create(
            **kwargs,
            extra_body=_openrouter_request_body(include_reasoning=False),
        )


class _OpenRouterEmbeddingClient:
    """Minimal client surface required by Ragas' modern embedding adapter."""

    def __init__(self, client: Any) -> None:
        self.embeddings = _OpenRouterEmbeddingsResource(client.embeddings)

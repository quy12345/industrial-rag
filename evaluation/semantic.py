"""Ragas quality-score contracts for offline ATV320 evaluation.

Raw inputs exist only for one judge call. Persisted payloads contain scores,
applicability, latency, and sanitized errors.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from evaluation.dataset import EvaluationItem

RAGAS_VERSION = "0.4.3"
OPENAI_VERSION = "2.53.0"
LANGCHAIN_COMMUNITY_VERSION = "0.3.31"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_UPSTREAM_PROVIDER = "openai"
OPENROUTER_ROUTING_POLICY_ID = "openai-only-no-fallback-v1"
JUDGE_MODEL = "openai/gpt-5.6-luna"
JUDGE_REASONING_EFFORT = "low"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
JUDGE_TIMEOUT_SECONDS = 60.0
JUDGE_MAX_RETRIES = 1
JUDGE_MAX_COMPLETION_TOKENS = 4096
REFERENCE_POLICY_ID = "reviewed_expected_fact_reference_v1"
CORE_METRICS = ("factual_correctness", "faithfulness", "context_recall")
DIAGNOSTIC_METRICS = ("answer_relevancy", "context_precision_without_reference")


@dataclass(frozen=True)
class RagasEvaluationInput:
    """Raw, in-memory-only inputs for one applicable Ragas invocation."""

    item_id: str
    question: str
    answer: str
    reference: str
    cited_contexts: tuple[str, ...]
    generation_context: str
    final_contexts: tuple[str, ...]


@dataclass(frozen=True)
class MetricScore:
    """One sanitized metric result."""

    status: str
    score: float | None
    latency_ms: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("Metric latency must be finite and non-negative.")
        if self.status in {"complete", "empty_context", "invalid_citations"}:
            if self.score is None or not math.isfinite(self.score):
                raise ValueError("A scored metric requires a finite score.")
            if not -1.0 <= self.score <= 1.0:
                raise ValueError("Metric score must be between minus one and one.")
        elif self.score is not None:
            raise ValueError("A non-complete metric cannot contain a score.")

    @classmethod
    def not_applicable(cls) -> MetricScore:
        return cls("not_applicable", None, 0.0)

    @classmethod
    def deterministic_zero(cls, status: str) -> MetricScore:
        return cls(status, 0.0, 0.0)

    def to_payload(self) -> dict[str, str | float | None]:
        return {"status": self.status, "score": self.score, "latency_ms": self.latency_ms}


@dataclass(frozen=True)
class RagasScores:
    """Applicable core scores plus optional diagnostics for one case."""

    factual_correctness: MetricScore
    faithfulness: MetricScore
    context_recall: MetricScore
    answer_relevancy: MetricScore
    context_precision: MetricScore

    def to_payload(self) -> dict[str, dict[str, str | float | None]]:
        return {
            "factual_correctness": self.factual_correctness.to_payload(),
            "faithfulness": self.faithfulness.to_payload(),
            "context_recall": self.context_recall.to_payload(),
            "answer_relevancy": self.answer_relevancy.to_payload(),
            "context_precision_without_reference": self.context_precision.to_payload(),
        }


class RagasJudge(Protocol):
    """Port implemented by the OpenRouter adapter and offline fakes."""

    def score(
        self,
        evaluation_input: RagasEvaluationInput,
        *,
        score_factual_correctness: bool,
        score_faithfulness: bool,
        score_context_recall: bool,
        include_diagnostics: bool,
    ) -> RagasScores: ...


class SemanticConfigurationError(RuntimeError):
    """Raised before egress when the pinned judge runtime is unavailable."""


class SemanticJudgeError(RuntimeError):
    """A sanitized provider or structured-output failure."""

    def __init__(self, *, item_id: str, metric: str, category: str) -> None:
        self.item_id = item_id
        self.metric = metric
        self.category = category
        super().__init__(f"Ragas judge failed for item {item_id}, metric {metric}.")


def render_reviewed_reference(item: EvaluationItem) -> str:
    """Render exactly the first reviewed alias of every expected fact."""

    return "\n".join(f"- {fact.aliases[0]}" for fact in item.expected_answer_facts)


class RagasScorecardJudge:
    """Pinned Ragas Collections adapter over OpenRouter."""

    def __init__(self, *, api_key: str, include_diagnostics: bool = False) -> None:
        _require_exact_dependency("ragas", RAGAS_VERSION)
        _require_exact_dependency("openai", OPENAI_VERSION)
        _require_exact_dependency("langchain-community", LANGCHAIN_COMMUNITY_VERSION)
        os.environ["RAGAS_DO_NOT_TRACK"] = "true"
        try:
            from openai import AsyncOpenAI
            from ragas.embeddings.base import embedding_factory
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AnswerRelevancy,
                ContextPrecisionWithoutReference,
                ContextRecall,
                FactualCorrectness,
                Faithfulness,
            )
        except (ImportError, AttributeError):
            raise SemanticConfigurationError(
                "Pinned Ragas Collections imports are unavailable; evaluation stopped."
            ) from None

        client: Any = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=JUDGE_TIMEOUT_SECONDS,
            max_retries=JUDGE_MAX_RETRIES,
        )
        llm: Any = llm_factory("openai/gpt-5.6-luna", provider="openai", client=client, cache=None)
        _configure_openrouter_llm(llm)
        self._factual_correctness: Any = FactualCorrectness(llm=llm, mode="f1")
        self._faithfulness: Any = Faithfulness(llm=llm)
        self._context_recall: Any = ContextRecall(llm=llm)
        self._answer_relevancy: Any | None = None
        self._context_precision: Any | None = None
        if include_diagnostics:
            embeddings: Any = embedding_factory(
                "openai",
                model=EMBEDDING_MODEL,
                client=_OpenRouterEmbeddingClient(client),
            )
            self._answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
            self._context_precision = ContextPrecisionWithoutReference(llm=llm)

    def score(
        self,
        evaluation_input: RagasEvaluationInput,
        *,
        score_factual_correctness: bool,
        score_faithfulness: bool,
        score_context_recall: bool,
        include_diagnostics: bool,
    ) -> RagasScores:
        factual = self._optional_metric(
            evaluation_input,
            enabled=score_factual_correctness,
            name="factual_correctness",
            metric=self._factual_correctness,
            response=evaluation_input.answer,
            reference=evaluation_input.reference,
        )
        faithfulness = self._optional_metric(
            evaluation_input,
            enabled=score_faithfulness,
            name="faithfulness",
            metric=self._faithfulness,
            user_input=evaluation_input.question,
            response=evaluation_input.answer,
            retrieved_contexts=list(evaluation_input.cited_contexts),
        )
        recall = self._optional_metric(
            evaluation_input,
            enabled=score_context_recall,
            name="context_recall",
            metric=self._context_recall,
            user_input=evaluation_input.question,
            retrieved_contexts=[evaluation_input.generation_context],
            reference=evaluation_input.reference,
        )
        relevancy = self._optional_metric(
            evaluation_input,
            enabled=include_diagnostics and score_factual_correctness,
            name="answer_relevancy",
            metric=self._answer_relevancy,
            user_input=evaluation_input.question,
            response=evaluation_input.answer,
        )
        precision = self._optional_metric(
            evaluation_input,
            enabled=include_diagnostics and score_factual_correctness,
            name="context_precision_without_reference",
            metric=self._context_precision,
            user_input=evaluation_input.question,
            response=evaluation_input.answer,
            retrieved_contexts=list(evaluation_input.final_contexts),
        )
        return RagasScores(factual, faithfulness, recall, relevancy, precision)

    def _optional_metric(
        self,
        evaluation_input: RagasEvaluationInput,
        *,
        enabled: bool,
        name: str,
        metric: Any,
        **inputs: Any,
    ) -> MetricScore:
        if not enabled:
            return MetricScore.not_applicable()
        if metric is None:
            raise SemanticConfigurationError(f"Ragas diagnostic metric is unavailable: {name}")
        started = perf_counter()
        value = self._score_metric(evaluation_input.item_id, name, metric, **inputs)
        return MetricScore("complete", value, (perf_counter() - started) * 1000)

    @staticmethod
    def _score_metric(item_id: str, metric_name: str, metric: Any, **inputs: Any) -> float:
        try:
            value = float(metric.score(**inputs).value)
        except Exception:
            raise SemanticJudgeError(
                item_id=item_id,
                metric=metric_name,
                category="provider_or_structured_output_error",
            ) from None
        lower_bound = -1.0 if metric_name == "answer_relevancy" else 0.0
        if not math.isfinite(value) or not lower_bound <= value <= 1.0:
            raise SemanticJudgeError(item_id=item_id, metric=metric_name, category="invalid_score")
        return value


def semantic_run_identity(*, include_diagnostics: bool) -> dict[str, object]:
    """Return raw-content-free identity used to reject incompatible resumes."""

    configuration = {
        "ragas_version": RAGAS_VERSION,
        "openai_version": OPENAI_VERSION,
        "langchain_community_version": LANGCHAIN_COMMUNITY_VERSION,
        "provider": "openrouter",
        "base_url": OPENROUTER_BASE_URL,
        "upstream_provider": OPENROUTER_UPSTREAM_PROVIDER,
        "routing_policy_id": OPENROUTER_ROUTING_POLICY_ID,
        "provider_fallbacks": False,
        "require_parameters": True,
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": JUDGE_REASONING_EFFORT,
        "timeout_seconds": JUDGE_TIMEOUT_SECONDS,
        "maximum_retries": JUDGE_MAX_RETRIES,
        "maximum_completion_tokens": JUDGE_MAX_COMPLETION_TOKENS,
        "store": False,
        "reference_policy_id": REFERENCE_POLICY_ID,
        "core_metrics": list(CORE_METRICS),
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS) if include_diagnostics else [],
        "embedding_model": EMBEDDING_MODEL if include_diagnostics else None,
    }
    encoded = json.dumps(configuration, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return configuration | {"configuration_sha256": hashlib.sha256(encoded).hexdigest()}


def _require_exact_dependency(name: str, expected: str) -> None:
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        raise SemanticConfigurationError(f"Evaluation requires {name}=={expected}.") from None
    if actual != expected:
        raise SemanticConfigurationError(f"Evaluation requires {name}=={expected}; found {actual}.")


def _configure_openrouter_llm(llm: Any) -> None:
    try:
        model_args = dict(llm.model_args)
    except (AttributeError, TypeError, ValueError):
        raise SemanticConfigurationError(
            "Pinned Ragas LLM arguments are unavailable; evaluation stopped."
        ) from None
    for key in ("max_completion_tokens", "temperature", "top_p", "reasoning_effort"):
        model_args.pop(key, None)
    model_args.update(
        {
            "max_tokens": JUDGE_MAX_COMPLETION_TOKENS,
            "store": False,
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
        body["reasoning"] = {"effort": JUDGE_REASONING_EFFORT, "exclude": True}
    return body


class _OpenRouterEmbeddingsResource:
    def __init__(self, resource: Any) -> None:
        self._resource = resource

    async def create(self, **kwargs: Any) -> Any:
        if "extra_body" in kwargs:
            raise SemanticConfigurationError(
                "Unexpected embedding routing override; evaluation stopped."
            )
        return await self._resource.create(
            **kwargs,
            extra_body=_openrouter_request_body(include_reasoning=False),
        )


class _OpenRouterEmbeddingClient:
    def __init__(self, client: Any) -> None:
        self.embeddings = _OpenRouterEmbeddingsResource(client.embeddings)

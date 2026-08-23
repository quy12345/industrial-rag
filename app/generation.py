"""Compatibility facade and lazy structured-generation adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import Lock
from typing import Any

from pydantic import ValidationError

from app.application.generation_prompt import HUMAN_PROMPT as HUMAN_PROMPT
from app.application.generation_prompt import SYSTEM_PROMPT as SYSTEM_PROMPT
from app.application.generation_prompt import TRUNCATION_MARKER as TRUNCATION_MARKER
from app.application.generation_prompt import build_correction_text
from app.application.generation_prompt import format_evidence as format_evidence
from app.config import Settings
from app.domain.generation import AnswerGenerator as AnswerGenerator
from app.domain.generation import EvidenceBundle as EvidenceBundle
from app.domain.generation import GeneratedAnswer as GeneratedAnswer
from app.domain.generation import GenerationResult as GenerationResult
from app.domain.generation import TokenUsage as TokenUsage
from app.errors import (
    GenerationValidationError,
    LLMNotConfiguredError,
    LLMRefusalError,
    LLMTimeoutError,
    LLMUnavailableError,
)


class LangChainOpenAIGenerator:
    """Lazy OpenAI/compatible adapter with provider-native structured output."""

    def __init__(
        self,
        settings: Settings,
        *,
        model_factory: Callable[..., Any] | None = None,
        prompt_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self._model_factory = model_factory
        self._prompt_factory = prompt_factory
        self._structured_model: Any | None = None
        self._prompt: Any | None = None
        self._lock = Lock()

    def ensure_configured(self) -> None:
        if self.settings.generation_api_key is None:
            key_name = (
                "GEMINI_API_KEY"
                if self.settings.generation_provider == "gemini"
                else "OPENAI_API_KEY"
            )
            raise LLMNotConfiguredError(f"{key_name} is not configured.")
        if self.settings.openai_store:
            raise LLMNotConfiguredError("OPENAI_STORE must remain false for grounded queries.")

    def generate(
        self,
        *,
        question: str,
        evidence: EvidenceBundle,
        validation_errors: Sequence[str] = (),
    ) -> GenerationResult:
        self.ensure_configured()
        prompt = self._get_prompt().invoke(
            {
                "question": question,
                "evidence": evidence.text,
                "allowed_source_ids": ", ".join(evidence.allowed_source_ids),
                "correction": build_correction_text(validation_errors),
            }
        )
        try:
            result = self._get_structured_model().invoke(prompt)
        except (LLMNotConfiguredError, LLMRefusalError, LLMTimeoutError, LLMUnavailableError):
            raise
        except Exception as exc:
            _raise_provider_error(exc)
        if not isinstance(result, dict):
            raise GenerationValidationError("Structured provider result must be a mapping.")
        parsing_error = result.get("parsing_error")
        if parsing_error is not None:
            if "refusal" in type(parsing_error).__name__.casefold():
                raise LLMRefusalError("The generation provider refused the request.")
            raise GenerationValidationError(
                "Provider structured output could not be parsed.",
                errors=(type(parsing_error).__name__,),
            )
        parsed = result.get("parsed")
        if parsed is None:
            if _raw_has_refusal(result.get("raw")):
                raise LLMRefusalError("The generation provider refused the request.")
            raise GenerationValidationError("Provider returned no structured answer.")
        try:
            output = (
                parsed
                if isinstance(parsed, GeneratedAnswer)
                else GeneratedAnswer.model_validate(parsed)
            )
        except ValidationError as exc:
            raise GenerationValidationError(
                "Provider output does not match GeneratedAnswer.",
                errors=tuple(error["type"] for error in exc.errors()),
            ) from exc
        return GenerationResult(output=output, usage=_extract_usage(result.get("raw")))

    def _get_structured_model(self) -> Any:
        if self._structured_model is None:
            with self._lock:
                if self._structured_model is None:
                    factory = self._model_factory
                    if factory is None:
                        try:
                            from langchain_openai import ChatOpenAI
                        except ImportError as exc:
                            raise LLMNotConfiguredError(
                                "langchain-openai is not installed."
                            ) from exc
                        factory = ChatOpenAI
                    api_key = self.settings.generation_api_key
                    if api_key is None:
                        raise LLMNotConfiguredError(
                            f"{self.settings.generation_provider} API key is not configured."
                        )
                    model_kwargs: dict[str, Any] = {
                        "model": self.settings.generation_model,
                        "api_key": api_key.get_secret_value(),
                        "max_tokens": self.settings.openai_max_output_tokens,
                        "timeout": self.settings.openai_timeout_seconds,
                        "max_retries": self.settings.openai_max_retries,
                    }
                    if self.settings.generation_provider == "gemini":
                        model_kwargs.update(
                            {
                                "base_url": self.settings.gemini_base_url,
                                "use_responses_api": False,
                                "reasoning_effort": self.settings.gemini_reasoning_effort,
                                "temperature": self.settings.gemini_temperature,
                            }
                        )
                    else:
                        model_kwargs.update(
                            {
                                "use_responses_api": True,
                                "output_version": "responses/v1",
                                "store": False,
                                "reasoning": {
                                    "effort": self.settings.openai_reasoning_effort
                                },
                            }
                        )
                    model = factory(**model_kwargs)
                    self._structured_model = model.with_structured_output(
                        GeneratedAnswer,
                        method="json_schema",
                        strict=True,
                        include_raw=True,
                    )
        return self._structured_model

    def _get_prompt(self) -> Any:
        if self._prompt is None:
            factory = self._prompt_factory
            if factory is None:
                try:
                    from langchain_core.prompts import ChatPromptTemplate
                except ImportError as exc:
                    raise LLMNotConfiguredError("langchain-core is not installed.") from exc
                factory = ChatPromptTemplate.from_messages
            self._prompt = factory(
                [
                    ("system", SYSTEM_PROMPT),
                    ("human", HUMAN_PROMPT),
                ]
            )
        return self._prompt


def _extract_usage(raw: Any) -> TokenUsage | None:
    metadata = getattr(raw, "usage_metadata", None)
    if not isinstance(metadata, dict):
        return None
    input_details = metadata.get("input_token_details")
    cached = input_details.get("cache_read") if isinstance(input_details, dict) else None
    return TokenUsage(
        input_tokens=_optional_int(metadata.get("input_tokens")),
        output_tokens=_optional_int(metadata.get("output_tokens")),
        cached_input_tokens=_optional_int(cached),
    )


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) and value >= 0 else None


def _raw_has_refusal(raw: Any) -> bool:
    additional = getattr(raw, "additional_kwargs", None)
    return isinstance(additional, dict) and bool(additional.get("refusal"))


def _raise_provider_error(exc: Exception) -> None:
    name = type(exc).__name__.casefold()
    if isinstance(exc, TimeoutError) or "timeout" in name:
        raise LLMTimeoutError("Generation provider timed out.") from exc
    if "refusal" in name:
        raise LLMRefusalError("Generation provider refused the request.") from exc
    raise LLMUnavailableError("Generation provider is unavailable.") from exc

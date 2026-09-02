"""Lazy LangChain adapter for OpenAI-compatible structured generation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import Lock
from typing import Any

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.domain.generation import EvidenceBundle, GeneratedAnswer, GenerationResult, TokenUsage
from app.errors import (
    GenerationValidationError,
    LLMNotConfiguredError,
    LLMRefusalError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.infrastructure.generation.endpoints import (
    GenerationEndpointConfigurationError,
    generation_model_kwargs,
    validate_generation_endpoint,
)


class LangChainStructuredGenerator:
    """Lazy OpenAI-compatible adapter with provider-native structured output."""

    def __init__(
        self,
        settings: Settings,
        *,
        system_prompt: str,
        human_prompt: str,
        correction_text_builder: Callable[[Sequence[str]], str],
        output_schema: type[BaseModel] = GeneratedAnswer,
        output_mapper: Callable[[BaseModel, EvidenceBundle], GeneratedAnswer] | None = None,
        model_factory: Callable[..., Any] | None = None,
        prompt_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self.system_prompt = system_prompt
        self.human_prompt = human_prompt
        self.correction_text_builder = correction_text_builder
        if output_schema is not GeneratedAnswer and output_mapper is None:
            raise ValueError("A custom structured output schema requires an output mapper.")
        self.output_schema = output_schema
        self.output_mapper = output_mapper
        self._model_factory = model_factory
        self._prompt_factory = prompt_factory
        self._structured_model: Any | None = None
        self._prompt: Any | None = None
        self._lock = Lock()

    def ensure_configured(self) -> None:
        if self.settings.generation_api_key is None:
            raise LLMNotConfiguredError("GENERATION_API_KEY is not configured.")
        try:
            validate_generation_endpoint(self.settings)
        except GenerationEndpointConfigurationError as exc:
            raise LLMNotConfiguredError(str(exc)) from None

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
                "correction": self.correction_text_builder(validation_errors),
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
            structured_output = (
                parsed
                if isinstance(parsed, self.output_schema)
                else self.output_schema.model_validate(parsed)
            )
        except ValidationError as exc:
            raise GenerationValidationError(
                f"Provider output does not match {self.output_schema.__name__}.",
                errors=tuple(error["type"] for error in exc.errors()),
            ) from exc
        if self.output_mapper is None:
            if not isinstance(structured_output, GeneratedAnswer):
                raise GenerationValidationError(
                    "Structured output has no configured mapper.",
                    errors=("output_mapper_missing",),
                )
            output = structured_output
        else:
            try:
                output = self.output_mapper(structured_output, evidence)
            except GenerationValidationError:
                raise
            except Exception as exc:
                raise GenerationValidationError(
                    "Structured output mapping failed.",
                    errors=(type(exc).__name__,),
                ) from None
            if not isinstance(output, GeneratedAnswer):
                raise GenerationValidationError(
                    "Structured output mapper returned an invalid result.",
                    errors=("output_mapper_result_invalid",),
                )
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
                        raise LLMNotConfiguredError("GENERATION_API_KEY is not configured.")
                    try:
                        model_kwargs = generation_model_kwargs(
                            self.settings,
                            api_key=api_key.get_secret_value(),
                        )
                    except GenerationEndpointConfigurationError as exc:
                        raise LLMNotConfiguredError(str(exc)) from None
                    model = factory(**model_kwargs)
                    self._structured_model = model.with_structured_output(
                        self.output_schema,
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
                    ("system", self.system_prompt),
                    ("human", self.human_prompt),
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

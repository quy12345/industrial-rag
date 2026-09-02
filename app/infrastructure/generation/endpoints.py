"""Resolved endpoint profiles for OpenAI-compatible structured generation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

if TYPE_CHECKING:
    from app.config import Settings

GenerationApiStyle = Literal["responses", "chat_completions"]


class GenerationEndpointConfigurationError(ValueError):
    """Raised when a selected endpoint cannot honor the generation contract."""


@dataclass(frozen=True)
class GenerationEndpointProfile:
    """Provider transport capabilities that are independent of a model ID."""

    identifier: str
    provider: str
    base_url: str
    api_style: GenerationApiStyle
    reasoning_style: Literal["openai", "gemini", "openrouter"]
    supported_reasoning_modes: frozenset[str]
    supports_temperature: bool
    routing_policy_id: str


ENDPOINT_PROFILES = MappingProxyType(
    {
        "openai-responses": GenerationEndpointProfile(
            identifier="openai-responses",
            provider="openai",
            base_url="https://api.openai.com/v1",
            api_style="responses",
            reasoning_style="openai",
            supported_reasoning_modes=frozenset({"low", "medium", "high"}),
            supports_temperature=False,
            routing_policy_id="direct-openai-v1",
        ),
        "gemini-chat": GenerationEndpointProfile(
            identifier="gemini-chat",
            provider="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_style="chat_completions",
            reasoning_style="gemini",
            supported_reasoning_modes=frozenset({"minimal", "low", "medium", "high"}),
            supports_temperature=True,
            routing_policy_id="direct-gemini-v1",
        ),
        "openrouter-chat": GenerationEndpointProfile(
            identifier="openrouter-chat",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_style="chat_completions",
            reasoning_style="openrouter",
            supported_reasoning_modes=frozenset(
                {"omit", "disabled", "low", "medium", "high", "max"}
            ),
            supports_temperature=True,
            routing_policy_id="require-parameters-no-fallback-v3",
        ),
    }
)


def endpoint_profile_for(identifier: str) -> GenerationEndpointProfile:
    """Return a declared endpoint profile or reject the configuration."""

    try:
        return ENDPOINT_PROFILES[identifier]
    except KeyError:
        supported = ", ".join(sorted(ENDPOINT_PROFILES))
        raise GenerationEndpointConfigurationError(
            f"Unsupported generation endpoint profile: {identifier}. Supported: {supported}."
        ) from None


def validate_generation_endpoint(settings: Settings) -> GenerationEndpointProfile:
    """Validate portable settings against the selected endpoint capabilities."""

    profile = endpoint_profile_for(settings.generation_endpoint_profile)
    if settings.generation_reasoning not in profile.supported_reasoning_modes:
        modes = ", ".join(sorted(profile.supported_reasoning_modes))
        raise GenerationEndpointConfigurationError(
            f"Generation reasoning mode {settings.generation_reasoning!r} is unsupported by "
            f"{profile.identifier}; supported modes: {modes}."
        )
    if settings.generation_store:
        raise GenerationEndpointConfigurationError(
            "GENERATION_STORE must remain false for grounded queries."
        )
    return profile


def generation_model_kwargs(settings: Settings, *, api_key: str) -> dict[str, Any]:
    """Build SDK arguments at the isolated untyped provider boundary."""

    profile = validate_generation_endpoint(settings)
    model_kwargs: dict[str, Any] = {
        "model": settings.generation_model,
        "api_key": api_key,
        "timeout": settings.generation_timeout_seconds,
        "max_retries": settings.generation_max_retries,
    }
    if profile.api_style == "responses":
        model_kwargs.update(
            {
                "max_tokens": settings.generation_max_output_tokens,
                "use_responses_api": True,
                "output_version": "responses/v1",
                "store": False,
                "reasoning": {"effort": settings.generation_reasoning},
            }
        )
        return model_kwargs

    model_kwargs.update(
        {
            "base_url": profile.base_url,
            "use_responses_api": False,
            "temperature": settings.generation_temperature,
        }
    )
    if profile.reasoning_style == "gemini":
        model_kwargs["max_tokens"] = settings.generation_max_output_tokens
        model_kwargs["reasoning_effort"] = settings.generation_reasoning
        return model_kwargs

    extra_body: dict[str, Any] = {
        # ChatOpenAI 1.4 maps its max_tokens argument to max_completion_tokens.
        # OpenRouter capability routing advertises max_tokens for this endpoint,
        # so pass the portable wire parameter through the SDK's extra body.
        "max_tokens": settings.generation_max_output_tokens,
        "provider": {
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    }
    if settings.generation_reasoning != "omit":
        if settings.generation_reasoning == "disabled":
            effort = "none"
        else:
            effort = (
                "xhigh"
                if settings.generation_reasoning == "max"
                else settings.generation_reasoning
            )
        extra_body["reasoning"] = {"effort": effort}
    model_kwargs["extra_body"] = extra_body
    return model_kwargs


def generation_endpoint_identity(settings: Settings) -> dict[str, Any]:
    """Return sanitized endpoint identity shared by diagnostics and evaluation."""

    profile = validate_generation_endpoint(settings)
    return {
        "endpoint_profile": profile.identifier,
        "provider": profile.provider,
        "model": settings.generation_model,
        "api_style": profile.api_style,
        "base_url_host": urlparse(profile.base_url).netloc,
        "reasoning_mode": settings.generation_reasoning,
        "temperature": (
            settings.generation_temperature if profile.supports_temperature else None
        ),
        "maximum_output_tokens": settings.generation_max_output_tokens,
        "timeout_seconds": settings.generation_timeout_seconds,
        "provider_max_retries": settings.generation_max_retries,
        "store": settings.generation_store,
        "routing_policy_id": profile.routing_policy_id,
    }

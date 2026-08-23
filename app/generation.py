"""Compatibility exports for generation contracts, policy, and infrastructure."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
from app.infrastructure.generation.langchain_structured import (
    LangChainStructuredGenerator as LangChainStructuredGenerator,
)


class LangChainOpenAIGenerator(LangChainStructuredGenerator):
    """Compatibility name preserving the historical constructor defaults."""

    def __init__(
        self,
        settings: Settings,
        *,
        model_factory: Callable[..., Any] | None = None,
        prompt_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            settings,
            system_prompt=SYSTEM_PROMPT,
            human_prompt=HUMAN_PROMPT,
            correction_text_builder=build_correction_text,
            model_factory=model_factory,
            prompt_factory=prompt_factory,
        )

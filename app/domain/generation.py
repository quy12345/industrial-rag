"""Provider-neutral structured generation contracts and ports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from app.domain.retrieval import RetrievalCandidate


class GeneratedAnswer(BaseModel):
    """Structured output whose source labels still require application validation."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    source_ids: list[str]
    insufficient_evidence: bool


@dataclass(frozen=True)
class TokenUsage:
    """Optional normalized usage metadata from a generation adapter."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None


@dataclass(frozen=True)
class EvidenceBundle:
    """Rendered untrusted evidence plus the authoritative source-label mapping."""

    text: str
    source_map: dict[str, RetrievalCandidate]

    @property
    def allowed_source_ids(self) -> tuple[str, ...]:
        return tuple(self.source_map)


@dataclass(frozen=True)
class GenerationResult:
    """One parsed generation result and optional provider usage."""

    output: GeneratedAnswer
    usage: TokenUsage | None = None


class AnswerGenerator(Protocol):
    """Generation port used by the grounded query application service."""

    def ensure_configured(self) -> None: ...

    def generate(
        self,
        *,
        question: str,
        evidence: EvidenceBundle,
        validation_errors: Sequence[str] = (),
    ) -> GenerationResult: ...

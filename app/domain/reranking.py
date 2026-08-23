"""Framework-neutral cross-encoder contracts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol


class RerankingError(ValueError):
    """Raised when candidate construction or cross-encoder output is invalid."""


@dataclass(frozen=True)
class CrossEncoderScore:
    """One model score mapped to the corresponding input candidate index."""

    candidate_index: int
    score: float


class CrossEncoder(Protocol):
    """Dependency-injection boundary used by real and fake cross-encoders."""

    def score(
        self, query: str, documents: Sequence[str], *, batch_size: int
    ) -> Iterable[CrossEncoderScore]: ...

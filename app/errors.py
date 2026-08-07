"""Domain exceptions used by the Phase 6 query pipeline."""

from __future__ import annotations


class QueryPipelineError(Exception):
    """Base class for expected query-pipeline failures."""


class RetrievalUnavailableError(QueryPipelineError):
    """Raised when Qdrant or a retrieval model cannot serve a query."""


class RerankerUnavailableError(QueryPipelineError):
    """Raised when configured cross-encoder reranking fails."""


class LLMNotConfiguredError(QueryPipelineError):
    """Raised when grounded generation has no safe provider configuration."""


class LLMUnavailableError(QueryPipelineError):
    """Raised when the configured generation provider is unavailable."""


class LLMTimeoutError(QueryPipelineError):
    """Raised when the configured generation provider times out."""


class LLMRefusalError(QueryPipelineError):
    """Raised when the provider returns a safety refusal."""


class GenerationValidationError(QueryPipelineError):
    """Raised when provider structured output cannot be parsed or validated."""

    def __init__(self, message: str, *, errors: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.errors = errors or (message,)


class CitationValidationError(QueryPipelineError):
    """Raised when generated source IDs violate the evidence contract."""

    def __init__(self, errors: list[str] | tuple[str, ...]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))

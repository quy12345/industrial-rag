"""Pydantic response and ingestion models."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    status: Literal["ok"]
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Response from the Qdrant-backed readiness check."""

    status: Literal["ok"]
    service: str
    version: str


class DocumentChunk(BaseModel):
    """JSON-serializable representation of one structure-aware document chunk."""

    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_numbers: list[int] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    content_type: str = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """One ranked chunk returned by dense similarity search."""

    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_numbers: list[int]
    headings: list[str]
    content_type: str
    score: float


class RetrievalCandidate(BaseModel):
    """One sparse, dense, or RRF-fused retrieval candidate.

    All ranks are one-based. Scores are retrieval ranking signals, never probabilities.
    """

    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_numbers: list[int]
    headings: list[str]
    content_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float
    dense_score: float | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    sparse_score: float | None = None
    sparse_rank: int | None = Field(default=None, ge=1)
    rrf_score: float | None = None
    rrf_rank: int | None = Field(default=None, ge=1)
    rerank_score: float | None = None
    rerank_rank: int | None = Field(default=None, ge=1)


class QueryRequest(BaseModel):
    """Validated request body for the grounded query endpoint."""

    model_config = ConfigDict(extra="forbid")

    question: str
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be empty")
        return normalized

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("document_id must not be empty")
        return normalized


class Citation(BaseModel):
    """Trusted citation metadata built from a retrieved Qdrant payload."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    filename: str
    page_numbers: list[int]
    headings: list[str]
    excerpt: str


class QueryResponse(BaseModel):
    """Public grounded-answer response."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    abstained: bool
    abstention_reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)

"""Pydantic response and ingestion models."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.contracts.query import Citation as Citation
from app.contracts.query import QueryRequest as QueryRequest
from app.contracts.query import QueryResponse as QueryResponse
from app.domain.documents import DocumentChunk as DocumentChunk


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

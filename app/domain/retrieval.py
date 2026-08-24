"""Framework-neutral retrieval candidate assembly policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field


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


@dataclass(frozen=True)
class QueryRetrievalResult:
    """Final ordered candidates plus independently measured stage latency."""

    candidates: list[RetrievalCandidate]
    retrieval_ms: float
    rerank_ms: float
    candidate_pool: list[RetrievalCandidate] | None = None


class QueryRetriever(Protocol):
    """Port supplying retrieved candidates to the grounded-query use case."""

    def retrieve(self, question: str, *, document_id: str | None) -> QueryRetrievalResult: ...


class DenseSearchPort(Protocol):
    """Retrieve dense results through an adapter with bound infrastructure state."""

    def search(
        self,
        query: str,
        *,
        collection_name: str,
        limit: int,
        document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]: ...


class SparseSearchPort(Protocol):
    """Retrieve sparse candidates through an adapter with bound infrastructure state."""

    def search(
        self,
        query: str,
        *,
        collection_name: str,
        limit: int,
        document_id: str | None = None,
    ) -> list[RetrievalCandidate]: ...


def dense_results_to_candidates(
    results: Sequence[RetrievedChunk],
) -> list[RetrievalCandidate]:
    """Map dense results to candidates with deterministic one-based dense ranks."""

    candidates = [
        RetrievalCandidate(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            filename=result.filename,
            text=result.text,
            page_numbers=result.page_numbers,
            headings=result.headings,
            content_type=result.content_type,
            score=result.score,
            dense_score=result.score,
        )
        for result in results
    ]
    return [
        candidate.model_copy(update={"dense_rank": rank})
        for rank, candidate in enumerate(
            sorted(
                candidates,
                key=lambda candidate: (-(candidate.dense_score or 0.0), candidate.chunk_id),
            ),
            start=1,
        )
    ]


def union_dense_sparse_candidates(
    dense_candidates: Sequence[RetrievalCandidate],
    sparse_candidates: Sequence[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    """Return a deterministic, unranked-for-fusion union without mixing raw scores."""

    merged: dict[str, RetrievalCandidate] = {}
    for candidate in dense_candidates:
        merged[candidate.chunk_id] = candidate
    for candidate in sparse_candidates:
        existing = merged.get(candidate.chunk_id)
        if existing is None:
            merged[candidate.chunk_id] = candidate
        else:
            merged[candidate.chunk_id] = existing.model_copy(
                update={
                    "sparse_score": candidate.sparse_score,
                    "sparse_rank": candidate.sparse_rank,
                }
            )
    return sorted(
        merged.values(),
        key=lambda candidate: (
            min(rank for rank in (candidate.dense_rank, candidate.sparse_rank) if rank is not None),
            candidate.chunk_id,
        ),
    )

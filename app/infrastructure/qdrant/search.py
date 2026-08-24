"""State-bound Qdrant implementations of the retrieval search ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.domain.retrieval import RetrievalCandidate, RetrievedChunk
from app.infrastructure.qdrant.dense import dense_search
from app.infrastructure.qdrant.hybrid import sparse_search


@dataclass(frozen=True)
class QdrantDenseSearcher:
    """Bind the Qdrant client, vector name, and dense model once."""

    client: Any
    vector_name: str
    embedding_model: Any

    def search(
        self,
        query: str,
        *,
        collection_name: str,
        limit: int,
        document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        return dense_search(
            self.client,
            query,
            collection_name=collection_name,
            vector_name=self.vector_name,
            embedding_model=self.embedding_model,
            limit=limit,
            document_id=document_id,
            score_threshold=score_threshold,
        )


@dataclass(frozen=True)
class QdrantSparseSearcher:
    """Bind the Qdrant client, vector name, and sparse model once."""

    client: Any
    vector_name: str
    embedding_model: Any

    def search(
        self,
        query: str,
        *,
        collection_name: str,
        limit: int,
        document_id: str | None = None,
    ) -> list[RetrievalCandidate]:
        return sparse_search(
            self.client,
            query,
            collection_name=collection_name,
            sparse_vector_name=self.vector_name,
            sparse_embedding_model=self.embedding_model,
            limit=limit,
            document_id=document_id,
        )

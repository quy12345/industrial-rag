"""Application service for safe active-corpus indexing."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.domain.documents import DocumentChunk


class IndexingSafetyError(ValueError):
    """Raised when corpus or collection safety invariants fail."""


class IndexingSettings(Protocol):
    """Settings required by the dense and hybrid indexing use case."""

    qdrant_collection: str
    qdrant_hybrid_collection: str
    dense_vector_name: str
    sparse_vector_name: str
    embedding_batch_size: int
    sparse_embedding_batch_size: int


class DenseIndexer(Protocol):
    def __call__(
        self,
        client: Any,
        chunks: Sequence[DocumentChunk],
        *,
        collection_name: str,
        vector_name: str,
        embedding_model: Any,
        embedding_batch_size: int,
        vector_size: int,
    ) -> int: ...


class HybridIndexer(Protocol):
    def __call__(
        self,
        client: Any,
        chunks: Sequence[DocumentChunk],
        *,
        collection_name: str,
        dense_vector_name: str,
        sparse_vector_name: str,
        dense_embedding_model: Any,
        sparse_embedding_model: Any,
        dense_embedding_batch_size: int,
        sparse_embedding_batch_size: int,
        dense_vector_size: int,
    ) -> int: ...


class IndexedChunkReader(Protocol):
    def __call__(
        self,
        client: Any,
        *,
        collection_name: str,
        document_id: str,
    ) -> set[str]: ...


def validate_collection_targets(
    dense_collection: str,
    hybrid_collection: str,
    *,
    protected_collections: Collection[str],
) -> None:
    """Reject protected or overlapping collection targets before external access."""

    if dense_collection in protected_collections or hybrid_collection in protected_collections:
        raise IndexingSafetyError("Active indexing refuses protected historical collections.")
    if dense_collection == hybrid_collection:
        raise IndexingSafetyError("Dense and hybrid collection names must differ.")


def validate_chunk_preview(
    chunks_by_document: Mapping[str, Sequence[DocumentChunk]],
) -> None:
    """Validate the two-document normalized chunk contract before persistence."""

    if len(chunks_by_document) != 2:
        raise IndexingSafetyError("Both ATV320 manuals must produce chunks.")
    for document_id, chunks in chunks_by_document.items():
        indices = [chunk.metadata.get("chunk_index") for chunk in chunks]
        if indices != list(range(len(chunks))):
            raise IndexingSafetyError(f"{document_id} has non-contiguous chunk indices.")
        if not all(
            chunk.filename and chunk.page_numbers and chunk.text.strip() for chunk in chunks
        ):
            raise IndexingSafetyError(f"{document_id} has incomplete citation metadata.")
        header_like = sum(1 for chunk in chunks if len(chunk.text) < 120 and not chunk.headings)
        if header_like / len(chunks) > 0.25:
            raise IndexingSafetyError(
                f"{document_id} has too many likely header/footer-only chunks."
            )


@dataclass(frozen=True)
class CorpusIndexingService:
    """Coordinate injected dense/hybrid mutation and verification operations."""

    client: Any
    settings: IndexingSettings
    dense_model: Any
    sparse_model: Any
    dense_dimension: int
    dense_indexer: DenseIndexer
    hybrid_indexer: HybridIndexer
    indexed_chunk_reader: IndexedChunkReader

    def index_once(
        self,
        chunks_by_document: Mapping[str, Sequence[DocumentChunk]],
    ) -> None:
        """Index dense then hybrid representations for each document in stable order."""

        for chunks in chunks_by_document.values():
            self.dense_indexer(
                self.client,
                chunks,
                collection_name=self.settings.qdrant_collection,
                vector_name=self.settings.dense_vector_name,
                embedding_model=self.dense_model,
                embedding_batch_size=self.settings.embedding_batch_size,
                vector_size=self.dense_dimension,
            )
            self.hybrid_indexer(
                self.client,
                chunks,
                collection_name=self.settings.qdrant_hybrid_collection,
                dense_vector_name=self.settings.dense_vector_name,
                sparse_vector_name=self.settings.sparse_vector_name,
                dense_embedding_model=self.dense_model,
                sparse_embedding_model=self.sparse_model,
                dense_embedding_batch_size=self.settings.embedding_batch_size,
                sparse_embedding_batch_size=self.settings.sparse_embedding_batch_size,
                dense_vector_size=self.dense_dimension,
            )

    def verify_index(
        self,
        chunks_by_document: Mapping[str, Sequence[DocumentChunk]],
    ) -> None:
        """Require exact total counts and per-document stable chunk-ID sets."""

        expected_total = sum(len(chunks) for chunks in chunks_by_document.values())
        for collection_name in (
            self.settings.qdrant_collection,
            self.settings.qdrant_hybrid_collection,
        ):
            if self.client.count(collection_name, exact=True).count != expected_total:
                raise IndexingSafetyError(
                    f"{collection_name} does not contain the expected total point count."
                )
            for document_id, chunks in chunks_by_document.items():
                actual = self.indexed_chunk_reader(
                    self.client,
                    collection_name=collection_name,
                    document_id=document_id,
                )
                expected = {chunk.chunk_id for chunk in chunks}
                if actual != expected:
                    raise IndexingSafetyError(
                        f"{collection_name} chunk IDs mismatch for {document_id}."
                    )

    def verify_protected_collections(
        self,
        *,
        dense_collection: str,
        hybrid_collection: str,
        expected_count: int,
    ) -> None:
        """Confirm both historical archive collections retain their frozen point counts."""

        dense_count = self.client.count(dense_collection, exact=True).count
        hybrid_count = self.client.count(hybrid_collection, exact=True).count
        if (dense_count, hybrid_count) != (expected_count, expected_count):
            raise IndexingSafetyError(
                "Protected collections are not frozen at "
                f"{expected_count}/{expected_count}: {dense_count}/{hybrid_count}"
            )

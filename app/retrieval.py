"""Compatibility facade for dense search and extracted indexing infrastructure."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
from qdrant_client import QdrantClient

from app.errors import RetrievalError as RetrievalError
from app.infrastructure.qdrant import client as qdrant_client_adapter
from app.infrastructure.qdrant import dense as dense_infrastructure
from app.infrastructure.qdrant import manifests as dense_manifests
from app.models import RetrievedChunk

create_qdrant_client = qdrant_client_adapter.create_qdrant_client
build_embedding_text = dense_infrastructure.build_embedding_text
build_point_id = dense_infrastructure.build_point_id
create_embedding_model = dense_infrastructure.create_embedding_model
ensure_dense_collection = dense_infrastructure.ensure_dense_collection
get_embedding_dimension = dense_infrastructure.get_embedding_dimension
get_indexed_chunk_ids = dense_infrastructure.get_indexed_chunk_ids
index_chunks = dense_infrastructure.index_chunks
validate_dense_collection = dense_infrastructure.validate_dense_collection
_document_filter = dense_infrastructure.document_filter
_to_float_vector = dense_infrastructure.to_float_vector
INDEX_MANIFEST_PATH = dense_manifests.INDEX_MANIFEST_PATH
validate_index_manifest = dense_manifests.validate_index_manifest
write_index_manifest = dense_manifests.write_index_manifest


def dense_search(
    client: QdrantClient,
    question: str,
    *,
    collection_name: str,
    vector_name: str,
    embedding_model: Any,
    limit: int,
    document_id: str | None = None,
    score_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """Return ranked chunks for one dense similarity query."""

    normalized_question = question.strip()
    if not normalized_question:
        raise RetrievalError("Dense search question must not be empty.")
    if limit <= 0:
        raise RetrievalError("Dense search limit must be greater than 0.")

    try:
        raw_vector = next(iter(embedding_model.query_embed(normalized_question)))
        query_vector = _to_float_vector(raw_vector)
    except StopIteration as exc:
        raise RetrievalError("Embedding model returned no query vector.") from exc
    except Exception as exc:
        raise RetrievalError(f"Failed to embed dense search question: {exc}") from exc

    query_filter = _document_filter(document_id) if document_id else None
    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )
    except Exception as exc:
        raise RetrievalError(f"Dense search failed in collection {collection_name}: {exc}") from exc

    results: list[RetrievedChunk] = []
    for point in response.points:
        try:
            results.append(_retrieved_chunk_from_payload(point.payload, point.score))
        except (KeyError, TypeError, ValidationError) as exc:
            raise RetrievalError(f"Invalid payload for Qdrant point {point.id}: {exc}") from exc
    return results


def _retrieved_chunk_from_payload(payload: Any, score: float) -> RetrievedChunk:
    """Validate a Qdrant payload and map it to a ranked chunk."""

    if not isinstance(payload, dict):
        raise TypeError("payload must be a JSON object")
    return RetrievedChunk(
        chunk_id=payload["chunk_id"],
        document_id=payload["document_id"],
        filename=payload["filename"],
        text=payload["text"],
        page_numbers=payload["page_numbers"],
        headings=payload["headings"],
        content_type=payload["content_type"],
        score=score,
    )

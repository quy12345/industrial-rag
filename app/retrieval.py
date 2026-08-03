"""Dense embedding, Qdrant indexing, and similarity search utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import UUID, uuid5

from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from app.config import Settings
from app.models import DocumentChunk, RetrievedChunk

POINT_NAMESPACE = UUID("91bf9b94-7641-5d4f-9e2a-d76c9d358c7d")


class RetrievalError(Exception):
    """Raised when dense indexing or retrieval fails."""


def build_embedding_text(chunk: DocumentChunk) -> str:
    """Build deterministic passage text with optional heading context."""

    parts: list[str] = []
    if chunk.headings:
        parts.append(f"Section: {' > '.join(chunk.headings)}")
    parts.append(f"Content:\n{chunk.text}")
    return "\n".join(parts)


def build_point_id(chunk_id: str) -> str:
    """Build a deterministic Qdrant-compatible UUID from a chunk ID."""

    return str(uuid5(POINT_NAMESPACE, chunk_id))


def create_embedding_model(model_name: str) -> Any:
    """Validate and initialize a supported FastEmbed text model."""

    from fastembed import TextEmbedding

    supported_names = {
        model["model"]
        for model in TextEmbedding.list_supported_models()
        if isinstance(model.get("model"), str)
    }
    if model_name not in supported_names:
        raise RetrievalError(
            f"Embedding model is not supported by the installed FastEmbed version: {model_name}"
        )

    try:
        return TextEmbedding(model_name=model_name)
    except Exception as exc:
        raise RetrievalError(f"Failed to initialize embedding model {model_name}: {exc}") from exc


def get_embedding_dimension(embedding_model: Any) -> int:
    """Determine vector dimension from one non-persisted probe embedding."""

    try:
        vector = next(iter(embedding_model.passage_embed(["dimension probe"])))
        dimension = len(vector)
    except (StopIteration, TypeError, ValueError) as exc:
        raise RetrievalError("Embedding model did not produce a valid dimension probe.") from exc
    except Exception as exc:
        raise RetrievalError(f"Failed to determine embedding dimension: {exc}") from exc

    if dimension <= 0:
        raise RetrievalError("Embedding model produced an empty dimension probe.")
    return dimension


def create_qdrant_client(settings: Settings) -> QdrantClient:
    """Create a Qdrant client and confirm the configured endpoint is reachable."""

    endpoint = f"{settings.qdrant_url}:{settings.qdrant_port}"
    try:
        client = QdrantClient(url=settings.qdrant_url, port=settings.qdrant_port)
        client.get_collections()
        return client
    except Exception as exc:
        raise RetrievalError(f"Qdrant is unavailable at {endpoint}") from exc


def ensure_dense_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    vector_name: str,
    vector_size: int,
) -> None:
    """Create or validate a named cosine-vector collection."""

    if vector_size <= 0:
        raise RetrievalError("Vector size must be greater than 0.")

    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )
            return

        collection = client.get_collection(collection_name)
    except Exception as exc:
        raise RetrievalError(
            f"Failed to create or inspect Qdrant collection {collection_name}: {exc}"
        ) from exc

    vectors = collection.config.params.vectors
    if not isinstance(vectors, dict) or vector_name not in vectors:
        raise RetrievalError(
            f"Collection {collection_name} does not define named vector {vector_name}."
        )

    vector_config = vectors[vector_name]
    if vector_config.size != vector_size:
        raise RetrievalError(
            f"Collection {collection_name} uses vector size {vector_config.size}, "
            f"but model produces {vector_size}."
        )
    if vector_config.distance != models.Distance.COSINE:
        raise RetrievalError(
            f"Collection {collection_name} vector {vector_name} must use cosine distance."
        )


def index_chunks(
    client: QdrantClient,
    chunks: Sequence[DocumentChunk],
    *,
    collection_name: str,
    vector_name: str,
    embedding_model: Any,
    embedding_batch_size: int,
    vector_size: int | None = None,
) -> int:
    """Embed and replace document chunks in a shared dense collection."""

    if not chunks:
        raise RetrievalError("Cannot index an empty chunk list.")
    if embedding_batch_size <= 0:
        raise RetrievalError("Embedding batch size must be greater than 0.")

    expected_size = vector_size or get_embedding_dimension(embedding_model)
    point_batches: list[list[models.PointStruct]] = []

    try:
        for chunk_batch in _batched(chunks, embedding_batch_size):
            embedding_texts = [build_embedding_text(chunk) for chunk in chunk_batch]
            vectors = list(embedding_model.passage_embed(embedding_texts))
            if len(vectors) != len(chunk_batch):
                raise RetrievalError(
                    "Embedding model returned a different number of vectors than input chunks."
                )

            points: list[models.PointStruct] = []
            for chunk, raw_vector in zip(chunk_batch, vectors, strict=True):
                vector = _to_float_vector(raw_vector)
                if len(vector) != expected_size:
                    raise RetrievalError(
                        f"Chunk {chunk.chunk_id} produced vector size {len(vector)}, "
                        f"expected {expected_size}."
                    )
                points.append(
                    models.PointStruct(
                        id=build_point_id(chunk.chunk_id),
                        vector={vector_name: vector},
                        payload=_build_payload(chunk),
                    )
                )
            point_batches.append(points)
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(f"Failed to embed document chunks: {exc}") from exc

    ensure_dense_collection(
        client,
        collection_name=collection_name,
        vector_name=vector_name,
        vector_size=expected_size,
    )

    document_ids = {chunk.document_id for chunk in chunks}
    try:
        for document_id in document_ids:
            client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=_document_filter(document_id),
                ),
                wait=True,
            )
        for points in point_batches:
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
    except Exception as exc:
        raise RetrievalError(
            f"Failed to update Qdrant collection {collection_name}: {exc}"
        ) from exc

    return sum(len(points) for points in point_batches)


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


def _batched(
    values: Sequence[DocumentChunk],
    batch_size: int,
) -> Iterable[Sequence[DocumentChunk]]:
    """Yield deterministic slices without copying the full sequence."""

    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _to_float_vector(raw_vector: Any) -> list[float]:
    """Convert NumPy or list-like vectors to Qdrant-compatible floats."""

    return [float(value) for value in raw_vector]


def _build_payload(chunk: DocumentChunk) -> dict[str, Any]:
    """Build a flat, JSON-serializable citation-ready payload."""

    source_path = str(chunk.metadata.get("source_path", chunk.filename))
    if Path(source_path).is_absolute() or PureWindowsPath(source_path).is_absolute():
        source_path = PureWindowsPath(source_path).name

    character_count = chunk.metadata.get("character_count", len(chunk.text))
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "text": chunk.text,
        "page_numbers": list(chunk.page_numbers),
        "headings": list(chunk.headings),
        "content_type": chunk.content_type,
        "source_path": source_path,
        "character_count": int(character_count),
    }


def _document_filter(document_id: str) -> models.Filter:
    """Build a server-side Qdrant document payload filter."""

    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            )
        ]
    )


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

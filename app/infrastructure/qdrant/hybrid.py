"""Sparse embedding and hybrid Qdrant indexing infrastructure."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from app.domain.documents import DocumentChunk
from app.errors import RetrievalError
from app.infrastructure.qdrant.dense import (
    batched,
    build_embedding_text,
    build_payload,
    build_point_id,
    scroll_document_point_ids,
    to_float_vector,
)


def create_sparse_embedding_model(
    model_name: str,
    cache_dir: str | None,
    *,
    disable_stemmer: bool,
    k: float,
    b: float,
    avg_len: float,
) -> Any:
    """Create the installed FastEmbed sparse BM25 model with explicit settings."""

    from fastembed import SparseTextEmbedding

    supported_names = {
        model["model"]
        for model in SparseTextEmbedding.list_supported_models()
        if isinstance(model.get("model"), str)
    }
    if model_name not in supported_names:
        raise RetrievalError(
            f"Sparse embedding model is not supported by installed FastEmbed: {model_name}"
        )
    try:
        return SparseTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            disable_stemmer=disable_stemmer,
            k=k,
            b=b,
            avg_len=avg_len,
        )
    except Exception as exc:
        raise RetrievalError(f"Failed to initialize sparse model {model_name}: {exc}") from exc


def compute_bm25_average_length(
    sparse_model: Any,
    chunks: Sequence[DocumentChunk],
) -> float:
    """Compute BM25 document length from FastEmbed's active preprocessing."""

    if not chunks:
        raise RetrievalError("Cannot calculate BM25 average length for no chunks.")
    backend = getattr(sparse_model, "model", None)
    tokenizer = getattr(backend, "tokenizer", None)
    stem = getattr(backend, "_stem", None)
    if tokenizer is None or not callable(stem):
        raise RetrievalError(
            "Installed sparse model does not expose the FastEmbed BM25 tokenizer needed "
            "to calculate avg_len."
        )
    try:
        from fastembed.sparse.bm25 import remove_non_alphanumeric

        lengths = []
        for chunk in chunks:
            normalized = remove_non_alphanumeric(build_embedding_text(chunk))
            lengths.append(len(stem(tokenizer.tokenize(normalized))))
    except Exception as exc:
        raise RetrievalError(f"Failed to calculate BM25 average length: {exc}") from exc

    average = sum(lengths) / len(lengths)
    if average <= 0:
        raise RetrievalError("BM25 average length must be greater than zero.")
    return average


def ensure_hybrid_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    dense_vector_name: str,
    dense_vector_size: int,
    sparse_vector_name: str,
) -> None:
    """Create or validate a v2 dense+sparse collection without touching v1."""

    if dense_vector_size <= 0:
        raise RetrievalError("Dense vector size must be greater than 0.")
    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    dense_vector_name: models.VectorParams(
                        size=dense_vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    sparse_vector_name: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        else:
            validate_hybrid_collection(
                client,
                collection_name=collection_name,
                dense_vector_name=dense_vector_name,
                dense_vector_size=dense_vector_size,
                sparse_vector_name=sparse_vector_name,
            )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(
            f"Failed to create or inspect hybrid collection {collection_name}: {exc}"
        ) from exc


def validate_hybrid_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    dense_vector_name: str,
    dense_vector_size: int,
    sparse_vector_name: str,
) -> None:
    """Validate the hybrid collection schema without creating or replacing it."""

    try:
        if not client.collection_exists(collection_name):
            raise RetrievalError(f"Hybrid Qdrant collection {collection_name} does not exist.")
        collection = client.get_collection(collection_name)
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(
            f"Failed to inspect hybrid collection {collection_name}: {exc}"
        ) from exc

    vectors = collection.config.params.vectors
    if not isinstance(vectors, dict) or dense_vector_name not in vectors:
        raise RetrievalError(
            f"Collection {collection_name} does not define dense vector {dense_vector_name}."
        )
    dense_config = vectors[dense_vector_name]
    if dense_config.size != dense_vector_size:
        raise RetrievalError(
            f"Collection {collection_name} uses dense vector size {dense_config.size}, "
            f"but model produces {dense_vector_size}."
        )
    if dense_config.distance != models.Distance.COSINE:
        raise RetrievalError(f"Collection {collection_name} dense vector must use cosine distance.")

    sparse_vectors = collection.config.params.sparse_vectors or {}
    if sparse_vector_name not in sparse_vectors:
        raise RetrievalError(
            f"Collection {collection_name} does not define sparse vector {sparse_vector_name}."
        )
    if sparse_vectors[sparse_vector_name].modifier != models.Modifier.IDF:
        raise RetrievalError(
            f"Collection {collection_name} sparse vector {sparse_vector_name} must use IDF."
        )

    payload_schema = getattr(collection, "payload_schema", None)
    if payload_schema:
        document_index = payload_schema.get("document_id")
        if document_index is None:
            raise RetrievalError(f"Collection {collection_name} has no document_id payload index.")
        data_type = getattr(document_index, "data_type", None)
        if data_type not in {None, models.PayloadSchemaType.KEYWORD}:
            raise RetrievalError(
                f"Collection {collection_name} document_id payload index must be keyword."
            )


def index_hybrid_chunks(
    client: QdrantClient,
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
) -> int:
    """Replace document vectors only after both embedding passes succeed."""

    if not chunks:
        raise RetrievalError("Cannot index an empty chunk list.")
    if dense_embedding_batch_size <= 0 or sparse_embedding_batch_size <= 0:
        raise RetrievalError("Embedding batch sizes must be greater than 0.")

    dense_vectors = _embed_dense_chunks(
        chunks, dense_embedding_model, dense_embedding_batch_size, dense_vector_size
    )
    sparse_vectors = _embed_sparse_chunks(
        chunks, sparse_embedding_model, sparse_embedding_batch_size
    )
    point_batches: list[list[models.PointStruct]] = []
    for chunk_batch in batched(chunks, dense_embedding_batch_size):
        points: list[models.PointStruct] = []
        for chunk in chunk_batch:
            points.append(
                models.PointStruct(
                    id=build_point_id(chunk.chunk_id),
                    vector={
                        dense_vector_name: dense_vectors[chunk.chunk_id],
                        sparse_vector_name: sparse_vectors[chunk.chunk_id],
                    },
                    payload=build_payload(chunk),
                )
            )
        point_batches.append(points)

    ensure_hybrid_collection(
        client,
        collection_name=collection_name,
        dense_vector_name=dense_vector_name,
        dense_vector_size=dense_vector_size,
        sparse_vector_name=sparse_vector_name,
    )
    document_ids = {chunk.document_id for chunk in chunks}
    existing_by_document = {
        document_id: scroll_document_point_ids(client, collection_name, document_id)
        for document_id in document_ids
    }
    new_by_document: dict[str, set[str]] = {document_id: set() for document_id in document_ids}
    for chunk in chunks:
        new_by_document[chunk.document_id].add(build_point_id(chunk.chunk_id))

    try:
        for points in point_batches:
            client.upsert(collection_name=collection_name, points=points, wait=True)
        for document_id in document_ids:
            stale = existing_by_document[document_id] - new_by_document[document_id]
            if stale:
                client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=sorted(stale)),
                    wait=True,
                )
    except Exception as exc:
        raise RetrievalError(
            f"Failed to update hybrid Qdrant collection {collection_name}: {exc}"
        ) from exc
    return len(chunks)


def to_sparse_vector(raw_vector: Any) -> models.SparseVector:
    """Convert a sparse model result into a validated Qdrant vector."""

    indices = [int(value) for value in getattr(raw_vector, "indices", [])]
    values = [float(value) for value in getattr(raw_vector, "values", [])]
    if not indices or not values:
        raise RetrievalError("Sparse embedding model produced an empty sparse vector.")
    if len(indices) != len(values):
        raise RetrievalError("Sparse embedding indices and values have different lengths.")
    return models.SparseVector(indices=indices, values=values)


def _embed_dense_chunks(
    chunks: Sequence[DocumentChunk], model: Any, batch_size: int, vector_size: int
) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    try:
        for batch in batched(chunks, batch_size):
            raw_vectors = list(
                model.passage_embed([build_embedding_text(chunk) for chunk in batch])
            )
            if len(raw_vectors) != len(batch):
                raise RetrievalError(
                    "Dense model returned a different number of vectors than chunks."
                )
            for chunk, raw_vector in zip(batch, raw_vectors, strict=True):
                vector = to_float_vector(raw_vector)
                if len(vector) != vector_size:
                    raise RetrievalError(
                        f"Chunk {chunk.chunk_id} produced dense vector size {len(vector)}, "
                        f"expected {vector_size}."
                    )
                vectors[chunk.chunk_id] = vector
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(f"Failed to embed dense hybrid passages: {exc}") from exc
    return vectors


def _embed_sparse_chunks(
    chunks: Sequence[DocumentChunk], model: Any, batch_size: int
) -> dict[str, models.SparseVector]:
    vectors: dict[str, models.SparseVector] = {}
    try:
        for batch in batched(chunks, batch_size):
            raw_vectors = list(
                model.passage_embed([build_embedding_text(chunk) for chunk in batch])
            )
            if len(raw_vectors) != len(batch):
                raise RetrievalError(
                    "Sparse model returned a different number of vectors than chunks."
                )
            for chunk, raw_vector in zip(batch, raw_vectors, strict=True):
                vectors[chunk.chunk_id] = to_sparse_vector(raw_vector)
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(f"Failed to embed sparse hybrid passages: {exc}") from exc
    return vectors

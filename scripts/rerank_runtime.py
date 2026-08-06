"""Shared validated runtime construction for Phase 5 integration CLIs."""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.evaluation import EvaluationError
from app.hybrid_retrieval import (
    HYBRID_INDEX_MANIFEST_PATH,
    create_sparse_embedding_model,
    validate_hybrid_collection,
    validate_hybrid_index_manifest,
)
from app.models import DocumentChunk
from app.reranking import FastEmbedCrossEncoder, RerankPipeline, fastembed_model_metadata
from app.retrieval import (
    INDEX_MANIFEST_PATH,
    create_embedding_model,
    create_qdrant_client,
    get_embedding_dimension,
    get_indexed_chunk_ids,
    validate_dense_collection,
    validate_index_manifest,
)


def build_rerank_runtime(
    settings: Settings,
    frozen_metadata: dict[str, Any],
    chunks: list[DocumentChunk],
) -> tuple[RerankPipeline, dict[str, Any]]:
    """Validate both immutable collections and construct lazy Phase 5 dependencies."""

    dense_model = create_embedding_model(
        settings.embedding_model, cache_dir=settings.embedding_cache_dir
    )
    dense_dimension = get_embedding_dimension(dense_model)
    client = create_qdrant_client(settings)
    validate_index_manifest(
        INDEX_MANIFEST_PATH,
        collection_name=settings.qdrant_collection,
        vector_name=settings.dense_vector_name,
        embedding_model=settings.embedding_model,
        embedding_dimension=dense_dimension,
    )
    validate_dense_collection(
        client,
        collection_name=settings.qdrant_collection,
        vector_name=settings.dense_vector_name,
        vector_size=dense_dimension,
    )
    hybrid_manifest = validate_hybrid_index_manifest(
        HYBRID_INDEX_MANIFEST_PATH,
        settings=settings,
        dense_dimension=dense_dimension,
        frozen_chunk_set=frozen_metadata,
    )
    validate_hybrid_collection(
        client,
        collection_name=settings.qdrant_hybrid_collection,
        dense_vector_name=settings.dense_vector_name,
        dense_vector_size=dense_dimension,
        sparse_vector_name=settings.sparse_vector_name,
    )
    _validate_frozen_ids(client, settings.qdrant_collection, chunks)
    _validate_frozen_ids(client, settings.qdrant_hybrid_collection, chunks)
    sparse_model = create_sparse_embedding_model(
        settings.sparse_model,
        settings.embedding_cache_dir,
        disable_stemmer=settings.bm25_disable_stemmer,
        k=settings.bm25_k,
        b=settings.bm25_b,
        avg_len=float(hybrid_manifest["bm25_avg_len"]),
    )
    cross_encoder = FastEmbedCrossEncoder(
        settings.rerank_model,
        cache_dir=settings.rerank_cache_dir or settings.embedding_cache_dir,
    )
    pipeline = RerankPipeline(
        client=client,
        dense_embedding_model=dense_model,
        sparse_embedding_model=sparse_model,
        cross_encoder=cross_encoder,
        dense_collection=settings.qdrant_collection,
        hybrid_collection=settings.qdrant_hybrid_collection,
        dense_vector_name=settings.dense_vector_name,
        sparse_vector_name=settings.sparse_vector_name,
        dense_candidate_limit=settings.dense_candidate_limit,
        sparse_candidate_limit=settings.sparse_candidate_limit,
        rrf_k=settings.rrf_k,
        rerank_batch_size=settings.rerank_batch_size,
    )
    return pipeline, {
        "collections": {
            "dense_v1": settings.qdrant_collection,
            "hybrid_v2": settings.qdrant_hybrid_collection,
        },
        "dense_model": settings.embedding_model,
        "dense_dimension": dense_dimension,
        "sparse_model": settings.sparse_model,
        "bm25_avg_len": hybrid_manifest["bm25_avg_len"],
        "rrf_k": settings.rrf_k,
        "rerank_model": settings.rerank_model,
        "rerank_model_metadata": fastembed_model_metadata(settings.rerank_model),
    }


def _validate_frozen_ids(client: Any, collection_name: str, chunks: list[DocumentChunk]) -> None:
    grouped: dict[str, set[str]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document_id, set()).add(chunk.chunk_id)
    for document_id, expected in grouped.items():
        actual = get_indexed_chunk_ids(
            client, collection_name=collection_name, document_id=document_id
        )
        if actual != expected:
            raise EvaluationError(
                f"Collection {collection_name} differs from frozen chunks for {document_id}."
            )

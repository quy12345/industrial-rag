"""Shared validated runtime construction for Phase 5 integration CLIs."""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.evaluation import EvaluationError
from app.hybrid_retrieval import HYBRID_INDEX_MANIFEST_PATH, validate_hybrid_index_manifest
from app.models import DocumentChunk
from app.reranking import RerankPipeline, fastembed_model_metadata
from app.retrieval import (
    INDEX_MANIFEST_PATH,
    validate_index_manifest,
)
from app.retrieval_runtime import (
    FrozenRetrievalContract,
    build_union_rerank_runtime,
)

ARCHIVED_PHASE6_RETRIEVAL_CONTRACT = FrozenRetrievalContract(
    document_id="manual-77d5dae4c2c5",
    chunk_count=99,
    chunk_ids_sha256="bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be",
    dense_collection="industrial_manual_chunks",
    hybrid_collection="industrial_manual_chunks_v2",
    bm25_avg_len=72.83838383838383,
    dense_candidate_limit=20,
    sparse_candidate_limit=20,
    rrf_k=60,
)


def build_rerank_runtime(
    settings: Settings,
    frozen_metadata: dict[str, Any],
    chunks: list[DocumentChunk],
) -> tuple[RerankPipeline, dict[str, Any]]:
    """Validate both immutable collections and construct lazy Phase 5 dependencies."""

    contract = ARCHIVED_PHASE6_RETRIEVAL_CONTRACT
    if frozen_metadata != {
        "chunk_count": contract.chunk_count,
        "document_ids": [contract.document_id],
        "chunk_ids_sha256": contract.chunk_ids_sha256,
    }:
        raise EvaluationError("Supplied chunks differ from the frozen Phase 6 contract.")
    runtime_settings = settings.model_copy(
        update={
            "qdrant_collection": contract.dense_collection,
            "qdrant_hybrid_collection": contract.hybrid_collection,
            "dense_vector_name": contract.dense_vector_name,
            "sparse_vector_name": contract.sparse_vector_name,
            "embedding_model": contract.dense_model,
            "sparse_model": contract.sparse_model,
            "rerank_model": contract.rerank_model,
            "dense_candidate_limit": contract.dense_candidate_limit,
            "sparse_candidate_limit": contract.sparse_candidate_limit,
            "rrf_k": contract.rrf_k,
            "bm25_k": contract.bm25_k,
            "bm25_b": contract.bm25_b,
            "bm25_avg_len": contract.bm25_avg_len,
            "bm25_disable_stemmer": contract.bm25_disable_stemmer,
            "rerank_deduplicate_content": False,
            "retrieval_strategy": "union",
            "rerank_enabled": True,
        }
    )
    validate_index_manifest(
        INDEX_MANIFEST_PATH,
        collection_name=runtime_settings.qdrant_collection,
        vector_name=runtime_settings.dense_vector_name,
        embedding_model=runtime_settings.embedding_model,
        embedding_dimension=contract.dense_dimension,
    )
    hybrid_manifest = validate_hybrid_index_manifest(
        HYBRID_INDEX_MANIFEST_PATH,
        settings=runtime_settings,
        dense_dimension=contract.dense_dimension,
        frozen_chunk_set=frozen_metadata,
    )
    pipeline, metadata = build_union_rerank_runtime(runtime_settings, contract=contract)
    metadata["bm25_avg_len"] = hybrid_manifest["bm25_avg_len"]
    metadata["rerank_model_metadata"] = fastembed_model_metadata(runtime_settings.rerank_model)
    return pipeline, metadata

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
    PHASE6_RETRIEVAL_CONTRACT,
    build_union_rerank_runtime,
)


def build_rerank_runtime(
    settings: Settings,
    frozen_metadata: dict[str, Any],
    chunks: list[DocumentChunk],
) -> tuple[RerankPipeline, dict[str, Any]]:
    """Validate both immutable collections and construct lazy Phase 5 dependencies."""

    contract = PHASE6_RETRIEVAL_CONTRACT
    if frozen_metadata != {
        "chunk_count": contract.chunk_count,
        "document_ids": [contract.document_id],
        "chunk_ids_sha256": contract.chunk_ids_sha256,
    }:
        raise EvaluationError("Supplied chunks differ from the frozen Phase 6 contract.")
    validate_index_manifest(
        INDEX_MANIFEST_PATH,
        collection_name=settings.qdrant_collection,
        vector_name=settings.dense_vector_name,
        embedding_model=settings.embedding_model,
        embedding_dimension=contract.dense_dimension,
    )
    hybrid_manifest = validate_hybrid_index_manifest(
        HYBRID_INDEX_MANIFEST_PATH,
        settings=settings,
        dense_dimension=contract.dense_dimension,
        frozen_chunk_set=frozen_metadata,
    )
    runtime_settings = settings.model_copy(
        update={"retrieval_strategy": "union", "rerank_enabled": True}
    )
    pipeline, metadata = build_union_rerank_runtime(runtime_settings)
    metadata["bm25_avg_len"] = hybrid_manifest["bm25_avg_len"]
    metadata["rerank_model_metadata"] = fastembed_model_metadata(settings.rerank_model)
    return pipeline, metadata

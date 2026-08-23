"""Compatibility facade for sparse search and client-side RRF fusion."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

import app.domain.policies.fusion as fusion_policy
import app.infrastructure.qdrant.dense as dense_infrastructure
import app.infrastructure.qdrant.hybrid as hybrid_infrastructure
from app.errors import RetrievalError
from app.infrastructure.qdrant import manifests as index_manifests
from app.models import RetrievalCandidate

HYBRID_INDEX_MANIFEST_PATH = index_manifests.HYBRID_INDEX_MANIFEST_PATH
HYBRID_SCHEMA_VERSION = index_manifests.HYBRID_SCHEMA_VERSION
SPARSE_MODIFIER = index_manifests.SPARSE_MODIFIER
create_sparse_embedding_model = hybrid_infrastructure.create_sparse_embedding_model
compute_bm25_average_length = hybrid_infrastructure.compute_bm25_average_length
ensure_hybrid_collection = hybrid_infrastructure.ensure_hybrid_collection
validate_hybrid_collection = hybrid_infrastructure.validate_hybrid_collection
index_hybrid_chunks = hybrid_infrastructure.index_hybrid_chunks
write_hybrid_index_manifest = index_manifests.write_hybrid_index_manifest
validate_hybrid_index_manifest = index_manifests.validate_hybrid_index_manifest
_document_filter = dense_infrastructure.document_filter
_to_sparse_vector = hybrid_infrastructure.to_sparse_vector
_runtime_versions = index_manifests.runtime_versions
dense_search = dense_infrastructure.dense_search
sparse_search = hybrid_infrastructure.sparse_search
_candidate_from_dense = hybrid_infrastructure._candidate_from_dense
_candidate_from_payload = hybrid_infrastructure._candidate_from_payload
_assign_component_ranks = hybrid_infrastructure._assign_component_ranks
fuse_rrf = fusion_policy.fuse_rrf


def hybrid_search(
    client: QdrantClient,
    question: str,
    *,
    collection_name: str,
    dense_vector_name: str,
    sparse_vector_name: str,
    dense_embedding_model: Any,
    sparse_embedding_model: Any,
    dense_candidate_limit: int,
    sparse_candidate_limit: int,
    final_limit: int,
    rrf_k: int,
    document_id: str | None = None,
) -> list[RetrievalCandidate]:
    """Retrieve dense and sparse candidates then apply deterministic client-side RRF."""

    if final_limit <= 0:
        raise RetrievalError("Hybrid final limit must be greater than 0.")
    dense_results = dense_search(
        client,
        question,
        collection_name=collection_name,
        vector_name=dense_vector_name,
        embedding_model=dense_embedding_model,
        limit=dense_candidate_limit,
        document_id=document_id,
    )
    dense_candidates = _assign_component_ranks(
        [_candidate_from_dense(result) for result in dense_results], component="dense"
    )
    sparse_candidates = sparse_search(
        client,
        question,
        collection_name=collection_name,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding_model=sparse_embedding_model,
        limit=sparse_candidate_limit,
        document_id=document_id,
    )
    return fuse_rrf(dense_candidates, sparse_candidates, rrf_k=rrf_k, final_limit=final_limit)

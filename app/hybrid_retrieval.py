"""Compatibility facade for sparse search and client-side RRF fusion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError
from qdrant_client import QdrantClient

import app.domain.policies.fusion as fusion_policy
from app.errors import RetrievalError
from app.infrastructure.qdrant import dense as dense_infrastructure
from app.infrastructure.qdrant import hybrid as hybrid_infrastructure
from app.infrastructure.qdrant import manifests as index_manifests
from app.models import RetrievalCandidate, RetrievedChunk
from app.retrieval import dense_search

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
fuse_rrf = fusion_policy.fuse_rrf


def sparse_search(
    client: QdrantClient,
    question: str,
    *,
    collection_name: str,
    sparse_vector_name: str,
    sparse_embedding_model: Any,
    limit: int,
    document_id: str | None = None,
) -> list[RetrievalCandidate]:
    """Return sparse BM25 candidates with one-based deterministic ranks."""

    normalized_question = question.strip()
    if not normalized_question:
        raise RetrievalError("Sparse search question must not be empty.")
    if limit <= 0:
        raise RetrievalError("Sparse search limit must be greater than 0.")
    try:
        raw_vector = next(iter(sparse_embedding_model.query_embed(normalized_question)))
        query_vector = _to_sparse_vector(raw_vector)
    except StopIteration as exc:
        raise RetrievalError("Sparse embedding model returned no query vector.") from exc
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(f"Failed to embed sparse search question: {exc}") from exc

    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=sparse_vector_name,
            query_filter=_document_filter(document_id) if document_id else None,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        raise RetrievalError(
            f"Sparse search failed in collection {collection_name}: {exc}"
        ) from exc

    candidates = []
    for point in response.points:
        try:
            candidates.append(
                _candidate_from_payload(point.payload, sparse_score=float(point.score))
            )
        except (KeyError, TypeError, ValidationError) as exc:
            raise RetrievalError(f"Invalid payload for Qdrant point {point.id}: {exc}") from exc
    return _assign_component_ranks(candidates, component="sparse")


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


def _candidate_from_dense(result: RetrievedChunk) -> RetrievalCandidate:
    return RetrievalCandidate(
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


def _candidate_from_payload(payload: Any, *, sparse_score: float) -> RetrievalCandidate:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a JSON object")
    metadata = {key: payload[key] for key in ("source_path", "character_count") if key in payload}
    return RetrievalCandidate(
        chunk_id=payload["chunk_id"],
        document_id=payload["document_id"],
        filename=payload["filename"],
        text=payload["text"],
        page_numbers=payload["page_numbers"],
        headings=payload["headings"],
        content_type=payload["content_type"],
        metadata=metadata,
        score=sparse_score,
        sparse_score=sparse_score,
    )


def _assign_component_ranks(
    candidates: Sequence[RetrievalCandidate], *, component: str
) -> list[RetrievalCandidate]:
    if component not in {"dense", "sparse"}:
        raise ValueError(f"Unknown retrieval component: {component}")
    score_field = f"{component}_score"
    rank_field = f"{component}_rank"
    ordered = sorted(
        candidates,
        key=lambda candidate: (-(getattr(candidate, score_field) or 0.0), candidate.chunk_id),
    )
    return [
        candidate.model_copy(update={rank_field: rank})
        for rank, candidate in enumerate(ordered, start=1)
    ]

"""Artifact-independent Phase 7 retrieval runtime construction."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Any, Protocol

from app import config as runtime_config
from app.domain import retrieval_contracts
from app.domain.policies.query_analysis import (
    QUERY_EXPANSION_PROFILE,
    augment_vietnamese_technical_query,
)
from app.errors import RerankerUnavailableError, RetrievalUnavailableError
from app.hybrid_retrieval import (
    create_sparse_embedding_model,
    sparse_search,
    validate_hybrid_collection,
)
from app.models import RetrievalCandidate
from app.reranking import FastEmbedCrossEncoder, RerankingError, RerankPipeline
from app.retrieval import (
    RetrievalError,
    create_embedding_model,
    create_qdrant_client,
    get_embedding_dimension,
    get_indexed_chunk_ids,
    validate_dense_collection,
)

# Compatibility exports keep existing runtime and script imports stable until R07.
Settings = runtime_config.Settings
FrozenDocumentContext = retrieval_contracts.FrozenDocumentContext
FrozenRetrievalContract = retrieval_contracts.FrozenRetrievalContract
PHASE7_RETRIEVAL_CONTRACT = retrieval_contracts.PHASE7_RETRIEVAL_CONTRACT
resolve_retrieval_runtime = runtime_config.resolve_retrieval_runtime
_validate_settings = runtime_config.validate_retrieval_settings


@dataclass(frozen=True)
class QueryRetrievalResult:
    """Final ordered candidates plus independently measured stage latency."""

    candidates: list[RetrievalCandidate]
    retrieval_ms: float
    rerank_ms: float
    candidate_pool: list[RetrievalCandidate] | None = None


class QueryRetriever(Protocol):
    """Injectable retrieval boundary used by QueryService."""

    def retrieve(self, question: str, *, document_id: str | None) -> QueryRetrievalResult: ...


class UnionRerankRetriever:
    """Accuracy-first dense/sparse union followed by cross-encoder reranking."""

    def __init__(self, pipeline: RerankPipeline) -> None:
        self.pipeline = pipeline

    def retrieve(self, question: str, *, document_id: str | None) -> QueryRetrievalResult:
        try:
            execution = self.pipeline.search(question, strategy="union", document_id=document_id)
        except RetrievalError as exc:
            raise RetrievalUnavailableError("Union retrieval failed.") from exc
        except RerankingError as exc:
            raise RerankerUnavailableError("Cross-encoder reranking failed.") from exc
        retrieval_ms = sum(
            value
            for name, value in execution.stage_latency_ms.items()
            if name
            in {
                "dense_retrieval",
                "sparse_retrieval",
                "content_deduplication",
                "union_preparation",
                "query_expansion",
                "rrf_pruning",
                "query_role_inference",
                "coverage_preserving_weighted_rrf",
                "role_aware_rank_fusion",
            }
        )
        return QueryRetrievalResult(
            candidates=execution.candidates_after_rerank,
            retrieval_ms=retrieval_ms,
            rerank_ms=execution.stage_latency_ms.get("rerank", 0.0),
            candidate_pool=list(
                getattr(execution, "candidates_before_rerank", execution.candidates_after_rerank)
            ),
        )


class SparseRollbackRetriever:
    """Explicit no-reranker operational rollback using the v2 sparse vector."""

    def __init__(self, *, client: Any, sparse_embedding_model: Any, settings: Settings) -> None:
        self.client = client
        self.sparse_embedding_model = sparse_embedding_model
        self.settings = settings

    def retrieve(self, question: str, *, document_id: str | None) -> QueryRetrievalResult:
        started = perf_counter()
        try:
            candidates = sparse_search(
                self.client,
                question,
                collection_name=self.settings.qdrant_hybrid_collection,
                sparse_vector_name=self.settings.sparse_vector_name,
                sparse_embedding_model=self.sparse_embedding_model,
                limit=self.settings.sparse_candidate_limit,
                document_id=document_id,
            )
        except RetrievalError as exc:
            raise RetrievalUnavailableError("Sparse retrieval failed.") from exc
        return QueryRetrievalResult(
            candidates=candidates,
            retrieval_ms=(perf_counter() - started) * 1000,
            rerank_ms=0.0,
            candidate_pool=list(candidates),
        )


class LazyQueryRetriever:
    """Construct and cache heavy retrieval dependencies on their first real use."""

    def __init__(self, factory: Callable[[], QueryRetriever]) -> None:
        self._factory = factory
        self._delegate: QueryRetriever | None = None
        self._lock = Lock()

    def retrieve(self, question: str, *, document_id: str | None) -> QueryRetrievalResult:
        return self._get_delegate().retrieve(question, document_id=document_id)

    def _get_delegate(self) -> QueryRetriever:
        if self._delegate is None:
            with self._lock:
                if self._delegate is None:
                    self._delegate = self._factory()
        return self._delegate


def build_query_retriever(
    settings: Settings,
    *,
    contract: FrozenRetrievalContract,
) -> QueryRetriever:
    """Validate the selected runtime and build only dependencies that strategy needs."""

    _validate_settings(settings, contract)
    if settings.retrieval_strategy == "union":
        pipeline, _ = build_union_rerank_runtime(settings, contract=contract)
        return UnionRerankRetriever(pipeline)
    try:
        client = create_qdrant_client(settings)
        validate_hybrid_collection(
            client,
            collection_name=settings.qdrant_hybrid_collection,
            dense_vector_name=settings.dense_vector_name,
            dense_vector_size=contract.dense_dimension,
            sparse_vector_name=settings.sparse_vector_name,
        )
        _validate_frozen_collection(client, settings.qdrant_hybrid_collection, contract)
        sparse_model = create_sparse_embedding_model(
            settings.sparse_model,
            settings.embedding_cache_dir,
            disable_stemmer=settings.bm25_disable_stemmer,
            k=settings.bm25_k,
            b=settings.bm25_b,
            avg_len=contract.bm25_avg_len,
        )
        return SparseRollbackRetriever(
            client=client, sparse_embedding_model=sparse_model, settings=settings
        )
    except RetrievalError as exc:
        raise RetrievalUnavailableError("Unable to initialize frozen retrieval runtime.") from exc
    except RerankingError as exc:
        raise RerankerUnavailableError("Unable to initialize reranker runtime.") from exc


def build_union_rerank_runtime(
    settings: Settings,
    *,
    contract: FrozenRetrievalContract,
) -> tuple[RerankPipeline, dict[str, Any]]:
    """Build one explicit frozen union pipeline without host artifact dependencies."""

    _validate_settings(settings, contract)
    if settings.retrieval_strategy != "union" or not settings.rerank_enabled:
        raise RetrievalUnavailableError("Union runtime requires configured reranking.")
    try:
        client = create_qdrant_client(settings)
        validate_hybrid_collection(
            client,
            collection_name=settings.qdrant_hybrid_collection,
            dense_vector_name=settings.dense_vector_name,
            dense_vector_size=contract.dense_dimension,
            sparse_vector_name=settings.sparse_vector_name,
        )
        validate_dense_collection(
            client,
            collection_name=settings.qdrant_collection,
            vector_name=settings.dense_vector_name,
            vector_size=contract.dense_dimension,
        )
        _validate_frozen_collection(client, settings.qdrant_hybrid_collection, contract)
        _validate_frozen_collection(client, settings.qdrant_collection, contract)
        dense_model = create_embedding_model(
            settings.embedding_model, cache_dir=settings.embedding_cache_dir
        )
        dimension = get_embedding_dimension(dense_model)
        if dimension != contract.dense_dimension:
            raise RetrievalError(
                f"Dense model dimension {dimension} does not match frozen dimension "
                f"{contract.dense_dimension}."
            )
        sparse_model = create_sparse_embedding_model(
            settings.sparse_model,
            settings.embedding_cache_dir,
            disable_stemmer=settings.bm25_disable_stemmer,
            k=settings.bm25_k,
            b=settings.bm25_b,
            avg_len=contract.bm25_avg_len,
        )
        pipeline = RerankPipeline(
            client=client,
            dense_embedding_model=dense_model,
            sparse_embedding_model=sparse_model,
            cross_encoder=FastEmbedCrossEncoder(
                settings.rerank_model,
                cache_dir=settings.rerank_cache_dir or settings.embedding_cache_dir,
                threads=(
                    contract.frozen_rerank_threads
                    if contract.freeze_rerank_threads
                    else settings.rerank_threads
                ),
            ),
            dense_collection=settings.qdrant_collection,
            hybrid_collection=settings.qdrant_hybrid_collection,
            dense_vector_name=settings.dense_vector_name,
            sparse_vector_name=settings.sparse_vector_name,
            dense_candidate_limit=settings.dense_candidate_limit,
            sparse_candidate_limit=settings.sparse_candidate_limit,
            rrf_k=settings.rrf_k,
            rerank_batch_size=contract.frozen_rerank_batch_size or settings.rerank_batch_size,
            deduplicate_content=settings.rerank_deduplicate_content,
            document_contexts=contract.document_context_by_id,
            sparse_query_transform=(
                _expand_phase7_query
                if contract.query_expansion_profile == QUERY_EXPANSION_PROFILE
                else None
            ),
            union_rrf_prune_limit=contract.union_rrf_prune_limit,
            phase7_fusion_profile=contract.phase7_fusion_profile,
        )
        return pipeline, {
            "collections": {
                "dense_v1": settings.qdrant_collection,
                "hybrid_v2": settings.qdrant_hybrid_collection,
            },
            "dense_model": settings.embedding_model,
            "dense_dimension": dimension,
            "sparse_model": settings.sparse_model,
            "bm25_avg_len": contract.bm25_avg_len,
            "rrf_k": settings.rrf_k,
            "rerank_model": settings.rerank_model,
            "rerank_batch_size": contract.frozen_rerank_batch_size or settings.rerank_batch_size,
            "rerank_threads": (
                contract.frozen_rerank_threads
                if contract.freeze_rerank_threads
                else settings.rerank_threads
            ),
            "deduplicate_content": settings.rerank_deduplicate_content,
            "query_expansion_profile": contract.query_expansion_profile,
            "union_rrf_prune_limit": contract.union_rrf_prune_limit,
            "phase7_fusion_profile": (
                None
                if contract.phase7_fusion_profile is None
                else {
                    "name": contract.phase7_fusion_profile.name,
                    "rrf_k": contract.phase7_fusion_profile.rrf_k,
                    "dense_weight": contract.phase7_fusion_profile.dense_weight,
                    "sparse_weight": contract.phase7_fusion_profile.sparse_weight,
                    "fusion_role_multiplier": contract.phase7_fusion_profile.fusion_role_multiplier,
                    "dense_reserve": contract.phase7_fusion_profile.dense_reserve,
                    "sparse_reserve": contract.phase7_fusion_profile.sparse_reserve,
                    "max_candidates": contract.phase7_fusion_profile.max_candidates,
                    "post_rerank_role_multiplier": (
                        contract.phase7_fusion_profile.post_rerank_role_multiplier
                    ),
                    "post_rerank_rrf_multiplier": (
                        contract.phase7_fusion_profile.post_rerank_rrf_multiplier
                    ),
                    "post_rerank_rank_offset": (
                        contract.phase7_fusion_profile.post_rerank_rank_offset
                    ),
                    "post_rerank_confidence_mode": (
                        contract.phase7_fusion_profile.post_rerank_confidence_mode
                    ),
                    "list_completeness_enabled": (
                        contract.phase7_fusion_profile.list_completeness_enabled
                    ),
                    "relation_list_completeness_enabled": (
                        contract.phase7_fusion_profile.relation_list_completeness_enabled
                    ),
                }
            ),
        }
    except RetrievalUnavailableError:
        raise
    except RetrievalError as exc:
        raise RetrievalUnavailableError("Unable to initialize frozen retrieval runtime.") from exc
    except RerankingError as exc:
        raise RerankerUnavailableError("Unable to initialize reranker runtime.") from exc


def validate_frozen_runtime(
    client: Any,
    *,
    collection_names: tuple[str, ...],
    contract: FrozenRetrievalContract,
) -> None:
    """Public read-only validation helper used by scripts and integration checks."""

    for collection_name in collection_names:
        _validate_frozen_collection(client, collection_name, contract)


def _expand_phase7_query(question: str) -> str:
    """Apply the frozen lexical profile to sparse retrieval only."""

    expanded, _ = augment_vietnamese_technical_query(question)
    return expanded


def _validate_frozen_collection(
    client: Any, collection_name: str, contract: FrozenRetrievalContract
) -> None:
    try:
        collection = client.get_collection(collection_name)
        point_count = collection.points_count
    except Exception as exc:
        raise RetrievalError(f"Unable to inspect collection {collection_name}.") from exc
    if point_count != contract.chunk_count:
        raise RetrievalError(
            f"Collection {collection_name} has {point_count} points; "
            f"expected {contract.chunk_count}."
        )
    chunk_ids: set[str] = set()
    for document_id in contract.indexed_document_ids:
        chunk_ids.update(
            get_indexed_chunk_ids(
                client,
                collection_name=collection_name,
                document_id=document_id,
            )
        )
    fingerprint = hashlib.sha256("\n".join(sorted(chunk_ids)).encode("utf-8")).hexdigest()
    if len(chunk_ids) != contract.chunk_count or fingerprint != contract.chunk_ids_sha256:
        raise RetrievalError(f"Collection {collection_name} does not match the frozen chunk set.")

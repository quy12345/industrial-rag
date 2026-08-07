"""Artifact-independent Phase 6 retrieval runtime construction."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Any, Protocol

from app.config import Settings
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


@dataclass(frozen=True)
class FrozenRetrievalContract:
    """Immutable index identity required by the Phase 6 runtime."""

    document_id: str = "manual-77d5dae4c2c5"
    chunk_count: int = 99
    chunk_ids_sha256: str = (
        "bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be"
    )
    dense_collection: str = "industrial_manual_chunks"
    hybrid_collection: str = "industrial_manual_chunks_v2"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    dense_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    dense_dimension: int = 384
    dense_candidate_limit: int = 20
    sparse_candidate_limit: int = 20
    rrf_k: int = 60
    bm25_k: float = 1.2
    bm25_b: float = 0.75
    bm25_avg_len: float = 72.83838383838383
    bm25_disable_stemmer: bool = True


PHASE6_RETRIEVAL_CONTRACT = FrozenRetrievalContract()


@dataclass(frozen=True)
class QueryRetrievalResult:
    """Final ordered candidates plus independently measured stage latency."""

    candidates: list[RetrievalCandidate]
    retrieval_ms: float
    rerank_ms: float


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
            if name in {"dense_retrieval", "sparse_retrieval", "union_preparation"}
        )
        return QueryRetrievalResult(
            candidates=execution.candidates_after_rerank,
            retrieval_ms=retrieval_ms,
            rerank_ms=execution.stage_latency_ms.get("rerank", 0.0),
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
    contract: FrozenRetrievalContract = PHASE6_RETRIEVAL_CONTRACT,
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
    contract: FrozenRetrievalContract = PHASE6_RETRIEVAL_CONTRACT,
) -> tuple[RerankPipeline, dict[str, Any]]:
    """Build the shared Phase 5/6 union pipeline without host artifact dependencies."""

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
            ),
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
            "dense_dimension": dimension,
            "sparse_model": settings.sparse_model,
            "bm25_avg_len": contract.bm25_avg_len,
            "rrf_k": settings.rrf_k,
            "rerank_model": settings.rerank_model,
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
    contract: FrozenRetrievalContract = PHASE6_RETRIEVAL_CONTRACT,
) -> None:
    """Public read-only validation helper used by scripts and integration checks."""

    for collection_name in collection_names:
        _validate_frozen_collection(client, collection_name, contract)


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
    chunk_ids = get_indexed_chunk_ids(
        client,
        collection_name=collection_name,
        document_id=contract.document_id,
    )
    fingerprint = hashlib.sha256("\n".join(sorted(chunk_ids)).encode("utf-8")).hexdigest()
    if len(chunk_ids) != contract.chunk_count or fingerprint != contract.chunk_ids_sha256:
        raise RetrievalError(
            f"Collection {collection_name} does not match the frozen Phase 6 chunk set."
        )


def _validate_settings(settings: Settings, contract: FrozenRetrievalContract) -> None:
    expected = {
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
        "bm25_disable_stemmer": contract.bm25_disable_stemmer,
    }
    mismatches = [
        name
        for name, expected_value in expected.items()
        if getattr(settings, name) != expected_value
    ]
    if mismatches:
        raise RetrievalUnavailableError(
            "Runtime settings differ from the frozen retrieval contract: "
            + ", ".join(sorted(mismatches))
        )
    combination = (settings.retrieval_strategy, settings.rerank_enabled)
    if combination not in {("union", True), ("sparse", False)}:
        raise RetrievalUnavailableError(
            "Supported runtime combinations are union+rerank or sparse without reranking."
        )

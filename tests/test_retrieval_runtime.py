"""Offline tests for lazy Phase 7 retrieval composition and frozen identity."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.composition import retrieval as runtime
from app.composition.retrieval import (
    LazyQueryRetriever,
    UnionRerankRetriever,
    _expand_phase7_query,
    _validate_frozen_collection,
)
from app.config import Settings, resolve_retrieval_runtime, validate_retrieval_settings
from app.domain.reranking import RerankingError
from app.domain.retrieval import QueryRetrievalResult, RetrievalCandidate
from app.domain.retrieval_contracts import PHASE7_RETRIEVAL_CONTRACT
from app.errors import RerankerUnavailableError, RetrievalError, RetrievalUnavailableError


def _candidate(chunk_id="a"):
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="manual-a",
        filename="manual.pdf",
        text="evidence",
        page_numbers=[1],
        headings=[],
        content_type="text",
        score=1.0,
    )


def test_lazy_retriever_builds_once_and_preserves_document_filter() -> None:
    calls = []

    class Delegate:
        def retrieve(self, question, *, document_id):
            calls.append((question, document_id))
            return QueryRetrievalResult([_candidate()], 1, 2)

    builds = []
    lazy = LazyQueryRetriever(lambda: builds.append(True) or Delegate())
    lazy.retrieve("q1", document_id="manual-a")
    lazy.retrieve("q2", document_id=None)
    assert builds == [True]
    assert calls == [("q1", "manual-a"), ("q2", None)]


def test_union_adapter_preserves_full_order_and_stage_timings() -> None:
    class Pipeline:
        def search(self, question, *, strategy, document_id):
            assert (strategy, document_id) == ("union", "manual-a")
            return SimpleNamespace(
                candidates_after_rerank=[_candidate("b"), _candidate("a")],
                stage_latency_ms={
                    "dense_retrieval": 2.0,
                    "sparse_retrieval": 1.0,
                    "union_preparation": 0.5,
                    "content_deduplication": 0.25,
                    "rerank": 4.0,
                    "total": 7.5,
                },
            )

    result = UnionRerankRetriever(Pipeline()).retrieve("q", document_id="manual-a")
    assert [item.chunk_id for item in result.candidates] == ["b", "a"]
    assert result.retrieval_ms == 3.75
    assert result.rerank_ms == 4.0


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RetrievalError("down"), RetrievalUnavailableError),
        (RerankingError("bad"), RerankerUnavailableError),
    ],
)
def test_union_adapter_does_not_fallback_on_failures(error, expected) -> None:
    class Pipeline:
        def search(self, *args, **kwargs):
            raise error

    with pytest.raises(expected):
        UnionRerankRetriever(Pipeline()).retrieve("q", document_id=None)


def test_sparse_rollback_uses_bound_searcher_and_preserves_error_mapping() -> None:
    calls: list[tuple[str, str, int, str | None]] = []

    class Searcher:
        def search(self, query, *, collection_name, limit, document_id=None):
            calls.append((query, collection_name, limit, document_id))
            return [_candidate("sparse")]

    settings, _ = resolve_retrieval_runtime(Settings())
    settings = settings.model_copy(
        update={"retrieval_strategy": "sparse", "rerank_enabled": False}
    )
    retriever = runtime.SparseRollbackRetriever(searcher=Searcher(), settings=settings)

    result = retriever.retrieve("q", document_id="manual-a")

    assert [candidate.chunk_id for candidate in result.candidates] == ["sparse"]
    assert result.rerank_ms == 0.0
    assert calls == [("q", settings.qdrant_hybrid_collection, 40, "manual-a")]

    class FailingSearcher:
        @staticmethod
        def search(*args, **kwargs):
            raise RetrievalError("down")

    with pytest.raises(RetrievalUnavailableError, match="Sparse retrieval failed"):
        runtime.SparseRollbackRetriever(searcher=FailingSearcher(), settings=settings).retrieve(
            "q", document_id=None
        )


def test_runtime_settings_accept_only_default_and_explicit_rollback() -> None:
    settings, contract = resolve_retrieval_runtime(Settings())
    validate_retrieval_settings(settings, contract)
    validate_retrieval_settings(
        settings.model_copy(update={"retrieval_strategy": "sparse", "rerank_enabled": False}),
        contract,
    )
    with pytest.raises(RetrievalUnavailableError, match="combinations"):
        validate_retrieval_settings(
            settings.model_copy(update={"retrieval_strategy": "sparse", "rerank_enabled": True}),
            contract,
        )
    with pytest.raises(RetrievalUnavailableError, match="frozen"):
        validate_retrieval_settings(
            settings.model_copy(update={"dense_candidate_limit": 61}), contract
        )


def test_runtime_profile_is_phase7_only_and_overrides_mutable_values() -> None:
    phase7_settings, phase7_contract = resolve_retrieval_runtime(
        Settings(
            qdrant_collection="ignored-dense",
            qdrant_hybrid_collection="ignored-hybrid",
            dense_candidate_limit=999,
            sparse_candidate_limit=999,
            rrf_k=999,
        )
    )
    assert phase7_contract is PHASE7_RETRIEVAL_CONTRACT
    assert phase7_settings.qdrant_collection == "industrial_manual_phase7_dense_v1"
    assert phase7_settings.qdrant_hybrid_collection == "industrial_manual_phase7_hybrid_v1"
    assert phase7_settings.dense_candidate_limit == 60
    assert phase7_settings.sparse_candidate_limit == 40
    assert phase7_settings.rrf_k == 40
    assert phase7_settings.bm25_avg_len == 81.33599709407919
    assert phase7_settings.rerank_deduplicate_content is True

    with pytest.raises(ValidationError, match="retrieval_profile"):
        Settings(retrieval_profile="phase6")


def test_runtime_metadata_preserves_the_exact_frozen_profile() -> None:
    settings, contract = resolve_retrieval_runtime(Settings())

    assert runtime._runtime_metadata(settings, contract, dense_dimension=384) == {
        "collections": {
            "dense_v1": "industrial_manual_phase7_dense_v1",
            "hybrid_v2": "industrial_manual_phase7_hybrid_v1",
        },
        "dense_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dense_dimension": 384,
        "sparse_model": "Qdrant/bm25",
        "bm25_avg_len": 81.33599709407919,
        "rrf_k": 40,
        "rerank_model": "jinaai/jina-reranker-v2-base-multilingual",
        "rerank_batch_size": 8,
        "rerank_threads": None,
        "deduplicate_content": True,
        "query_expansion_profile": "vi_technical_glossary_v1",
        "union_rrf_prune_limit": 30,
        "phase7_fusion_profile": {
            "name": (
                "weighted_rrf_k40_s1.25_frole0.1_prole0.5_offset40_"
                "strong_and_weak_d5_s24_relation_list_v1"
            ),
            "rrf_k": 40,
            "dense_weight": 1.0,
            "sparse_weight": 1.25,
            "fusion_role_multiplier": 0.1,
            "dense_reserve": 5,
            "sparse_reserve": 24,
            "max_candidates": 30,
            "post_rerank_role_multiplier": 0.5,
            "post_rerank_rrf_multiplier": 0.0,
            "post_rerank_rank_offset": 40,
            "post_rerank_confidence_mode": "strong_and_weak",
            "list_completeness_enabled": False,
            "relation_list_completeness_enabled": True,
        },
    }


def test_frozen_collection_rejects_count_and_hash_mismatch(monkeypatch) -> None:
    contract = replace(
        PHASE7_RETRIEVAL_CONTRACT,
        document_id="test-doc",
        document_ids=(),
        document_contexts=(),
        chunk_count=2,
        chunk_ids_sha256=hashlib.sha256(b"a\nb").hexdigest(),
    )

    class Client:
        def __init__(self, count):
            self.count = count

        def get_collection(self, name):
            return SimpleNamespace(points_count=self.count)

    with pytest.raises(RetrievalError, match="points"):
        _validate_frozen_collection(Client(1), "v1", contract)
    monkeypatch.setattr(
        "app.composition.retrieval.get_indexed_chunk_ids", lambda *args, **kwargs: {"wrong"}
    )
    with pytest.raises(RetrievalError, match="frozen"):
        _validate_frozen_collection(Client(2), "v1", contract)


def test_multi_document_frozen_contract_hashes_the_union_of_stable_ids(monkeypatch) -> None:
    contract = replace(
        PHASE7_RETRIEVAL_CONTRACT,
        document_id="a-doc",
        document_ids=("a-doc", "b-doc"),
        document_contexts=(),
        chunk_count=2,
        chunk_ids_sha256=hashlib.sha256(b"a\nb").hexdigest(),
    )

    class Client:
        def get_collection(self, name):
            return SimpleNamespace(points_count=2)

    monkeypatch.setattr(
        "app.composition.retrieval.get_indexed_chunk_ids",
        lambda *args, document_id, **kwargs: {"a"} if document_id == "a-doc" else {"b"},
    )
    _validate_frozen_collection(Client(), "phase7", contract)
    assert PHASE7_RETRIEVAL_CONTRACT.chunk_count == 2753
    assert (
        PHASE7_RETRIEVAL_CONTRACT.document_context_by_id[PHASE7_RETRIEVAL_CONTRACT.document_ids[0]][
            "document_role"
        ]
        == "installation"
    )
    assert PHASE7_RETRIEVAL_CONTRACT.dense_candidate_limit == 60
    assert PHASE7_RETRIEVAL_CONTRACT.sparse_candidate_limit == 40
    assert PHASE7_RETRIEVAL_CONTRACT.union_rrf_prune_limit == 30
    assert PHASE7_RETRIEVAL_CONTRACT.rrf_k == 40
    assert PHASE7_RETRIEVAL_CONTRACT.phase7_fusion_profile is not None
    assert PHASE7_RETRIEVAL_CONTRACT.phase7_fusion_profile.name == (
        "weighted_rrf_k40_s1.25_frole0.1_prole0.5_offset40_"
        "strong_and_weak_d5_s24_relation_list_v1"
    )
    assert (
        PHASE7_RETRIEVAL_CONTRACT.phase7_fusion_profile.relation_list_completeness_enabled
        is True
    )
    assert PHASE7_RETRIEVAL_CONTRACT.frozen_rerank_batch_size == 8
    assert PHASE7_RETRIEVAL_CONTRACT.freeze_rerank_threads is True
    assert _expand_phase7_query("Phím MODE chuyển nhóm menu") != ("Phím MODE chuyển nhóm menu")


def test_importing_canonical_runtime_does_not_construct_models() -> None:
    assert PHASE7_RETRIEVAL_CONTRACT.chunk_count == 2753
    assert not hasattr(runtime, "PHASE6_RETRIEVAL_CONTRACT")


def test_legacy_runtime_anchor_retains_the_frozen_contract() -> None:
    import app.retrieval_runtime as legacy_runtime

    assert legacy_runtime.PHASE7_RETRIEVAL_CONTRACT is PHASE7_RETRIEVAL_CONTRACT
    assert legacy_runtime._expand_phase7_query("MODE menu") == _expand_phase7_query("MODE menu")


def test_runtime_composition_injects_canonical_search_adapters(monkeypatch) -> None:
    settings, contract = resolve_retrieval_runtime(Settings())
    captured: dict[str, object] = {}
    pipeline = object()
    dense_searcher = object()
    sparse_searcher = object()

    monkeypatch.setattr(runtime, "create_qdrant_client", lambda settings: object())
    monkeypatch.setattr(runtime, "validate_hybrid_collection", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "validate_dense_collection", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "_validate_frozen_collection", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "create_embedding_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime, "get_embedding_dimension", lambda model: contract.dense_dimension)
    monkeypatch.setattr(runtime, "create_sparse_embedding_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime, "FastEmbedCrossEncoder", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime, "QdrantDenseSearcher", lambda *args, **kwargs: dense_searcher)
    monkeypatch.setattr(runtime, "QdrantSparseSearcher", lambda *args, **kwargs: sparse_searcher)

    class PipelineFactory:
        @classmethod
        def from_searchers(cls, **kwargs):
            captured.update(kwargs)
            return pipeline

    monkeypatch.setattr(runtime, "RerankPipeline", PipelineFactory)

    built, _ = runtime.build_union_rerank_runtime(settings, contract=contract)

    assert built is pipeline
    assert captured["dense_searcher"] is dense_searcher
    assert captured["sparse_searcher"] is sparse_searcher
    assert "client" not in captured
    assert "dense_embedding_model" not in captured
    assert "sparse_embedding_model" not in captured

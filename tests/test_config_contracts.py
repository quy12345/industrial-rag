"""Characterization for canonical settings and the frozen Phase 7 contract."""

from __future__ import annotations

from dataclasses import asdict

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings, resolve_retrieval_runtime
from app.domain.retrieval_contracts import (
    PHASE7_RETRIEVAL_CONTRACT,
    FrozenRetrievalContract,
)
from app.errors import RetrievalUnavailableError

INSTALLATION_DOCUMENT_ID = "atv320-installation-manual-en-nve41289-09-c181b4d7f11b"
PROGRAMMING_DOCUMENT_ID = "atv320-programming-manual-en-nve41295-06-f5e9bb48167a"


def test_settings_precedence_and_cached_identity(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://environment-qdrant")
    assert Settings(_env_file=None).qdrant_url == "http://environment-qdrant"
    assert Settings(qdrant_url="http://explicit-qdrant", _env_file=None).qdrant_url == (
        "http://explicit-qdrant"
    )

    get_settings.cache_clear()
    try:
        first = get_settings()
        second = get_settings()
        assert first is second
        assert first.qdrant_url == "http://environment-qdrant"
    finally:
        get_settings.cache_clear()


def test_empty_environment_value_keeps_the_canonical_default(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "")
    assert Settings(_env_file=None).qdrant_url == "http://localhost"


def test_phase7_contract_snapshot_is_exact_and_immutable() -> None:
    assert asdict(PHASE7_RETRIEVAL_CONTRACT) == {
        "document_id": INSTALLATION_DOCUMENT_ID,
        "chunk_count": 2753,
        "chunk_ids_sha256": (
            "2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d"
        ),
        "dense_collection": "industrial_manual_phase7_dense_v1",
        "hybrid_collection": "industrial_manual_phase7_hybrid_v1",
        "bm25_avg_len": 81.33599709407919,
        "dense_candidate_limit": 60,
        "sparse_candidate_limit": 40,
        "rrf_k": 40,
        "dense_vector_name": "dense",
        "sparse_vector_name": "sparse",
        "dense_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sparse_model": "Qdrant/bm25",
        "rerank_model": "jinaai/jina-reranker-v2-base-multilingual",
        "dense_dimension": 384,
        "bm25_k": 1.2,
        "bm25_b": 0.75,
        "bm25_disable_stemmer": True,
        "document_ids": (INSTALLATION_DOCUMENT_ID, PROGRAMMING_DOCUMENT_ID),
        "document_contexts": (
            {
                "document_id": INSTALLATION_DOCUMENT_ID,
                "document_title": (
                    "Altivar Machine ATV320 Variable Speed Drives Installation Manual"
                ),
                "document_role": "installation",
            },
            {
                "document_id": PROGRAMMING_DOCUMENT_ID,
                "document_title": (
                    "Altivar Machine ATV320 Variable Speed Drives Programming Manual"
                ),
                "document_role": "programming",
            },
        ),
        "union_rrf_prune_limit": 30,
        "query_expansion_profile": "vi_technical_glossary_v1",
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
        "frozen_rerank_batch_size": 8,
        "freeze_rerank_threads": True,
        "frozen_rerank_threads": None,
    }

    with pytest.raises(AttributeError):
        PHASE7_RETRIEVAL_CONTRACT.chunk_count = 1  # type: ignore[misc]


def test_profile_resolution_is_atomic_and_preserves_non_profile_settings() -> None:
    source = Settings(
        qdrant_url="http://qdrant.internal",
        qdrant_collection="ignored-dense",
        qdrant_hybrid_collection="ignored-hybrid",
        embedding_model="ignored-dense-model",
        sparse_model="ignored-sparse-model",
        rerank_model="ignored-reranker",
        dense_candidate_limit=999,
        sparse_candidate_limit=999,
        rrf_k=999,
        api_auth_enabled=True,
    )

    resolved, contract = resolve_retrieval_runtime(source)

    assert contract is PHASE7_RETRIEVAL_CONTRACT
    assert resolved is not source
    assert resolved.qdrant_url == "http://qdrant.internal"
    assert resolved.api_auth_enabled is True
    assert resolved.qdrant_collection == contract.dense_collection
    assert resolved.qdrant_hybrid_collection == contract.hybrid_collection
    assert resolved.embedding_model == contract.dense_model
    assert resolved.sparse_model == contract.sparse_model
    assert resolved.rerank_model == contract.rerank_model
    assert resolved.dense_candidate_limit == contract.dense_candidate_limit
    assert resolved.sparse_candidate_limit == contract.sparse_candidate_limit
    assert resolved.rrf_k == contract.rrf_k
    assert resolved.bm25_avg_len == contract.bm25_avg_len
    assert resolved.rerank_deduplicate_content is True


@pytest.mark.parametrize(
    ("strategy", "rerank_enabled"),
    [("union", True), ("sparse", False)],
)
def test_supported_runtime_matrix_is_preserved(strategy, rerank_enabled) -> None:
    resolved, _ = resolve_retrieval_runtime(
        Settings(retrieval_strategy=strategy, rerank_enabled=rerank_enabled)
    )
    assert (resolved.retrieval_strategy, resolved.rerank_enabled) == (
        strategy,
        rerank_enabled,
    )


@pytest.mark.parametrize(
    ("strategy", "rerank_enabled"),
    [("union", False), ("sparse", True)],
)
def test_unsupported_runtime_matrix_fails_without_fallback(strategy, rerank_enabled) -> None:
    with pytest.raises(RetrievalUnavailableError, match="combinations"):
        resolve_retrieval_runtime(
            Settings(retrieval_strategy=strategy, rerank_enabled=rerank_enabled)
        )


def test_retired_phase6_profile_is_rejected_by_settings() -> None:
    with pytest.raises(ValidationError, match="retrieval_profile"):
        Settings(retrieval_profile="phase6")  # type: ignore[arg-type]


def test_retrieval_runtime_compatibility_exports_are_identity_preserving() -> None:
    import app.retrieval_runtime as compatibility

    assert compatibility.FrozenRetrievalContract is FrozenRetrievalContract
    assert compatibility.PHASE7_RETRIEVAL_CONTRACT is PHASE7_RETRIEVAL_CONTRACT
    assert compatibility.resolve_retrieval_runtime is resolve_retrieval_runtime

"""Offline tests for lazy Phase 6 retrieval composition and frozen identity."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.errors import RerankerUnavailableError, RetrievalUnavailableError
from app.models import RetrievalCandidate
from app.reranking import RerankingError
from app.retrieval import RetrievalError
from app.retrieval_runtime import (
    PHASE6_RETRIEVAL_CONTRACT,
    PHASE7_RETRIEVAL_CONTRACT,
    FrozenRetrievalContract,
    LazyQueryRetriever,
    QueryRetrievalResult,
    UnionRerankRetriever,
    _expand_phase7_query,
    _validate_frozen_collection,
    _validate_settings,
)


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


def test_runtime_settings_accept_only_default_and_explicit_rollback() -> None:
    contract = PHASE6_RETRIEVAL_CONTRACT
    _validate_settings(Settings(), contract)
    _validate_settings(Settings(retrieval_strategy="sparse", rerank_enabled=False), contract)
    with pytest.raises(RetrievalUnavailableError, match="combinations"):
        _validate_settings(Settings(retrieval_strategy="sparse", rerank_enabled=True), contract)
    with pytest.raises(RetrievalUnavailableError, match="frozen"):
        _validate_settings(Settings(dense_candidate_limit=21), contract)


def test_frozen_collection_rejects_count_and_hash_mismatch(monkeypatch) -> None:
    contract = PHASE6_RETRIEVAL_CONTRACT

    class Client:
        def __init__(self, count):
            self.count = count

        def get_collection(self, name):
            return SimpleNamespace(points_count=self.count)

    with pytest.raises(RetrievalError, match="points"):
        _validate_frozen_collection(Client(98), "v1", contract)
    monkeypatch.setattr(
        "app.retrieval_runtime.get_indexed_chunk_ids", lambda *args, **kwargs: {"wrong"}
    )
    with pytest.raises(RetrievalError, match="frozen"):
        _validate_frozen_collection(Client(99), "v1", contract)


def test_multi_document_frozen_contract_hashes_the_union_of_stable_ids(monkeypatch) -> None:
    contract = FrozenRetrievalContract(
        document_id="a-doc",
        document_ids=("a-doc", "b-doc"),
        chunk_count=2,
        chunk_ids_sha256=hashlib.sha256(b"a\nb").hexdigest(),
    )

    class Client:
        def get_collection(self, name):
            return SimpleNamespace(points_count=2)

    monkeypatch.setattr(
        "app.retrieval_runtime.get_indexed_chunk_ids",
        lambda *args, document_id, **kwargs: {"a"} if document_id == "a-doc" else {"b"},
    )
    _validate_frozen_collection(Client(), "phase7", contract)
    assert PHASE7_RETRIEVAL_CONTRACT.chunk_count == 2753
    assert PHASE7_RETRIEVAL_CONTRACT.document_context_by_id[
        PHASE7_RETRIEVAL_CONTRACT.document_ids[0]
    ]["document_role"] == "installation"
    assert PHASE7_RETRIEVAL_CONTRACT.dense_candidate_limit == 60
    assert PHASE7_RETRIEVAL_CONTRACT.sparse_candidate_limit == 40
    assert PHASE7_RETRIEVAL_CONTRACT.union_rrf_prune_limit == 30
    assert PHASE7_RETRIEVAL_CONTRACT.rrf_k == 40
    assert PHASE7_RETRIEVAL_CONTRACT.phase7_fusion_profile is not None
    assert PHASE7_RETRIEVAL_CONTRACT.phase7_fusion_profile.name == (
        "weighted_rrf_k40_s1.25_role0.1_d5_s24"
    )
    assert _expand_phase7_query("Phím MODE chuyển nhóm menu") != (
        "Phím MODE chuyển nhóm menu"
    )


def test_importing_runtime_does_not_construct_models() -> None:
    import app.retrieval_runtime as runtime

    assert runtime.PHASE6_RETRIEVAL_CONTRACT.chunk_count == 99

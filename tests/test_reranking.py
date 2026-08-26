"""Offline tests for multilingual cross-encoder reranking and diagnostics."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from types import ModuleType

import pytest
from pydantic import ValidationError

from app.application.reranking_service import (
    CANDIDATE_TEXT_FORMAT,
    RerankExecution,
    RerankPipeline,
    build_candidate_pool,
    build_candidate_text,
    deduplicate_candidates_by_content,
    rerank_candidates,
)
from app.config import Settings
from app.domain.reranking import CrossEncoderScore, RerankingError
from app.domain.retrieval import RetrievalCandidate, RetrievedChunk
from app.infrastructure.models.reranker import (
    FastEmbedCrossEncoder,
    fastembed_model_metadata,
)
from evaluation.reranking import (
    classify_rerank_failure,
    evaluate_reranked_cases,
)
from evaluation.retrieval import EvaluationCase


class FakeCrossEncoder:
    def __init__(self, outputs: list[CrossEncoderScore] | None = None) -> None:
        self.outputs = outputs
        self.calls: list[tuple[str, list[str], int]] = []

    def score(self, query, documents, *, batch_size):
        self.calls.append((query, list(documents), batch_size))
        return self.outputs or [
            CrossEncoderScore(candidate_index=index, score=float(index))
            for index in range(len(documents))
        ]


class FailingCrossEncoder:
    def score(self, query, documents, *, batch_size):
        raise RuntimeError("model exploded")


class DenseSearchStub:
    def __init__(
        self,
        callback: Callable[..., list[RetrievedChunk]],
    ) -> None:
        self._callback = callback

    def search(
        self,
        query: str,
        *,
        collection_name: str,
        limit: int,
        document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        return self._callback(
            query,
            collection_name=collection_name,
            limit=limit,
            document_id=document_id,
            score_threshold=score_threshold,
        )


class SparseSearchStub:
    def __init__(
        self,
        callback: Callable[..., list[RetrievalCandidate]],
    ) -> None:
        self._callback = callback

    def search(
        self,
        query: str,
        *,
        collection_name: str,
        limit: int,
        document_id: str | None = None,
    ) -> list[RetrievalCandidate]:
        return self._callback(
            query,
            collection_name=collection_name,
            limit=limit,
            document_id=document_id,
        )


def _candidate(
    chunk_id: str,
    *,
    sparse_rank: int | None = None,
    dense_rank: int | None = None,
    rrf_rank: int | None = None,
    document_id: str = "manual-a",
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        filename="manual.pdf",
        text=f"raw {chunk_id}",
        page_numbers=[1],
        headings=["Safety", "Limits"],
        content_type="text",
        metadata={"marker": chunk_id},
        score=0.5,
        dense_score=0.8 if dense_rank is not None else None,
        dense_rank=dense_rank,
        sparse_score=3.0 if sparse_rank is not None else None,
        sparse_rank=sparse_rank,
        rrf_score=0.02 if rrf_rank is not None else None,
        rrf_rank=rrf_rank,
    )


def _dense(chunk_id: str, score: float = 0.8, document_id: str = "manual-a") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename="manual.pdf",
        text=f"raw {chunk_id}",
        page_numbers=[1],
        headings=["Safety"],
        content_type="text",
        score=score,
    )


def _case(case_id: str = "case", relevant: str = "evidence", *, critical: bool = False):
    return EvaluationCase(
        id=case_id,
        language="vi",
        question=case_id,
        relevant_chunk_ids=[relevant],
        expected_phrases=["raw"],
        expected_pages=[1],
        category="semantic_paraphrase",
        critical=critical,
        document_id="manual-a",
    )


def test_settings_and_candidate_model_are_backward_compatible() -> None:
    settings = Settings()
    assert settings.rerank_model == "jinaai/jina-reranker-v2-base-multilingual"
    assert settings.rerank_candidate_strategy is None
    assert settings.rerank_batch_size == 16
    assert settings.rerank_deduplicate_content is False
    assert _candidate("a", sparse_rank=1).rerank_score is None
    with pytest.raises(ValidationError):
        Settings(rerank_batch_size=0)


def test_fastembed_adapter_is_lazy_reused_and_preserves_sdk_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[dict[str, object]] = []
    rerank_calls: list[tuple[str, list[str], int]] = []

    class FakeTextCrossEncoder:
        def __init__(self, **kwargs) -> None:
            constructor_calls.append(kwargs)

        @staticmethod
        def list_supported_models() -> list[dict[str, str]]:
            return [{"model": "model-a", "license": "apache-2.0"}]

        def rerank(self, query, documents, *, batch_size):
            rerank_calls.append((query, list(documents), batch_size))
            return [0.25, -1.5]

    fastembed_module = ModuleType("fastembed")
    fastembed_module.__path__ = []  # type: ignore[attr-defined]
    rerank_module = ModuleType("fastembed.rerank")
    rerank_module.__path__ = []  # type: ignore[attr-defined]
    cross_encoder_module = ModuleType("fastembed.rerank.cross_encoder")
    cross_encoder_module.TextCrossEncoder = FakeTextCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed_module)
    monkeypatch.setitem(sys.modules, "fastembed.rerank", rerank_module)
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", cross_encoder_module)

    adapter = FastEmbedCrossEncoder("model-a", cache_dir="cache", threads=3)
    assert adapter._model is None
    assert list(adapter.score(" query ", ["first", "second"], batch_size=4)) == [
        CrossEncoderScore(candidate_index=0, score=0.25),
        CrossEncoderScore(candidate_index=1, score=-1.5),
    ]
    assert list(adapter.score("again", ["first", "second"], batch_size=2)) == [
        CrossEncoderScore(candidate_index=0, score=0.25),
        CrossEncoderScore(candidate_index=1, score=-1.5),
    ]
    assert constructor_calls == [
        {
            "model_name": "model-a",
            "cache_dir": "cache",
            "threads": 3,
            "cuda": False,
            "lazy_load": True,
        }
    ]
    assert rerank_calls == [
        (" query ", ["first", "second"], 4),
        ("again", ["first", "second"], 2),
    ]
    assert fastembed_model_metadata("MODEL-A") == {
        "model": "model-a",
        "license": "apache-2.0",
    }


def test_candidate_text_uses_heading_breadcrumb_without_mutating_raw_text() -> None:
    candidate = _candidate("a", sparse_rank=1)
    assert CANDIDATE_TEXT_FORMAT == "document_context_heading_content_v2"
    assert build_candidate_text(candidate) == "Safety > Limits\n\nraw a"
    assert candidate.text == "raw a"
    assert build_candidate_text(candidate.model_copy(update={"headings": []})) == "raw a"


def test_candidate_text_includes_trusted_document_context_when_present() -> None:
    candidate = _candidate("a", sparse_rank=1).model_copy(
        update={
            "metadata": {
                "document_title": "ATV320 Installation Manual",
                "document_role": "installation",
            }
        }
    )
    assert build_candidate_text(candidate) == (
        "Document title: ATV320 Installation Manual\n"
        "Document role: installation\n\n"
        "Safety > Limits\n\nraw a"
    )


def test_rerank_orders_scores_and_preserves_all_metadata_and_component_signals() -> None:
    candidates = [_candidate("a", sparse_rank=1), _candidate("b", sparse_rank=2)]
    model = FakeCrossEncoder([CrossEncoderScore(0, -2.0), CrossEncoderScore(1, 4.0)])
    results = rerank_candidates(" query ", candidates, model, strategy="sparse", batch_size=8)

    assert [item.chunk_id for item in results] == ["b", "a"]
    assert [item.rerank_rank for item in results] == [1, 2]
    assert results[0].score == results[0].rerank_score == 4.0
    assert results[0].sparse_rank == 2
    assert results[0].metadata == {"marker": "b"}
    assert results[0].text == "raw b"
    assert model.calls[0][0] == "query"
    assert len(results) == len(candidates)


def test_rerank_ties_use_previous_rank_then_chunk_id() -> None:
    candidates = [
        _candidate("z", sparse_rank=2),
        _candidate("b", sparse_rank=1),
        _candidate("a", sparse_rank=1),
    ]
    model = FakeCrossEncoder([CrossEncoderScore(index, 1.0) for index in range(3)])
    results = rerank_candidates("q", candidates, model, strategy="sparse", batch_size=4)
    assert [item.chunk_id for item in results] == ["a", "b", "z"]


def test_empty_candidates_return_empty_but_empty_query_and_bad_batch_fail() -> None:
    model = FakeCrossEncoder()
    assert rerank_candidates("q", [], model, strategy="union", batch_size=1) == []
    with pytest.raises(RerankingError, match="query"):
        rerank_candidates(" ", [], model, strategy="union", batch_size=1)
    with pytest.raises(RerankingError, match="batch"):
        rerank_candidates("q", [], model, strategy="union", batch_size=0)


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ([CrossEncoderScore(0, 1.0)], "1 scores for 2"),
        ([CrossEncoderScore(0, 1.0), CrossEncoderScore(3, 2.0)], "invalid candidate index"),
        ([CrossEncoderScore(0, 1.0), CrossEncoderScore(0, 2.0)], "duplicate candidate index"),
        ([CrossEncoderScore(0, 1.0), CrossEncoderScore(2, 2.0)], "invalid candidate index"),
        ([CrossEncoderScore(0, math.nan), CrossEncoderScore(1, 1.0)], "finite"),
        ([CrossEncoderScore(0, math.inf), CrossEncoderScore(1, 1.0)], "finite"),
        ([CrossEncoderScore(0, -math.inf), CrossEncoderScore(1, 1.0)], "finite"),
    ],
)
def test_invalid_model_outputs_fail_without_fallback(outputs, message) -> None:
    with pytest.raises(RerankingError, match=message):
        rerank_candidates(
            "q",
            [_candidate("a", sparse_rank=1), _candidate("b", sparse_rank=2)],
            FakeCrossEncoder(outputs),
            strategy="sparse",
            batch_size=2,
        )


def test_model_exception_is_wrapped_with_preserved_cause() -> None:
    with pytest.raises(RerankingError, match="model exploded") as error:
        rerank_candidates(
            "q",
            [_candidate("a", sparse_rank=1)],
            FailingCrossEncoder(),
            strategy="sparse",
            batch_size=1,
        )
    assert isinstance(error.value.__cause__, RuntimeError)


def test_duplicate_candidates_and_missing_previous_ranks_fail() -> None:
    model = FakeCrossEncoder()
    with pytest.raises(RerankingError, match="unique"):
        rerank_candidates(
            "q",
            [_candidate("a", sparse_rank=1), _candidate("a", sparse_rank=2)],
            model,
            strategy="sparse",
            batch_size=2,
        )
    with pytest.raises(RerankingError, match="sparse_rank"):
        rerank_candidates("q", [_candidate("a")], model, strategy="sparse", batch_size=1)
    with pytest.raises(RerankingError, match="rrf_rank"):
        rerank_candidates("q", [_candidate("a")], model, strategy="hybrid", batch_size=1)


def test_sparse_hybrid_and_union_pool_construction_preserve_signals() -> None:
    dense = [_dense("both", 0.9), _dense("dense-only", 0.8)]
    sparse = [
        _candidate("both", sparse_rank=1),
        _candidate("sparse-only", sparse_rank=2),
    ]
    assert [item.chunk_id for item in build_candidate_pool("sparse", sparse_candidates=sparse)] == [
        "both",
        "sparse-only",
    ]
    hybrid = build_candidate_pool(
        "hybrid", dense_results=dense, sparse_candidates=sparse, hybrid_limit=20
    )
    assert hybrid[0].chunk_id == "both"
    assert hybrid[0].dense_rank == 1 and hybrid[0].sparse_rank == 1
    union = build_candidate_pool("union", dense_results=dense, sparse_candidates=sparse)
    assert {item.chunk_id for item in union} == {"both", "dense-only", "sparse-only"}
    merged = next(item for item in union if item.chunk_id == "both")
    assert merged.dense_rank == 1 and merged.sparse_rank == 1


def test_content_dedup_collapses_only_exact_normalized_text_and_preserves_signals() -> None:
    first = _candidate("a", dense_rank=3).model_copy(update={"text": "  SAME\ntext  "})
    second = _candidate("b", sparse_rank=2).model_copy(update={"text": "same text"})
    similar = _candidate("c", sparse_rank=1).model_copy(update={"text": "same texts"})
    other_document = _candidate("d", sparse_rank=4, document_id="manual-b").model_copy(
        update={"text": "same text"}
    )

    result = deduplicate_candidates_by_content([first, second, similar, other_document])

    assert [item.chunk_id for item in result] == ["a", "c", "d"]
    assert result[0].dense_rank == 3
    assert result[0].sparse_rank == 2
    assert result[0].metadata["equivalent_chunk_ids"] == ["a", "b"]
    assert result[0].metadata["equivalent_chunk_count"] == 2
    assert first.metadata == {"marker": "a"}


def test_pipeline_content_dedup_is_opt_in() -> None:
    def fake_dense(*args, **kwargs):
        return [_dense("dense").model_copy(update={"text": "same"})]

    def fake_sparse(*args, **kwargs):
        return [_candidate("sparse", sparse_rank=1).model_copy(update={"text": " SAME "})]

    pipeline = RerankPipeline(
        dense_searcher=DenseSearchStub(fake_dense),
        sparse_searcher=SparseSearchStub(fake_sparse),
        cross_encoder=FakeCrossEncoder(),
        dense_collection="v1",
        hybrid_collection="v2",
        deduplicate_content=True,
    )
    execution = pipeline.search("q", strategy="union")
    assert len(execution.candidates_before_rerank) == 1
    assert "content_deduplication" in execution.stage_latency_ms


def test_pipeline_preserves_document_filter_and_uses_correct_dense_collection() -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_dense(*args, **kwargs):
        calls.append((kwargs["collection_name"], kwargs["document_id"]))
        return [_dense("dense", document_id=kwargs["document_id"])]

    def fake_sparse(*args, **kwargs):
        calls.append((kwargs["collection_name"], kwargs["document_id"]))
        return [_candidate("sparse", sparse_rank=1, document_id=kwargs["document_id"])]

    pipeline = RerankPipeline(
        dense_searcher=DenseSearchStub(fake_dense),
        sparse_searcher=SparseSearchStub(fake_sparse),
        cross_encoder=FakeCrossEncoder(),
        dense_collection="v1",
        hybrid_collection="v2",
    )
    pipeline.prepare_pool("q", strategy="union", document_id="manual-b")
    assert calls == [("v1", "manual-b"), ("v2", "manual-b")]
    calls.clear()
    pipeline.prepare_pool("q", strategy="hybrid", document_id="manual-b")
    assert calls == [("v2", "manual-b"), ("v2", "manual-b")]


def test_pipeline_preserves_bound_adapter_contract() -> None:
    calls: list[tuple[str, str, str, int, str | None, float | None]] = []

    class DensePort:
        @staticmethod
        def search(
            query,
            *,
            collection_name,
            limit,
            document_id=None,
            score_threshold=None,
        ):
            calls.append(
                ("dense", query, collection_name, limit, document_id, score_threshold)
            )
            return [_dense("dense", document_id=document_id)]

    class SparsePort:
        @staticmethod
        def search(query, *, collection_name, limit, document_id=None):
            calls.append(("sparse", query, collection_name, limit, document_id, None))
            return [_candidate("sparse", sparse_rank=1, document_id=document_id)]

    pipeline = RerankPipeline(
        dense_searcher=DensePort(),
        sparse_searcher=SparsePort(),
        cross_encoder=FakeCrossEncoder(),
        dense_collection="v1",
        hybrid_collection="v2",
        dense_candidate_limit=7,
        sparse_candidate_limit=9,
    )

    pool = pipeline.prepare_pool("q", strategy="union", document_id="manual-b")

    assert {candidate.chunk_id for candidate in pool.candidates} == {"dense", "sparse"}
    assert calls == [
        ("dense", "q", "v1", 7, "manual-b", None),
        ("sparse", "q", "v2", 9, "manual-b", None),
    ]


def test_pipeline_attaches_only_configured_trusted_document_context() -> None:
    def fake_dense(*args, **kwargs):
        return [_dense("dense", document_id="manual-a")]

    def fake_sparse(*args, **kwargs):
        return [_candidate("sparse", sparse_rank=1, document_id="manual-b")]

    pipeline = RerankPipeline(
        dense_searcher=DenseSearchStub(fake_dense),
        sparse_searcher=SparseSearchStub(fake_sparse),
        cross_encoder=FakeCrossEncoder(),
        dense_collection="v1",
        hybrid_collection="v2",
        document_contexts={
            "manual-a": {
                "document_title": "Installation Manual",
                "document_role": "installation",
            }
        },
    )
    pool = pipeline.prepare_pool("q", strategy="union").candidates
    dense = next(candidate for candidate in pool if candidate.chunk_id == "dense")
    sparse = next(candidate for candidate in pool if candidate.chunk_id == "sparse")
    assert dense.metadata["document_role"] == "installation"
    assert dense.metadata["document_title"] == "Installation Manual"
    assert "document_role" not in sparse.metadata


def test_pipeline_expands_sparse_query_then_rrf_prunes_before_fixed_rerank_budget() -> None:
    sparse_queries: list[str] = []

    def fake_dense(*args, **kwargs):
        return [_dense(f"d{index}", 1.0 - index / 10) for index in range(4)]

    def fake_sparse(query, **kwargs):
        sparse_queries.append(query)
        return [_candidate(f"s{index}", sparse_rank=index + 1) for index in range(4)]

    pipeline = RerankPipeline(
        dense_searcher=DenseSearchStub(fake_dense),
        sparse_searcher=SparseSearchStub(fake_sparse),
        cross_encoder=FakeCrossEncoder(),
        dense_collection="v1",
        hybrid_collection="v2",
        sparse_query_transform=lambda query: f"{query} expanded",
        union_rrf_prune_limit=3,
    )
    pool = pipeline.prepare_pool("q", strategy="union")
    assert sparse_queries == ["q expanded"]
    assert len(pool.candidates) == 3
    assert all(candidate.rrf_rank is not None for candidate in pool.candidates)
    assert "query_expansion" in pool.stage_latency_ms
    assert "rrf_pruning" in pool.stage_latency_ms


def test_evaluator_classifies_candidate_and_ordering_failures_with_stage_metrics() -> None:
    cases = [
        _case("hit", "hit"),
        _case("top5-miss", "late"),
        _case("top20-miss", "beyond"),
        _case("candidate-miss", "absent", critical=True),
    ]

    def search(question, document_id):
        case_id = next(case.id for case in cases if case.question == question)
        if case_id == "hit":
            before = [_candidate("hit", sparse_rank=1)]
            after = [_candidate("hit", sparse_rank=1).model_copy(update={"rerank_rank": 1})]
        elif case_id == "top5-miss":
            before = [_candidate("late", sparse_rank=1)]
            after = [_candidate(str(index), sparse_rank=index) for index in range(1, 6)] + [
                _candidate("late", sparse_rank=6)
            ]
        elif case_id == "top20-miss":
            before = [_candidate("beyond", sparse_rank=1)]
            after = [_candidate(str(index), sparse_rank=index) for index in range(1, 21)] + [
                _candidate("beyond", sparse_rank=21)
            ]
        else:
            before = [_candidate("wrong", sparse_rank=1)]
            after = before
        return RerankExecution(before, after, {"rerank": 2.0, "total": 3.0})

    report = evaluate_reranked_cases(cases, search, cutoff=20)
    assert [row["failure_class"] for row in report["per_query"]] == [
        "hit",
        "reranker_miss_top5",
        "reranker_miss_top20",
        "candidate_miss",
    ]
    assert report["overall"]["candidate_recall"] == 0.75
    assert report["overall"]["stage_latency_ms"]["total"]["p95"] == 3.0


@pytest.mark.parametrize(
    ("candidate_rank", "final_rank", "expected"),
    [
        (None, None, "candidate_miss"),
        (1, 3, "hit"),
        (1, 8, "reranker_miss_top5"),
        (1, 21, "reranker_miss_top20"),
    ],
)
def test_failure_classification(candidate_rank, final_rank, expected) -> None:
    assert classify_rerank_failure(candidate_rank=candidate_rank, final_rank=final_rank) == expected


def test_importing_module_does_not_construct_fastembed_model() -> None:
    adapter = FastEmbedCrossEncoder("not-loaded")
    assert adapter._model is None

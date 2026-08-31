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
    RerankPipeline,
    build_candidate_pool,
    build_candidate_text,
    deduplicate_candidates_by_content,
    refill_candidates_after_content_dedup,
    rerank_candidates,
)
from app.config import Settings
from app.domain.policies.fusion import FusionProfile
from app.domain.reranking import CrossEncoderScore, RerankingError
from app.domain.retrieval import RetrievalCandidate, RetrievedChunk
from app.infrastructure.models.reranker import (
    FastEmbedCrossEncoder,
    fastembed_model_metadata,
)


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


def test_refill_skips_same_document_duplicates_but_keeps_provenance_and_near_matches() -> None:
    selected_same = _candidate("selected-same", rrf_rank=2).model_copy(
        update={"text": " SAME\ncontent "}
    )
    selected_other = _candidate("selected-other", rrf_rank=4).model_copy(
        update={"text": "other content"}
    )
    deduplicated = deduplicate_candidates_by_content([selected_same, selected_other])
    fused = [
        _candidate("same-document", rrf_rank=1).model_copy(update={"text": "same content"}),
        selected_same,
        _candidate("cross-document", rrf_rank=3, document_id="manual-b").model_copy(
            update={"text": "same content"}
        ),
        selected_other,
        _candidate("near-match", rrf_rank=5).model_copy(update={"text": "same contents"}),
    ]

    result, refilled_ids = refill_candidates_after_content_dedup(
        deduplicated,
        selected_candidate_ids={"selected-same", "selected-other"},
        fused_candidates=fused,
        max_candidates=4,
    )

    assert [candidate.chunk_id for candidate in result] == [
        "selected-same",
        "cross-document",
        "selected-other",
        "near-match",
    ]
    assert refilled_ids == ("cross-document", "near-match")
    assert result[1].metadata["equivalent_chunk_ids"] == ["cross-document"]


def test_refill_uses_rrf_rank_then_chunk_id_and_never_exceeds_budget() -> None:
    selected = _candidate("selected", rrf_rank=1)
    deduplicated = deduplicate_candidates_by_content([selected])
    fused = [
        _candidate("tail-b", rrf_rank=2),
        selected,
        _candidate("tail-a", rrf_rank=2),
    ]

    result, refilled_ids = refill_candidates_after_content_dedup(
        deduplicated,
        selected_candidate_ids={"selected"},
        fused_candidates=fused,
        max_candidates=2,
    )

    assert [candidate.chunk_id for candidate in result] == ["selected", "tail-a"]
    assert refilled_ids == ("tail-a",)


def test_refill_restores_a_realistic_thirty_candidate_budget_after_dedup() -> None:
    selected = [
        _candidate(f"selected-{index:02d}", rrf_rank=index + 1).model_copy(
            update={"text": f"unique content {index}"}
        )
        for index in range(26)
    ]
    selected.extend(
        _candidate(f"duplicate-{index:02d}", rrf_rank=27 + index).model_copy(
            update={"text": f" UNIQUE   CONTENT {index} "}
        )
        for index in range(4)
    )
    fused = [
        *selected,
        *[
            _candidate(f"tail-{index:02d}", rrf_rank=31 + index).model_copy(
                update={"text": f"tail content {index}"}
            )
            for index in range(4)
        ],
    ]

    result, refilled_ids = refill_candidates_after_content_dedup(
        deduplicate_candidates_by_content(selected),
        selected_candidate_ids={candidate.chunk_id for candidate in selected},
        fused_candidates=fused,
        max_candidates=30,
    )

    assert len(deduplicate_candidates_by_content(selected)) == 26
    assert len(result) == 30
    assert refilled_ids == ("tail-00", "tail-01", "tail-02", "tail-03")


def test_refill_returns_smaller_pool_when_the_fused_tail_is_exhausted() -> None:
    selected = _candidate("selected", rrf_rank=1).model_copy(update={"text": "same"})
    duplicate = _candidate("duplicate", rrf_rank=2).model_copy(update={"text": " SAME "})

    result, refilled_ids = refill_candidates_after_content_dedup(
        deduplicate_candidates_by_content([selected]),
        selected_candidate_ids={"selected"},
        fused_candidates=[selected, duplicate],
        max_candidates=3,
    )

    assert [candidate.chunk_id for candidate in result] == ["selected"]
    assert refilled_ids == ()


@pytest.mark.parametrize(
    ("candidates", "fused", "max_candidates", "message"),
    [
        ([], [], 0, "greater than 0"),
        (
            [_candidate("a", rrf_rank=1), _candidate("b", rrf_rank=2)],
            [],
            1,
            "exceeds the refill budget",
        ),
        ([], [_candidate("missing-rank")], 1, "one-based RRF ranks"),
    ],
)
def test_refill_rejects_invalid_candidate_contracts(
    candidates: list[RetrievalCandidate],
    fused: list[RetrievalCandidate],
    max_candidates: int,
    message: str,
) -> None:
    with pytest.raises(RerankingError, match=message):
        refill_candidates_after_content_dedup(
            candidates,
            selected_candidate_ids={candidate.chunk_id for candidate in candidates},
            fused_candidates=fused,
            max_candidates=max_candidates,
        )


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


def test_pipeline_refill_requires_content_deduplication() -> None:
    with pytest.raises(RerankingError, match="requires exact-content deduplication"):
        RerankPipeline(
            dense_searcher=DenseSearchStub(lambda *args, **kwargs: []),
            sparse_searcher=SparseSearchStub(lambda *args, **kwargs: []),
            cross_encoder=FakeCrossEncoder(),
            dense_collection="v1",
            hybrid_collection="v2",
            refill_after_content_dedup=True,
        )


def test_diagnostic_component_probe_requires_trace_and_a_larger_limit() -> None:
    common = {
        "dense_searcher": DenseSearchStub(lambda *args, **kwargs: []),
        "sparse_searcher": SparseSearchStub(lambda *args, **kwargs: []),
        "cross_encoder": FakeCrossEncoder(),
        "dense_collection": "v1",
        "hybrid_collection": "v2",
    }

    with pytest.raises(RerankingError, match="requires trace capture"):
        RerankPipeline(**common, diagnostic_component_limit=100)
    with pytest.raises(RerankingError, match="exceed both normal component limits"):
        RerankPipeline(
            **common,
            dense_candidate_limit=60,
            sparse_candidate_limit=40,
            capture_trace=True,
            diagnostic_component_limit=60,
        )


def test_diagnostic_component_probe_does_not_expand_the_reranker_pool() -> None:
    calls: list[tuple[str, int]] = []

    def fake_dense(*args, limit, **kwargs):
        calls.append(("dense", limit))
        results = [_dense("dense-normal", 0.9)]
        if limit == 100:
            results.append(_dense("dense-probe", 0.8))
        return results

    def fake_sparse(*args, limit, **kwargs):
        calls.append(("sparse", limit))
        results = [_candidate("sparse-normal", sparse_rank=1)]
        if limit == 100:
            results.append(_candidate("sparse-probe", sparse_rank=2))
        return results

    baseline = RerankPipeline(
        dense_searcher=DenseSearchStub(fake_dense),
        sparse_searcher=SparseSearchStub(fake_sparse),
        cross_encoder=FakeCrossEncoder(),
        dense_collection="v1",
        hybrid_collection="v2",
        dense_candidate_limit=60,
        sparse_candidate_limit=40,
        capture_trace=True,
    ).prepare_pool("q", strategy="union")
    calls.clear()
    pool = RerankPipeline(
        dense_searcher=DenseSearchStub(fake_dense),
        sparse_searcher=SparseSearchStub(fake_sparse),
        cross_encoder=FakeCrossEncoder(),
        dense_collection="v1",
        hybrid_collection="v2",
        dense_candidate_limit=60,
        sparse_candidate_limit=40,
        capture_trace=True,
        diagnostic_component_limit=100,
    ).prepare_pool("q", strategy="union")

    assert calls == [("dense", 60), ("sparse", 40), ("dense", 100), ("sparse", 100)]
    assert [candidate.model_dump() for candidate in pool.candidates] == [
        candidate.model_dump() for candidate in baseline.candidates
    ]
    assert len(pool.candidates) <= 30
    assert pool.trace is not None
    assert [candidate.chunk_id for candidate in pool.trace.dense_candidates] == [
        "dense-normal",
        "dense-probe",
    ]
    assert [candidate.chunk_id for candidate in pool.trace.sparse_candidates] == [
        "sparse-normal",
        "sparse-probe",
    ]
    assert "diagnostic_dense_retrieval" in pool.stage_latency_ms
    assert "diagnostic_sparse_retrieval" in pool.stage_latency_ms
    assert "diagnostic_candidate_context" in pool.stage_latency_ms


def test_pipeline_refill_is_opt_in_and_trace_preserves_dedup_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_dense(*args, **kwargs):
        return [
            _dense("dense-duplicate", 0.9).model_copy(update={"text": "same content"}),
            _dense("dense-tail", 0.8).model_copy(update={"text": "dense tail"}),
        ]

    def fake_sparse(*args, **kwargs):
        return [
            _candidate("sparse-duplicate", sparse_rank=1).model_copy(
                update={"text": " SAME  CONTENT "}
            ),
            _candidate("sparse-tail", sparse_rank=2).model_copy(
                update={"text": "sparse tail"}
            ),
        ]

    profile = FusionProfile(
        name="refill-test",
        rrf_k=40,
        dense_weight=1.0,
        sparse_weight=1.0,
        fusion_role_multiplier=0.0,
        dense_reserve=1,
        sparse_reserve=1,
        max_candidates=3,
        post_rerank_role_multiplier=0.0,
    )

    def prepare(*, refill: bool, capture_trace: bool):
        return RerankPipeline(
            dense_searcher=DenseSearchStub(fake_dense),
            sparse_searcher=SparseSearchStub(fake_sparse),
            cross_encoder=FakeCrossEncoder(),
            dense_collection="v1",
            hybrid_collection="v2",
            deduplicate_content=True,
            refill_after_content_dedup=refill,
            fusion_profile=profile,
            capture_trace=capture_trace,
        ).prepare_pool("q", strategy="union")

    baseline = prepare(refill=False, capture_trace=False)
    without_trace = prepare(refill=True, capture_trace=False)
    fusion_calls = 0
    from app.application import reranking_service

    original_fuse = reranking_service.fuse_weighted_rrf

    def counted_fuse(*args, **kwargs):
        nonlocal fusion_calls
        fusion_calls += 1
        return original_fuse(*args, **kwargs)

    monkeypatch.setattr(reranking_service, "fuse_weighted_rrf", counted_fuse)
    traced = prepare(refill=True, capture_trace=True)

    assert [candidate.chunk_id for candidate in baseline.candidates] == [
        "dense-duplicate",
        "dense-tail",
    ]
    assert [candidate.chunk_id for candidate in without_trace.candidates] == [
        "dense-duplicate",
        "dense-tail",
        "sparse-tail",
    ]
    assert [candidate.model_dump() for candidate in traced.candidates] == [
        candidate.model_dump() for candidate in without_trace.candidates
    ]
    assert fusion_calls == 1
    assert traced.trace is not None
    assert traced.trace.candidate_count_pre_dedup == 3
    assert traced.trace.candidate_count_post_dedup == 2
    assert traced.trace.candidate_count_before_rerank == 3
    assert traced.trace.refilled_candidate_ids == ("sparse-tail",)
    assert "candidate_refill" in traced.stage_latency_ms
    assert [candidate.chunk_id for candidate in traced.trace.candidate_pool_post_dedup] == [
        "dense-duplicate",
        "dense-tail",
    ]
    assert traced.trace.candidates_before_rerank == tuple(traced.candidates)


def test_pipeline_trace_captures_boundaries_without_changing_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_dense(*args, **kwargs):
        return [
            _dense("dense-a", 0.9).model_copy(update={"text": "same content"}),
            _dense("shared", 0.8),
        ]

    def fake_sparse(*args, **kwargs):
        return [
            _candidate("sparse-a", sparse_rank=1).model_copy(
                update={"text": " SAME  CONTENT "}
            ),
            _candidate("shared", sparse_rank=2),
        ]

    profile = FusionProfile(
        name="trace-test",
        rrf_k=40,
        dense_weight=1.0,
        sparse_weight=1.25,
        fusion_role_multiplier=0.0,
        dense_reserve=1,
        sparse_reserve=1,
        max_candidates=3,
        post_rerank_role_multiplier=0.0,
    )

    def execute(*, capture_trace: bool):
        return RerankPipeline(
            dense_searcher=DenseSearchStub(fake_dense),
            sparse_searcher=SparseSearchStub(fake_sparse),
            cross_encoder=FakeCrossEncoder(),
            dense_collection="v1",
            hybrid_collection="v2",
            deduplicate_content=True,
            fusion_profile=profile,
            capture_trace=capture_trace,
        ).search("q", strategy="union")

    baseline = execute(capture_trace=False)
    fusion_calls = 0
    from app.application import reranking_service

    original_fuse = reranking_service.fuse_weighted_rrf

    def counted_fuse(*args, **kwargs):
        nonlocal fusion_calls
        fusion_calls += 1
        return original_fuse(*args, **kwargs)

    monkeypatch.setattr(reranking_service, "fuse_weighted_rrf", counted_fuse)
    traced = execute(capture_trace=True)

    assert traced.trace is not None
    trace = traced.trace
    assert baseline.trace is None
    assert fusion_calls == 1
    assert [candidate.model_dump() for candidate in traced.candidates_after_rerank] == [
        candidate.model_dump() for candidate in baseline.candidates_after_rerank
    ]
    assert [candidate.chunk_id for candidate in trace.dense_candidates] == [
        "dense-a",
        "shared",
    ]
    assert [candidate.chunk_id for candidate in trace.sparse_candidates] == [
        "sparse-a",
        "shared",
    ]
    assert {candidate.chunk_id for candidate in trace.fused_candidates} == {
        "dense-a",
        "sparse-a",
        "shared",
    }
    assert trace.candidate_count_pre_dedup == 3
    assert trace.candidate_count_post_dedup == 2
    assert [candidate.chunk_id for candidate in trace.candidate_pool_post_dedup] == [
        candidate.chunk_id for candidate in traced.candidates_before_rerank
    ]
    assert [candidate.chunk_id for candidate in trace.raw_cross_encoder_candidates] == [
        candidate.chunk_id for candidate in traced.candidates_after_rerank
    ]
    assert [candidate.chunk_id for candidate in trace.final_candidates] == [
        candidate.chunk_id for candidate in traced.candidates_after_rerank
    ]


def test_pipeline_trace_separates_cross_encoder_and_post_policy_order() -> None:
    def fake_dense(*args, **kwargs):
        return [
            _dense("installation", 0.9, document_id="manual-installation"),
            _dense("programming", 0.8, document_id="manual-programming"),
        ]

    pipeline = RerankPipeline(
        dense_searcher=DenseSearchStub(fake_dense),
        sparse_searcher=SparseSearchStub(lambda *args, **kwargs: []),
        cross_encoder=FakeCrossEncoder(),
        dense_collection="v1",
        hybrid_collection="v2",
        document_contexts={
            "manual-installation": {
                "document_title": "Installation Manual",
                "document_role": "installation",
            },
            "manual-programming": {
                "document_title": "Programming Manual",
                "document_role": "programming",
            },
        },
        fusion_profile=FusionProfile(
            name="trace-role-test",
            rrf_k=40,
            dense_weight=1.0,
            sparse_weight=1.0,
            fusion_role_multiplier=0.0,
            dense_reserve=2,
            sparse_reserve=0,
            max_candidates=2,
            post_rerank_role_multiplier=0.5,
            post_rerank_rank_offset=10,
        ),
        capture_trace=True,
    )

    execution = pipeline.search("electrical installation", strategy="union")

    assert execution.trace is not None
    assert [
        candidate.chunk_id for candidate in execution.trace.raw_cross_encoder_candidates
    ] == ["programming", "installation"]
    assert [candidate.chunk_id for candidate in execution.trace.final_candidates] == [
        "installation",
        "programming",
    ]
    assert [candidate.chunk_id for candidate in execution.candidates_after_rerank] == [
        "installation",
        "programming",
    ]


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
        capture_trace=True,
    )
    pool = pipeline.prepare_pool("q", strategy="union")
    assert sparse_queries == ["q expanded"]
    assert len(pool.candidates) == 3
    assert all(candidate.rrf_rank is not None for candidate in pool.candidates)
    assert "query_expansion" in pool.stage_latency_ms
    assert "rrf_pruning" in pool.stage_latency_ms
    assert pool.trace is not None
    assert pool.trace.candidate_count_pre_dedup == 8
    assert pool.trace.candidate_count_post_dedup == 8
    assert pool.trace.candidate_count_before_rerank == 3
    assert len(pool.trace.candidate_pool_post_dedup) == 8
    assert pool.trace.candidates_before_rerank == tuple(pool.candidates)


def test_importing_module_does_not_construct_fastembed_model() -> None:
    adapter = FastEmbedCrossEncoder("not-loaded")
    assert adapter._model is None

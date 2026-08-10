"""Offline tests for multilingual cross-encoder reranking and diagnostics."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.evaluation import EvaluationCase
from app.models import RetrievalCandidate, RetrievedChunk
from app.reranking import (
    CANDIDATE_TEXT_FORMAT,
    CrossEncoderScore,
    RerankExecution,
    RerankingError,
    RerankPipeline,
    build_candidate_pool,
    build_candidate_text,
    classify_rerank_failure,
    deduplicate_candidates_by_content,
    evaluate_reranked_cases,
    rerank_candidates,
)
from scripts import evaluate_reranking, search_reranked


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


def test_candidate_text_uses_heading_breadcrumb_without_mutating_raw_text() -> None:
    candidate = _candidate("a", sparse_rank=1)
    assert CANDIDATE_TEXT_FORMAT == "heading_content_v1"
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
        client=object(),
        dense_embedding_model=object(),
        sparse_embedding_model=object(),
        cross_encoder=FakeCrossEncoder(),
        dense_collection="v1",
        hybrid_collection="v2",
        dense_vector_name="dense",
        sparse_vector_name="sparse",
        deduplicate_content=True,
        dense_search_fn=fake_dense,
        sparse_search_fn=fake_sparse,
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
        client=object(),
        dense_embedding_model=object(),
        sparse_embedding_model=object(),
        cross_encoder=FakeCrossEncoder(),
        dense_collection="v1",
        hybrid_collection="v2",
        dense_vector_name="dense",
        sparse_vector_name="sparse",
        dense_search_fn=fake_dense,
        sparse_search_fn=fake_sparse,
    )
    pipeline.prepare_pool("q", strategy="union", document_id="manual-b")
    assert calls == [("v1", "manual-b"), ("v2", "manual-b")]
    calls.clear()
    pipeline.prepare_pool("q", strategy="hybrid", document_id="manual-b")
    assert calls == [("v2", "manual-b"), ("v2", "manual-b")]


def test_pipeline_attaches_only_configured_trusted_document_context() -> None:
    def fake_dense(*args, **kwargs):
        return [_dense("dense", document_id="manual-a")]

    def fake_sparse(*args, **kwargs):
        return [_candidate("sparse", sparse_rank=1, document_id="manual-b")]

    pipeline = RerankPipeline(
        client=object(),
        dense_embedding_model=object(),
        sparse_embedding_model=object(),
        cross_encoder=FakeCrossEncoder(),
        dense_collection="v1",
        hybrid_collection="v2",
        dense_vector_name="dense",
        sparse_vector_name="sparse",
        document_contexts={
            "manual-a": {
                "document_title": "Installation Manual",
                "document_role": "installation",
            }
        },
        dense_search_fn=fake_dense,
        sparse_search_fn=fake_sparse,
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

    def fake_sparse(*args, **kwargs):
        sparse_queries.append(args[1])
        return [_candidate(f"s{index}", sparse_rank=index + 1) for index in range(4)]

    pipeline = RerankPipeline(
        client=object(),
        dense_embedding_model=object(),
        sparse_embedding_model=object(),
        cross_encoder=FakeCrossEncoder(),
        dense_collection="v1",
        hybrid_collection="v2",
        dense_vector_name="dense",
        sparse_vector_name="sparse",
        sparse_query_transform=lambda query: f"{query} expanded",
        union_rrf_prune_limit=3,
        dense_search_fn=fake_dense,
        sparse_search_fn=fake_sparse,
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


def test_cli_requires_strategy_and_rejects_invalid_final_limit() -> None:
    parser = search_reranked._build_parser()
    args = parser.parse_args(["question", "--strategy", "union", "--limit", "5"])
    assert args.strategy == "union"
    with pytest.raises(SystemExit):
        parser.parse_args(["question"])
    with pytest.raises(SystemExit):
        parser.parse_args(["question", "--strategy", "sparse", "--limit", "0"])


def test_evaluation_cli_supports_model_free_comparison_rebuild() -> None:
    args = evaluate_reranking._build_parser().parse_args(["--comparison-only"])
    assert args.comparison_only is True
    assert args.strategy == "all"


def test_ranking_gate_can_pass_when_latency_gate_fails() -> None:
    summary = {
        "quality_gate": {
            "critical_pairs_top5": {"actual": 3, "target": 3},
            "hit_rate_at_5": {"actual": 0.733, "target": 0.633},
            "mrr_at_5": {"actual": 0.529, "target": 0.485},
            "warm_total_p95_ms": {"actual": 9879.69, "target_less_than": 1500.0},
            "pass": False,
        }
    }
    assert evaluate_reranking._ranking_gates_pass(summary)


def test_importing_module_does_not_construct_fastembed_model() -> None:
    import app.reranking as reranking

    adapter = reranking.FastEmbedCrossEncoder("not-loaded")
    assert adapter._model is None

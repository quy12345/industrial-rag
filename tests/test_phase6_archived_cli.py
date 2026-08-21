"""Offline characterization tests for archived Phase 4/5 command-line tools."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.archive.phase6 import (
    audit_candidate_pools,
    evaluate,
    generate_phase5_readiness,
    index_document,
    index_hybrid,
    search_dense,
    search_hybrid,
)


def test_archived_retrieval_evaluation_defaults_remain_frozen() -> None:
    args = evaluate._build_parser().parse_args([])

    assert args.dataset == Path("data/eval/dense_smoke.jsonl")
    assert args.chunks == Path("artifacts/manual-batched.jsonl")
    assert args.strategy == "dense"
    assert args.limit == 20
    assert args.output is None


def test_archived_candidate_audit_defaults_remain_frozen() -> None:
    args = audit_candidate_pools._build_parser().parse_args([])

    assert args.dataset == Path("data/eval/dense_smoke.jsonl")
    assert args.chunks == Path("artifacts/manual-batched.jsonl")
    assert args.limit == 20
    assert args.output == Path("artifacts/metrics/candidate-pool-audit.json")


def test_archived_phase5_readiness_defaults_remain_frozen() -> None:
    args = generate_phase5_readiness._build_parser().parse_args([])

    assert args.chunks == Path("artifacts/manual-batched.jsonl")
    assert args.dense_metrics == Path("artifacts/metrics/dense-baseline-closure.json")
    assert args.candidate_audit == Path("artifacts/metrics/candidate-pool-audit.json")
    assert args.output == Path("artifacts/metrics/phase-5-readiness.json")
    assert args.status == "ready_with_documented_deviation"


def test_archived_dense_search_parser_and_preview_remain_stable() -> None:
    args = search_dense._build_parser().parse_args(["historical question"])

    assert args.question == "historical question"
    assert args.limit is None
    assert args.document_id is None
    assert args.score_threshold is None
    assert args.preview_chars == 500
    assert search_dense._truncate("abcdef", 3) == ("abc", True)
    assert search_dense._truncate("abc", 3) == ("abc", False)


def test_archived_hybrid_search_defaults_remain_frozen() -> None:
    args = search_hybrid._build_parser().parse_args(["historical question"])

    assert args.question == "historical question"
    assert args.frozen_chunks == "artifacts/manual-batched.jsonl"
    assert args.limit is None
    assert args.dense_candidate_limit is None
    assert args.sparse_candidate_limit is None
    assert args.preview_chars == 500


def test_archived_hybrid_search_rejects_missing_chunks_before_external_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_external_access(*args: object, **kwargs: object) -> None:
        pytest.fail("external search dependency was initialized")

    monkeypatch.setattr(search_hybrid, "create_embedding_model", unexpected_external_access)
    monkeypatch.setattr(search_hybrid, "create_qdrant_client", unexpected_external_access)

    result = search_hybrid.main(
        ["historical question", "--frozen-chunks", str(tmp_path / "missing.jsonl")]
    )

    assert result == 1
    assert capsys.readouterr().err.startswith("Error:")


def test_archived_dense_index_parser_and_page_range_remain_stable() -> None:
    args = index_document._build_parser().parse_args(["data/raw/manual.pdf"])

    assert args.input == "data/raw/manual.pdf"
    assert args.page_start is None
    assert args.page_end is None
    assert args.page_batch_size is None
    assert args.embedding_batch_size is None
    assert index_document._parse_page_range(None, None) is None
    assert index_document._parse_page_range(1, 21) == (1, 21)
    with pytest.raises(index_document.IngestionError, match="provided together"):
        index_document._parse_page_range(1, None)


def test_archived_hybrid_index_defaults_and_page_range_remain_stable() -> None:
    args = index_hybrid._build_parser().parse_args(["data/raw/manual.pdf"])

    assert args.input == "data/raw/manual.pdf"
    assert args.frozen_chunks == Path("artifacts/manual-batched.jsonl")
    assert args.page_start is None
    assert args.page_end is None
    assert args.page_batch_size is None
    assert args.embedding_batch_size is None
    assert args.sparse_embedding_batch_size is None
    assert index_hybrid._parse_page_range(None, None) is None
    assert index_hybrid._parse_page_range(1, 21) == (1, 21)
    with pytest.raises(index_hybrid.IngestionError, match="provided together"):
        index_hybrid._parse_page_range(None, 21)


@pytest.mark.parametrize(
    ("main", "path_option"),
    [
        (evaluate.main, None),
        (audit_candidate_pools.main, "--dataset"),
    ],
)
def test_archived_retrieval_clis_fail_before_external_runtime_access(
    main: Callable[[list[str]], int],
    path_option: str | None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_path = tmp_path / "missing.jsonl"
    args = [str(missing_path)] if path_option is None else [path_option, str(missing_path)]

    assert main(args) == 1
    assert capsys.readouterr().err.startswith("Error:")

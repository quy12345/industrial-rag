"""Unit tests for direct-evidence dense evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.evaluation import (
    EvaluationCase,
    EvaluationError,
    aggregate_rows,
    direct_evidence_rank,
    evaluate_cases,
    load_evaluation_cases,
    percentile_nearest_rank,
    validate_cases_against_chunks,
)
from app.models import DocumentChunk
from scripts import evaluate as evaluate_cli


def _case(**overrides: object) -> EvaluationCase:
    payload: dict[str, object] = {
        "id": "case-1",
        "language": "en",
        "question": "Which threshold is used?",
        "relevant_chunk_ids": ["evidence"],
        "expected_phrases": ["threshold is 3.338"],
        "expected_pages": [11],
        "category": "numeric_unit",
        "document_id": "manual-1",
    }
    payload.update(overrides)
    return EvaluationCase.model_validate(payload)


def _chunk(
    chunk_id: str = "evidence",
    *,
    text: str = "The threshold is 3.338.",
    page_numbers: list[int] | None = None,
    document_id: str = "manual-1",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename="manual.pdf",
        text=text,
        page_numbers=page_numbers or [11],
        headings=["Algorithm"],
        content_type="text",
    )


def _result(
    chunk_id: str,
    *,
    text: str = "unrelated",
    page_numbers: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        chunk_id=chunk_id,
        document_id="manual-1",
        text=text,
        page_numbers=page_numbers or [1],
        headings=["Heading"],
        score=0.9,
    )


def test_dataset_loader_rejects_missing_qrels_duplicate_ids_and_bad_json(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"id": "missing-qrels", "question": "x"}),
                json.dumps(_case().model_dump()),
                json.dumps(_case().model_dump()),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError, match="Invalid evaluation record on line 1"):
        load_evaluation_cases(dataset)

    dataset.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="Invalid JSON on line 1"):
        load_evaluation_cases(dataset)

    dataset.write_text("\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="Blank evaluation record on line 1"):
        load_evaluation_cases(dataset)

    dataset.write_text(
        "\n".join([json.dumps(_case().model_dump()), json.dumps(_case().model_dump())]),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="Duplicate evaluation ID"):
        load_evaluation_cases(dataset)


def test_schema_rejects_empty_qrels_and_unsupported_language_or_category() -> None:
    with pytest.raises(ValueError):
        _case(relevant_chunk_ids=[])
    with pytest.raises(ValueError):
        _case(language="fr")
    with pytest.raises(ValueError):
        _case(category="invented")


def test_schema_derives_cross_lingual_scenario_and_rejects_inconsistent_metadata() -> None:
    assert _case(language="en").retrieval_scenario == "cross_lingual"
    assert _case(language="vi").retrieval_scenario == "monolingual"
    with pytest.raises(ValueError, match="retrieval_scenario"):
        _case(language="en", retrieval_scenario="monolingual")


def test_qrel_validation_requires_existing_direct_evidence_phrase_and_page() -> None:
    case = _case()
    validate_cases_against_chunks([case], [_chunk()])

    with pytest.raises(EvaluationError, match="missing frozen chunk ID"):
        validate_cases_against_chunks([_case(relevant_chunk_ids=["missing"])], [_chunk()])
    with pytest.raises(EvaluationError, match="expected phrase is absent"):
        validate_cases_against_chunks([_case(expected_phrases=["not in chunk"])], [_chunk()])
    with pytest.raises(EvaluationError, match="expected pages do not match"):
        validate_cases_against_chunks([_case(expected_pages=[2])], [_chunk()])


def test_same_page_unrelated_chunk_is_diagnostic_only_not_direct_hit() -> None:
    case = _case()
    rows = evaluate_cases(
        [case],
        lambda question, limit, document_id: [_result("wrong", page_numbers=[11])],
        candidate_limit=5,
    )["per_query"]

    assert rows[0]["direct_evidence_rank"] is None
    assert rows[0]["diagnostic_page_rank"] == 1
    assert rows[0]["diagnostic_phrase_rank"] is None
    assert aggregate_rows(rows, candidate_limit=5)["hit_rate_at_5"] == 0.0


def test_direct_rank_uses_first_of_multiple_relevant_chunk_ids() -> None:
    results = [_result("wrong"), _result("second"), _result("first")]

    assert direct_evidence_rank(results, {"first", "second"}) == 2


def test_evaluator_calculates_hit_mrr_group_metrics_and_failure_diagnostics() -> None:
    cases = [
        _case(
            id="vi-hit",
            language="vi",
            category="exact_technical_term",
            question="Vietnamese hit",
        ),
        _case(id="en-rank-3", relevant_chunk_ids=["third"], question="English rank three"),
        _case(id="miss", relevant_chunk_ids=["missing"], critical=True, question="English miss"),
    ]
    results = {
        "Vietnamese hit": [_result("evidence", text="threshold is 3.338")],
        "English rank three": [_result("one"), _result("two"), _result("third")],
        "English miss": [_result("wrong")],
    }

    report = evaluate_cases(
        cases,
        lambda question, limit, document_id: results[question],
        candidate_limit=20,
    )

    assert report["overall"]["hit_rate_at_1"] == pytest.approx(1 / 3)
    assert report["overall"]["hit_rate_at_3"] == pytest.approx(2 / 3)
    assert report["overall"]["hit_rate_at_5"] == pytest.approx(2 / 3)
    assert report["overall"]["hit_rate_at_candidate_limit"] == pytest.approx(2 / 3)
    assert report["overall"]["candidate_recall_at_candidate_limit"] == pytest.approx(2 / 3)
    assert report["overall"]["mrr_at_5"] == pytest.approx((1 + 1 / 3) / 3)
    assert report["overall"]["mrr_at_candidate_limit"] == pytest.approx((1 + 1 / 3) / 3)
    assert report["per_language"]["vi"]["query_count"] == 1
    assert report["per_language"]["en"]["query_count"] == 2
    assert report["per_retrieval_scenario"]["cross_lingual"]["query_count"] == 2
    assert report["per_retrieval_scenario"]["monolingual"]["query_count"] == 1
    assert [row["id"] for row in report["failure_cases"]] == ["miss"]
    assert report["critical_questions"][0]["direct_evidence_rank"] is None
    assert report["critical_metrics"]["hit_rate_at_5"] == 0.0


def test_percentiles_use_nearest_rank_and_reject_invalid_inputs() -> None:
    assert percentile_nearest_rank([1, 2, 3, 4, 5], 50) == 3
    assert percentile_nearest_rank([1, 2, 3, 4, 5], 95) == 5
    with pytest.raises(EvaluationError, match="empty sample"):
        percentile_nearest_rank([], 50)
    with pytest.raises(EvaluationError, match="range"):
        percentile_nearest_rank([1], 0)


def test_evaluator_requires_at_least_five_candidates() -> None:
    with pytest.raises(EvaluationError, match="at least 5"):
        evaluate_cases([_case()], lambda question, limit, document_id: [], candidate_limit=4)


@pytest.mark.parametrize("strategy", ["dense", "sparse", "hybrid"])
def test_evaluation_cli_accepts_all_comparable_retrieval_strategies(strategy: str) -> None:
    args = evaluate_cli._build_parser().parse_args(["--strategy", strategy, "--limit", "20"])

    assert args.strategy == strategy
    assert args.limit == 20

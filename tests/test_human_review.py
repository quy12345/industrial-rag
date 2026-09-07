"""Offline tests for deterministic review cohorts and judge agreement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.application.query_service import QueryExecution, QueryTimings
from app.contracts.query import Citation, QueryResponse
from app.domain.retrieval import RetrievalCandidate
from evaluation.dataset import read_dataset
from evaluation.human_review import (
    HumanReviewError,
    PrivateReviewPack,
    select_judge_validation_cohort,
    selected_ids_sha256,
    validate_scorecard_agreement,
)
from evaluation.semantic import MetricScore, RagasScores

DEVELOPMENT_PATH = Path("data/eval/atv320/development-v1.jsonl")


def _candidate() -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id="c1",
        document_id="manual",
        filename="manual.pdf",
        text="RAW_EVIDENCE_SENTINEL",
        page_numbers=[1],
        headings=["Safety"],
        content_type="text",
        score=1,
    )


def _execution() -> QueryExecution:
    candidate = _candidate()
    return QueryExecution(
        response=QueryResponse(
            answer="RAW_ANSWER_SENTINEL",
            abstained=False,
            citations=[
                Citation(
                    chunk_id="c1",
                    document_id="manual",
                    filename="manual.pdf",
                    page_numbers=[1],
                    headings=["Safety"],
                    excerpt="excerpt",
                )
            ],
        ),
        timings=QueryTimings(1, 2, 3, 4, 5, 15),
        usage=None,
        candidates=(candidate,),
        candidate_pool=(candidate,),
        evidence_candidates=(candidate,),
        evidence_context_chars=20,
    )


def _scores(value: float = 1.0) -> RagasScores:
    metric = MetricScore("complete", value, 1)
    return RagasScores(metric, metric, metric, metric, metric)


def test_fixed_cohort_is_deterministic_balanced_paired_and_covers_types() -> None:
    items = read_dataset(DEVELOPMENT_PATH)

    first = select_judge_validation_cohort(items)
    second = select_judge_validation_cohort(items)

    assert [item.id for item in first] == [item.id for item in second]
    assert len(first) == 40
    assert sum(item.answerable for item in first) == 30
    assert sum(not item.answerable for item in first) == 10
    assert sum(item.language == "en" for item in first) == 20
    assert len({item.fact_family_id for item in first}) == 20
    assert {item.question_type for item in first if item.answerable} == {
        "mounting",
        "safety",
        "menu_navigation",
        "parameter_code",
        "fault_diagnosis",
        "wiring",
        "table_lookup",
        "ordered_procedure",
        "cross_document",
    }
    assert len(selected_ids_sha256(first)) == 64


def test_private_pack_is_atomic_private_and_resumable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    item = next(
        row for row in read_dataset(Path(__file__).parents[1] / DEVELOPMENT_PATH) if row.answerable
    ).model_copy(update={"reference_answer_review_status": "approved"})
    path = Path("artifacts/local-review/judge-validation-v1/review.md")
    pack = PrivateReviewPack.open(path, selected_ids=[item.id], checkpoint_ids=[])

    pack.append(item=item, execution=_execution(), scores=_scores(), max_context_chars=4_000)

    markdown = path.read_text(encoding="utf-8")
    labels = path.with_name("labels.jsonl").read_text(encoding="utf-8")
    assert item.question in markdown
    assert "RAW_ANSWER_SENTINEL" in markdown
    assert "RAW_EVIDENCE_SENTINEL" in markdown
    assert json.loads(labels)["id"] == item.id
    assert not list(path.parent.glob(".*.tmp"))
    resumed = PrivateReviewPack.open(path, selected_ids=[item.id], checkpoint_ids=[item.id])
    assert resumed._completed_ids == [item.id]


def test_private_pack_rejects_outside_path_and_unrelated_existing_files(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(HumanReviewError, match="artifacts/local-review"):
        PrivateReviewPack.open(Path("review.md"), selected_ids=["a"], checkpoint_ids=[])

    path = Path("artifacts/local-review/review.md")
    path.parent.mkdir(parents=True)
    path.write_text("unrelated", encoding="utf-8")
    with pytest.raises(HumanReviewError, match="incomplete existing"):
        PrivateReviewPack.open(path, selected_ids=["a"], checkpoint_ids=[])


def _case(
    item_id: str,
    *,
    answerable: bool,
    abstained: bool,
    factual: float | None = 1.0,
    faithfulness: float | None = 1.0,
    recall: float | None = 1.0,
) -> dict:
    def metric(value: float | None) -> dict:
        return {
            "status": "complete" if value is not None else "not_applicable",
            "score": value,
            "latency_ms": 1 if value is not None else 0,
        }

    return {
        "id": item_id,
        "answerable": answerable,
        "abstained": abstained,
        "ragas": {
            "factual_correctness": metric(factual),
            "faithfulness": metric(faithfulness),
            "context_recall": metric(recall),
        },
    }


def _labels(cases: list[dict]) -> list[dict]:
    return [
        {
            "id": case["id"],
            "answer_correct": True if case["answerable"] and not case["abstained"] else None,
            "citation_supported": True if case["answerable"] and not case["abstained"] else None,
            "context_complete": True if case["answerable"] else None,
            "abstention_correct": True,
        }
        for case in cases
    ]


def _artifact(cases: list[dict]) -> dict:
    return {
        "schema_version": 17,
        "run_identity": {"dataset_sha256": "a" * 64, "private": "NO_RAW_CONTENT"},
        "cases": cases,
    }


def test_assessment_is_descriptive_even_with_perfect_agreement_and_is_sanitized() -> None:
    cases = [_case(f"a{index}", answerable=True, abstained=False) for index in range(30)] + [
        _case(
            f"u{index}",
            answerable=False,
            abstained=True,
            factual=None,
            faithfulness=None,
            recall=None,
        )
        for index in range(10)
    ]
    result = validate_scorecard_agreement(_artifact(cases), _labels(cases))

    assert result["status"] == "REVIEW_REQUIRED"
    assert result["schema_version"] == 2
    assert result["answered_answerable_count"] == 30
    assert all(row["agreement_rate"] == 1 for row in result["rubrics"].values())
    encoded = json.dumps(result)
    assert "NO_RAW_CONTENT" not in encoded


def test_agreement_reports_confusion_and_rejects_incomplete_applicable_labels() -> None:
    cases = [_case(f"a{index}", answerable=True, abstained=False) for index in range(20)]
    labels = _labels(cases)
    labels[0]["answer_correct"] = False
    result = validate_scorecard_agreement(_artifact(cases), labels)
    assert "answer_correct" not in result["rubrics"]
    assert result["factual_correctness_by_human_label"]["human_incorrect"] == {
        "count": 1,
        "mean": 1.0,
        "median": 1.0,
    }
    assert result["factual_correctness_review_item_ids"] == ["a0"]

    labels[0]["answer_correct"] = None
    with pytest.raises(HumanReviewError, match="incomplete"):
        validate_scorecard_agreement(_artifact(cases), labels)


def test_low_f1_is_not_a_binary_wrong_answer_verdict_and_citation_support_is_separate():
    cases = [_case("a1", answerable=True, abstained=False, factual=0.4)]
    labels = _labels(cases)
    labels[0]["citation_supported"] = False  # valid ID, wrong purpose according to human
    result = validate_scorecard_agreement(_artifact(cases), labels)
    assert result["factual_correctness_by_human_label"]["human_correct"] == {
        "count": 1,
        "mean": 0.4,
        "median": 0.4,
    }
    assert result["cases"][0]["human_answer_correct"] is True
    assert "automatic_answer_correct" not in result["cases"][0]
    assert result["rubrics"]["citation_supported"]["fp"] == 1
    assert result["status"] == "REVIEW_REQUIRED"


@pytest.mark.parametrize("score", [None, float("nan"), float("inf")])
def test_assessment_rejects_missing_or_nonfinite_applicable_scores(score):
    cases = [_case("a1", answerable=True, abstained=False, factual=score)]
    with pytest.raises(HumanReviewError, match="missing or invalid"):
        validate_scorecard_agreement(_artifact(cases), _labels(cases))


def test_answerable_abstention_is_a_system_miss_even_if_generator_refusal_was_safe():
    cases = [_case("a1", answerable=True, abstained=True, factual=None, faithfulness=None)]
    labels = _labels(cases)
    labels[0]["abstention_correct"] = False
    result = validate_scorecard_agreement(_artifact(cases), labels)
    assert result["cases"][0]["automatic_answerability_outcome"] is False
    assert result["rubrics"]["abstention_correct"]["tn"] == 1

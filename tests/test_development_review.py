"""Offline contracts for local development failure review."""

from __future__ import annotations

from copy import deepcopy
from io import StringIO

import pytest

from app.domain.documents import DocumentChunk
from app.infrastructure.corpus_artifacts import chunk_set_metadata
from evaluation.dataset import EvaluationItem, dataset_sha256
from evaluation.development_review import DevelopmentReviewError, build_development_review
from scripts.evaluation import review_development


def _item() -> EvaluationItem:
    return EvaluationItem(
        id="dev-001",
        fact_family_id="family-001",
        question="SENTINEL_QUESTION",
        language="en",
        answerable=True,
        scenario="en_to_en",
        question_type="safety",
        expected_document_ids=["installation"],
        relevant_chunk_ids=["qrel"],
        expected_pages=[7],
        expected_phrases=["SENTINEL_EXPECTED_PHRASE"],
        expected_answer_facts=[
            {"id": "fact-001", "aliases": ["SENTINEL_EXPECTED_FACT"]}
        ],
        citation_required=True,
        review_status="approved",
    )


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="qrel",
            document_id="installation",
            filename="installation.pdf",
            text="SENTINEL_QREL_TEXT",
            page_numbers=[7],
            headings=["Safety"],
        ),
        DocumentChunk(
            chunk_id="evidence",
            document_id="programming",
            filename="programming.pdf",
            text="SENTINEL_EVIDENCE_TEXT",
            page_numbers=[8],
            headings=["Procedure"],
        ),
    ]


def _artifact(item: EvaluationItem, chunks: list[DocumentChunk]) -> dict:
    return {
        "schema_version": 11,
        "run_identity": {
            "dataset": "development",
            "dataset_sha256": dataset_sha256([item]),
            "corpus": chunk_set_metadata(chunks),
            "generation_model": "fake-model",
            "generation_experiment": {"id": "baseline-v1"},
        },
        "per_query": [
            {
                "id": item.id,
                "repeat_index": 1,
                "fact_family_id": item.fact_family_id,
                "language": item.language,
                "question_type": item.question_type,
                "scenario": item.scenario,
                "answerable": True,
                "abstained": False,
                "abstention_reason": None,
                "failure_class": "reranker_miss_top5",
                "citation_ids": ["evidence"],
                "citation_document_ids": ["programming"],
                "missing_answer_fact_ids": ["fact-001"],
                "answer_fact_results": [{"id": "fact-001", "deterministic_matched": False}],
                "candidate_direct_evidence_rank": 2,
                "ranked_direct_evidence_rank": 8,
                "direct_evidence_rank": None,
                "unexpected_citation_document_ids": ["programming"],
                "evidence_candidate_ids": ["evidence"],
                "evidence_rendering": {"truncated": False},
                "retrieval_trace": {
                    "removed_by_candidate_selection_ids": [],
                    "removed_by_same_document_dedup_ids": [],
                },
            }
        ],
    }


def test_builds_terminal_only_review_with_local_question_facts_and_evidence() -> None:
    item = _item()
    chunks = _chunks()

    review = build_development_review(
        dataset=[item],
        chunks=chunks,
        artifact=_artifact(item, chunks),
        item_ids=[item.id],
    )

    assert review["review_policy"] == {
        "dataset": "development",
        "destination": "terminal_stdout_only",
        "provider_calls": "none",
        "raw_answer": "unavailable_in_sanitized_artifact",
        "interpretation": "diagnostic_hints_require_human_confirmation",
    }
    case = review["cases"][0]
    assert case["item"]["question"] == "SENTINEL_QUESTION"
    assert case["expected"]["answer_facts"][0]["aliases"] == [
        "SENTINEL_EXPECTED_FACT"
    ]
    assert case["expected"]["qrel_chunks"][0]["text"] == "SENTINEL_QREL_TEXT"
    assert case["generation_evidence"][0]["source_id"] == "S1"
    assert case["generation_evidence"][0]["text"] == "SENTINEL_EVIDENCE_TEXT"
    assert case["observed"]["raw_answer_available"] is False
    assert case["diagnostic_hints"] == [
        "qrel_did_not_reach_generation_evidence",
        "citation_provenance_mismatch",
    ]


def test_rejects_non_development_artifact_before_exposing_local_content() -> None:
    item = _item()
    chunks = _chunks()
    artifact = _artifact(item, chunks)
    artifact["run_identity"]["dataset"] = "confirmation"

    with pytest.raises(DevelopmentReviewError, match="Only development"):
        build_development_review(
            dataset=[item],
            chunks=chunks,
            artifact=artifact,
            item_ids=[item.id],
        )


@pytest.mark.parametrize("identity_field", ["dataset_sha256", "chunk_ids_sha256"])
def test_rejects_mismatched_dataset_or_corpus_identity(identity_field: str) -> None:
    item = _item()
    chunks = _chunks()
    artifact = _artifact(item, chunks)
    target = (
        artifact["run_identity"]
        if identity_field == "dataset_sha256"
        else artifact["run_identity"]["corpus"]
    )
    target[identity_field] = "0" * 64

    with pytest.raises(DevelopmentReviewError, match="does not match"):
        build_development_review(
            dataset=[item],
            chunks=chunks,
            artifact=artifact,
            item_ids=[item.id],
        )


def test_rejects_duplicate_ids_and_missing_repeat_records() -> None:
    item = _item()
    chunks = _chunks()
    artifact = _artifact(item, chunks)

    with pytest.raises(DevelopmentReviewError, match="must be unique"):
        build_development_review(
            dataset=[item],
            chunks=chunks,
            artifact=artifact,
            item_ids=[item.id, item.id],
        )
    with pytest.raises(DevelopmentReviewError, match="repeat 2"):
        build_development_review(
            dataset=[item],
            chunks=chunks,
            artifact=artifact,
            item_ids=[item.id],
            repeat_index=2,
        )


def test_rejects_artifact_record_that_disagrees_with_dataset_contract() -> None:
    item = _item()
    chunks = _chunks()
    artifact = deepcopy(_artifact(item, chunks))
    artifact["per_query"][0]["language"] = "vi"

    with pytest.raises(DevelopmentReviewError, match="does not match dataset fields"):
        build_development_review(
            dataset=[item],
            chunks=chunks,
            artifact=artifact,
            item_ids=[item.id],
        )


def test_cli_json_output_preserves_unicode_when_stdout_cannot_be_reconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = StringIO()
    monkeypatch.setattr(review_development.sys, "stdout", stream)

    review_development._print_utf8_json({"question": "Cần ngắt nguồn điện nào?"})

    assert "Cần ngắt nguồn điện nào?" in stream.getvalue()
    assert "\\u1ea7" not in stream.getvalue()

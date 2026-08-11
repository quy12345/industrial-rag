"""Offline tests for the explicit calibration-v3 approval boundary."""

from __future__ import annotations

import pytest

from app.phase7 import Phase7DatasetItem, Phase7Error
from scripts.freeze_phase7_calibration_v3 import _approve_draft, _validate_ground_truth_preserved


def _item(*, review_status: str = "needs_human_review") -> Phase7DatasetItem:
    return Phase7DatasetItem(
        id="calibration-row",
        question="What is the identifier?",
        language="en",
        answerable=True,
        scenario="en_to_en",
        question_type="parameter_code",
        expected_document_ids=["document"],
        relevant_chunk_ids=["chunk"],
        expected_pages=[1],
        expected_phrases=["identifier"],
        expected_answer_facts=[
            {
                "id": "identifier",
                "aliases": ["ABC"],
                "type": "identifier",
                "acceptable_values": ["ABC"],
            }
        ],
        citation_required=True,
        review_status=review_status,
    )


def test_approval_only_changes_review_status() -> None:
    draft = [_item()]
    frozen = _approve_draft(draft)
    assert frozen[0].review_status == "approved"
    _validate_ground_truth_preserved(draft, frozen)


def test_approval_refuses_preapproved_draft_or_ground_truth_change() -> None:
    with pytest.raises(Phase7Error, match="needs_human_review"):
        _approve_draft([_item(review_status="approved")])
    with pytest.raises(Phase7Error, match="ground truth"):
        _validate_ground_truth_preserved(
            [_item()], [_item().model_copy(update={"expected_pages": [2]})]
        )

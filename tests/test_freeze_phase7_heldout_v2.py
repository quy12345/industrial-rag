"""Offline tests for the private replacement held-out approval boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.phase7 import Phase7DatasetItem, Phase7Error
from scripts.freeze_phase7_heldout_v2 import (
    _approve_draft,
    _git_commit,
    _require_private_path,
    _validate_ground_truth_preserved,
)


def _item(*, review_status: str = "needs_human_review") -> Phase7DatasetItem:
    return Phase7DatasetItem(
        id="heldout-v2-row",
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


def test_heldout_v2_approval_only_changes_review_status() -> None:
    draft = [_item()]
    frozen = _approve_draft(draft)
    assert frozen[0].review_status == "approved"
    _validate_ground_truth_preserved(draft, frozen)


def test_heldout_v2_approval_rejects_preapproved_or_mutated_ground_truth() -> None:
    with pytest.raises(Phase7Error, match="needs_human_review"):
        _approve_draft([_item(review_status="approved")])
    with pytest.raises(Phase7Error, match="ground truth"):
        _validate_ground_truth_preserved(
            [_item()], [_item().model_copy(update={"expected_pages": [2]})]
        )


def test_heldout_v2_paths_must_remain_private() -> None:
    _require_private_path(Path("data/eval/phase7/private-heldout-v2/draft.jsonl"))
    with pytest.raises(Phase7Error, match="must remain under"):
        _require_private_path(Path("data/eval/phase7/test.jsonl"))


def test_heldout_v2_manifest_uses_explicit_host_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHASE7_GIT_COMMIT", "abc123")
    assert _git_commit() == "abc123"

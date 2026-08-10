"""Offline tests for typed-fact draft activation safety."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.generate_phase7_fact_evaluator_readiness import _validate_preserved_ground_truth


def _item(**updates):
    values = {
        "id": "row",
        "question": "q",
        "language": "en",
        "answerable": True,
        "citation_required": True,
        "relevant_chunk_ids": ["chunk"],
        "expected_document_ids": ["document"],
        "expected_pages": [1],
        "expected_phrases": ["phrase"],
        "phrase_match_mode": "all",
        "question_type": "safety",
        "scenario": "en_to_en",
        "unanswerable_reason": None,
        "review_status": "needs_human_review",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_fact_readiness_requires_preserved_ground_truth_and_human_review() -> None:
    _validate_preserved_ground_truth([_item()], [_item()])
    with pytest.raises(ValueError, match="expected_pages"):
        _validate_preserved_ground_truth([_item()], [_item(expected_pages=[2])])
    with pytest.raises(ValueError, match="human review"):
        _validate_preserved_ground_truth([_item()], [_item(review_status="approved")])

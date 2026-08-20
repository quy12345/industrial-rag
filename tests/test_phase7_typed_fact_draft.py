"""Offline tests for the review-required Phase 7 typed calibration draft."""

from __future__ import annotations

import json

from app.phase7 import read_phase7_dataset
from scripts import draft_phase7_calibration_fact_types as draft


def test_typed_draft_preserves_retrieval_ground_truth_and_requires_review(
    monkeypatch, tmp_path
) -> None:
    source = "data/eval/phase7/calibration.jsonl"
    output = tmp_path / "calibration-v3-draft.jsonl"
    monkeypatch.setattr("sys.argv", ["draft", "--input", source, "--output", str(output)])
    assert draft.main() == 0
    original = {item.id: item for item in read_phase7_dataset(source)}
    drafted = {item.id: item for item in read_phase7_dataset(output)}
    assert drafted.keys() == original.keys()
    for identifier, item in drafted.items():
        assert item.relevant_chunk_ids == original[identifier].relevant_chunk_ids
        assert item.expected_pages == original[identifier].expected_pages
        assert item.expected_phrases == original[identifier].expected_phrases
    assert drafted["phase7_calibration_009"].expected_answer_facts[0].type == "identifier"
    assert drafted["phase7_calibration_001"].expected_answer_facts[0].required_token_groups
    assert drafted["phase7_calibration_001"].review_status == "needs_human_review"
    assert "Phase 7.4 typed-fact draft" in drafted["phase7_calibration_001"].annotation_notes
    assert all(json.loads(line) for line in output.read_text(encoding="utf-8").splitlines())

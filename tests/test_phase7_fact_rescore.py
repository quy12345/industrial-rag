"""Offline tests for sanitized Phase 7 fact rescoring."""

from __future__ import annotations

import pytest

from scripts.rescore_phase7_calibration_facts import rescore_records


def test_rescore_separates_strict_phrase_from_deterministic_token_set() -> None:
    records = [
        {
            "id": "a",
            "answerable": True,
            "answer_fact_results": [
                {"id": "fact", "matched": False, "max_alias_token_recall": 1.0}
            ],
        }
    ]
    rows = rescore_records(records, {"a": object()})
    assert rows == [
        {
            "id": "a",
            "strict_phrase_match": False,
            "deterministic_fact_match": True,
            "strict_missing_fact_ids": ["fact"],
            "deterministic_missing_fact_ids": [],
        }
    ]


def test_rescore_rejects_missing_or_mismatched_source_records() -> None:
    with pytest.raises(ValueError, match="IDs differ"):
        rescore_records([], {"a": object()})
    with pytest.raises(ValueError, match="no fact diagnostics"):
        rescore_records([{"id": "a", "answerable": True}], {"a": object()})

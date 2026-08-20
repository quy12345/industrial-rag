"""Offline tests for worst-run Phase 7 calibration stability aggregation."""

from __future__ import annotations

import json

import pytest

from scripts.aggregate_phase7_calibration_stability import aggregate_stability_runs


def _payload(run: int, *, failed_ids=(), identity_suffix="same") -> dict:
    rows = []
    for index in range(1, 13):
        identifier = f"phase7_calibration_{index:03d}"
        rows.append(
            {
                "id": identifier,
                "answerable": True,
                "deterministic_fact_match": identifier not in failed_ids,
                "citation_document_correct": True,
                "candidate_direct_evidence_rank": 1,
                "ranked_direct_evidence_rank": 1,
                "direct_evidence_rank": 1,
                "evidence_candidate_ids": [f"chunk-{index}"],
            }
        )
    for index in range(13, 21):
        rows.append(
            {
                "id": f"phase7_calibration_{index:03d}",
                "answerable": False,
                "deterministic_fact_match": None,
                "citation_document_correct": None,
                "candidate_direct_evidence_rank": None,
                "ranked_direct_evidence_rank": None,
                "direct_evidence_rank": None,
                "evidence_candidate_ids": [],
            }
        )
    return {
        "schema_version": 5,
        "timestamp": f"run-{run}",
        "run_identity": {"dataset": "calibration", "frozen": identity_suffix},
        "overall": {
            "citations": {
                "referential_valid_rate_when_answered": 1.0,
                "unsupported_citation_count": 0,
                "wrong_document_citation_count": 0,
            },
            "abstention": {"precision": 1.0, "recall": 1.0},
        },
        "per_query": rows,
        "sanitization": {
            "raw_question": "excluded",
            "raw_answer": "excluded",
            "evidence_text": "excluded",
            "provider_response": "excluded",
        },
    }


def _write_runs(tmp_path, payloads):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, payload in enumerate(payloads, start=1):
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    return paths


def test_stability_uses_all_three_runs_and_worst_run_gate(tmp_path) -> None:
    paths = _write_runs(tmp_path, [_payload(index) for index in range(1, 4)])
    result = aggregate_stability_runs(paths)
    assert result["quality_gates"]["overall_pass"] is True
    assert result["fact_accuracy"] == {
        "counts": [12, 12, 12],
        "minimum": 12,
        "maximum": 12,
        "mean": 12,
        "denominator": 12,
    }
    assert result["per_item"]["phase7_calibration_010"][
        "deterministic_fact_pass_count"
    ] == 3


def test_stability_fails_when_worst_run_has_only_ten_fact_matches(tmp_path) -> None:
    failed = {"phase7_calibration_001", "phase7_calibration_002"}
    paths = _write_runs(
        tmp_path,
        [_payload(1), _payload(2, failed_ids=failed), _payload(3)],
    )
    result = aggregate_stability_runs(paths)
    assert result["fact_accuracy"]["minimum"] == 10
    assert result["quality_gates"]["overall_pass"] is False


def test_stability_rejects_identity_mismatch_duplicate_and_raw_answer(tmp_path) -> None:
    mismatched = _write_runs(
        tmp_path,
        [_payload(1), _payload(2, identity_suffix="changed"), _payload(3)],
    )
    with pytest.raises(ValueError, match="identities"):
        aggregate_stability_runs(mismatched)
    with pytest.raises(ValueError, match="distinct"):
        aggregate_stability_runs([mismatched[0], mismatched[0], mismatched[2]])

    raw = _payload(2)
    raw["per_query"][0]["answer"] = "must not be stored"
    paths = _write_runs(tmp_path / "raw", [_payload(1), raw, _payload(3)])
    with pytest.raises(ValueError, match="forbidden"):
        aggregate_stability_runs(paths)

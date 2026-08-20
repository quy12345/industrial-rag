"""Offline tests for the fixed-evidence calibration-005 diagnostic."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.errors import LLMTimeoutError
from app.generation import EvidenceBundle, GeneratedAnswer, GenerationResult
from app.models import RetrievalCandidate
from app.phase7 import ExpectedAnswerFact
from scripts.diagnose_phase7_calibration_005 import (
    _run_fixed_evidence_attempts,
    _validated_private_debug_path,
)


class FakeGenerator:
    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)
        self.evidence_objects = []

    def generate(self, *, question, evidence, validation_errors=()):
        self.evidence_objects.append(evidence)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def _evidence() -> EvidenceBundle:
    candidate = RetrievalCandidate(
        chunk_id="chunk-a",
        document_id="manual-a",
        filename="manual.pdf",
        text="Protective equipment is installed.",
        page_numbers=[1],
        headings=["Safety"],
        content_type="text",
        score=1.0,
    )
    return EvidenceBundle(text="fixed", source_map={"S1": candidate})


def _item():
    return type(
        "DiagnosticItem",
        (),
        {
            "question": "Is it installed?",
            "expected_answer_facts": [
                ExpectedAnswerFact(id="guard", aliases=["protective equipment installed"])
            ],
        },
    )()


def test_three_attempts_reuse_one_evidence_and_keep_raw_answer_private() -> None:
    generator = FakeGenerator(
        [
            GenerationResult(
                GeneratedAnswer(
                    answer="Protective equipment is installed.",
                    source_ids=["S1"],
                    insufficient_evidence=False,
                )
            ),
            GenerationResult(
                GeneratedAnswer(
                    answer="Insufficient evidence.",
                    source_ids=[],
                    insufficient_evidence=True,
                )
            ),
            LLMTimeoutError("timeout"),
        ]
    )
    evidence = _evidence()
    sanitized, private = _run_fixed_evidence_attempts(
        generator,
        item=_item(),
        evidence=evidence,
        attempts=3,
    )
    assert generator.evidence_objects == [evidence, evidence, evidence]
    assert [row["status"] for row in sanitized] == [
        "completed",
        "model_abstention",
        "provider_timeout",
    ]
    assert all("answer" not in row for row in sanitized)
    assert private[0]["answer"] == "Protective equipment is installed."


def test_diagnostic_rejects_non_three_attempt_count() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        _run_fixed_evidence_attempts(
            FakeGenerator([]),
            item=_item(),
            evidence=_evidence(),
            attempts=2,
        )


def test_raw_debug_path_must_remain_in_ignored_private_directory(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    allowed = Path("artifacts/private-debug/raw.json")
    assert _validated_private_debug_path(allowed) == (tmp_path / allowed).resolve()
    with pytest.raises(ValueError, match="private-debug"):
        _validated_private_debug_path(Path("artifacts/metrics/raw.json"))

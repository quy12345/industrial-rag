"""Offline contracts for preregistered generation experiments."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.application.generation_prompt import HUMAN_PROMPT, SYSTEM_PROMPT
from app.domain.citations import validate_generated_answer
from app.domain.generation import EvidenceBundle, GeneratedAnswer
from app.domain.retrieval import RetrievalCandidate
from app.errors import GenerationValidationError
from evaluation.generation_experiments import (
    BASELINE_GENERATION_VARIANT,
    CLAIM_COVERAGE_GENERATION_VARIANT,
    CLAIM_COVERAGE_SYSTEM_PROMPT,
    GENERATION_EXPERIMENTS,
    ClaimCoverageAnswer,
    GroundedClaim,
    generation_experiment_for,
    map_claim_coverage_output,
)


def _candidate(chunk_id: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="manual-a",
        filename="manual.pdf",
        text=f"Evidence for {chunk_id}.",
        page_numbers=[1],
        headings=["Evidence"],
        content_type="text",
        score=1.0,
        rerank_score=1.0,
        rerank_rank=1,
    )


def _evidence() -> EvidenceBundle:
    return EvidenceBundle(
        text="sanitized-evidence",
        source_map={
            "S1": _candidate("chunk-1"),
            "S2": _candidate("chunk-2"),
            "S3": _candidate("chunk-3"),
        },
    )


def test_registry_has_exactly_two_immutable_preregistered_experiments() -> None:
    assert tuple(GENERATION_EXPERIMENTS) == (
        BASELINE_GENERATION_VARIANT,
        CLAIM_COVERAGE_GENERATION_VARIANT,
    )
    baseline = generation_experiment_for(BASELINE_GENERATION_VARIANT)
    claim_coverage = generation_experiment_for(CLAIM_COVERAGE_GENERATION_VARIANT)
    assert baseline.system_prompt == SYSTEM_PROMPT
    assert baseline.human_prompt == HUMAN_PROMPT
    assert baseline.output_schema is GeneratedAnswer
    assert baseline.output_mapper is None
    assert claim_coverage.system_prompt == CLAIM_COVERAGE_SYSTEM_PROMPT
    assert claim_coverage.output_schema is ClaimCoverageAnswer
    assert claim_coverage.output_mapper is map_claim_coverage_output
    with pytest.raises(TypeError):
        GENERATION_EXPERIMENTS["extra"] = baseline  # type: ignore[index]
    with pytest.raises(ValueError, match="Unknown generation experiment"):
        generation_experiment_for("unknown")


def test_claim_prompt_is_generic_and_contains_no_evaluation_answers() -> None:
    normalized = CLAIM_COVERAGE_SYSTEM_PROMPT.casefold()
    assert "condition, value, unit, identifier" in normalized
    assert "ordered step" in normalized
    for forbidden in (
        "expected_answer_facts",
        "expected phrase",
        "relevant_chunk_ids",
        "qrel",
    ):
        assert forbidden not in normalized


@pytest.mark.parametrize(
    "payload",
    [
        {"claims": [], "insufficient_evidence": False},
        {
            "claims": [{"text": " ", "source_ids": ["S1"]}],
            "insufficient_evidence": False,
        },
        {
            "claims": [{"text": "Claim", "source_ids": []}],
            "insufficient_evidence": False,
        },
        {
            "claims": [{"text": "Claim", "source_ids": ["S1", "S1"]}],
            "insufficient_evidence": False,
        },
        {
            "claims": [{"text": "Claim", "source_ids": ["S1"]}],
            "insufficient_evidence": True,
        },
    ],
)
def test_claim_schema_rejects_incomplete_or_contradictory_output(payload) -> None:
    with pytest.raises(ValidationError):
        ClaimCoverageAnswer.model_validate(payload)


def test_claim_mapper_preserves_claim_and_first_source_order() -> None:
    output = ClaimCoverageAnswer(
        claims=[
            GroundedClaim(text="First fact.", source_ids=["S2", "S1"]),
            GroundedClaim(text="Second fact.", source_ids=["S1", "S3"]),
        ],
        insufficient_evidence=False,
    )

    mapped = map_claim_coverage_output(output, _evidence())

    assert mapped == GeneratedAnswer(
        answer="First fact.\nSecond fact.",
        source_ids=["S2", "S1", "S3"],
        insufficient_evidence=False,
    )
    validated = validate_generated_answer(mapped, source_map=_evidence().source_map)
    assert validated.source_ids == ("S2", "S1", "S3")


def test_claim_mapper_rejects_unknown_sources_without_exposing_claim_text() -> None:
    secret = "secret-claim-text"
    output = ClaimCoverageAnswer(
        claims=[GroundedClaim(text=secret, source_ids=["S9"])],
        insufficient_evidence=False,
    )

    with pytest.raises(GenerationValidationError) as caught:
        map_claim_coverage_output(output, _evidence())

    assert caught.value.errors == ("claim 1 uses unknown source ID: S9",)
    assert secret not in str(caught.value)


def test_claim_mapper_maps_valid_abstention_to_existing_contract() -> None:
    mapped = map_claim_coverage_output(
        ClaimCoverageAnswer(claims=[], insufficient_evidence=True),
        _evidence(),
    )

    assert mapped == GeneratedAnswer(
        answer="",
        source_ids=[],
        insufficient_evidence=True,
    )

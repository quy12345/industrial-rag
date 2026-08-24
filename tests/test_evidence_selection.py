"""Offline tests for deterministic final evidence selection."""

from __future__ import annotations

import pytest

import app.content_identity as compatibility_content_identity
import app.evidence_selection as evidence_facade
from app.domain import content_identity
from app.domain import evidence as evidence_policy
from app.domain.retrieval import RetrievalCandidate
from app.evidence_selection import EvidenceSelectionError, select_evidence_candidates
from app.query_service import EvidenceGate, EvidenceGateDecision


def test_evidence_content_identity_normalizes_and_hashes_exactly() -> None:
    text = "  Disconnect\tPOWER\r\n before  service. "

    assert compatibility_content_identity.normalize_evidence_content is (
        content_identity.normalize_evidence_content
    )
    assert compatibility_content_identity.evidence_content_fingerprint is (
        content_identity.evidence_content_fingerprint
    )
    assert content_identity.normalize_evidence_content(text) == (
        "disconnect power before service."
    )
    assert content_identity.evidence_content_fingerprint(text) == (
        "5e8167a98d18e8c913f4731198d5c80e1600085e409bc53d09ec13142fcd9575"
    )


def _candidate(
    chunk_id: str,
    *,
    document_id: str,
    role: str,
    text: str = "Disconnect the supply before wiring.",
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"{document_id}.pdf",
        text=text,
        page_numbers=[1],
        headings=["Safety"],
        content_type="text",
        score=1.0,
        rerank_rank=1,
        metadata={"document_role": role},
    )


def test_evidence_compatibility_exports_resolve_to_domain_policy() -> None:
    assert evidence_facade.EvidenceSelectionError is evidence_policy.EvidenceSelectionError
    assert evidence_facade.EvidenceDuplicateGroup is evidence_policy.EvidenceDuplicateGroup
    assert evidence_facade.EvidenceSelection is evidence_policy.EvidenceSelection
    assert evidence_facade.select_evidence_candidates is evidence_policy.select_evidence_candidates
    assert evidence_facade.select_evidence_candidates_for_role is (
        evidence_policy.select_evidence_candidates_for_role
    )
    assert EvidenceGate is evidence_policy.EvidenceGate
    assert EvidenceGateDecision is evidence_policy.EvidenceGateDecision


def test_cross_document_exact_duplicate_prefers_query_role_then_fills_top_k() -> None:
    programming = _candidate("programming-copy", document_id="programming", role="programming")
    installation = _candidate(
        "installation-copy", document_id="installation", role="installation"
    )
    unique = _candidate(
        "unique", document_id="installation", role="installation", text="Use terminal X1."
    )
    selection = select_evidence_candidates(
        "Which wiring terminal is required?",
        [programming, installation, unique],
        top_k=2,
    )
    assert [candidate.chunk_id for candidate in selection.candidates] == [
        "installation-copy",
        "unique",
    ]
    assert selection.candidates[0].metadata["equivalent_chunk_ids"] == (
        "installation-copy",
        "programming-copy",
    )
    assert selection.duplicate_groups[0].representative_chunk_id == "installation-copy"


def test_neutral_query_uses_best_rank_and_near_duplicates_are_not_collapsed() -> None:
    first = _candidate("first", document_id="programming", role="programming")
    second = _candidate("second", document_id="installation", role="installation")
    near = _candidate(
        "near",
        document_id="installation",
        role="installation",
        text="Disconnect the supply before wiring!",
    )
    selection = select_evidence_candidates(
        "Give technical information.", [first, second, near], top_k=3
    )
    assert [candidate.chunk_id for candidate in selection.candidates] == ["first", "near"]


def test_same_document_duplicate_content_is_preserved() -> None:
    first = _candidate("first", document_id="installation", role="installation")
    second = _candidate("second", document_id="installation", role="installation")
    selection = select_evidence_candidates("wiring", [first, second], top_k=2)
    assert [candidate.chunk_id for candidate in selection.candidates] == ["first", "second"]
    assert selection.duplicate_groups == ()


def test_selection_rejects_invalid_inputs_without_fallback() -> None:
    candidate = _candidate("first", document_id="installation", role="installation")

    with pytest.raises(EvidenceSelectionError, match="top_k"):
        select_evidence_candidates("wiring", [candidate], top_k=0)
    with pytest.raises(EvidenceSelectionError, match="blank"):
        select_evidence_candidates(" ", [candidate], top_k=1)
    with pytest.raises(EvidenceSelectionError, match="unique"):
        select_evidence_candidates("wiring", [candidate, candidate], top_k=2)


def test_evidence_gate_returns_exact_metadata_and_threshold_decisions() -> None:
    candidate = _candidate("first", document_id="installation", role="installation")
    invalid_page = candidate.model_copy(update={"page_numbers": [0]})
    invalid_heading = candidate.model_copy(update={"headings": [" "]})
    non_finite = candidate.model_copy(update={"score": float("nan")})

    gate = EvidenceGate()
    assert gate.evaluate([candidate], requested_document_id="installation") == (
        EvidenceGateDecision(True)
    )
    assert gate.evaluate([], requested_document_id=None) == EvidenceGateDecision(
        False, "no_candidates"
    )
    for candidates, requested_document_id in (
        ([candidate], "programming"),
        ([candidate, candidate], None),
        ([invalid_page], None),
        ([invalid_heading], None),
        ([non_finite], None),
    ):
        assert gate.evaluate(candidates, requested_document_id=requested_document_id) == (
            EvidenceGateDecision(False, "invalid_candidate_metadata")
        )
    assert EvidenceGate(score_threshold=1.1).evaluate(
        [candidate], requested_document_id=None
    ) == EvidenceGateDecision(False, "configured_score_gate_failed")

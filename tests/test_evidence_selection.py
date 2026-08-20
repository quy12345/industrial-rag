"""Offline tests for deterministic final evidence selection."""

from __future__ import annotations

from app.evidence_selection import select_evidence_candidates
from app.models import RetrievalCandidate


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

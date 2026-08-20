"""Offline tests for referential citation validation and trusted metadata."""

from __future__ import annotations

import pytest

from app.citations import build_citations, validate_generated_answer
from app.errors import CitationValidationError
from app.generation import GeneratedAnswer
from app.models import RetrievalCandidate


def _candidate(chunk_id: str = "chunk-a", *, document_id: str = "manual-a"):
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        filename="manual.pdf",
        text="  Nội dung kỹ thuật 24 VDC và ký tự Unicode động cơ.  ",
        page_numbers=[3, 1, 3],
        headings=[" Safety ", "Limits"],
        content_type="text",
        score=1.0,
    )


def _output(answer="Có.", source_ids=None, insufficient=False):
    return GeneratedAnswer(
        answer=answer,
        source_ids=["S1"] if source_ids is None else source_ids,
        insufficient_evidence=insufficient,
    )


def test_valid_single_multiple_and_duplicate_source_ids() -> None:
    source_map = {"S1": _candidate("a"), "S2": _candidate("b")}
    validated = validate_generated_answer(
        _output(source_ids=["S2", "S1", "S2"]), source_map=source_map
    )
    assert validated.source_ids == ("S2", "S1")
    citations = build_citations(
        validated.source_ids,
        source_map=source_map,
        requested_document_id="manual-a",
        excerpt_max_chars=400,
    )
    assert [citation.chunk_id for citation in citations] == ["b", "a"]


@pytest.mark.parametrize(
    "output",
    [
        _output(source_ids=["S9"]),
        _output(source_ids=[]),
        _output(answer="   "),
        _output(source_ids=["S1"], insufficient=True),
    ],
)
def test_unknown_missing_empty_and_abstention_with_citation_are_rejected(output) -> None:
    with pytest.raises(CitationValidationError):
        validate_generated_answer(output, source_map={"S1": _candidate()})


def test_valid_model_abstention_has_no_sources() -> None:
    validated = validate_generated_answer(
        _output(answer="not enough", source_ids=[], insufficient=True),
        source_map={"S1": _candidate()},
    )
    assert validated.insufficient_evidence is True
    assert validated.source_ids == ()


def test_builder_normalizes_pages_headings_excerpt_and_unicode() -> None:
    citation = build_citations(
        ["S1"],
        source_map={"S1": _candidate()},
        requested_document_id="manual-a",
        excerpt_max_chars=24,
    )[0]
    assert citation.page_numbers == [1, 3]
    assert citation.headings == ["Safety", "Limits"]
    assert len(citation.excerpt) == 24
    assert citation.excerpt.endswith("…")
    citation.excerpt.encode("utf-8")


def test_builder_rejects_cross_document_unknown_and_empty_text() -> None:
    with pytest.raises(CitationValidationError, match="another document"):
        build_citations(
            ["S1"],
            source_map={"S1": _candidate(document_id="manual-b")},
            requested_document_id="manual-a",
            excerpt_max_chars=100,
        )
    with pytest.raises(CitationValidationError, match="unknown"):
        build_citations(
            ["S9"], source_map={}, requested_document_id=None, excerpt_max_chars=100
        )
    candidate = _candidate().model_copy(update={"text": " "})
    with pytest.raises(CitationValidationError, match="no citation text"):
        build_citations(
            ["S1"],
            source_map={"S1": candidate},
            requested_document_id=None,
            excerpt_max_chars=100,
        )


def test_builder_deduplicates_same_chunk_from_distinct_source_labels() -> None:
    candidate = _candidate()
    citations = build_citations(
        ["S1", "S2"],
        source_map={"S1": candidate, "S2": candidate},
        requested_document_id=None,
        excerpt_max_chars=100,
    )
    assert len(citations) == 1

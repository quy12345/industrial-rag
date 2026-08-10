"""Offline tests for provider-free Phase 7 retrieval calibration profiles."""

from __future__ import annotations

from app.models import RetrievalCandidate
from app.phase7 import Phase7DatasetItem
from scripts.calibrate_phase7_retrieval import (
    CandidateProfile,
    _build_profile,
    _score_profile,
    _summarize_profile,
)


def _candidate(
    chunk_id: str,
    *,
    dense_rank: int | None = None,
    sparse_rank: int | None = None,
    text: str | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="manual",
        filename="manual.pdf",
        text=text or chunk_id,
        page_numbers=[1],
        headings=[],
        content_type="text",
        score=1.0,
        dense_score=1.0 if dense_rank is not None else None,
        dense_rank=dense_rank,
        sparse_score=1.0 if sparse_rank is not None else None,
        sparse_rank=sparse_rank,
    )


def _item(relevant: str, *, language: str = "en") -> Phase7DatasetItem:
    return Phase7DatasetItem(
        id=f"item-{language}",
        question="Question",
        language=language,
        answerable=True,
        scenario="vi_to_en" if language == "vi" else "en_to_en",
        question_type="installation",
        expected_document_ids=["manual"],
        relevant_chunk_ids=[relevant],
        expected_pages=[1],
        expected_phrases=["evidence"],
        expected_answer_facts=[{"id": "fact", "aliases": ["answer"]}],
        citation_required=True,
        review_status="approved",
    )


def test_union_profile_slices_components_and_deduplicates_exact_content() -> None:
    dense = [
        _candidate("dense-1", dense_rank=1, text="same"),
        _candidate("dense-2", dense_rank=2),
    ]
    sparse = [
        _candidate("sparse-1", sparse_rank=1, text=" SAME "),
        _candidate("sparse-2", sparse_rank=2),
    ]
    profile = CandidateProfile("union", "union", 1, 2)
    result = _build_profile(profile, dense, sparse)
    assert [candidate.chunk_id for candidate in result] == ["dense-1", "sparse-2"]
    assert result[0].metadata["equivalent_chunk_ids"] == ["dense-1", "sparse-1"]


def test_rrf_profile_applies_final_limit_and_preserves_component_diagnostics() -> None:
    dense = [_candidate(f"d{rank}", dense_rank=rank) for rank in range(1, 4)]
    sparse = [_candidate(f"s{rank}", sparse_rank=rank) for rank in range(1, 4)]
    profile = CandidateProfile("rrf", "rrf", 3, 3, 2)
    candidates = _build_profile(profile, dense, sparse)
    assert len(candidates) == 2
    assert all(candidate.rrf_rank is not None for candidate in candidates)


def test_profile_scoring_and_summary_use_only_stable_qrel_ids() -> None:
    profile = CandidateProfile("union", "union", 20, 20)
    hit = _score_profile(
        _item("qrel"), profile, [_candidate("same-page"), _candidate("qrel", dense_rank=2)]
    )
    miss = _score_profile(_item("absent", language="vi"), profile, [_candidate("other")])
    summary = _summarize_profile(profile, [hit, miss])
    assert hit["direct_evidence_rank"] == 2
    assert hit["qrel_component_ranks"][0]["dense_rank"] == 2
    assert miss["direct_evidence_rank"] is None
    assert summary["candidate_recall"] == 0.5
    assert summary["missing_query_ids"] == ["item-vi"]
    assert summary["per_language"]["en"]["candidate_recall"] == 1.0

"""Framework-neutral retrieval candidate assembly policies."""

from __future__ import annotations

from collections.abc import Sequence

from app.models import RetrievalCandidate, RetrievedChunk


def dense_results_to_candidates(
    results: Sequence[RetrievedChunk],
) -> list[RetrievalCandidate]:
    """Map dense results to candidates with deterministic one-based dense ranks."""

    candidates = [
        RetrievalCandidate(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            filename=result.filename,
            text=result.text,
            page_numbers=result.page_numbers,
            headings=result.headings,
            content_type=result.content_type,
            score=result.score,
            dense_score=result.score,
        )
        for result in results
    ]
    return [
        candidate.model_copy(update={"dense_rank": rank})
        for rank, candidate in enumerate(
            sorted(
                candidates,
                key=lambda candidate: (-(candidate.dense_score or 0.0), candidate.chunk_id),
            ),
            start=1,
        )
    ]


def union_dense_sparse_candidates(
    dense_candidates: Sequence[RetrievalCandidate],
    sparse_candidates: Sequence[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    """Return a deterministic, unranked-for-fusion union without mixing raw scores."""

    merged: dict[str, RetrievalCandidate] = {}
    for candidate in dense_candidates:
        merged[candidate.chunk_id] = candidate
    for candidate in sparse_candidates:
        existing = merged.get(candidate.chunk_id)
        if existing is None:
            merged[candidate.chunk_id] = candidate
        else:
            merged[candidate.chunk_id] = existing.model_copy(
                update={
                    "sparse_score": candidate.sparse_score,
                    "sparse_rank": candidate.sparse_rank,
                }
            )
    return sorted(
        merged.values(),
        key=lambda candidate: (
            min(rank for rank in (candidate.dense_rank, candidate.sparse_rank) if rank is not None),
            candidate.chunk_id,
        ),
    )

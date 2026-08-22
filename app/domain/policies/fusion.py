"""Deterministic reciprocal-rank fusion policies."""

from __future__ import annotations

from collections.abc import Sequence

from app.errors import RetrievalError
from app.models import RetrievalCandidate


def fuse_rrf(
    dense_candidates: Sequence[RetrievalCandidate],
    sparse_candidates: Sequence[RetrievalCandidate],
    *,
    rrf_k: int,
    final_limit: int,
) -> list[RetrievalCandidate]:
    """Fuse component lists by one-based reciprocal rank without combining raw scores."""

    if rrf_k <= 0:
        raise RetrievalError("RRF k must be greater than 0.")
    if final_limit <= 0:
        raise RetrievalError("Hybrid final limit must be greater than 0.")

    merged: dict[str, RetrievalCandidate] = {}
    for candidate in dense_candidates:
        if candidate.dense_rank is None:
            raise RetrievalError("Dense RRF candidate has no one-based dense rank.")
        merged[candidate.chunk_id] = candidate.model_copy(
            update={"rrf_score": 1 / (rrf_k + candidate.dense_rank)}
        )
    for candidate in sparse_candidates:
        if candidate.sparse_rank is None:
            raise RetrievalError("Sparse RRF candidate has no one-based sparse rank.")
        contribution = 1 / (rrf_k + candidate.sparse_rank)
        existing = merged.get(candidate.chunk_id)
        if existing is None:
            merged[candidate.chunk_id] = candidate.model_copy(update={"rrf_score": contribution})
        else:
            merged[candidate.chunk_id] = existing.model_copy(
                update={
                    "sparse_score": candidate.sparse_score,
                    "sparse_rank": candidate.sparse_rank,
                    "rrf_score": (existing.rrf_score or 0.0) + contribution,
                }
            )

    ordered = sorted(
        merged.values(),
        key=lambda candidate: (
            -(candidate.rrf_score or 0.0),
            min(rank for rank in (candidate.dense_rank, candidate.sparse_rank) if rank is not None),
            candidate.chunk_id,
        ),
    )
    return [
        candidate.model_copy(update={"score": candidate.rrf_score, "rrf_rank": rank})
        for rank, candidate in enumerate(ordered[:final_limit], start=1)
    ]

"""Sanitized Phase 7 cross-encoder snapshot validation and rank-only replay."""

from __future__ import annotations

import math
from typing import Any

from app.models import RetrievalCandidate
from app.phase7_optimization import (
    Phase7OptimizationError,
    PostRerankConfidenceMode,
    QueryRole,
    RoleConfidence,
    apply_role_aware_rank_fusion,
)


class Phase7ReplayError(ValueError):
    """Raised when a sanitized reranker snapshot cannot be replayed safely."""


def snapshot_candidates_to_retrieval(
    values: list[dict[str, Any]],
) -> list[RetrievalCandidate]:
    """Validate sanitized candidate records and reconstruct rank-only inputs."""

    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    candidates: list[RetrievalCandidate] = []
    for value in values:
        try:
            chunk_id = str(value["chunk_id"])
            document_id = str(value["document_id"])
            document_role = str(value["document_role"])
            rank = int(value["cross_encoder_rank"])
            score = float(value["rerank_score"])
            dense_rank = _optional_rank(value.get("dense_rank"))
            sparse_rank = _optional_rank(value.get("sparse_rank"))
            rrf_rank = _optional_rank(value.get("rrf_rank"))
            content_fingerprint = value.get("content_fingerprint_sha256")
            query_identifier_matches = _non_negative_count(
                value.get("query_identifier_match_count")
            )
            bracketed_pair_count = _non_negative_count(
                value.get("bracketed_label_code_pair_count")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Phase7ReplayError("Snapshot candidate is malformed.") from exc
        if not chunk_id or not document_id or document_role not in {"installation", "programming"}:
            raise Phase7ReplayError("Snapshot candidate has invalid identity metadata.")
        if rank <= 0 or rank in seen_ranks or chunk_id in seen_ids:
            raise Phase7ReplayError(
                "Snapshot candidate ranks and chunk IDs must be unique and one-based."
            )
        if not math.isfinite(score):
            raise Phase7ReplayError("Snapshot rerank score must be finite.")
        if content_fingerprint is not None and (
            not isinstance(content_fingerprint, str)
            or len(content_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in content_fingerprint)
        ):
            raise Phase7ReplayError("Snapshot content fingerprint must be lowercase SHA-256.")
        seen_ids.add(chunk_id)
        seen_ranks.add(rank)
        candidates.append(
            RetrievalCandidate(
                chunk_id=chunk_id,
                document_id=document_id,
                filename="snapshot",
                text="",
                page_numbers=[],
                headings=[],
                content_type="snapshot",
                metadata={
                    "document_role": document_role,
                    "query_identifier_match_count": query_identifier_matches,
                    "bracketed_label_code_pair_count": bracketed_pair_count,
                    **(
                        {"content_fingerprint_sha256": content_fingerprint}
                        if content_fingerprint is not None
                        else {}
                    ),
                },
                score=score,
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
                rrf_rank=rrf_rank,
                rerank_score=score,
                rerank_rank=rank,
            )
        )
    if seen_ranks != set(range(1, len(candidates) + 1)):
        raise Phase7ReplayError("Snapshot cross-encoder ranks must be contiguous from one.")
    return sorted(candidates, key=lambda candidate: candidate.rerank_rank or 2**31)


def replay_role_prior(
    values: list[dict[str, Any]],
    *,
    query_role: QueryRole,
    confidence: RoleConfidence,
    role_multiplier: float,
    rrf_rank_multiplier: float = 0.0,
    rank_offset: int,
    confidence_mode: PostRerankConfidenceMode,
) -> list[RetrievalCandidate]:
    """Replay only the deterministic post-rerank role prior from a snapshot."""

    candidates = snapshot_candidates_to_retrieval(values)
    try:
        return apply_role_aware_rank_fusion(
            candidates,
            query_role=query_role,
            confidence=confidence,
            confidence_mode=confidence_mode,
            role_multiplier=role_multiplier,
            rrf_rank_multiplier=rrf_rank_multiplier,
            rank_offset=rank_offset,
        )
    except Phase7OptimizationError as exc:
        raise Phase7ReplayError(str(exc)) from exc


def _optional_rank(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Phase7ReplayError("Snapshot component ranks must be positive integers or null.")
    return value


def _non_negative_count(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Phase7ReplayError("Snapshot structural feature counts must be non-negative integers.")
    return value

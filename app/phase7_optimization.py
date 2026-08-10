"""Deterministic Phase 7.4 retrieval-profile helpers.

The module deliberately contains no Qdrant, embedding, reranker, dataset, or
provider dependency.  Query-role inference uses only query text and the
candidate selector only uses component ranks plus trusted document metadata.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

from app.models import RetrievalCandidate

QueryRole = Literal["installation", "programming", "neutral"]
QUERY_ROLE_PROFILE = "phase7_query_role_v1"


class Phase7OptimizationError(ValueError):
    """Raised when a bounded Phase 7 retrieval profile is invalid."""


@dataclass(frozen=True)
class QueryRoleInference:
    """Auditable role classification derived solely from the user query."""

    role: QueryRole
    installation_cues: tuple[str, ...]
    programming_cues: tuple[str, ...]


@dataclass(frozen=True)
class Phase7FusionProfile:
    """A bounded weighted-RRF profile used before cross-encoder reranking."""

    name: str
    rrf_k: int
    dense_weight: float
    sparse_weight: float
    role_multiplier: float
    dense_reserve: int
    sparse_reserve: int
    max_candidates: int = 30

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Phase 7 fusion profile name must not be blank.")
        if self.rrf_k <= 0 or self.max_candidates <= 0:
            raise ValueError("RRF k and candidate budget must be greater than zero.")
        if self.dense_weight <= 0 or self.sparse_weight <= 0:
            raise ValueError("Fusion component weights must be greater than zero.")
        if not 0 <= self.role_multiplier <= 0.25:
            raise ValueError("Role multiplier must be between 0 and 0.25.")
        if self.dense_reserve < 0 or self.sparse_reserve < 0:
            raise ValueError("Component reserves must not be negative.")


PHASE7_CALIBRATION_FUSION_PROFILE = Phase7FusionProfile(
    name="weighted_rrf_k40_s1.25_role0.1_d5_s24",
    rrf_k=40,
    dense_weight=1.0,
    sparse_weight=1.25,
    role_multiplier=0.10,
    dense_reserve=5,
    sparse_reserve=24,
)


INSTALLATION_CUES: tuple[str, ...] = (
    "safety",
    "mount",
    "install",
    "wiring",
    "wire",
    "contact",
    "run",
    "prevent",
    "rotate",
    "rotation",
    "motor",
    "shaft",
    "power",
    "terminal",
    "protection",
    "protective",
    "electrical",
    "an toan",
    "tranh",
    "truc",
    "dong co",
    "tiep diem",
    "lenh chay",
    "lap",
    "dau day",
    "nguon",
    "dau cuc",
    "bao ve",
    "dien",
)
PROGRAMMING_CUES: tuple[str, ...] = (
    "mode",
    "menu",
    "parameter",
    "configuration",
    "monitoring",
    "fault",
    "reference",
    "programming",
    "tham so",
    "cau hinh",
    "giam sat",
    "loi",
    "tham chieu",
)


def phase7_fusion_profile_grid() -> tuple[Phase7FusionProfile, ...]:
    """Return the finite, documented provider-free Phase 7.4 ablation grid."""

    profiles: list[Phase7FusionProfile] = []
    for rrf_k in (20, 40, 60, 80):
        for sparse_weight in (1.0, 1.25, 1.5):
            for role_multiplier in (0.0, 0.05, 0.10):
                for dense_reserve, sparse_reserve in ((5, 24), (7, 24), (7, 26), (10, 26)):
                    profiles.append(
                        Phase7FusionProfile(
                            name=(
                                f"weighted_rrf_k{rrf_k}_s{sparse_weight:g}_role"
                                f"{role_multiplier:g}_d{dense_reserve}_s{sparse_reserve}"
                            ),
                            rrf_k=rrf_k,
                            dense_weight=1.0,
                            sparse_weight=sparse_weight,
                            role_multiplier=role_multiplier,
                            dense_reserve=dense_reserve,
                            sparse_reserve=sparse_reserve,
                        )
                    )
    return tuple(profiles)


def infer_query_role(query: str) -> QueryRoleInference:
    """Infer a conservative manual role; conflicting/no cues remain neutral."""

    normalized = _normalized_query(query)
    installation = tuple(cue for cue in INSTALLATION_CUES if _cue_present(normalized, cue))
    programming = tuple(cue for cue in PROGRAMMING_CUES if _cue_present(normalized, cue))
    if installation and not programming:
        role: QueryRole = "installation"
    elif programming and not installation:
        role = "programming"
    else:
        role = "neutral"
    return QueryRoleInference(role, installation, programming)


def fuse_weighted_rrf(
    dense_candidates: list[RetrievalCandidate],
    sparse_candidates: list[RetrievalCandidate],
    *,
    profile: Phase7FusionProfile,
    query_role: QueryRole,
) -> list[RetrievalCandidate]:
    """Fuse one-based component ranks without combining incomparable raw scores."""

    merged: dict[str, RetrievalCandidate] = {}
    for candidate in dense_candidates:
        if candidate.dense_rank is None:
            raise Phase7OptimizationError(
                "Dense weighted-RRF candidate has no one-based dense rank."
            )
        merged[candidate.chunk_id] = candidate.model_copy(
            update={"rrf_score": profile.dense_weight / (profile.rrf_k + candidate.dense_rank)}
        )
    for candidate in sparse_candidates:
        if candidate.sparse_rank is None:
            raise Phase7OptimizationError(
                "Sparse weighted-RRF candidate has no one-based sparse rank."
            )
        contribution = profile.sparse_weight / (profile.rrf_k + candidate.sparse_rank)
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

    scored: list[RetrievalCandidate] = []
    for candidate in merged.values():
        score = candidate.rrf_score or 0.0
        if query_role != "neutral" and candidate.metadata.get("document_role") == query_role:
            score *= 1 + profile.role_multiplier
        scored.append(candidate.model_copy(update={"rrf_score": score, "score": score}))
    ordered = sorted(
        scored,
        key=lambda candidate: (
            -(candidate.rrf_score or 0.0),
            min(rank for rank in (candidate.dense_rank, candidate.sparse_rank) if rank is not None),
            candidate.chunk_id,
        ),
    )
    return [
        candidate.model_copy(update={"rrf_rank": rank})
        for rank, candidate in enumerate(ordered, 1)
    ]


def select_coverage_preserving_candidates(
    dense_candidates: list[RetrievalCandidate],
    sparse_candidates: list[RetrievalCandidate],
    *,
    profile: Phase7FusionProfile,
    query_role: QueryRole,
) -> list[RetrievalCandidate]:
    """Keep bounded component reserves, then fill the remaining slots by weighted RRF."""

    fused = fuse_weighted_rrf(
        dense_candidates,
        sparse_candidates,
        profile=profile,
        query_role=query_role,
    )
    mandatory_ids = {
        candidate.chunk_id for candidate in dense_candidates[: profile.dense_reserve]
    }
    mandatory_ids.update(
        candidate.chunk_id for candidate in sparse_candidates[: profile.sparse_reserve]
    )
    if len(mandatory_ids) > profile.max_candidates:
        raise Phase7OptimizationError(
            "Coverage-preserving reserves exceed the fixed reranker candidate budget."
        )
    mandatory = [candidate for candidate in fused if candidate.chunk_id in mandatory_ids]
    optional_slots = profile.max_candidates - len(mandatory)
    optional = [
        candidate for candidate in fused if candidate.chunk_id not in mandatory_ids
    ][:optional_slots]
    # Keep the fused ranking as the ordering signal; reserves affect membership, not an arbitrary
    # component-first ordering.
    selected = mandatory + optional
    selected.sort(
        key=lambda candidate: (
            candidate.rrf_rank if candidate.rrf_rank is not None else 2**31,
            candidate.chunk_id,
        )
    )
    return selected


def apply_role_aware_rank_fusion(
    candidates: list[RetrievalCandidate],
    *,
    query_role: QueryRole,
    role_multiplier: float,
    rank_offset: int = 10,
) -> list[RetrievalCandidate]:
    """Apply a bounded rank-only document-role prior after cross-encoder scoring.

    ``rerank_score`` remains the unmodified cross-encoder signal.  The final
    ``score`` is an explicit rank-derived ordering signal, never a probability
    and never a mixture of raw retrieval and model scores.
    """

    if not candidates or query_role == "neutral" or role_multiplier == 0:
        return list(candidates)
    if rank_offset <= 0:
        raise Phase7OptimizationError("Role-aware rank offset must be greater than zero.")
    if not 0 < role_multiplier <= 0.25:
        raise Phase7OptimizationError("Role-aware multiplier must be in the range (0, 0.25].")
    adjusted: list[RetrievalCandidate] = []
    for candidate in candidates:
        if candidate.rerank_rank is None:
            raise Phase7OptimizationError("Role-aware rank fusion requires one-based rerank ranks.")
        multiplier = (
            1 + role_multiplier
            if candidate.metadata.get("document_role") == query_role
            else 1.0
        )
        score = multiplier / (rank_offset + candidate.rerank_rank)
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "cross_encoder_rank": candidate.rerank_rank,
                "role_aware_rank_score": score,
            }
        )
        adjusted.append(candidate.model_copy(update={"metadata": metadata, "score": score}))
    ordered = sorted(
        adjusted,
        key=lambda candidate: (
            -candidate.score,
            int(candidate.metadata["cross_encoder_rank"]),
            candidate.chunk_id,
        ),
    )
    return [
        candidate.model_copy(update={"rerank_rank": rank})
        for rank, candidate in enumerate(ordered, start=1)
    ]


def _normalized_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return normalized.replace("đ", "d")


def _cue_present(query: str, cue: str) -> bool:
    return cue in query

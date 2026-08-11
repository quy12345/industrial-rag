"""Deterministic, query-only Phase 7 retrieval optimisation helpers.

This module deliberately has no Qdrant, embedding, reranker, dataset, or
provider dependency.  It may use only user-query text, component ranks, and
trusted document metadata; evaluation labels never reach runtime logic.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from app.models import RetrievalCandidate

QueryRole = Literal["installation", "programming", "neutral"]
RoleConfidence = Literal["strong", "weak", "neutral"]
PostRerankConfidenceMode = Literal["strong_only", "strong_and_weak"]
QUERY_ROLE_PROFILE = "phase7_query_role_v2"
LIST_COMPLETENESS_PROFILE = "phase7_list_completeness_v1"
_LIST_INTENT_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("which_groups", ("which groups", "what groups", "which categories", "what categories")),
    ("which_menus", ("which menus", "what menus")),
    ("vi_groups", ("nhung nhom nao", "cac nhom nao", "nhom menu nao")),
)
_TECHNICAL_IDENTIFIER_PATTERN = re.compile(r"(?<![\w])[A-Z][A-Z0-9_-]{2,}(?![\w])")
_BRACKETED_LABEL_CODE_PATTERN = re.compile(
    r"\[([^\]\n]{1,80})\]\s*([A-Za-z][A-Za-z0-9_.-]{1,15})(?=\s|[-,.;:)]|$)"
)


class Phase7OptimizationError(ValueError):
    """Raised when a bounded Phase 7 optimisation profile is invalid."""


@dataclass(frozen=True)
class QueryRoleCue:
    """One bilingual query-only cue; IDs are safe to record in artifacts."""

    identifier: str
    role: Literal["installation", "programming"]
    strength: Literal["strong", "weak"]
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class QueryRoleInference:
    """Auditable role inference derived solely from the query."""

    role: QueryRole
    confidence: RoleConfidence
    cue_ids: tuple[str, ...]
    installation_cues: tuple[str, ...]
    programming_cues: tuple[str, ...]


@dataclass(frozen=True)
class ListIntentInference:
    """Query-only signal for bounded list-completeness ordering."""

    enabled: bool
    cue_ids: tuple[str, ...]
    technical_identifiers: tuple[str, ...]


@dataclass(frozen=True)
class Phase7FusionProfile:
    """Bounded weighted-RRF and post-rerank document-role configuration."""

    name: str
    rrf_k: int
    dense_weight: float
    sparse_weight: float
    fusion_role_multiplier: float
    dense_reserve: int
    sparse_reserve: int
    max_candidates: int = 30
    post_rerank_role_multiplier: float = 0.10
    post_rerank_rrf_multiplier: float = 0.0
    post_rerank_rank_offset: int = 10
    post_rerank_confidence_mode: PostRerankConfidenceMode = "strong_only"
    list_completeness_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Phase 7 fusion profile name must not be blank.")
        if self.rrf_k <= 0 or self.max_candidates <= 0:
            raise ValueError("RRF k and candidate budget must be greater than zero.")
        if self.dense_weight <= 0 or self.sparse_weight <= 0:
            raise ValueError("Fusion component weights must be greater than zero.")
        if not 0 <= self.fusion_role_multiplier <= 0.25:
            raise ValueError("Fusion role multiplier must be between 0 and 0.25.")
        if not 0 <= self.post_rerank_role_multiplier <= 0.50:
            raise ValueError("Post-rerank role multiplier must be between 0 and 0.50.")
        if not 0 <= self.post_rerank_rrf_multiplier <= 2.0:
            raise ValueError("Post-rerank RRF multiplier must be between 0 and 2.0.")
        if self.post_rerank_rank_offset <= 0:
            raise ValueError("Post-rerank rank offset must be greater than zero.")
        if self.post_rerank_confidence_mode not in ("strong_only", "strong_and_weak"):
            raise ValueError("Unsupported post-rerank confidence mode.")
        if not isinstance(self.list_completeness_enabled, bool):
            raise ValueError("List-completeness activation must be a boolean.")
        if self.dense_reserve < 0 or self.sparse_reserve < 0:
            raise ValueError("Component reserves must not be negative.")


PHASE7_CALIBRATION_FUSION_PROFILE = Phase7FusionProfile(
    name="weighted_rrf_k40_s1.25_frole0.1_prole0.5_offset20_strong_and_weak_d5_s24",
    rrf_k=40,
    dense_weight=1.0,
    sparse_weight=1.25,
    fusion_role_multiplier=0.10,
    dense_reserve=5,
    sparse_reserve=24,
    post_rerank_role_multiplier=0.50,
    post_rerank_rank_offset=20,
    post_rerank_confidence_mode="strong_and_weak",
)


# Generic bilingual technical cues only.  These contain no dataset IDs, qrels,
# expected pages, expected documents, or answer facts.
QUERY_ROLE_CUES: tuple[QueryRoleCue, ...] = (
    QueryRoleCue("safety", "installation", "strong", ("safety", "an toan")),
    QueryRoleCue(
        "prevent_rotation",
        "installation",
        "strong",
        ("prevent", "tranh", "rotate", "rotation", "shaft", "truc"),
    ),
    QueryRoleCue("electrical", "installation", "strong", ("electrical", "dien", "power", "nguon")),
    QueryRoleCue(
        "installation",
        "installation",
        "strong",
        ("install", "installed", "installing", "installation", "lap dat"),
    ),
    QueryRoleCue(
        "wiring", "installation", "strong", ("wiring", "wire", "terminal", "dau day", "dau cuc")
    ),
    QueryRoleCue(
        "contacts",
        "installation",
        "weak",
        ("contact", "tiep diem", "run", "lenh chay", "motor", "dong co"),
    ),
    QueryRoleCue("protection", "installation", "weak", ("protection", "protective", "bao ve")),
    QueryRoleCue("menu", "programming", "strong", ("mode", "menu", "configuration", "cau hinh")),
    QueryRoleCue("parameter", "programming", "strong", ("parameter", "tham so", "programming")),
    QueryRoleCue(
        "monitoring", "programming", "strong", ("monitoring", "giam sat", "reference", "tham chieu")
    ),
    QueryRoleCue("fault", "programming", "weak", ("fault", "loi")),
)


def phase7_fusion_profile_grid() -> tuple[Phase7FusionProfile, ...]:
    """Return the finite provider-free Phase 7.4 weighted-RRF grid."""

    profiles: list[Phase7FusionProfile] = []
    for rrf_k in (20, 40, 60, 80):
        for sparse_weight in (1.0, 1.25, 1.5):
            for fusion_role_multiplier in (0.0, 0.05, 0.10):
                for dense_reserve, sparse_reserve in ((5, 24), (7, 24), (7, 26), (10, 26)):
                    profiles.append(
                        Phase7FusionProfile(
                            name=(
                                f"weighted_rrf_k{rrf_k}_s{sparse_weight:g}_frole"
                                f"{fusion_role_multiplier:g}_d{dense_reserve}_s{sparse_reserve}"
                            ),
                            rrf_k=rrf_k,
                            dense_weight=1.0,
                            sparse_weight=sparse_weight,
                            fusion_role_multiplier=fusion_role_multiplier,
                            dense_reserve=dense_reserve,
                            sparse_reserve=sparse_reserve,
                        )
                    )
    return tuple(profiles)


def phase7_profile_from_mapping(value: dict[str, object]) -> Phase7FusionProfile:
    """Load a profile while accepting the historical Phase 7.4 artifact field."""

    normalized = dict(value)
    legacy_multiplier = normalized.pop("role_multiplier", None)
    if legacy_multiplier is not None and "fusion_role_multiplier" not in normalized:
        normalized["fusion_role_multiplier"] = legacy_multiplier
    return Phase7FusionProfile(**normalized)  # type: ignore[arg-type]


def infer_query_role(query: str) -> QueryRoleInference:
    """Infer a conservative role using normalized token/phrase boundaries."""

    normalized = _normalized_query(query)
    installation: list[str] = []
    programming: list[str] = []
    installation_strengths: list[str] = []
    programming_strengths: list[str] = []
    cue_ids: list[str] = []
    for cue in QUERY_ROLE_CUES:
        if not any(_cue_present(normalized, phrase) for phrase in cue.phrases):
            continue
        cue_ids.append(cue.identifier)
        if cue.role == "installation":
            installation.append(cue.identifier)
            installation_strengths.append(cue.strength)
        else:
            programming.append(cue.identifier)
            programming_strengths.append(cue.strength)
    if installation and not programming:
        role: QueryRole = "installation"
        confidence: RoleConfidence = "strong" if "strong" in installation_strengths else "weak"
    elif programming and not installation:
        role = "programming"
        confidence = "strong" if "strong" in programming_strengths else "weak"
    else:
        role = "neutral"
        confidence = "neutral"
    return QueryRoleInference(
        role=role,
        confidence=confidence,
        cue_ids=tuple(cue_ids),
        installation_cues=tuple(installation),
        programming_cues=tuple(programming),
    )


def infer_list_intent(query: str) -> ListIntentInference:
    """Infer a generic bilingual list request without evaluation-label access."""

    normalized = _normalized_query(query)
    cue_ids = tuple(
        identifier
        for identifier, phrases in _LIST_INTENT_CUES
        if any(_cue_present(normalized, phrase) for phrase in phrases)
    )
    identifiers = tuple(
        sorted(
            {
                match.group(0).casefold()
                for match in _TECHNICAL_IDENTIFIER_PATTERN.finditer(
                    unicodedata.normalize("NFKC", query)
                )
            }
        )
    )
    return ListIntentInference(bool(cue_ids), cue_ids, identifiers)


def list_completeness_features(
    text: str, *, technical_identifiers: tuple[str, ...]
) -> dict[str, int]:
    """Return bounded structural counts derived only from query and candidate text."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    pairs = {
        (" ".join(label.casefold().split()), code.casefold())
        for label, code in _BRACKETED_LABEL_CODE_PATTERN.findall(text)
    }
    identifier_matches = sum(
        re.search(rf"(?<![\w]){re.escape(identifier)}(?![\w])", normalized) is not None
        for identifier in technical_identifiers
    )
    return {
        "query_identifier_match_count": identifier_matches,
        "bracketed_label_code_pair_count": len(pairs),
    }


def apply_list_completeness_fallback(
    candidates: list[RetrievalCandidate], *, query: str
) -> list[RetrievalCandidate]:
    """Apply the registered ranks-5-to-10 list fallback from raw runtime inputs."""

    inference = infer_list_intent(query)
    enriched = []
    for candidate in candidates:
        metadata = dict(candidate.metadata)
        metadata.update(
            list_completeness_features(
                candidate.text,
                technical_identifiers=inference.technical_identifiers,
            )
        )
        metadata["list_intent_cue_ids"] = inference.cue_ids
        metadata["query_technical_identifiers"] = inference.technical_identifiers
        enriched.append(candidate.model_copy(update={"metadata": metadata}))
    return apply_list_completeness_from_metadata(enriched, enabled=inference.enabled)


def apply_list_completeness_from_metadata(
    candidates: list[RetrievalCandidate], *, enabled: bool
) -> list[RetrievalCandidate]:
    """Replay the fallback from sanitized structural feature counts."""

    if not enabled:
        return list(candidates)
    ranks = [candidate.rerank_rank for candidate in candidates]
    if any(rank is None or rank <= 0 for rank in ranks) or set(ranks) != set(
        range(1, len(candidates) + 1)
    ):
        raise Phase7OptimizationError(
            "List completeness requires unique, contiguous one-based rerank ranks."
        )
    if len(candidates) < 5:
        return list(candidates)
    ordered = sorted(candidates, key=lambda item: (item.rerank_rank or 2**31, item.chunk_id))
    window = ordered[4:10]
    for candidate in window:
        for field in ("query_identifier_match_count", "bracketed_label_code_pair_count"):
            value = candidate.metadata.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise Phase7OptimizationError(
                    "List completeness requires non-negative sanitized feature counts."
                )
    window.sort(
        key=lambda candidate: (
            -int(candidate.metadata["query_identifier_match_count"]),
            -int(candidate.metadata["bracketed_label_code_pair_count"]),
            int(candidate.rerank_rank or 2**31),
            candidate.chunk_id,
        )
    )
    reordered = [*ordered[:4], *window, *ordered[10:]]
    result: list[RetrievalCandidate] = []
    for rank, candidate in enumerate(reordered, start=1):
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "pre_list_completeness_rank": candidate.rerank_rank,
                "pre_list_completeness_rank_score": candidate.score,
                "list_completeness_profile": LIST_COMPLETENESS_PROFILE,
            }
        )
        result.append(
            candidate.model_copy(
                update={"metadata": metadata, "rerank_rank": rank, "score": 1 / (100 + rank)}
            )
        )
    return result


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
            score *= 1 + profile.fusion_role_multiplier
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
        candidate.model_copy(update={"rrf_rank": rank}) for rank, candidate in enumerate(ordered, 1)
    ]


def select_coverage_preserving_candidates(
    dense_candidates: list[RetrievalCandidate],
    sparse_candidates: list[RetrievalCandidate],
    *,
    profile: Phase7FusionProfile,
    query_role: QueryRole,
) -> list[RetrievalCandidate]:
    """Keep bounded component reserves, then fill remaining slots by weighted RRF."""

    fused = fuse_weighted_rrf(
        dense_candidates, sparse_candidates, profile=profile, query_role=query_role
    )
    mandatory_ids = {candidate.chunk_id for candidate in dense_candidates[: profile.dense_reserve]}
    mandatory_ids.update(
        candidate.chunk_id for candidate in sparse_candidates[: profile.sparse_reserve]
    )
    if len(mandatory_ids) > profile.max_candidates:
        raise Phase7OptimizationError(
            "Coverage-preserving reserves exceed the fixed reranker candidate budget."
        )
    mandatory = [candidate for candidate in fused if candidate.chunk_id in mandatory_ids]
    optional_slots = profile.max_candidates - len(mandatory)
    optional = [candidate for candidate in fused if candidate.chunk_id not in mandatory_ids][
        :optional_slots
    ]
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
    rrf_rank_multiplier: float = 0.0,
    rank_offset: int = 10,
    confidence: RoleConfidence = "strong",
    confidence_mode: PostRerankConfidenceMode = "strong_only",
) -> list[RetrievalCandidate]:
    """Add a bounded rank-only role prior after the cross-encoder.

    ``rerank_score`` remains unmodified.  The final ``score`` is a rank-derived
    ordering signal, not a probability and not a mixture of raw model scores.
    """

    if not candidates:
        return list(candidates)
    if rank_offset <= 0:
        raise Phase7OptimizationError("Role-aware rank offset must be greater than zero.")
    if not 0 <= role_multiplier <= 0.50:
        raise Phase7OptimizationError("Role-aware multiplier must be in the range [0, 0.50].")
    if not 0 <= rrf_rank_multiplier <= 2.0:
        raise Phase7OptimizationError("Post-rerank RRF multiplier must be in the range [0, 2.0].")
    if confidence_mode not in ("strong_only", "strong_and_weak"):
        raise Phase7OptimizationError("Unsupported post-rerank confidence mode.")
    role_enabled = (
        query_role != "neutral"
        and role_multiplier > 0
        and confidence != "neutral"
        and not (confidence == "weak" and confidence_mode == "strong_only")
    )
    if not role_enabled and rrf_rank_multiplier == 0:
        return list(candidates)

    role_rank = 0
    adjusted: list[RetrievalCandidate] = []
    ordered_input = sorted(candidates, key=lambda item: (item.rerank_rank or 2**31, item.chunk_id))
    for candidate in ordered_input:
        if candidate.rerank_rank is None:
            raise Phase7OptimizationError("Role-aware rank fusion requires one-based rerank ranks.")
        base_score = 1 / (rank_offset + candidate.rerank_rank)
        document_role = candidate.metadata.get("document_role")
        role_prior = 0.0
        matching_role_rank: int | None = None
        if role_enabled and document_role == query_role:
            role_rank += 1
            matching_role_rank = role_rank
            role_prior = role_multiplier / (rank_offset + role_rank)
        if rrf_rank_multiplier > 0 and candidate.rrf_rank is None:
            raise Phase7OptimizationError(
                "Post-rerank RRF fusion requires one-based pre-rerank RRF ranks."
            )
        rrf_rank_prior = (
            rrf_rank_multiplier / (rank_offset + candidate.rrf_rank)
            if rrf_rank_multiplier > 0 and candidate.rrf_rank is not None
            else 0.0
        )
        score = base_score + role_prior + rrf_rank_prior
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "cross_encoder_rank": candidate.rerank_rank,
                "query_role": query_role,
                "query_role_confidence": confidence,
                "role_rank": matching_role_rank,
                "role_prior_score": role_prior,
                "rrf_rank_prior_score": rrf_rank_prior,
                "role_aware_rank_score": score,
            }
        )
        adjusted.append(candidate.model_copy(update={"metadata": metadata, "score": score}))
    ordered = sorted(
        adjusted,
        key=lambda candidate: (
            -candidate.score,
            int(candidate.metadata["cross_encoder_rank"]),
            candidate.rrf_rank if candidate.rrf_rank is not None else 2**31,
            candidate.chunk_id,
        ),
    )
    return [
        candidate.model_copy(update={"rerank_rank": rank})
        for rank, candidate in enumerate(ordered, 1)
    ]


def _normalized_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    normalized = normalized.replace("\u0111", "d")
    return " ".join(re.findall(r"[a-z0-9]+", normalized))


def _cue_present(query: str, cue: str) -> bool:
    phrase = _normalized_query(cue)
    if not phrase:
        return False
    tokens = query.split()
    phrase_tokens = phrase.split()
    width = len(phrase_tokens)
    return any(tokens[index : index + width] == phrase_tokens for index in range(len(tokens)))

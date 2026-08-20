"""Deterministic final evidence selection without evaluation-label access."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.content_identity import evidence_content_fingerprint
from app.models import RetrievalCandidate
from app.phase7_optimization import QueryRole, QueryRoleInference, infer_query_role


class EvidenceSelectionError(ValueError):
    """Raised when a final evidence set cannot be selected safely."""


@dataclass(frozen=True)
class EvidenceDuplicateGroup:
    """Sanitized provenance for exact content repeated across source documents."""

    representative_chunk_id: str
    equivalent_chunk_ids: tuple[str, ...]
    equivalent_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceSelection:
    """Actual generation context plus exact-duplicate provenance diagnostics."""

    candidates: tuple[RetrievalCandidate, ...]
    duplicate_groups: tuple[EvidenceDuplicateGroup, ...]


def select_evidence_candidates(
    question: str,
    candidates: Sequence[RetrievalCandidate],
    *,
    top_k: int,
    query_role_inferer: Callable[[str], QueryRoleInference] = infer_query_role,
) -> EvidenceSelection:
    """Collapse exact content across documents, then apply the final context limit.

    The selector may inspect only query-derived role signals, ranking order, stable
    candidate identity, raw candidate content, and trusted document metadata. It
    never receives qrels, expected facts, pages, or expected document IDs.
    """

    if top_k <= 0:
        raise EvidenceSelectionError("Evidence top_k must be greater than zero.")
    if not question.strip():
        raise EvidenceSelectionError("Evidence selection question must not be blank.")
    if len({candidate.chunk_id for candidate in candidates}) != len(candidates):
        raise EvidenceSelectionError("Evidence selection candidates must have unique chunk IDs.")

    inference = query_role_inferer(question)
    return select_evidence_candidates_for_role(
        candidates,
        top_k=top_k,
        query_role=inference.role,
    )


def select_evidence_candidates_for_role(
    candidates: Sequence[RetrievalCandidate],
    *,
    top_k: int,
    query_role: QueryRole,
) -> EvidenceSelection:
    """Select evidence when the query-derived role has already been sealed."""

    if top_k <= 0:
        raise EvidenceSelectionError("Evidence top_k must be greater than zero.")
    if query_role not in {"installation", "programming", "neutral"}:
        raise EvidenceSelectionError("Evidence selection received an invalid query role.")
    if len({candidate.chunk_id for candidate in candidates}) != len(candidates):
        raise EvidenceSelectionError("Evidence selection candidates must have unique chunk IDs.")

    grouped: dict[str, list[tuple[int, RetrievalCandidate]]] = {}
    group_order: list[str] = []
    for ordinal, candidate in enumerate(candidates, start=1):
        fingerprint = candidate.metadata.get("content_fingerprint_sha256")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            fingerprint = evidence_content_fingerprint(candidate.text)
        if fingerprint not in grouped:
            grouped[fingerprint] = []
            group_order.append(fingerprint)
        grouped[fingerprint].append((ordinal, candidate))

    selected: list[tuple[int, RetrievalCandidate]] = []
    diagnostics: list[EvidenceDuplicateGroup] = []
    for fingerprint in group_order:
        members = grouped[fingerprint]
        document_ids = {member[1].document_id for member in members}
        if len(document_ids) == 1:
            selected.extend(members)
            continue
        preferred = [
            member
            for member in members
            if query_role != "neutral"
            and member[1].metadata.get("document_role") == query_role
        ]
        representative_ordinal, representative = min(
            preferred or members,
            key=lambda member: (member[0], member[1].chunk_id),
        )
        equivalent_ids = tuple(sorted(member[1].chunk_id for member in members))
        equivalent_documents = tuple(sorted(document_ids))
        metadata = dict(representative.metadata)
        metadata.update(
            {
                "evidence_group_rank": min(member[0] for member in members),
                "evidence_representative_original_rank": representative_ordinal,
                "equivalent_chunk_ids": equivalent_ids,
                "equivalent_document_ids": equivalent_documents,
                "exact_cross_document_duplicate": len(equivalent_documents) > 1,
            }
        )
        selected.append(
            (
                min(member[0] for member in members),
                representative.model_copy(update={"metadata": metadata}),
            )
        )
        diagnostics.append(
            EvidenceDuplicateGroup(
                representative_chunk_id=representative.chunk_id,
                equivalent_chunk_ids=equivalent_ids,
                equivalent_document_ids=equivalent_documents,
            )
        )

    selected.sort(key=lambda member: (member[0], member[1].chunk_id))
    return EvidenceSelection(
        candidates=tuple(candidate for _, candidate in selected[:top_k]),
        duplicate_groups=tuple(diagnostics),
    )

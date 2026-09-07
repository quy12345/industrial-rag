"""Conservative lexical scope checks over evidence supplied to generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.domain.policies.query_roles import normalize_query_text, query_contains_phrase

EvidenceScopeStatus = Literal["not_applicable", "covered", "missing_scope"]
EVIDENCE_SCOPE_PROFILE = "protected_technical_concepts_v1"


@dataclass(frozen=True)
class ProtectedTechnicalConcept:
    """A reviewed bilingual technical concept with boundary-safe aliases."""

    identifier: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceScopeDecision:
    """Scope presence only; this decision does not establish entailment."""

    status: EvidenceScopeStatus
    query_concept_ids: tuple[str, ...]
    missing_concept_ids: tuple[str, ...]


# This registry intentionally keeps related but non-equivalent concepts separate.
# It contains no dataset IDs, answer text, source IDs, pages, or secret values.
PROTECTED_TECHNICAL_CONCEPTS: tuple[ProtectedTechnicalConcept, ...] = (
    ProtectedTechnicalConcept("bluetooth", ("bluetooth",)),
    ProtectedTechnicalConcept(
        "credential",
        ("pin", "pin code", "password", "access code", "mat khau", "ma pin", "ma truy cap"),
    ),
    ProtectedTechnicalConcept("qr", ("qr", "qr code", "ma qr")),
    ProtectedTechnicalConcept("pairing", ("pairing", "pair", "paired", "ghep noi")),
    ProtectedTechnicalConcept(
        "commissioning", ("commissioning", "commission", "chay thu", "dua vao van hanh")
    ),
    ProtectedTechnicalConcept(
        "mobile_application",
        ("mobile app", "mobile application", "ung dung di dong", "ung dung dien thoai"),
    ),
    ProtectedTechnicalConcept("unlocking", ("unlock", "unlocking", "mo khoa")),
    ProtectedTechnicalConcept("configuration", ("configuration", "configure", "cau hinh")),
    ProtectedTechnicalConcept(
        "product_data_sheet",
        ("product data sheet", "datasheet", "bang du lieu san pham", "bang thong so san pham"),
    ),
)


def evaluate_evidence_scope(
    question: str, sources: Sequence[str]
) -> EvidenceScopeDecision:
    """Require every protected query concept to appear in rendered evidence.

    The guard is intentionally inapplicable until at least two distinct protected
    concepts occur in the question. Matching concepts in evidence is a necessary
    scope check, not a general-purpose relation or entailment verifier.
    """

    query = normalize_query_text(question)
    source_text = normalize_query_text("\n".join(sources))
    query_concept_ids = tuple(
        concept.identifier
        for concept in PROTECTED_TECHNICAL_CONCEPTS
        if _contains_any_alias(query, concept.aliases)
    )
    if len(query_concept_ids) < 2:
        return EvidenceScopeDecision("not_applicable", query_concept_ids, ())
    missing = tuple(
        concept.identifier
        for concept in PROTECTED_TECHNICAL_CONCEPTS
        if concept.identifier in query_concept_ids
        and not _contains_any_alias(source_text, concept.aliases)
    )
    return EvidenceScopeDecision(
        "missing_scope" if missing else "covered",
        query_concept_ids,
        missing,
    )


def _contains_any_alias(text: str, aliases: Sequence[str]) -> bool:
    return any(query_contains_phrase(text, alias) for alias in aliases)

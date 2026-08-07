"""Referential citation validation and trusted citation construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.errors import CitationValidationError
from app.generation import GeneratedAnswer
from app.models import Citation, RetrievalCandidate


@dataclass(frozen=True)
class ValidatedGeneration:
    """Normalized generated output after referential validation."""

    answer: str
    source_ids: tuple[str, ...]
    insufficient_evidence: bool


def validate_generated_answer(
    output: GeneratedAnswer,
    *,
    source_map: dict[str, RetrievalCandidate],
) -> ValidatedGeneration:
    """Validate source labels without trusting model-created citation metadata."""

    errors: list[str] = []
    answer = output.answer.strip()
    unique_source_ids = tuple(_deduplicate(output.source_ids))
    if output.insufficient_evidence:
        if output.source_ids:
            errors.append("insufficient_evidence output must not cite sources")
    else:
        if not answer:
            errors.append("non-abstained answer must not be empty")
        if not output.source_ids:
            errors.append("non-abstained answer must cite at least one source")
    for source_id in unique_source_ids:
        if source_id not in source_map:
            errors.append(f"unknown source ID: {source_id}")
    if errors:
        raise CitationValidationError(errors)
    return ValidatedGeneration(
        answer=answer,
        source_ids=unique_source_ids,
        insufficient_evidence=output.insufficient_evidence,
    )


def build_citations(
    source_ids: Sequence[str],
    *,
    source_map: dict[str, RetrievalCandidate],
    requested_document_id: str | None,
    excerpt_max_chars: int,
) -> list[Citation]:
    """Build deterministic public citations from authoritative retrieved candidates."""

    if excerpt_max_chars <= 0:
        raise CitationValidationError(["citation excerpt limit must be positive"])
    citations: list[Citation] = []
    seen_chunks: set[str] = set()
    for source_id in source_ids:
        candidate = source_map.get(source_id)
        if candidate is None:
            raise CitationValidationError([f"unknown source ID: {source_id}"])
        if requested_document_id is not None and candidate.document_id != requested_document_id:
            raise CitationValidationError([f"source {source_id} belongs to another document"])
        if candidate.chunk_id in seen_chunks:
            continue
        seen_chunks.add(candidate.chunk_id)
        excerpt = candidate.text.strip()
        if not excerpt:
            raise CitationValidationError([f"source {source_id} has no citation text"])
        if len(excerpt) > excerpt_max_chars:
            excerpt = excerpt[: excerpt_max_chars - 1] + "…"
        citations.append(
            Citation(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                filename=candidate.filename,
                page_numbers=sorted(set(candidate.page_numbers)),
                headings=[heading.strip() for heading in candidate.headings],
                excerpt=excerpt,
            )
        )
    return citations


def _deduplicate(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))

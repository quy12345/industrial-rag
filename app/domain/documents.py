"""Document chunk records and stable identity policies."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SUPPORTED_EXTENSIONS = (".pdf", ".docx")


class IngestionError(Exception):
    """Raised when document ingestion fails."""


class DocumentChunk(BaseModel):
    """JSON-serializable representation of one structure-aware document chunk."""

    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_numbers: list[int] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    content_type: str = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_document_id(file_path: Path) -> str:
    """Build a stable ID from the normalized filename stem and file content."""

    path = Path(file_path)
    stem = unicodedata.normalize("NFKD", path.stem).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-") or "document"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{slug}-{digest}"


def build_chunk_id(
    document_id: str,
    page_numbers: Sequence[int],
    headings: Sequence[str],
    text: str,
    occurrence_index: int = 0,
) -> str:
    """Build a stable ID from chunk content and its duplicate occurrence."""

    first_page = str(min(page_numbers)) if page_numbers else "unknown"
    if occurrence_index < 0:
        raise IngestionError("Chunk occurrence index must not be negative.")
    canonical = canonical_chunk_key(document_id, page_numbers, headings, text)
    digest = hashlib.sha256(
        json.dumps(
            [canonical, occurrence_index],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"{document_id}_p{first_page}_h{digest}"


def build_page_batches(
    start_page: int,
    end_page: int,
    batch_size: int,
) -> list[tuple[int, int]]:
    """Split an inclusive page range into validated, page-aligned batches."""

    if start_page < 1:
        raise IngestionError("Page start must be greater than or equal to 1.")
    if end_page < start_page:
        raise IngestionError("Page end must be greater than or equal to page start.")
    if batch_size <= 0:
        raise IngestionError("Batch size must be greater than 0.")

    return [
        (batch_start, min(batch_start + batch_size - 1, end_page))
        for batch_start in range(start_page, end_page + 1, batch_size)
    ]


def canonical_chunk_key(
    document_id: str,
    page_numbers: Sequence[int],
    headings: Sequence[str],
    text: str,
) -> str:
    """Return the canonical identity fields used for stable chunk IDs."""

    normalized_text = (
        unicodedata.normalize("NFKC", text)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )
    normalized_headings = [
        unicodedata.normalize("NFKC", heading).strip() for heading in headings
    ]
    return json.dumps(
        [document_id, sorted(set(page_numbers)), normalized_headings, normalized_text],
        ensure_ascii=False,
        separators=(",", ":"),
    )

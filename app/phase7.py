"""Frozen-corpus and held-out-dataset helpers for Phase 7.

This module is deliberately offline.  It validates annotations against an already
frozen JSONL chunk export and never talks to a model, provider, or Qdrant server.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.evaluation import chunk_set_metadata, phrase_matches
from app.models import DocumentChunk

PHASE7_DENSE_COLLECTION = "industrial_manual_phase7_dense_v1"
PHASE7_HYBRID_COLLECTION = "industrial_manual_phase7_hybrid_v1"
PROTECTED_COLLECTIONS = {"industrial_manual_chunks", "industrial_manual_chunks_v2"}
PHASE7_CORPUS_VERSION = "atv320-2025-04-v1"

DatasetKind = Literal["calibration", "test"]
QuestionLanguage = Literal["vi", "en"]
Scenario = Literal["vi_to_en", "en_to_en"]
QuestionType = Literal[
    "safety",
    "installation",
    "mounting",
    "wiring",
    "terminal",
    "electrical_specification",
    "environmental_condition",
    "parameter_code",
    "menu_navigation",
    "fault_diagnosis",
    "maintenance",
    "ordered_procedure",
    "table_lookup",
    "cross_document",
    "unanswerable",
]
ReviewStatus = Literal["needs_human_review", "approved"]
PhraseMatchMode = Literal["all", "any"]


class Phase7Error(ValueError):
    """Raised when Phase 7 frozen inputs do not satisfy their contract."""


class Phase7Source(BaseModel):
    """Redistribution-safe source metadata; raw vendor content is never stored here."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    manufacturer: str
    title: str
    document_reference: str
    version: str
    language: Literal["en"]
    document_role: Literal["installation", "programming"]
    official_url: str
    sha256: str
    file_size_bytes: int = Field(gt=0)
    page_count: int = Field(gt=0)
    retrieved_at: str
    redistribution_note: str

    @field_validator(
        "filename", "manufacturer", "title", "document_reference", "version", "official_url",
        "retrieved_at", "redistribution_note",
    )
    @classmethod
    def non_empty_source_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", normalized):
            raise ValueError("must be a lowercase SHA-256 hex digest")
        return normalized


def validate_source_records(records: Sequence[Phase7Source]) -> None:
    """Require one unique installation and programming source manifest entry."""

    if len(records) != 2:
        raise Phase7Error("Phase 7 source manifest requires exactly two ATV320 manuals.")
    filenames = [record.filename for record in records]
    if len(set(filenames)) != len(filenames):
        raise Phase7Error("Phase 7 source manifest contains duplicate filenames.")
    roles = {record.document_role for record in records}
    if roles != {"installation", "programming"}:
        raise Phase7Error("Phase 7 source manifest requires installation and programming roles.")


class Phase7DatasetItem(BaseModel):
    """One answerable or deliberately unsupported Phase 7 question."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    language: QuestionLanguage
    answerable: bool
    scenario: Scenario
    question_type: QuestionType
    expected_document_ids: list[str]
    relevant_chunk_ids: list[str]
    expected_pages: list[int]
    expected_phrases: list[str]
    phrase_match_mode: PhraseMatchMode = "all"
    citation_required: bool
    annotation_notes: str | None = None
    unanswerable_reason: str | None = None
    review_status: ReviewStatus = "needs_human_review"

    @field_validator("id", "question")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("expected_document_ids", "relevant_chunk_ids", "expected_phrases")
    @classmethod
    def normalized_unique_texts(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("must not contain blank values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("must not contain duplicate values")
        return normalized

    @field_validator("expected_pages")
    @classmethod
    def valid_pages(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError("must contain only positive page numbers")
        return sorted(set(values))

    def model_post_init(self, __context: Any) -> None:
        if self.language == "vi" and self.scenario != "vi_to_en":
            raise ValueError("Vietnamese questions must use scenario vi_to_en")
        if self.language == "en" and self.scenario != "en_to_en":
            raise ValueError("English questions must use scenario en_to_en")
        if self.answerable:
            if not self.relevant_chunk_ids:
                raise ValueError("answerable items require relevant_chunk_ids")
            if not self.expected_document_ids or not self.expected_phrases:
                raise ValueError("answerable items require documents and expected_phrases")
            if not self.expected_pages or not self.citation_required:
                raise ValueError("answerable items require pages and citations")
            if self.unanswerable_reason is not None:
                raise ValueError("answerable items cannot have unanswerable_reason")
        else:
            if any(
                (
                    self.expected_document_ids,
                    self.relevant_chunk_ids,
                    self.expected_pages,
                    self.expected_phrases,
                )
            ):
                raise ValueError("unanswerable items cannot contain qrels or diagnostics")
            if self.question_type != "unanswerable" or self.citation_required:
                raise ValueError("unanswerable items must use unanswerable type and no citation")
            if not self.unanswerable_reason or not self.unanswerable_reason.strip():
                raise ValueError("unanswerable items require unanswerable_reason")


def chunk_ids_sha256(chunks: Iterable[DocumentChunk]) -> str:
    """Return the corpus identity used by manifests and datasets."""

    return str(chunk_set_metadata(chunks)["chunk_ids_sha256"])


def file_sha256(path: Path) -> str:
    """Hash a source file without loading the complete manual into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_phase7_dataset(path: Path) -> list[Phase7DatasetItem]:
    """Read strict JSONL and reject malformed or duplicated records."""

    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise Phase7Error(f"Unable to read Phase 7 dataset {path}: {exc}") from exc
    records: list[Phase7DatasetItem] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise Phase7Error(f"Blank dataset record on line {line_number}.")
        try:
            record = Phase7DatasetItem.model_validate_json(line)
        except ValidationError as exc:
            raise Phase7Error(f"Invalid dataset record on line {line_number}: {exc}") from exc
        if record.id in seen_ids:
            raise Phase7Error(f"Duplicate dataset ID on line {line_number}: {record.id}")
        seen_ids.add(record.id)
        records.append(record)
    if not records:
        raise Phase7Error(f"Phase 7 dataset is empty: {path}")
    return records


def validate_phase7_datasets(
    calibration: Sequence[Phase7DatasetItem],
    test: Sequence[Phase7DatasetItem],
    chunks: Sequence[DocumentChunk],
) -> dict[str, Any]:
    """Validate immutable annotation rules against the frozen corpus."""

    _validate_dataset_shape(calibration, kind="calibration", expected_answerable=12, expected_unanswerable=8)
    _validate_dataset_shape(test, kind="test", expected_answerable=30, expected_unanswerable=15)
    calibration_ids = {item.id for item in calibration}
    duplicate_ids = calibration_ids.intersection(item.id for item in test)
    if duplicate_ids:
        raise Phase7Error(f"Calibration/test IDs overlap: {sorted(duplicate_ids)}")
    calibration_questions = {_normalized_question(item.question) for item in calibration}
    duplicated_questions = calibration_questions.intersection(
        _normalized_question(item.question) for item in test
    )
    if duplicated_questions:
        raise Phase7Error("Calibration/test questions have normalized exact duplicates.")

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    for item in [*calibration, *test]:
        if not item.answerable:
            continue
        relevant = []
        for chunk_id in item.relevant_chunk_ids:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                raise Phase7Error(f"{item.id} references missing frozen chunk ID: {chunk_id}")
            if chunk.document_id not in item.expected_document_ids:
                raise Phase7Error(f"{item.id} qrel {chunk_id} belongs to an unexpected document")
            relevant.append(chunk)
        for phrase in item.expected_phrases:
            if not any(phrase_matches(chunk.text, phrase) for chunk in relevant):
                raise Phase7Error(f"{item.id} expected phrase is absent from direct-evidence qrels")
        if not any(set(item.expected_pages).intersection(chunk.page_numbers) for chunk in relevant):
            raise Phase7Error(f"{item.id} expected_pages do not intersect direct-evidence qrels")

    return {
        "calibration": _dataset_summary(calibration),
        "test": _dataset_summary(test),
        "corpus": chunk_set_metadata(chunks),
    }


def dataset_sha256(items: Sequence[Phase7DatasetItem]) -> str:
    """Hash canonical records, independent of JSONL line ending differences."""

    canonical = "\n".join(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for item in items
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON atomically for manifests and audit records."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_dataset_shape(
    items: Sequence[Phase7DatasetItem], *, kind: DatasetKind, expected_answerable: int,
    expected_unanswerable: int,
) -> None:
    counts = Counter(item.answerable for item in items)
    if counts[True] != expected_answerable or counts[False] != expected_unanswerable:
        raise Phase7Error(
            f"{kind} requires {expected_answerable} answerable and {expected_unanswerable} "
            f"unanswerable items; found {counts[True]} and {counts[False]}."
        )
    if kind == "test":
        vi_to_en = sum(item.scenario == "vi_to_en" and item.answerable for item in items)
        if vi_to_en < 10:
            raise Phase7Error("Held-out test requires at least 10 answerable vi_to_en questions.")
        answerable_types = Counter(item.question_type for item in items if item.answerable)
        required = {"installation", "safety", "wiring", "parameter_code", "fault_diagnosis", "cross_document"}
        missing = sorted(required - set(answerable_types))
        if missing:
            raise Phase7Error(f"Held-out test misses required question types: {missing}")


def _dataset_summary(items: Sequence[Phase7DatasetItem]) -> dict[str, Any]:
    return {
        "total": len(items),
        "answerable": sum(item.answerable for item in items),
        "unanswerable": sum(not item.answerable for item in items),
        "by_language": dict(sorted(Counter(item.language for item in items).items())),
        "by_scenario": dict(sorted(Counter(item.scenario for item in items).items())),
        "by_question_type": dict(sorted(Counter(item.question_type for item in items).items())),
        "sha256": dataset_sha256(items),
    }


def _normalized_question(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", normalized)).strip()

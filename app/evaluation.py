"""Typed, dependency-free helpers for dense retrieval evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.models import DocumentChunk

EvaluationLanguage = Literal["vi", "en"]
EvaluationCategory = Literal[
    "exact_technical_term",
    "semantic_paraphrase",
    "numeric_unit",
    "heading_dependent",
    "table_related",
    "known_failure",
]


class EvaluationError(ValueError):
    """Raised when an evaluation dataset or frozen chunk set is invalid."""


class EvaluationCase(BaseModel):
    """One manually verified retrieval-development query and its qrels."""

    model_config = ConfigDict(extra="forbid")

    id: str
    language: EvaluationLanguage
    question: str
    relevant_chunk_ids: list[str] = Field(min_length=1)
    expected_phrases: list[str] = Field(min_length=1)
    expected_pages: list[int] = Field(min_length=1)
    category: EvaluationCategory
    critical: bool = False
    document_id: str = "manual-77d5dae4c2c5"
    notes: str | None = None

    @field_validator("id", "question", "document_id")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        """Reject blank identity and query fields."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("relevant_chunk_ids", "expected_phrases")
    @classmethod
    def require_non_empty_values(cls, values: list[str]) -> list[str]:
        """Normalize list members and reject blank qrel evidence."""

        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("must not contain blank values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("must not contain duplicate values")
        return normalized

    @field_validator("expected_pages")
    @classmethod
    def require_positive_unique_pages(cls, values: list[int]) -> list[int]:
        """Keep page diagnostics deterministic and valid."""

        if any(value <= 0 for value in values):
            raise ValueError("must contain only positive page numbers")
        return sorted(set(values))


class RetrievedLike(Protocol):
    """Minimum retrieval result fields used by the evaluator."""

    chunk_id: str
    document_id: str
    text: str
    page_numbers: Sequence[int]
    headings: Sequence[str]
    score: float


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Load strict JSONL evaluation cases and reject duplicate IDs."""

    records: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationError(f"Unable to read evaluation dataset {path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise EvaluationError(f"Blank evaluation record on line {line_number}.")
        try:
            raw_record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"Invalid JSON on line {line_number}: {exc}") from exc
        try:
            record = EvaluationCase.model_validate(raw_record)
        except ValidationError as exc:
            raise EvaluationError(
                f"Invalid evaluation record on line {line_number}: {exc}"
            ) from exc
        if record.id in seen_ids:
            raise EvaluationError(f"Duplicate evaluation ID on line {line_number}: {record.id}")
        seen_ids.add(record.id)
        records.append(record)

    if not records:
        raise EvaluationError(f"Evaluation dataset is empty: {path}")
    return records


def load_frozen_chunks(path: Path) -> list[DocumentChunk]:
    """Load a frozen JSONL chunk set used to validate qrels and baseline identity."""

    chunks: list[DocumentChunk] = []
    seen_ids: set[str] = set()
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationError(f"Unable to read frozen chunk set {path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise EvaluationError(f"Blank frozen chunk on line {line_number}.")
        try:
            chunk = DocumentChunk.model_validate_json(line)
        except ValidationError as exc:
            raise EvaluationError(
                f"Invalid frozen chunk on line {line_number}: {exc}"
            ) from exc
        if chunk.chunk_id in seen_ids:
            raise EvaluationError(f"Duplicate chunk ID on line {line_number}: {chunk.chunk_id}")
        seen_ids.add(chunk.chunk_id)
        chunks.append(chunk)

    if not chunks:
        raise EvaluationError(f"Frozen chunk set is empty: {path}")
    return chunks


def validate_cases_against_chunks(
    cases: Iterable[EvaluationCase],
    chunks: Iterable[DocumentChunk],
) -> None:
    """Require every qrel phrase and page diagnostic to match frozen evidence."""

    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    for case in cases:
        relevant_chunks: list[DocumentChunk] = []
        for chunk_id in case.relevant_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                raise EvaluationError(
                    f"Evaluation case {case.id} references missing frozen chunk ID: {chunk_id}"
                )
            if chunk.document_id != case.document_id:
                raise EvaluationError(
                    f"Evaluation case {case.id} qrel {chunk_id} belongs to {chunk.document_id}, "
                    f"not {case.document_id}"
                )
            relevant_chunks.append(chunk)

        for phrase in case.expected_phrases:
            if not any(phrase_matches(chunk.text, phrase) for chunk in relevant_chunks):
                raise EvaluationError(
                    f"Evaluation case {case.id} expected phrase is absent from its qrels: "
                    f"{phrase!r}"
                )

        expected_pages = set(case.expected_pages)
        if not any(expected_pages.intersection(chunk.page_numbers) for chunk in relevant_chunks):
            raise EvaluationError(
                f"Evaluation case {case.id} expected pages do not match its qrels: "
                f"{case.expected_pages}"
            )


def chunk_set_metadata(chunks: Iterable[DocumentChunk]) -> dict[str, Any]:
    """Return a stable identity for a frozen chunk set without hashing chunk text."""

    chunk_list = list(chunks)
    chunk_ids = sorted(chunk.chunk_id for chunk in chunk_list)
    document_ids = sorted({chunk.document_id for chunk in chunk_list})
    return {
        "chunk_count": len(chunk_list),
        "document_ids": document_ids,
        "chunk_ids_sha256": hashlib.sha256(
            "\n".join(chunk_ids).encode("utf-8")
        ).hexdigest(),
    }


def phrase_matches(text: str, expected_phrase: str) -> bool:
    """Compare evidence text with Unicode and whitespace normalization."""

    return _normalize_text(expected_phrase) in _normalize_text(text)


def direct_evidence_rank(
    results: Sequence[RetrievedLike], relevant_chunk_ids: set[str]
) -> int | None:
    """Return the first one-based result rank whose stable chunk ID is a qrel."""

    return next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if result.chunk_id in relevant_chunk_ids
        ),
        None,
    )


def diagnostic_phrase_rank(results: Sequence[RetrievedLike], phrases: Sequence[str]) -> int | None:
    """Return a phrase diagnostic rank; it never affects retrieval metrics."""

    return next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if any(phrase_matches(result.text, phrase) for phrase in phrases)
        ),
        None,
    )


def diagnostic_page_rank(results: Sequence[RetrievedLike], expected_pages: set[int]) -> int | None:
    """Return a page diagnostic rank; it never affects retrieval metrics."""

    return next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if expected_pages.intersection(result.page_numbers)
        ),
        None,
    )


def evaluate_cases(
    cases: Sequence[EvaluationCase],
    search: Callable[[str, int, str], list[RetrievedLike]],
    *,
    candidate_limit: int,
) -> dict[str, Any]:
    """Measure direct-evidence retrieval after a caller-controlled warmup query."""

    if candidate_limit < 5:
        raise EvaluationError("Candidate limit must be at least 5 for Hit@5 and MRR@5.")
    if not cases:
        raise EvaluationError("Cannot evaluate an empty case list.")

    rows: list[dict[str, Any]] = []
    for case in cases:
        started = perf_counter()
        results = search(case.question, candidate_limit, case.document_id)
        latency_ms = (perf_counter() - started) * 1000
        direct_rank = direct_evidence_rank(results, set(case.relevant_chunk_ids))
        phrase_rank = diagnostic_phrase_rank(results, case.expected_phrases)
        page_rank = diagnostic_page_rank(results, set(case.expected_pages))
        rows.append(
            {
                "id": case.id,
                "language": case.language,
                "category": case.category,
                "critical": case.critical,
                "question": case.question,
                "document_id": case.document_id,
                "relevant_chunk_ids": case.relevant_chunk_ids,
                "expected_pages": case.expected_pages,
                "direct_evidence_rank": direct_rank,
                "diagnostic_phrase_rank": phrase_rank,
                "diagnostic_page_rank": page_rank,
                "latency_ms": latency_ms,
                "retrieved": [
                    _result_summary(result, rank)
                    for rank, result in enumerate(results, start=1)
                ],
            }
        )

    return {
        "candidate_limit": candidate_limit,
        "overall": aggregate_rows(rows, candidate_limit=candidate_limit),
        "per_language": _aggregate_groups(rows, "language", candidate_limit),
        "per_category": _aggregate_groups(rows, "category", candidate_limit),
        "critical_questions": [row for row in rows if row["critical"]],
        "failure_cases": [
            row
            for row in rows
            if row["direct_evidence_rank"] is None or row["direct_evidence_rank"] > 5
        ],
        "per_query": rows,
    }


def aggregate_rows(rows: Sequence[dict[str, Any]], *, candidate_limit: int) -> dict[str, Any]:
    """Aggregate one-based direct-evidence ranks into explicit retrieval metrics."""

    if not rows:
        raise EvaluationError("Cannot aggregate an empty result set.")
    ranks = [row["direct_evidence_rank"] for row in rows]
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "query_count": len(rows),
        "hit_rate_at_1": _hit_rate(ranks, 1),
        "hit_rate_at_3": _hit_rate(ranks, 3),
        "hit_rate_at_5": _hit_rate(ranks, 5),
        "mrr_at_5": _mrr(ranks, 5),
        "mrr_at_candidate_limit": _mrr(ranks, candidate_limit),
        "average_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": percentile_nearest_rank(latencies, 50),
        "p95_latency_ms": percentile_nearest_rank(latencies, 95),
    }


def percentile_nearest_rank(values: Sequence[float], percentile: int) -> float:
    """Calculate a deterministic nearest-rank percentile for a non-empty sample."""

    if not values:
        raise EvaluationError("Cannot calculate a percentile of an empty sample.")
    if not 0 < percentile <= 100:
        raise EvaluationError("Percentile must be in the range 1..100.")
    ordered = sorted(float(value) for value in values)
    index = math.ceil((percentile / 100) * len(ordered)) - 1
    return ordered[index]


def _aggregate_groups(
    rows: Sequence[dict[str, Any]],
    key: str,
    candidate_limit: int,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        name: aggregate_rows(group_rows, candidate_limit=candidate_limit)
        for name, group_rows in sorted(groups.items())
    }


def _result_summary(result: RetrievedLike, rank: int) -> dict[str, Any]:
    """Keep report diagnostics compact and JSON serializable."""

    return {
        "rank": rank,
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "score": float(result.score),
        "page_numbers": list(result.page_numbers),
        "headings": list(result.headings),
    }


def _hit_rate(ranks: Sequence[int | None], cutoff: int) -> float:
    return sum(rank is not None and rank <= cutoff for rank in ranks) / len(ranks)


def _mrr(ranks: Sequence[int | None], cutoff: int) -> float:
    return sum(1 / rank if rank is not None and rank <= cutoff else 0.0 for rank in ranks) / len(
        ranks
    )


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

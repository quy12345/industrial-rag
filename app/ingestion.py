"""Docling-based document validation, conversion, chunking, and normalization."""

from __future__ import annotations

import gc
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from app.domain.documents import (
    SUPPORTED_EXTENSIONS,
    DocumentChunk,
    IngestionError,
    build_chunk_id,
    build_document_id,
    build_page_batches,
    canonical_chunk_key,
)
from app.infrastructure.ingestion import docling as docling_adapter

get_pdf_page_count = docling_adapter.get_pdf_page_count
_convert_document = docling_adapter.convert_document
_validate_conversion_result = docling_adapter.validate_conversion_result
_canonical_chunk_key = canonical_chunk_key


def validate_input_path(file_path: Path) -> Path:
    """Validate and return a supported document path."""

    path = Path(file_path)
    if not path.exists():
        raise IngestionError(f"Input document does not exist: {path}")
    if not path.is_file():
        raise IngestionError(f"Input path is not a file: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise IngestionError(
            f"Unsupported document type: {extension or '<none>'}. Supported types: {supported}"
        )
    return path


def ingest_document(
    file_path: Path,
    *,
    page_range: tuple[int, int] | None = None,
    batch_size: int | None = None,
    chunker: str = "hierarchical",
) -> list[DocumentChunk]:
    """Convert a PDF or DOCX into normalized, structure-aware document chunks."""

    if chunker not in {"hierarchical", "hybrid"}:
        raise IngestionError("chunker must be 'hierarchical' or 'hybrid'.")
    path = validate_input_path(file_path)
    conversion_ranges = _resolve_conversion_ranges(path, page_range, batch_size)
    document_id = build_document_id(path)
    normalized_chunks: list[DocumentChunk] = []
    occurrence_counts: dict[str, int] = {}

    for conversion_range in conversion_ranges:
        # Preserve the historical helper call shape for existing integrations and fakes.
        raw_chunks = (
            _convert_document(path, page_range=conversion_range)
            if chunker == "hierarchical"
            else _convert_document(path, page_range=conversion_range, chunker=chunker)
        )
        _append_normalized_chunks(
            path,
            document_id,
            raw_chunks,
            normalized_chunks,
            occurrence_counts,
        )
        del raw_chunks
        gc.collect()

    if not normalized_chunks:
        raise IngestionError(f"Document produced no non-empty chunks: {path.name}")
    return normalized_chunks


def write_chunks_jsonl(output_path: Path, chunks: Iterable[DocumentChunk]) -> None:
    """Atomically replace an output path with UTF-8 JSON objects."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            for chunk in chunks:
                output_file.write(chunk.model_dump_json())
                output_file.write("\n")
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _resolve_conversion_ranges(
    file_path: Path,
    page_range: tuple[int, int] | None,
    batch_size: int | None,
) -> list[tuple[int, int] | None]:
    """Validate options and return the Docling conversion ranges to run."""

    if file_path.suffix.lower() != ".pdf":
        if page_range is not None:
            raise IngestionError("Page ranges are supported only for PDF documents.")
        if batch_size is not None:
            raise IngestionError("Batch size is supported only for PDF documents.")
        return [None]

    if batch_size is not None and batch_size <= 0:
        raise IngestionError("Batch size must be greater than 0.")

    page_count = get_pdf_page_count(file_path)
    effective_range = page_range or (1, page_count)
    start_page, end_page = effective_range
    _validate_page_range(start_page, end_page, page_count)

    if batch_size is not None:
        return build_page_batches(start_page, end_page, batch_size)
    return [effective_range]


def _validate_page_range(start_page: int, end_page: int, page_count: int) -> None:
    """Validate an inclusive page range against a known PDF page count."""

    if start_page < 1:
        raise IngestionError("Page start must be greater than or equal to 1.")
    if end_page < start_page:
        raise IngestionError("Page end must be greater than or equal to page start.")
    if end_page > page_count:
        raise IngestionError(
            f"Page end {end_page} exceeds the PDF page count of {page_count}."
        )


def _append_normalized_chunks(
    file_path: Path,
    document_id: str,
    raw_chunks: Iterable[Any],
    normalized_chunks: list[DocumentChunk],
    occurrence_counts: dict[str, int],
) -> None:
    """Append normalized chunks while preserving a document-wide chunk index."""

    for raw_chunk in raw_chunks:
        text = _extract_text(raw_chunk)
        if not text:
            continue

        page_numbers = _extract_page_numbers(raw_chunk)
        headings = _extract_headings(raw_chunk)
        chunk_index = len(normalized_chunks)
        canonical_key = _canonical_chunk_key(document_id, page_numbers, headings, text)
        occurrence_index = occurrence_counts.get(canonical_key, 0)
        occurrence_counts[canonical_key] = occurrence_index + 1
        metadata = {
            "source_path": _source_path(file_path),
            "file_extension": file_path.suffix.lower(),
            "chunk_index": chunk_index,
            "character_count": len(text),
        }
        normalized_chunks.append(
            DocumentChunk(
                chunk_id=build_chunk_id(
                    document_id,
                    page_numbers,
                    headings,
                    text,
                    occurrence_index,
                ),
                document_id=document_id,
                filename=file_path.name,
                text=text,
                page_numbers=page_numbers,
                headings=headings,
                content_type=_infer_content_type(raw_chunk),
                metadata=metadata,
            )
        )


def _extract_text(raw_chunk: Any) -> str:
    """Return stripped chunk text without truncating it."""

    text = getattr(raw_chunk, "text", "")
    return text.strip() if isinstance(text, str) else ""


def _extract_page_numbers(raw_chunk: Any) -> list[int]:
    """Extract sorted, unique page numbers from Docling item provenance."""

    pages: set[int] = set()
    for item in _doc_items(raw_chunk):
        for provenance in _as_sequence(getattr(item, "prov", [])):
            page_number = getattr(provenance, "page_no", None)
            if isinstance(page_number, int):
                pages.add(page_number)
    return sorted(pages)


def _extract_headings(raw_chunk: Any) -> list[str]:
    """Extract the heading breadcrumb supplied by Docling."""

    metadata = getattr(raw_chunk, "meta", None)
    headings: list[str] = []
    for heading in _as_sequence(getattr(metadata, "headings", [])):
        value = str(heading).strip()
        if value and (not headings or headings[-1] != value):
            headings.append(value)
    return headings


def _infer_content_type(raw_chunk: Any) -> str:
    """Infer a conservative content type from Docling item labels."""

    labels = {_label_name(item) for item in _doc_items(raw_chunk)}
    labels.discard("")
    if not labels:
        return "unknown"

    if "table" in labels:
        return "table" if labels == {"table"} else "mixed"
    if "code" in labels:
        return "code" if labels == {"code"} else "mixed"
    if labels.issubset({"list", "list_item"}):
        return "list"
    if labels.issubset(
        {
            "caption",
            "equation",
            "formula",
            "page_footer",
            "page_header",
            "paragraph",
            "section_header",
            "text",
            "title",
        }
    ):
        return "text"
    return "unknown"


def _doc_items(raw_chunk: Any) -> list[Any]:
    """Return Docling items attached to a chunk, if available."""

    metadata = getattr(raw_chunk, "meta", None)
    return list(_as_sequence(getattr(metadata, "doc_items", [])))


def _label_name(item: Any) -> str:
    """Normalize a Docling item label to a simple lowercase name."""

    label = getattr(item, "label", "")
    value = getattr(label, "value", label)
    return str(value).rsplit(".", maxsplit=1)[-1].lower()


def _as_sequence(value: Any) -> Sequence[Any]:
    """Return iterable metadata as a sequence while treating missing data as empty."""

    if value is None or isinstance(value, (str, bytes)):
        return []
    if isinstance(value, Sequence):
        return value
    if isinstance(value, Iterable):
        return list(value)
    return []


def _source_path(file_path: Path) -> str:
    """Return a stable, non-machine-specific source path for metadata."""

    if not file_path.is_absolute():
        return file_path.as_posix()
    try:
        return file_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return file_path.name

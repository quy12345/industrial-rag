"""Lazy Docling and PDFium adapter for document conversion."""

from __future__ import annotations

import gc
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from app.domain.documents import IngestionError

_OVERSIZED_METADATA_WARNING_PREFIX = (
    "Headers and captions for this chunk are longer than the total available size "
    "for the chunk, so they will be ignored:"
)


def get_pdf_page_count(file_path: Path) -> int:
    """Return a PDF page count using Docling's lightweight PDFium dependency."""

    try:
        import pypdfium2
    except ImportError as exc:
        raise IngestionError("Docling's PDFium backend is required to count PDF pages.") from exc

    document = None
    try:
        document = pypdfium2.PdfDocument(file_path)
        page_count = len(document)
    except Exception as exc:
        raise IngestionError(f"Failed to count pages in {file_path.name}: {exc}") from exc
    finally:
        if document is not None:
            document.close()

    if page_count < 1:
        raise IngestionError(f"PDF contains no pages: {file_path.name}")
    return page_count


def convert_document(
    file_path: Path,
    *,
    page_range: tuple[int, int] | None = None,
    chunker: str = "hierarchical",
) -> list[Any]:
    """Convert and chunk a document using Docling's native chunkers."""

    try:
        from docling.chunking import HierarchicalChunker, HybridChunker
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise IngestionError(
            "Docling is required for ingestion. Install the project dependencies first."
        ) from exc

    converter = None
    try:
        if file_path.suffix.lower() == ".pdf":
            pipeline_options = PdfPipelineOptions(
                do_ocr=False,
                ocr_batch_size=1,
                layout_batch_size=1,
                table_batch_size=1,
            )
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        else:
            converter = DocumentConverter()

        convert_kwargs: dict[str, Any] = {"source": file_path}
        if page_range is not None:
            convert_kwargs["page_range"] = page_range
        result = converter.convert(**convert_kwargs)
        validate_conversion_result(result, page_range)
        selected_chunker = HierarchicalChunker() if chunker == "hierarchical" else HybridChunker()
        return _chunk_document(selected_chunker, result.document)
    except IngestionError:
        raise
    except Exception as exc:
        range_context = _page_range_context(page_range)
        raise IngestionError(
            f"Failed to convert {range_context} from {file_path.name}: {exc}"
        ) from exc
    finally:
        del converter
        gc.collect()


def _chunk_document(chunker: Any, document: Any) -> list[Any]:
    """Chunk a document without echoing manual contents in Docling warnings."""

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        chunks = list(chunker.chunk(dl_doc=document))

    oversized_metadata_count = 0
    for caught_warning in caught_warnings:
        if (
            issubclass(caught_warning.category, UserWarning)
            and str(caught_warning.message).startswith(_OVERSIZED_METADATA_WARNING_PREFIX)
        ):
            oversized_metadata_count += 1
            continue
        warnings.warn(
            caught_warning.message,
            caught_warning.category,
            stacklevel=2,
        )

    if oversized_metadata_count:
        warnings.warn(
            "Docling omitted oversized heading/caption metadata for "
            f"{oversized_metadata_count} chunk(s); chunk body text was retained.",
            UserWarning,
            stacklevel=2,
        )
    return chunks


def validate_conversion_result(result: Any, page_range: tuple[int, int] | None) -> None:
    """Accept only a complete Docling conversion result."""

    from docling.datamodel.base_models import ConversionStatus

    if result.status == ConversionStatus.SUCCESS:
        return

    range_context = _page_range_context(page_range)
    details = _conversion_error_details(getattr(result, "errors", []))
    if result.status == ConversionStatus.PARTIAL_SUCCESS:
        message = (
            f"Docling returned PARTIAL_SUCCESS for {range_context}; "
            "refusing to create incomplete output."
        )
    elif result.status == ConversionStatus.FAILURE:
        message = f"Docling returned FAILURE for {range_context}."
    else:
        message = f"Docling returned unexpected status {result.status!s} for {range_context}."

    if details:
        message = f"{message} {details}"
    raise IngestionError(message)


def _conversion_error_details(errors: Iterable[Any]) -> str:
    """Summarize failed pages and unique Docling error messages."""

    error_items = list(errors)
    failed_pages = sorted(
        {
            page_number
            for error in error_items
            if isinstance((page_number := getattr(error, "page_no", None)), int)
        }
    )
    messages = list(
        dict.fromkeys(
            message
            for error in error_items
            if (message := str(getattr(error, "error_message", "")).strip())
        )
    )
    parts: list[str] = []
    if failed_pages:
        parts.append(f"Failed pages: {_compact_page_numbers(failed_pages)}.")
    if messages:
        parts.append(f"Details: {'; '.join(messages[:3])}")
    return " ".join(parts)


def _compact_page_numbers(page_numbers: Sequence[int]) -> str:
    """Format sorted page numbers as compact inclusive ranges."""

    ranges: list[str] = []
    range_start = range_end = page_numbers[0]
    for page_number in page_numbers[1:]:
        if page_number == range_end + 1:
            range_end = page_number
            continue
        ranges.append(_format_page_span(range_start, range_end))
        range_start = range_end = page_number
    ranges.append(_format_page_span(range_start, range_end))
    return ", ".join(ranges)


def _format_page_span(start_page: int, end_page: int) -> str:
    """Format one page or an inclusive page span."""

    return str(start_page) if start_page == end_page else f"{start_page}-{end_page}"


def _page_range_context(page_range: tuple[int, int] | None) -> str:
    """Return readable context for a Docling conversion run."""

    if page_range is None:
        return "the full document"
    return f"pages {page_range[0]}-{page_range[1]}"

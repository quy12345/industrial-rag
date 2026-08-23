"""CLI for inspecting Docling document ingestion and normalized chunk metadata."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from app.ingestion import (
    IngestionError,
    build_page_batches,
    get_pdf_page_count,
    ingest_document,
    write_chunks_jsonl,
)
from app.models import DocumentChunk


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ingestion preview CLI."""

    _configure_output_encoding()
    parser = _build_parser()
    args = parser.parse_args(argv)
    input_path = Path(args.input)

    try:
        page_range = _parse_page_range(args.page_start, args.page_end)
        plan = _build_pdf_plan(input_path, page_range, args.batch_size)
        chunks = ingest_document(
            input_path,
            page_range=page_range,
            batch_size=args.batch_size,
        )
        _print_summary(input_path, chunks, plan, args.batch_size)
        _print_previews(chunks[: args.limit], args.preview_chars)
        if args.output:
            output_path = Path(args.output)
            write_chunks_jsonl(output_path, chunks)
            print(f"JSONL output: {output_path}")
    except IngestionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error writing output: {exc}", file=sys.stderr)
        return 1
    return 0


def _configure_output_encoding() -> None:
    """Use UTF-8 for terminal output when the stream supports reconfiguration."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(description="Preview structure-aware Docling chunks.")
    parser.add_argument("input", help="Path to a PDF or DOCX document")
    parser.add_argument("--limit", type=_non_negative_int, default=10)
    parser.add_argument("--output", type=Path, help="Optional JSONL output path")
    parser.add_argument("--page-start", type=_positive_int)
    parser.add_argument("--page-end", type=_positive_int)
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--preview-chars", type=_positive_int, default=500)
    return parser


def _parse_page_range(
    page_start: int | None,
    page_end: int | None,
) -> tuple[int, int] | None:
    """Build a page range only when both CLI bounds are present."""

    if (page_start is None) != (page_end is None):
        raise IngestionError("--page-start and --page-end must be provided together.")
    if page_start is None or page_end is None:
        return None
    return page_start, page_end


def _build_pdf_plan(
    input_path: Path,
    page_range: tuple[int, int] | None,
    batch_size: int | None,
) -> list[tuple[int, int]] | None:
    """Build display metadata for a PDF ingestion run."""

    if input_path.suffix.lower() != ".pdf":
        return None
    page_count = get_pdf_page_count(input_path)
    effective_range = page_range or (1, page_count)
    if batch_size is None:
        return [effective_range]
    return build_page_batches(*effective_range, batch_size)


def _print_summary(
    input_path: Path,
    chunks: list[DocumentChunk],
    plan: list[tuple[int, int]] | None,
    batch_size: int | None,
) -> None:
    """Print document and chunk summary information."""

    document_id = chunks[0].document_id
    pages = sorted({page for chunk in chunks for page in chunk.page_numbers})
    content_types = Counter(chunk.content_type for chunk in chunks)
    page_summary = ", ".join(str(page) for page in pages) if pages else "unknown"

    print("Document ingestion completed")
    print(f"File: {input_path.name}")
    print(f"Document ID: {document_id}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Pages represented: {page_summary}")
    if plan is not None:
        print(f"Page range: {plan[0][0]}-{plan[-1][1]}")
        print(f"Batch size: {batch_size if batch_size is not None else 'not used'}")
        print(f"Batches processed: {len(plan)}")
    print("Content types:")
    for content_type, count in sorted(content_types.items()):
        print(f"  {content_type}: {count}")


def _print_previews(chunks: list[DocumentChunk], preview_chars: int) -> None:
    """Print a bounded preview for each selected chunk."""

    for index, chunk in enumerate(chunks, start=1):
        pages = ", ".join(str(page) for page in chunk.page_numbers) or "unknown"
        headings = " > ".join(chunk.headings) or "unknown"
        print("-" * 60)
        print(f"Chunk {index}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Pages: {pages}")
        print(f"Headings: {headings}")
        print(f"Content type: {chunk.content_type}")
        print(f"Characters: {len(chunk.text)}")
        print()
        print(chunk.text[:preview_chars])


def _non_negative_int(value: str) -> int:
    """Parse a non-negative integer CLI argument."""

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI argument."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())

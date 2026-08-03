"""CLI for ingesting a document and indexing dense vectors in Qdrant."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from app.config import get_settings
from app.ingestion import IngestionError, ingest_document
from app.retrieval import (
    RetrievalError,
    create_embedding_model,
    create_qdrant_client,
    get_embedding_dimension,
    index_chunks,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Ingest and index one PDF or DOCX document."""

    _configure_output_encoding()
    settings = get_settings()
    args = _build_parser().parse_args(argv)
    input_path = Path(args.input)

    try:
        page_range = _parse_page_range(args.page_start, args.page_end)
        chunks = ingest_document(
            input_path,
            page_range=page_range,
            batch_size=args.page_batch_size,
        )
        embedding_model = create_embedding_model(settings.embedding_model)
        vector_size = get_embedding_dimension(embedding_model)
        client = create_qdrant_client(settings)
        chunks_indexed = index_chunks(
            client,
            chunks,
            collection_name=settings.qdrant_collection,
            vector_name=settings.dense_vector_name,
            embedding_model=embedding_model,
            embedding_batch_size=(
                args.embedding_batch_size or settings.embedding_batch_size
            ),
            vector_size=vector_size,
        )
    except (IngestionError, RetrievalError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Dense indexing completed")
    print(f"File: {input_path.name}")
    print(f"Document ID: {chunks[0].document_id}")
    print(f"Chunks indexed: {chunks_indexed}")
    print(f"Collection: {settings.qdrant_collection}")
    print(f"Vector name: {settings.dense_vector_name}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Embedding dimension: {vector_size}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the indexing command parser."""

    parser = argparse.ArgumentParser(description="Index document chunks in Qdrant.")
    parser.add_argument("input", help="Path to a PDF or DOCX document")
    parser.add_argument("--page-start", type=_positive_int)
    parser.add_argument("--page-end", type=_positive_int)
    parser.add_argument("--page-batch-size", type=_positive_int)
    parser.add_argument("--embedding-batch-size", type=_positive_int)
    return parser


def _parse_page_range(
    page_start: int | None,
    page_end: int | None,
) -> tuple[int, int] | None:
    """Require both inclusive page bounds when either is supplied."""

    if (page_start is None) != (page_end is None):
        raise IngestionError("--page-start and --page-end must be provided together.")
    if page_start is None or page_end is None:
        return None
    return page_start, page_end


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI argument."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _configure_output_encoding() -> None:
    """Use UTF-8 terminal output where supported."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

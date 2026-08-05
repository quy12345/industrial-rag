"""CLI for dense similarity search over indexed document chunks."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from app.config import get_settings
from app.retrieval import (
    INDEX_MANIFEST_PATH,
    RetrievalError,
    create_embedding_model,
    create_qdrant_client,
    dense_search,
    get_embedding_dimension,
    validate_dense_collection,
    validate_index_manifest,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one dense query and print bounded ranked previews."""

    _configure_output_encoding()
    settings = get_settings()
    args = _build_parser().parse_args(argv)
    limit = args.limit or settings.retrieval_top_k
    score_threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else settings.retrieval_score_threshold
    )

    try:
        embedding_model = create_embedding_model(settings.embedding_model)
        vector_size = get_embedding_dimension(embedding_model)
        validate_index_manifest(
            INDEX_MANIFEST_PATH,
            collection_name=settings.qdrant_collection,
            vector_name=settings.dense_vector_name,
            embedding_model=settings.embedding_model,
            embedding_dimension=vector_size,
        )
        client = create_qdrant_client(settings)
        validate_dense_collection(
            client,
            collection_name=settings.qdrant_collection,
            vector_name=settings.dense_vector_name,
            vector_size=vector_size,
        )
        results = dense_search(
            client,
            args.question,
            collection_name=settings.qdrant_collection,
            vector_name=settings.dense_vector_name,
            embedding_model=embedding_model,
            limit=limit,
            document_id=args.document_id,
            score_threshold=score_threshold,
        )
    except RetrievalError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Dense search completed")
    print(f"Question: {args.question.strip()}")
    print(f"Results: {len(results)}")
    for rank, result in enumerate(results, start=1):
        pages = ", ".join(str(page) for page in result.page_numbers) or "unknown"
        headings = " > ".join(result.headings) or "unknown"
        preview, truncated = _truncate(result.text, args.preview_chars)
        print("-" * 60)
        print(f"Rank: {rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"File: {result.filename}")
        print(f"Pages: {pages}")
        print(f"Headings: {headings}")
        print(f"Content type: {result.content_type}")
        print()
        print(f"{preview}{' ... [truncated]' if truncated else ''}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the dense-search command parser."""

    parser = argparse.ArgumentParser(description="Search Qdrant dense vectors.")
    parser.add_argument("question", help="Natural-language search question")
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--document-id")
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--preview-chars", type=_positive_int, default=500)
    return parser


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """Return a bounded text preview and whether it was truncated."""

    if len(text) <= limit:
        return text, False
    return text[:limit], True


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

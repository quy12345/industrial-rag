"""Ingest one document and safely index dense plus BM25 sparse vectors into v2."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from app.config import get_settings
from app.evaluation import chunk_set_metadata, load_frozen_chunks
from app.hybrid_retrieval import (
    HYBRID_INDEX_MANIFEST_PATH,
    compute_bm25_average_length,
    create_sparse_embedding_model,
    index_hybrid_chunks,
    write_hybrid_index_manifest,
)
from app.ingestion import IngestionError, ingest_document
from app.retrieval import (
    RetrievalError,
    create_embedding_model,
    create_qdrant_client,
    get_embedding_dimension,
)

DEFAULT_FROZEN_CHUNKS = Path("artifacts/manual-batched.jsonl")


def main(argv: Sequence[str] | None = None) -> int:
    """Build v2 only after the incoming chunks match the dense frozen set."""

    _configure_output_encoding()
    settings = get_settings()
    args = _build_parser().parse_args(argv)
    try:
        chunks = ingest_document(
            Path(args.input),
            page_range=_parse_page_range(args.page_start, args.page_end),
            batch_size=args.page_batch_size,
        )
        frozen_chunks = load_frozen_chunks(args.frozen_chunks)
        frozen_metadata = chunk_set_metadata(frozen_chunks)
        if chunk_set_metadata(chunks) != frozen_metadata:
            raise RetrievalError(
                "Ingested chunk set does not equal the frozen dense baseline; v2 indexing stopped."
            )

        dense_model = create_embedding_model(
            settings.embedding_model, cache_dir=settings.embedding_cache_dir
        )
        dense_dimension = get_embedding_dimension(dense_model)
        # A probe instance gives access to the exact installed FastEmbed BM25 tokenizer.
        sparse_probe = create_sparse_embedding_model(
            settings.sparse_model,
            settings.embedding_cache_dir,
            disable_stemmer=settings.bm25_disable_stemmer,
            k=settings.bm25_k,
            b=settings.bm25_b,
            avg_len=settings.bm25_avg_len or 256.0,
        )
        computed_avg_len = compute_bm25_average_length(sparse_probe, chunks)
        if settings.bm25_avg_len is not None and settings.bm25_avg_len != computed_avg_len:
            raise RetrievalError(
                "BM25_AVG_LEN is configured but differs from the frozen corpus value: "
                f"configured={settings.bm25_avg_len}, computed={computed_avg_len}."
            )
        sparse_model = create_sparse_embedding_model(
            settings.sparse_model,
            settings.embedding_cache_dir,
            disable_stemmer=settings.bm25_disable_stemmer,
            k=settings.bm25_k,
            b=settings.bm25_b,
            avg_len=computed_avg_len,
        )
        client = create_qdrant_client(settings)
        indexed = index_hybrid_chunks(
            client,
            chunks,
            collection_name=settings.qdrant_hybrid_collection,
            dense_vector_name=settings.dense_vector_name,
            sparse_vector_name=settings.sparse_vector_name,
            dense_embedding_model=dense_model,
            sparse_embedding_model=sparse_model,
            dense_embedding_batch_size=args.embedding_batch_size or settings.embedding_batch_size,
            sparse_embedding_batch_size=args.sparse_embedding_batch_size
            or settings.sparse_embedding_batch_size,
            dense_vector_size=dense_dimension,
        )
        write_hybrid_index_manifest(
            HYBRID_INDEX_MANIFEST_PATH,
            settings=settings,
            dense_dimension=dense_dimension,
            bm25_avg_len=computed_avg_len,
            frozen_chunk_set=frozen_metadata,
            ingestion_profile={
                "ocr_mode": "off",
                "page_batch_size": args.page_batch_size,
                "chunker": "hierarchical",
            },
        )
    except (IngestionError, RetrievalError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Hybrid indexing completed")
    print(f"Document ID: {chunks[0].document_id}")
    print(f"Chunks indexed: {indexed}")
    print(f"Collection: {settings.qdrant_hybrid_collection}")
    print(f"BM25 avg_len: {computed_avg_len:.6f}")
    print(f"Manifest: {HYBRID_INDEX_MANIFEST_PATH}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index frozen chunks into the hybrid v2 collection."
    )
    parser.add_argument("input", help="Path to a PDF or DOCX document")
    parser.add_argument("--frozen-chunks", type=Path, default=DEFAULT_FROZEN_CHUNKS)
    parser.add_argument("--page-start", type=_positive_int)
    parser.add_argument("--page-end", type=_positive_int)
    parser.add_argument("--page-batch-size", type=_positive_int)
    parser.add_argument("--embedding-batch-size", type=_positive_int)
    parser.add_argument("--sparse-embedding-batch-size", type=_positive_int)
    return parser


def _parse_page_range(start: int | None, end: int | None) -> tuple[int, int] | None:
    if (start is None) != (end is None):
        raise IngestionError("--page-start and --page-end must be provided together.")
    return (start, end) if start is not None and end is not None else None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI for client-side RRF over dense and BM25 sparse Qdrant results."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from app.config import get_settings
from app.evaluation import chunk_set_metadata, load_frozen_chunks
from app.hybrid_retrieval import (
    HYBRID_INDEX_MANIFEST_PATH,
    create_sparse_embedding_model,
    hybrid_search,
    validate_hybrid_collection,
    validate_hybrid_index_manifest,
)
from app.retrieval import (
    RetrievalError,
    create_embedding_model,
    create_qdrant_client,
    get_embedding_dimension,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one validated hybrid query and print bounded component diagnostics."""

    _configure_output_encoding()
    settings = get_settings()
    args = _build_parser().parse_args(argv)
    try:
        frozen_metadata = chunk_set_metadata(load_frozen_chunks(args.frozen_chunks))
        dense_model = create_embedding_model(
            settings.embedding_model, cache_dir=settings.embedding_cache_dir
        )
        dense_dimension = get_embedding_dimension(dense_model)
        manifest = validate_hybrid_index_manifest(
            HYBRID_INDEX_MANIFEST_PATH,
            settings=settings,
            dense_dimension=dense_dimension,
            frozen_chunk_set=frozen_metadata,
        )
        sparse_model = create_sparse_embedding_model(
            settings.sparse_model,
            settings.embedding_cache_dir,
            disable_stemmer=settings.bm25_disable_stemmer,
            k=settings.bm25_k,
            b=settings.bm25_b,
            avg_len=float(manifest["bm25_avg_len"]),
        )
        client = create_qdrant_client(settings)
        validate_hybrid_collection(
            client,
            collection_name=settings.qdrant_hybrid_collection,
            dense_vector_name=settings.dense_vector_name,
            dense_vector_size=dense_dimension,
            sparse_vector_name=settings.sparse_vector_name,
        )
        results = hybrid_search(
            client,
            args.question,
            collection_name=settings.qdrant_hybrid_collection,
            dense_vector_name=settings.dense_vector_name,
            sparse_vector_name=settings.sparse_vector_name,
            dense_embedding_model=dense_model,
            sparse_embedding_model=sparse_model,
            dense_candidate_limit=args.dense_candidate_limit or settings.dense_candidate_limit,
            sparse_candidate_limit=args.sparse_candidate_limit or settings.sparse_candidate_limit,
            final_limit=args.limit or settings.hybrid_final_limit,
            rrf_k=settings.rrf_k,
            document_id=args.document_id,
        )
    except (RetrievalError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Hybrid search completed")
    for result in results:
        print("-" * 60)
        print(f"Rank: {result.rrf_rank}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Pages: {', '.join(str(page) for page in result.page_numbers) or 'unknown'}")
        print(f"Headings: {' > '.join(result.headings) or 'unknown'}")
        print(f"Dense rank / score: {result.dense_rank} / {result.dense_score}")
        print(f"Sparse rank / score: {result.sparse_rank} / {result.sparse_score}")
        print(f"RRF score: {result.rrf_score}")
        preview = result.text[: args.preview_chars]
        print(f"{preview}{' ... [truncated]' if len(result.text) > args.preview_chars else ''}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search hybrid dense + BM25 RRF results.")
    parser.add_argument("question")
    parser.add_argument("--document-id")
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--dense-candidate-limit", type=_positive_int)
    parser.add_argument("--sparse-candidate-limit", type=_positive_int)
    parser.add_argument("--frozen-chunks", default="artifacts/manual-batched.jsonl")
    parser.add_argument("--preview-chars", type=_positive_int, default=500)
    return parser


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

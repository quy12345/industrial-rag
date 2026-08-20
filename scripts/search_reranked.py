"""CLI for sparse, RRF-hybrid, or dense-sparse-union cross-encoder reranking."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from app.config import get_settings
from app.evaluation import chunk_set_metadata, load_frozen_chunks
from app.reranking import RerankingError
from app.retrieval import RetrievalError
from scripts.rerank_runtime import build_rerank_runtime


def main(argv: Sequence[str] | None = None) -> int:
    """Run one validated reranked query and print bounded diagnostics."""

    _configure_output_encoding()
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    try:
        chunks = load_frozen_chunks(args.frozen_chunks)
        pipeline, _ = build_rerank_runtime(settings, chunk_set_metadata(chunks), chunks)
        execution = pipeline.search(
            args.question,
            strategy=args.strategy,
            document_id=args.document_id,
        )
    except (OSError, RetrievalError, RerankingError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"{args.strategy.title()} candidate-pool reranking completed")
    print(f"Full candidate pool: {len(execution.candidates_after_rerank)}")
    print(f"Stage latency (ms): {execution.stage_latency_ms}")
    for result in execution.candidates_after_rerank[: args.limit]:
        print("-" * 60)
        print(f"Final rank: {result.rerank_rank}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Pages: {', '.join(str(page) for page in result.page_numbers) or 'unknown'}")
        print(f"Headings: {' > '.join(result.headings) or 'unknown'}")
        print(f"Rerank score: {result.rerank_score}")
        print(f"Dense rank / score: {result.dense_rank} / {result.dense_score}")
        print(f"Sparse rank / score: {result.sparse_rank} / {result.sparse_score}")
        print(f"RRF rank / score: {result.rrf_rank} / {result.rrf_score}")
        preview = result.text[: args.preview_chars]
        print(f"{preview}{' ... [truncated]' if len(result.text) > args.preview_chars else ''}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-encoder rerank one candidate pool.")
    parser.add_argument("question")
    parser.add_argument("--strategy", choices=("sparse", "hybrid", "union"), required=True)
    parser.add_argument("--document-id")
    parser.add_argument("--limit", type=_positive_int, default=5)
    parser.add_argument(
        "--frozen-chunks", type=Path, default=Path("artifacts/manual-batched.jsonl")
    )
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

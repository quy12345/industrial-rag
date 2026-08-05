"""Evaluate dense retrieval against a small JSONL smoke set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

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


def load_smoke_set(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate one JSON object per line."""

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("question"), str):
            raise ValueError(f"Smoke record on line {line_number} must contain a question.")
        records.append(record)
    if not records:
        raise ValueError(f"Smoke set is empty: {path}")
    return records


def is_relevant(result: Any, record: dict[str, Any]) -> bool:
    """Match a result by expected page or case-insensitive phrase."""

    expected_pages = {int(page) for page in record.get("expected_pages", [])}
    if expected_pages.intersection(result.page_numbers):
        return True
    text = result.text.casefold()
    return any(str(phrase).casefold() in text for phrase in record.get("expected_phrases", []))


def evaluate_records(
    records: Sequence[dict[str, Any]],
    search: Callable[[str, int, str | None], list[Any]],
    *,
    limit: int = 5,
) -> dict[str, Any]:
    """Run smoke queries and return ranks plus aggregate retrieval metrics."""

    if limit <= 0:
        raise ValueError("Evaluation limit must be greater than 0.")
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    reciprocal_ranks: list[float] = []
    for record in records:
        started = time.perf_counter()
        results = search(record["question"], limit, record.get("document_id"))
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        relevant_rank = next(
            (rank for rank, result in enumerate(results, start=1) if is_relevant(result, record)),
            None,
        )
        reciprocal_ranks.append(1 / relevant_rank if relevant_rank is not None else 0.0)
        rows.append(
            {
                "id": record.get("id", ""),
                "question": record["question"],
                "first_relevant_rank": relevant_rank,
                "top_5_hit": relevant_rank is not None,
                "results": results,
                "latency_ms": latency_ms,
            }
        )

    def hit_rate(cutoff: int) -> float:
        return sum(
            row["first_relevant_rank"] is not None
            and row["first_relevant_rank"] <= cutoff
            for row in rows
        ) / len(rows)

    return {
        "rows": rows,
        "hit_rate_at_1": hit_rate(1),
        "hit_rate_at_3": hit_rate(3),
        "hit_rate_at_5": hit_rate(5),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "average_latency_ms": sum(latencies) / len(latencies),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dense smoke evaluation against the configured Qdrant index."""

    _configure_output_encoding()
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    try:
        records = load_smoke_set(args.dataset)
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

        def search(question: str, limit: int, document_id: str | None) -> list[Any]:
            return dense_search(
                client,
                question,
                collection_name=settings.qdrant_collection,
                vector_name=settings.dense_vector_name,
                embedding_model=embedding_model,
                limit=limit,
                document_id=document_id,
            )

        metrics = evaluate_records(records, search, limit=args.limit)
    except (OSError, ValueError, RetrievalError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Question | First relevant rank | Relevant page/phrase | Top-5 hit")
    for row in metrics["rows"]:
        rank = row["first_relevant_rank"] or "none"
        match = "matched" if row["top_5_hit"] else "none"
        print(f"{row['question']} | {rank} | {match} | {row['top_5_hit']}")
    print(f"Hit Rate@1: {metrics['hit_rate_at_1']:.3f}")
    print(f"Hit Rate@3: {metrics['hit_rate_at_3']:.3f}")
    print(f"Hit Rate@5: {metrics['hit_rate_at_5']:.3f}")
    print(f"MRR: {metrics['mrr']:.3f}")
    print(f"Average latency: {metrics['average_latency_ms']:.2f} ms")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval on a JSONL smoke set.")
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("data/eval/dense_smoke.jsonl"),
    )
    parser.add_argument("--limit", type=_positive_int, default=5)
    return parser


def _positive_int(value: str) -> int:
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

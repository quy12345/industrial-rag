"""Run the direct-evidence dense retrieval development baseline."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.evaluation import (
    EvaluationError,
    chunk_set_metadata,
    evaluate_cases,
    load_evaluation_cases,
    load_frozen_chunks,
    validate_cases_against_chunks,
)
from app.retrieval import (
    INDEX_MANIFEST_PATH,
    RetrievalError,
    create_embedding_model,
    create_qdrant_client,
    dense_search,
    get_embedding_dimension,
    get_indexed_chunk_ids,
    validate_dense_collection,
    validate_index_manifest,
)

DEFAULT_DATASET = Path("data/eval/dense_smoke.jsonl")
DEFAULT_FROZEN_CHUNKS = Path("artifacts/manual-batched.jsonl")
DEFAULT_OUTPUT = Path("artifacts/metrics/dense-baseline.json")


def main(argv: list[str] | None = None) -> int:
    """Evaluate a frozen dense index with stable direct-evidence qrels."""

    _configure_output_encoding()
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    try:
        cases = load_evaluation_cases(args.dataset)
        frozen_chunks = load_frozen_chunks(args.chunks)
        validate_cases_against_chunks(cases, frozen_chunks)
        frozen_metadata = chunk_set_metadata(frozen_chunks)
        expected_ids_by_document = _chunk_ids_by_document(frozen_chunks)

        embedding_model = create_embedding_model(
            settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
        )
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
        _validate_index_matches_frozen_chunks(
            client,
            collection_name=settings.qdrant_collection,
            expected_ids_by_document=expected_ids_by_document,
        )

        def search(question: str, limit: int, document_id: str) -> list[Any]:
            return dense_search(
                client,
                question,
                collection_name=settings.qdrant_collection,
                vector_name=settings.dense_vector_name,
                embedding_model=embedding_model,
                limit=limit,
                document_id=document_id,
            )

        # This call warms the embedding and Qdrant request paths. It is intentionally
        # excluded from all latency samples in the report.
        first_case = cases[0]
        search(first_case.question, args.limit, first_case.document_id)
        evaluation = evaluate_cases(cases, search, candidate_limit=args.limit)
        report = _build_report(
            evaluation=evaluation,
            dataset_path=args.dataset,
            frozen_chunks_path=args.chunks,
            frozen_metadata=frozen_metadata,
            settings=settings,
            embedding_dimension=vector_size,
        )
        _write_json_atomic(args.output, report)
    except (EvaluationError, OSError, RetrievalError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_summary(report, args.output)
    return 0


def _build_report(
    *,
    evaluation: dict[str, Any],
    dataset_path: Path,
    frozen_chunks_path: Path,
    frozen_metadata: dict[str, Any],
    settings: Any,
    embedding_dimension: int,
) -> dict[str, Any]:
    """Build the machine-readable source of truth for one baseline run."""

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "evaluation_dataset": {
            "path": str(dataset_path),
            "sha256": _sha256_file(dataset_path),
            "purpose": "retrieval development set; not the Phase 6 held-out evaluation set",
        },
        "frozen_chunk_set": {
            "path": str(frozen_chunks_path),
            **frozen_metadata,
        },
        "runtime": {
            "python": platform.python_version(),
            "libraries": _library_versions(),
        },
        "index": {
            "collection": settings.qdrant_collection,
            "vector_name": settings.dense_vector_name,
            "distance": "cosine",
            "embedding_model": settings.embedding_model,
            "embedding_dimension": embedding_dimension,
        },
        "latency_methodology": {
            "unit": "milliseconds",
            "includes": ["query embedding", "Qdrant round trip", "result mapping"],
            "excludes": ["embedding model initialization", "model download", "one warmup query"],
            "percentile_method": "nearest-rank",
        },
        **evaluation,
    }


def _validate_index_matches_frozen_chunks(
    client: Any,
    *,
    collection_name: str,
    expected_ids_by_document: dict[str, set[str]],
) -> None:
    """Reject a baseline run when Qdrant does not equal its frozen chunk set."""

    for document_id, expected_ids in expected_ids_by_document.items():
        actual_ids = get_indexed_chunk_ids(
            client,
            collection_name=collection_name,
            document_id=document_id,
        )
        if actual_ids == expected_ids:
            continue
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise EvaluationError(
            f"Indexed chunk set differs from frozen chunks for {document_id}: "
            f"expected {len(expected_ids)}, found {len(actual_ids)}, "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}. Re-index the frozen set first."
        )


def _chunk_ids_by_document(chunks: list[Any]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document_id, set()).add(chunk.chunk_id)
    return grouped


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write reports atomically so a failed evaluation never publishes partial JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        temporary_path.replace(output_path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _print_summary(report: dict[str, Any], output_path: Path) -> None:
    """Print the compact human-facing view while JSON remains the source of truth."""

    overall = report["overall"]
    print("Dense direct-evidence baseline")
    print(f"Queries: {overall['query_count']}")
    print(f"Frozen chunks: {report['frozen_chunk_set']['chunk_count']}")
    print(f"Candidate limit: {report['candidate_limit']}")
    print(f"Hit Rate@1: {overall['hit_rate_at_1']:.3f}")
    print(f"Hit Rate@3: {overall['hit_rate_at_3']:.3f}")
    print(f"Hit Rate@5: {overall['hit_rate_at_5']:.3f}")
    print(f"MRR@5: {overall['mrr_at_5']:.3f}")
    print(f"MRR@{report['candidate_limit']}: {overall['mrr_at_candidate_limit']:.3f}")
    print(f"Latency average: {overall['average_latency_ms']:.2f} ms")
    print(f"Latency p50: {overall['p50_latency_ms']:.2f} ms")
    print(f"Latency p95: {overall['p95_latency_ms']:.2f} ms")
    print(f"Failure cases: {len(report['failure_cases'])}")
    print(f"Report: {output_path}")


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _library_versions() -> dict[str, str | None]:
    packages = ["docling", "docling-core", "qdrant-client", "fastembed"]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate dense retrieval against direct-evidence stable chunk qrels."
    )
    parser.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_FROZEN_CHUNKS,
        help="Frozen JSONL chunk set used to validate qrels and indexed chunk IDs.",
    )
    parser.add_argument(
        "--limit",
        type=_minimum_candidate_limit,
        default=20,
        help="Candidate limit; must be at least 5 (default: 20).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _minimum_candidate_limit(value: str) -> int:
    parsed = int(value)
    if parsed < 5:
        raise argparse.ArgumentTypeError("must be at least 5")
    return parsed


def _configure_output_encoding() -> None:
    """Use UTF-8 terminal output where supported."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

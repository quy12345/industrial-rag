"""Evaluate dense, sparse, or hybrid retrieval against immutable direct-evidence qrels."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.evaluation import (
    EvaluationError,
    chunk_set_metadata,
    evaluate_cases,
    load_evaluation_cases,
    load_frozen_chunks,
    validate_cases_against_chunks,
)
from app.hybrid_retrieval import (
    HYBRID_INDEX_MANIFEST_PATH,
    create_sparse_embedding_model,
    hybrid_search,
    sparse_search,
    validate_hybrid_collection,
    validate_hybrid_index_manifest,
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
DEFAULT_OUTPUTS = {
    "dense": Path("artifacts/metrics/dense-baseline-closure.json"),
    "sparse": Path("artifacts/metrics/sparse-baseline.json"),
    "hybrid": Path("artifacts/metrics/hybrid-baseline.json"),
}


def main(argv: list[str] | None = None) -> int:
    """Run one retrieval strategy over the same frozen development dataset."""

    _configure_output_encoding()
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    output = args.output or DEFAULT_OUTPUTS[args.strategy]
    try:
        cases = load_evaluation_cases(args.dataset)
        frozen_chunks = load_frozen_chunks(args.chunks)
        validate_cases_against_chunks(cases, frozen_chunks)
        frozen_metadata = chunk_set_metadata(frozen_chunks)
        expected_ids = _chunk_ids_by_document(frozen_chunks)
        search, index = _build_strategy_search(
            args.strategy,
            settings=settings,
            frozen_metadata=frozen_metadata,
            expected_ids=expected_ids,
        )

        first_case = cases[0]
        search(first_case.question, args.limit, first_case.document_id)
        evaluation = evaluate_cases(cases, search, candidate_limit=args.limit)
        report = _build_report(
            strategy=args.strategy,
            evaluation=evaluation,
            dataset_path=args.dataset,
            frozen_chunks_path=args.chunks,
            frozen_metadata=frozen_metadata,
            index=index,
        )
        if args.strategy == "hybrid":
            report["comparison"] = _comparison_if_available(report)
        _write_json_atomic(output, report)
    except (EvaluationError, OSError, RetrievalError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_summary(report, output)
    return 0


def _build_strategy_search(
    strategy: str,
    *,
    settings: Settings,
    frozen_metadata: dict[str, Any],
    expected_ids: dict[str, set[str]],
) -> tuple[Callable[[str, int, str], list[Any]], dict[str, Any]]:
    """Validate the chosen physical index and return its dependency-injected search."""

    dense_model = create_embedding_model(
        settings.embedding_model, cache_dir=settings.embedding_cache_dir
    )
    dense_dimension = get_embedding_dimension(dense_model)
    client = create_qdrant_client(settings)
    if strategy == "dense":
        validate_index_manifest(
            INDEX_MANIFEST_PATH,
            collection_name=settings.qdrant_collection,
            vector_name=settings.dense_vector_name,
            embedding_model=settings.embedding_model,
            embedding_dimension=dense_dimension,
        )
        validate_dense_collection(
            client,
            collection_name=settings.qdrant_collection,
            vector_name=settings.dense_vector_name,
            vector_size=dense_dimension,
        )
        _validate_index_matches_frozen_chunks(client, settings.qdrant_collection, expected_ids)

        def search(question: str, limit: int, document_id: str) -> list[Any]:
            return dense_search(
                client,
                question,
                collection_name=settings.qdrant_collection,
                vector_name=settings.dense_vector_name,
                embedding_model=dense_model,
                limit=limit,
                document_id=document_id,
            )

        return search, {
            "collection": settings.qdrant_collection,
            "dense_vector_name": settings.dense_vector_name,
            "dense_model": settings.embedding_model,
            "dense_dimension": dense_dimension,
            "dense_distance": "cosine",
        }

    manifest = validate_hybrid_index_manifest(
        HYBRID_INDEX_MANIFEST_PATH,
        settings=settings,
        dense_dimension=dense_dimension,
        frozen_chunk_set=frozen_metadata,
    )
    validate_hybrid_collection(
        client,
        collection_name=settings.qdrant_hybrid_collection,
        dense_vector_name=settings.dense_vector_name,
        dense_vector_size=dense_dimension,
        sparse_vector_name=settings.sparse_vector_name,
    )
    _validate_index_matches_frozen_chunks(client, settings.qdrant_hybrid_collection, expected_ids)
    sparse_model = create_sparse_embedding_model(
        settings.sparse_model,
        settings.embedding_cache_dir,
        disable_stemmer=settings.bm25_disable_stemmer,
        k=settings.bm25_k,
        b=settings.bm25_b,
        avg_len=float(manifest["bm25_avg_len"]),
    )
    if strategy == "sparse":

        def search(question: str, limit: int, document_id: str) -> list[Any]:
            return sparse_search(
                client,
                question,
                collection_name=settings.qdrant_hybrid_collection,
                sparse_vector_name=settings.sparse_vector_name,
                sparse_embedding_model=sparse_model,
                limit=limit,
                document_id=document_id,
            )

    else:

        def search(question: str, limit: int, document_id: str) -> list[Any]:
            return hybrid_search(
                client,
                question,
                collection_name=settings.qdrant_hybrid_collection,
                dense_vector_name=settings.dense_vector_name,
                sparse_vector_name=settings.sparse_vector_name,
                dense_embedding_model=dense_model,
                sparse_embedding_model=sparse_model,
                dense_candidate_limit=min(limit, settings.dense_candidate_limit),
                sparse_candidate_limit=min(limit, settings.sparse_candidate_limit),
                final_limit=limit,
                rrf_k=settings.rrf_k,
                document_id=document_id,
            )

    return search, {
        "collection": settings.qdrant_hybrid_collection,
        "dense_vector_name": settings.dense_vector_name,
        "dense_model": settings.embedding_model,
        "dense_dimension": dense_dimension,
        "dense_distance": "cosine",
        "sparse_vector_name": settings.sparse_vector_name,
        "sparse_model": settings.sparse_model,
        "sparse_modifier": "idf",
        "bm25_avg_len": manifest["bm25_avg_len"],
        "manifest": str(HYBRID_INDEX_MANIFEST_PATH),
    }


def _build_report(
    *,
    strategy: str,
    evaluation: dict[str, Any],
    dataset_path: Path,
    frozen_chunks_path: Path,
    frozen_metadata: dict[str, Any],
    index: dict[str, Any],
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "timestamp": datetime.now(UTC).isoformat(),
        "evaluation_dataset": {
            "path": str(dataset_path),
            "sha256": _sha256_file(dataset_path),
            "purpose": "retrieval development set; not the Phase 6 held-out evaluation set",
        },
        "frozen_chunk_set": {"path": str(frozen_chunks_path), **frozen_metadata},
        "runtime": {"python": platform.python_version(), "libraries": _library_versions()},
        "index": index,
        "latency_methodology": {
            "unit": "milliseconds",
            "includes": [
                "query embedding",
                "Qdrant round trip",
                "result mapping",
                "RRF for hybrid",
            ],
            "excludes": ["embedding model initialization", "model download", "one warmup query"],
            "percentile_method": "nearest-rank",
        },
        **evaluation,
    }


def _validate_index_matches_frozen_chunks(
    client: Any, collection_name: str, expected_ids: dict[str, set[str]]
) -> None:
    for document_id, expected in expected_ids.items():
        actual = get_indexed_chunk_ids(
            client, collection_name=collection_name, document_id=document_id
        )
        if actual != expected:
            raise EvaluationError(
                f"Indexed chunk set differs from frozen chunks for {document_id}: "
                f"expected {len(expected)}, found {len(actual)}. Re-index the frozen set first."
            )


def _chunk_ids_by_document(chunks: list[Any]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document_id, set()).add(chunk.chunk_id)
    return grouped


def _comparison_if_available(hybrid_report: dict[str, Any]) -> dict[str, Any] | None:
    """Add an evidence-based comparison only when the matching artifacts exist."""

    paths = {
        "dense": DEFAULT_OUTPUTS["dense"],
        "sparse": DEFAULT_OUTPUTS["sparse"],
    }
    try:
        prior = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    except (OSError, json.JSONDecodeError):
        return None
    metrics = [
        "hit_rate_at_1",
        "hit_rate_at_3",
        "hit_rate_at_5",
        "hit_rate_at_candidate_limit",
        "mrr_at_5",
        "mrr_at_candidate_limit",
        "p50_latency_ms",
        "p95_latency_ms",
    ]
    dense = prior["dense"].get("overall", {})
    sparse = prior["sparse"].get("overall", {})
    hybrid = hybrid_report["overall"]
    return {
        metric: {
            "dense": dense.get(metric),
            "sparse": sparse.get(metric),
            "hybrid": hybrid.get(metric),
            "hybrid_minus_dense": (
                hybrid[metric] - dense[metric]
                if isinstance(hybrid.get(metric), (int, float))
                and isinstance(dense.get(metric), (int, float))
                else None
            ),
        }
        for metric in metrics
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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
    overall = report["overall"]
    print(f"{report['strategy'].title()} direct-evidence retrieval evaluation")
    print(f"Queries: {overall['query_count']}")
    print(f"Hit Rate@1: {overall['hit_rate_at_1']:.3f}")
    print(f"Hit Rate@3: {overall['hit_rate_at_3']:.3f}")
    print(f"Hit Rate@5: {overall['hit_rate_at_5']:.3f}")
    print(f"Hit Rate@{report['candidate_limit']}: {overall['hit_rate_at_candidate_limit']:.3f}")
    print(f"MRR@5: {overall['mrr_at_5']:.3f}")
    print(f"MRR@{report['candidate_limit']}: {overall['mrr_at_candidate_limit']:.3f}")
    print(
        f"Latency p50 / p95: {overall['p50_latency_ms']:.2f} / {overall['p95_latency_ms']:.2f} ms"
    )
    print(f"Failure cases: {len(report['failure_cases'])}")
    print(f"Report: {output_path}")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("docling", "docling-core", "qdrant-client", "fastembed"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate dense, sparse, or hybrid retrieval with direct-evidence qrels."
    )
    parser.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)
    parser.add_argument("--strategy", choices=("dense", "sparse", "hybrid"), default="dense")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_FROZEN_CHUNKS)
    parser.add_argument("--limit", type=_minimum_candidate_limit, default=20)
    parser.add_argument("--output", type=Path)
    return parser


def _minimum_candidate_limit(value: str) -> int:
    parsed = int(value)
    if parsed < 5:
        raise argparse.ArgumentTypeError("must be at least 5")
    return parsed


def _configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

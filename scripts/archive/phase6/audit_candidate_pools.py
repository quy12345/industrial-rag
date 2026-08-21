"""Audit Phase 5 candidate-pool coverage without implementing a reranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from app.candidate_audit import aggregate_candidate_audit, audit_case, dense_results_to_candidates
from app.config import get_settings
from app.evaluation import (
    EvaluationError,
    chunk_set_metadata,
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
DEFAULT_CHUNKS = Path("artifacts/manual-batched.jsonl")
DEFAULT_OUTPUT = Path("artifacts/metrics/candidate-pool-audit.json")


def main(argv: list[str] | None = None) -> int:
    """Run dense/sparse/RRF pool coverage audit over the frozen development set."""

    _configure_output_encoding()
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    try:
        cases = load_evaluation_cases(args.dataset)
        chunks = load_frozen_chunks(args.chunks)
        validate_cases_against_chunks(cases, chunks)
        frozen_metadata = chunk_set_metadata(chunks)
        expected_ids = _chunk_ids_by_document(chunks)

        dense_model = create_embedding_model(
            settings.embedding_model, cache_dir=settings.embedding_cache_dir
        )
        dense_dimension = get_embedding_dimension(dense_model)
        client = create_qdrant_client(settings)
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
        hybrid_manifest = validate_hybrid_index_manifest(
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
        _validate_frozen_ids(client, settings.qdrant_collection, expected_ids)
        _validate_frozen_ids(client, settings.qdrant_hybrid_collection, expected_ids)
        sparse_model = create_sparse_embedding_model(
            settings.sparse_model,
            settings.embedding_cache_dir,
            disable_stemmer=settings.bm25_disable_stemmer,
            k=settings.bm25_k,
            b=settings.bm25_b,
            avg_len=float(hybrid_manifest["bm25_avg_len"]),
        )

        rows = []
        for case in cases:
            dense = dense_results_to_candidates(
                dense_search(
                    client,
                    case.question,
                    collection_name=settings.qdrant_collection,
                    vector_name=settings.dense_vector_name,
                    embedding_model=dense_model,
                    limit=args.limit,
                    document_id=case.document_id,
                )
            )
            sparse = sparse_search(
                client,
                case.question,
                collection_name=settings.qdrant_hybrid_collection,
                sparse_vector_name=settings.sparse_vector_name,
                sparse_embedding_model=sparse_model,
                limit=args.limit,
                document_id=case.document_id,
            )
            hybrid = hybrid_search(
                client,
                case.question,
                collection_name=settings.qdrant_hybrid_collection,
                dense_vector_name=settings.dense_vector_name,
                sparse_vector_name=settings.sparse_vector_name,
                dense_embedding_model=dense_model,
                sparse_embedding_model=sparse_model,
                dense_candidate_limit=args.limit,
                sparse_candidate_limit=args.limit,
                final_limit=args.limit,
                rrf_k=settings.rrf_k,
                document_id=case.document_id,
            )
            rows.append(
                audit_case(
                    case,
                    dense_candidates=dense,
                    sparse_candidates=sparse,
                    hybrid_candidates=hybrid,
                )
            )
        report = {
            "schema_version": 1,
            "timestamp": datetime.now(UTC).isoformat(),
            "document_id": chunks[0].document_id,
            "frozen_chunk_set": frozen_metadata,
            "evaluation_dataset": {"path": str(args.dataset), "sha256": _sha256_file(args.dataset)},
            "collections": {
                "dense_v1": settings.qdrant_collection,
                "hybrid_v2": settings.qdrant_hybrid_collection,
            },
            "dense_model": settings.embedding_model,
            "sparse": {
                "model": settings.sparse_model,
                "bm25_k": settings.bm25_k,
                "bm25_b": settings.bm25_b,
                "bm25_avg_len": hybrid_manifest["bm25_avg_len"],
                "disable_stemmer": settings.bm25_disable_stemmer,
            },
            "candidate_limits": {"dense": args.limit, "sparse": args.limit, "hybrid": args.limit},
            "rrf_k": settings.rrf_k,
            "runtime": {
                "python": platform.python_version(),
                "qdrant_client": _package_version("qdrant-client"),
                "fastembed": _package_version("fastembed"),
            },
            **aggregate_candidate_audit(rows),
        }
        _write_json_atomic(args.output, report)
    except (EvaluationError, RetrievalError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Candidate-pool audit completed: {args.output}")
    for name, metrics in report["aggregate_metrics"]["pools"].items():
        print(f"{name}: candidate recall={metrics['candidate_recall']:.3f}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit frozen Phase 5 candidate-pool coverage.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--limit", type=_positive_int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _validate_frozen_ids(client: Any, collection_name: str, expected: dict[str, set[str]]) -> None:
    for document_id, ids in expected.items():
        actual = get_indexed_chunk_ids(
            client, collection_name=collection_name, document_id=document_id
        )
        if actual != ids:
            raise EvaluationError(
                f"Collection {collection_name} does not match frozen chunks for {document_id}."
            )


def _chunk_ids_by_document(chunks: list[Any]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document_id, set()).add(chunk.chunk_id)
    return grouped


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


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


def _configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

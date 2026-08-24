"""Preview, freeze, and safely index the two ATV320 Phase 7 manuals.

This explicit integration command is intentionally separate from the Phase 3--6
collections.  It refuses their collection names before it creates or writes anything.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from app.application.indexing_service import (
    IndexingSafetyError,
    Phase7IndexingService,
    validate_chunk_preview,
    validate_collection_targets,
)
from app.config import get_settings
from app.domain.documents import IngestionError
from app.errors import RetrievalError
from app.infrastructure.corpus_artifacts import (
    PHASE7_CORPUS_VERSION,
    PHASE7_DENSE_COLLECTION,
    PHASE7_HYBRID_COLLECTION,
    PROTECTED_COLLECTIONS,
    chunk_set_metadata,
    file_sha256,
    load_frozen_chunks,
    write_json_atomic,
)
from app.infrastructure.ingestion.jsonl import write_chunks_jsonl
from app.infrastructure.ingestion.pipeline import ingest_document
from app.infrastructure.qdrant.client import create_qdrant_client
from app.infrastructure.qdrant.dense import (
    create_embedding_model,
    get_embedding_dimension,
    get_indexed_chunk_ids,
    index_chunks,
)
from app.infrastructure.qdrant.hybrid import (
    compute_bm25_average_length,
    create_sparse_embedding_model,
    index_hybrid_chunks,
)

DEFAULT_INPUTS = (
    Path("data/raw/ATV320_Installation_manual_EN_NVE41289_09.pdf"),
    Path("data/raw/ATV320_Programming_Manual_EN_NVE41295_06.pdf"),
)


def main() -> int:
    args = _parser().parse_args()
    try:
        validate_collection_targets(
            args.dense_collection,
            args.hybrid_collection,
            protected_collections=PROTECTED_COLLECTIONS,
        )
    except IndexingSafetyError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        if args.preview_only:
            chunks_by_document = {
                chunks[0].document_id: chunks
                for path in args.inputs
                if (
                    chunks := ingest_document(
                        path, batch_size=args.page_batch_size, chunker=args.chunker
                    )
                )
            }
        else:
            frozen_chunks = load_frozen_chunks(args.frozen_chunks)
            chunks_by_document = {}
            for chunk in frozen_chunks:
                chunks_by_document.setdefault(chunk.document_id, []).append(chunk)
        all_chunks = [chunk for chunks in chunks_by_document.values() for chunk in chunks]
        validate_chunk_preview(chunks_by_document)
        if args.preview_only:
            write_chunks_jsonl(args.chunks_output, all_chunks)
            _write_manifest(
                args, chunks_by_document, all_chunks, bm25_avg_len=None, dense_dimension=None
            )
            print(f"Phase 7 ingestion preview PASS: {args.chunks_output}")
            return 0

        settings = get_settings().model_copy(
            update={
                "qdrant_collection": args.dense_collection,
                "qdrant_hybrid_collection": args.hybrid_collection,
            }
        )
        dense_model = create_embedding_model(settings.embedding_model, settings.embedding_cache_dir)
        dense_dimension = get_embedding_dimension(dense_model)
        sparse_probe = create_sparse_embedding_model(
            settings.sparse_model,
            settings.embedding_cache_dir,
            disable_stemmer=settings.bm25_disable_stemmer,
            k=settings.bm25_k,
            b=settings.bm25_b,
            avg_len=256.0,
        )
        bm25_avg_len = compute_bm25_average_length(sparse_probe, all_chunks)
        sparse_model = create_sparse_embedding_model(
            settings.sparse_model,
            settings.embedding_cache_dir,
            disable_stemmer=settings.bm25_disable_stemmer,
            k=settings.bm25_k,
            b=settings.bm25_b,
            avg_len=bm25_avg_len,
        )
        client = create_qdrant_client(settings)
        indexing_service = Phase7IndexingService(
            client=client,
            settings=settings,
            dense_model=dense_model,
            sparse_model=sparse_model,
            dense_dimension=dense_dimension,
            dense_indexer=index_chunks,
            hybrid_indexer=index_hybrid_chunks,
            indexed_chunk_reader=get_indexed_chunk_ids,
        )
        indexing_service.verify_protected_collections(
            dense_collection="industrial_manual_chunks",
            hybrid_collection="industrial_manual_chunks_v2",
            expected_count=99,
        )
        indexing_service.index_once(chunks_by_document)
        indexing_service.verify_index(chunks_by_document)
        if args.verify_reindex:
            indexing_service.index_once(chunks_by_document)
            indexing_service.verify_index(chunks_by_document)
        _write_manifest(
            args,
            chunks_by_document,
            all_chunks,
            bm25_avg_len=bm25_avg_len,
            dense_dimension=dense_dimension,
        )
    except (IngestionError, RetrievalError, OSError, ValueError) as exc:
        print(f"Phase 7 corpus indexing FAILED: {exc}")
        return 1
    print(f"Phase 7 corpus indexing PASS: {args.manifest_output}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--page-batch-size", type=_positive_int, default=16)
    parser.add_argument("--chunker", choices=("hierarchical", "hybrid"), default="hybrid")
    parser.add_argument("--dense-collection", default=PHASE7_DENSE_COLLECTION)
    parser.add_argument("--hybrid-collection", default=PHASE7_HYBRID_COLLECTION)
    parser.add_argument(
        "--chunks-output", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl")
    )
    parser.add_argument(
        "--frozen-chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl")
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("artifacts/metrics/phase-7-corpus-manifest.json"),
    )
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--verify-reindex", action="store_true")
    return parser


def _write_manifest(args, chunks_by_document, all_chunks, *, bm25_avg_len, dense_dimension) -> None:
    document_entries = []
    for path in args.inputs:
        chunks = next(
            chunks for chunks in chunks_by_document.values() if chunks[0].filename == path.name
        )
        document_entries.append(
            {
                "document_id": chunks[0].document_id,
                "filename": path.name,
                "source_sha256": file_sha256(path),
                "chunk_count": len(chunks),
                "page_count": max(page for chunk in chunks for page in chunk.page_numbers),
            }
        )
    payload = {
        "schema_version": 1,
        "corpus_version": PHASE7_CORPUS_VERSION,
        "documents": sorted(document_entries, key=lambda item: item["filename"]),
        "total_chunk_count": len(all_chunks),
        "chunk_set": chunk_set_metadata(all_chunks),
        "ingestion_profile": {
            "ocr_mode": "off",
            "page_batch_size": args.page_batch_size,
            "chunker": args.chunker,
        },
        "dense": {
            "model": get_settings().embedding_model,
            "dimension": dense_dimension,
            "distance": "cosine",
            "collection": args.dense_collection,
        },
        "sparse": {
            "model": get_settings().sparse_model,
            "bm25_k": get_settings().bm25_k,
            "bm25_b": get_settings().bm25_b,
            "disable_stemmer": get_settings().bm25_disable_stemmer,
            "avg_len": bm25_avg_len,
            "collection": args.hybrid_collection,
        },
        "runtime_versions": {
            name: _version(name)
            for name in ("docling", "docling-core", "qdrant-client", "fastembed")
        },
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
    }
    write_json_atomic(args.manifest_output, payload)


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())

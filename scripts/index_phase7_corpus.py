"""Preview, freeze, and safely index the two ATV320 Phase 7 manuals.

This explicit integration command is intentionally separate from the Phase 3--6
collections.  It refuses their collection names before it creates or writes anything.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from app.config import get_settings
from app.evaluation import chunk_set_metadata
from app.hybrid_retrieval import compute_bm25_average_length, create_sparse_embedding_model, index_hybrid_chunks
from app.ingestion import IngestionError, ingest_document, write_chunks_jsonl
from app.phase7 import (
    PHASE7_CORPUS_VERSION,
    PHASE7_DENSE_COLLECTION,
    PHASE7_HYBRID_COLLECTION,
    PROTECTED_COLLECTIONS,
    Phase7Error,
    file_sha256,
    write_json_atomic,
)
from app.retrieval import (
    RetrievalError,
    create_embedding_model,
    create_qdrant_client,
    get_embedding_dimension,
    get_indexed_chunk_ids,
    index_chunks,
)

DEFAULT_INPUTS = (
    Path("data/raw/ATV320_Installation_manual_EN_NVE41289_09.pdf"),
    Path("data/raw/ATV320_Programming_Manual_EN_NVE41295_06.pdf"),
)


def main() -> int:
    args = _parser().parse_args()
    if args.dense_collection in PROTECTED_COLLECTIONS or args.hybrid_collection in PROTECTED_COLLECTIONS:
        raise SystemExit("Phase 7 refuses protected Phase 3--6 collection names.")
    if args.dense_collection == args.hybrid_collection:
        raise SystemExit("Phase 7 dense and hybrid collection names must differ.")
    try:
        chunks_by_document = {
            chunks[0].document_id: chunks
            for path in args.inputs
            if (chunks := ingest_document(path, batch_size=args.page_batch_size))
        }
        all_chunks = [chunk for chunks in chunks_by_document.values() for chunk in chunks]
        _validate_preview(chunks_by_document, args.page_batch_size)
        write_chunks_jsonl(args.chunks_output, all_chunks)
        if args.preview_only:
            _write_manifest(args, chunks_by_document, all_chunks, bm25_avg_len=None, dense_dimension=None)
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
            settings.sparse_model, settings.embedding_cache_dir,
            disable_stemmer=settings.bm25_disable_stemmer, k=settings.bm25_k,
            b=settings.bm25_b, avg_len=256.0,
        )
        bm25_avg_len = compute_bm25_average_length(sparse_probe, all_chunks)
        sparse_model = create_sparse_embedding_model(
            settings.sparse_model, settings.embedding_cache_dir,
            disable_stemmer=settings.bm25_disable_stemmer, k=settings.bm25_k,
            b=settings.bm25_b, avg_len=bm25_avg_len,
        )
        client = create_qdrant_client(settings)
        _old_collection_guard(client)
        _index_once(client, settings, chunks_by_document, dense_model, sparse_model, dense_dimension)
        _verify_index(client, settings, chunks_by_document)
        if args.verify_reindex:
            _index_once(client, settings, chunks_by_document, dense_model, sparse_model, dense_dimension)
            _verify_index(client, settings, chunks_by_document)
        _write_manifest(args, chunks_by_document, all_chunks, bm25_avg_len=bm25_avg_len, dense_dimension=dense_dimension)
    except (IngestionError, RetrievalError, Phase7Error, OSError, ValueError) as exc:
        print(f"Phase 7 corpus indexing FAILED: {exc}")
        return 1
    print(f"Phase 7 corpus indexing PASS: {args.manifest_output}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--page-batch-size", type=_positive_int, default=16)
    parser.add_argument("--dense-collection", default=PHASE7_DENSE_COLLECTION)
    parser.add_argument("--hybrid-collection", default=PHASE7_HYBRID_COLLECTION)
    parser.add_argument("--chunks-output", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument("--manifest-output", type=Path, default=Path("artifacts/metrics/phase-7-corpus-manifest.json"))
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--verify-reindex", action="store_true")
    return parser


def _index_once(client, settings, chunks_by_document, dense_model, sparse_model, dense_dimension: int) -> None:
    for chunks in chunks_by_document.values():
        index_chunks(client, chunks, collection_name=settings.qdrant_collection,
            vector_name=settings.dense_vector_name, embedding_model=dense_model,
            embedding_batch_size=settings.embedding_batch_size, vector_size=dense_dimension)
        index_hybrid_chunks(client, chunks, collection_name=settings.qdrant_hybrid_collection,
            dense_vector_name=settings.dense_vector_name, sparse_vector_name=settings.sparse_vector_name,
            dense_embedding_model=dense_model, sparse_embedding_model=sparse_model,
            dense_embedding_batch_size=settings.embedding_batch_size,
            sparse_embedding_batch_size=settings.sparse_embedding_batch_size,
            dense_vector_size=dense_dimension)


def _validate_preview(chunks_by_document, batch_size: int) -> None:
    if len(chunks_by_document) != 2:
        raise Phase7Error("Both ATV320 manuals must produce chunks.")
    for document_id, chunks in chunks_by_document.items():
        indices = [chunk.metadata.get("chunk_index") for chunk in chunks]
        if indices != list(range(len(chunks))):
            raise Phase7Error(f"{document_id} has non-contiguous chunk indices.")
        if not all(chunk.filename and chunk.page_numbers and chunk.text.strip() for chunk in chunks):
            raise Phase7Error(f"{document_id} has incomplete citation metadata.")
        header_like = sum(
            1 for chunk in chunks if len(chunk.text) < 120 and not chunk.headings
        )
        if header_like / len(chunks) > 0.25:
            raise Phase7Error(f"{document_id} has too many likely header/footer-only chunks.")


def _verify_index(client, settings, chunks_by_document) -> None:
    expected_total = sum(len(chunks) for chunks in chunks_by_document.values())
    for collection_name in (settings.qdrant_collection, settings.qdrant_hybrid_collection):
        if client.count(collection_name, exact=True).count != expected_total:
            raise Phase7Error(f"{collection_name} does not contain the expected total point count.")
        for document_id, chunks in chunks_by_document.items():
            actual = get_indexed_chunk_ids(client, collection_name=collection_name, document_id=document_id)
            expected = {chunk.chunk_id for chunk in chunks}
            if actual != expected:
                raise Phase7Error(f"{collection_name} chunk IDs mismatch for {document_id}.")


def _old_collection_guard(client) -> None:
    old_dense = client.count("industrial_manual_chunks", exact=True).count
    old_hybrid = client.count("industrial_manual_chunks_v2", exact=True).count
    if (old_dense, old_hybrid) != (99, 99):
        raise Phase7Error(f"Protected collections are not frozen at 99/99: {old_dense}/{old_hybrid}")


def _write_manifest(args, chunks_by_document, all_chunks, *, bm25_avg_len, dense_dimension) -> None:
    document_entries = []
    for path in args.inputs:
        chunks = next(chunks for chunks in chunks_by_document.values() if chunks[0].filename == path.name)
        document_entries.append({
            "document_id": chunks[0].document_id, "filename": path.name,
            "source_sha256": file_sha256(path), "chunk_count": len(chunks),
            "page_count": max(page for chunk in chunks for page in chunk.page_numbers),
        })
    payload = {
        "schema_version": 1, "corpus_version": PHASE7_CORPUS_VERSION,
        "documents": sorted(document_entries, key=lambda item: item["filename"]),
        "total_chunk_count": len(all_chunks), "chunk_set": chunk_set_metadata(all_chunks),
        "ingestion_profile": {"ocr_mode": "off", "page_batch_size": args.page_batch_size, "chunker": "hierarchical"},
        "dense": {"model": get_settings().embedding_model, "dimension": dense_dimension, "distance": "cosine", "collection": args.dense_collection},
        "sparse": {"model": get_settings().sparse_model, "bm25_k": get_settings().bm25_k, "bm25_b": get_settings().bm25_b, "disable_stemmer": get_settings().bm25_disable_stemmer, "avg_len": bm25_avg_len, "collection": args.hybrid_collection},
        "runtime_versions": {name: _version(name) for name in ("docling", "docling-core", "qdrant-client", "fastembed")},
        "created_at": datetime.now(UTC).isoformat(), "git_commit": _git_commit(),
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
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())

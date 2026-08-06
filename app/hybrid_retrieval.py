"""BM25 sparse retrieval, hybrid Qdrant indexing, and client-side RRF fusion."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from app.config import Settings
from app.models import DocumentChunk, RetrievalCandidate, RetrievedChunk
from app.retrieval import (
    RetrievalError,
    _batched,
    _build_payload,
    _document_filter,
    _scroll_document_point_ids,
    _to_float_vector,
    build_embedding_text,
    build_point_id,
    dense_search,
)

HYBRID_INDEX_MANIFEST_PATH = Path("artifacts/metrics/hybrid-index-manifest.json")
HYBRID_SCHEMA_VERSION = 2
SPARSE_MODIFIER = "idf"


def create_sparse_embedding_model(
    model_name: str,
    cache_dir: str | None,
    *,
    disable_stemmer: bool,
    k: float,
    b: float,
    avg_len: float,
) -> Any:
    """Create the installed FastEmbed sparse BM25 model with explicit settings."""

    from fastembed import SparseTextEmbedding

    supported_names = {
        model["model"]
        for model in SparseTextEmbedding.list_supported_models()
        if isinstance(model.get("model"), str)
    }
    if model_name not in supported_names:
        raise RetrievalError(
            f"Sparse embedding model is not supported by installed FastEmbed: {model_name}"
        )
    try:
        return SparseTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            disable_stemmer=disable_stemmer,
            k=k,
            b=b,
            avg_len=avg_len,
        )
    except Exception as exc:
        raise RetrievalError(f"Failed to initialize sparse model {model_name}: {exc}") from exc


def compute_bm25_average_length(
    sparse_model: Any,
    chunks: Sequence[DocumentChunk],
) -> float:
    """Compute BM25 document length from FastEmbed's active tokenizer/preprocessing.

    FastEmbed 0.8.0's public ``token_count`` counts tokens before punctuation and
    stemming filters. The BM25 denominator uses the post-filter token sequence, so
    this intentionally calls the installed Bm25 implementation's tokenizer and
    stemmer rather than approximating length with whitespace splitting.
    """

    if not chunks:
        raise RetrievalError("Cannot calculate BM25 average length for no chunks.")
    backend = getattr(sparse_model, "model", None)
    tokenizer = getattr(backend, "tokenizer", None)
    stem = getattr(backend, "_stem", None)
    if tokenizer is None or not callable(stem):
        raise RetrievalError(
            "Installed sparse model does not expose the FastEmbed BM25 tokenizer needed "
            "to calculate avg_len."
        )
    try:
        from fastembed.sparse.bm25 import remove_non_alphanumeric

        lengths = []
        for chunk in chunks:
            normalized = remove_non_alphanumeric(build_embedding_text(chunk))
            lengths.append(len(stem(tokenizer.tokenize(normalized))))
    except Exception as exc:
        raise RetrievalError(f"Failed to calculate BM25 average length: {exc}") from exc

    average = sum(lengths) / len(lengths)
    if average <= 0:
        raise RetrievalError("BM25 average length must be greater than zero.")
    return average


def ensure_hybrid_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    dense_vector_name: str,
    dense_vector_size: int,
    sparse_vector_name: str,
) -> None:
    """Create or validate a v2 dense+sparse collection without touching v1."""

    if dense_vector_size <= 0:
        raise RetrievalError("Dense vector size must be greater than 0.")
    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    dense_vector_name: models.VectorParams(
                        size=dense_vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    sparse_vector_name: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        else:
            validate_hybrid_collection(
                client,
                collection_name=collection_name,
                dense_vector_name=dense_vector_name,
                dense_vector_size=dense_vector_size,
                sparse_vector_name=sparse_vector_name,
            )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(
            f"Failed to create or inspect hybrid collection {collection_name}: {exc}"
        ) from exc


def validate_hybrid_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    dense_vector_name: str,
    dense_vector_size: int,
    sparse_vector_name: str,
) -> None:
    """Validate the hybrid collection schema without creating or replacing it."""

    try:
        if not client.collection_exists(collection_name):
            raise RetrievalError(f"Hybrid Qdrant collection {collection_name} does not exist.")
        collection = client.get_collection(collection_name)
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(
            f"Failed to inspect hybrid collection {collection_name}: {exc}"
        ) from exc

    vectors = collection.config.params.vectors
    if not isinstance(vectors, dict) or dense_vector_name not in vectors:
        raise RetrievalError(
            f"Collection {collection_name} does not define dense vector {dense_vector_name}."
        )
    dense_config = vectors[dense_vector_name]
    if dense_config.size != dense_vector_size:
        raise RetrievalError(
            f"Collection {collection_name} uses dense vector size {dense_config.size}, "
            f"but model produces {dense_vector_size}."
        )
    if dense_config.distance != models.Distance.COSINE:
        raise RetrievalError(f"Collection {collection_name} dense vector must use cosine distance.")

    sparse_vectors = collection.config.params.sparse_vectors or {}
    if sparse_vector_name not in sparse_vectors:
        raise RetrievalError(
            f"Collection {collection_name} does not define sparse vector {sparse_vector_name}."
        )
    if sparse_vectors[sparse_vector_name].modifier != models.Modifier.IDF:
        raise RetrievalError(
            f"Collection {collection_name} sparse vector {sparse_vector_name} must use IDF."
        )

    payload_schema = getattr(collection, "payload_schema", None)
    if payload_schema:
        document_index = payload_schema.get("document_id")
        if document_index is None:
            raise RetrievalError(f"Collection {collection_name} has no document_id payload index.")
        data_type = getattr(document_index, "data_type", None)
        if data_type not in {None, models.PayloadSchemaType.KEYWORD}:
            raise RetrievalError(
                f"Collection {collection_name} document_id payload index must be keyword."
            )


def index_hybrid_chunks(
    client: QdrantClient,
    chunks: Sequence[DocumentChunk],
    *,
    collection_name: str,
    dense_vector_name: str,
    sparse_vector_name: str,
    dense_embedding_model: Any,
    sparse_embedding_model: Any,
    dense_embedding_batch_size: int,
    sparse_embedding_batch_size: int,
    dense_vector_size: int,
) -> int:
    """Atomically replace one document's v2 vectors after both embeddings succeed."""

    if not chunks:
        raise RetrievalError("Cannot index an empty chunk list.")
    if dense_embedding_batch_size <= 0 or sparse_embedding_batch_size <= 0:
        raise RetrievalError("Embedding batch sizes must be greater than 0.")

    # Generate all representations before collection mutation or stale-point deletion.
    dense_vectors = _embed_dense_chunks(
        chunks, dense_embedding_model, dense_embedding_batch_size, dense_vector_size
    )
    sparse_vectors = _embed_sparse_chunks(
        chunks, sparse_embedding_model, sparse_embedding_batch_size
    )
    point_batches: list[list[models.PointStruct]] = []
    for chunk_batch in _batched(chunks, dense_embedding_batch_size):
        points: list[models.PointStruct] = []
        for chunk in chunk_batch:
            points.append(
                models.PointStruct(
                    id=build_point_id(chunk.chunk_id),
                    vector={
                        dense_vector_name: dense_vectors[chunk.chunk_id],
                        sparse_vector_name: sparse_vectors[chunk.chunk_id],
                    },
                    payload=_build_payload(chunk),
                )
            )
        point_batches.append(points)

    ensure_hybrid_collection(
        client,
        collection_name=collection_name,
        dense_vector_name=dense_vector_name,
        dense_vector_size=dense_vector_size,
        sparse_vector_name=sparse_vector_name,
    )
    document_ids = {chunk.document_id for chunk in chunks}
    existing_by_document = {
        document_id: _scroll_document_point_ids(client, collection_name, document_id)
        for document_id in document_ids
    }
    new_by_document: dict[str, set[str]] = {document_id: set() for document_id in document_ids}
    for chunk in chunks:
        new_by_document[chunk.document_id].add(build_point_id(chunk.chunk_id))

    try:
        for points in point_batches:
            client.upsert(collection_name=collection_name, points=points, wait=True)
        for document_id in document_ids:
            stale = existing_by_document[document_id] - new_by_document[document_id]
            if stale:
                client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=sorted(stale)),
                    wait=True,
                )
    except Exception as exc:
        raise RetrievalError(
            f"Failed to update hybrid Qdrant collection {collection_name}: {exc}"
        ) from exc
    return len(chunks)


def sparse_search(
    client: QdrantClient,
    question: str,
    *,
    collection_name: str,
    sparse_vector_name: str,
    sparse_embedding_model: Any,
    limit: int,
    document_id: str | None = None,
) -> list[RetrievalCandidate]:
    """Return sparse BM25 candidates with one-based deterministic ranks."""

    normalized_question = question.strip()
    if not normalized_question:
        raise RetrievalError("Sparse search question must not be empty.")
    if limit <= 0:
        raise RetrievalError("Sparse search limit must be greater than 0.")
    try:
        raw_vector = next(iter(sparse_embedding_model.query_embed(normalized_question)))
        query_vector = _to_sparse_vector(raw_vector)
    except StopIteration as exc:
        raise RetrievalError("Sparse embedding model returned no query vector.") from exc
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(f"Failed to embed sparse search question: {exc}") from exc

    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=sparse_vector_name,
            query_filter=_document_filter(document_id) if document_id else None,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        raise RetrievalError(
            f"Sparse search failed in collection {collection_name}: {exc}"
        ) from exc

    candidates = []
    for point in response.points:
        try:
            candidates.append(
                _candidate_from_payload(point.payload, sparse_score=float(point.score))
            )
        except (KeyError, TypeError, ValidationError) as exc:
            raise RetrievalError(f"Invalid payload for Qdrant point {point.id}: {exc}") from exc
    return _assign_component_ranks(candidates, component="sparse")


def hybrid_search(
    client: QdrantClient,
    question: str,
    *,
    collection_name: str,
    dense_vector_name: str,
    sparse_vector_name: str,
    dense_embedding_model: Any,
    sparse_embedding_model: Any,
    dense_candidate_limit: int,
    sparse_candidate_limit: int,
    final_limit: int,
    rrf_k: int,
    document_id: str | None = None,
) -> list[RetrievalCandidate]:
    """Retrieve dense and sparse candidates then apply deterministic client-side RRF."""

    if final_limit <= 0:
        raise RetrievalError("Hybrid final limit must be greater than 0.")
    dense_results = dense_search(
        client,
        question,
        collection_name=collection_name,
        vector_name=dense_vector_name,
        embedding_model=dense_embedding_model,
        limit=dense_candidate_limit,
        document_id=document_id,
    )
    dense_candidates = _assign_component_ranks(
        [_candidate_from_dense(result) for result in dense_results], component="dense"
    )
    sparse_candidates = sparse_search(
        client,
        question,
        collection_name=collection_name,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding_model=sparse_embedding_model,
        limit=sparse_candidate_limit,
        document_id=document_id,
    )
    return fuse_rrf(dense_candidates, sparse_candidates, rrf_k=rrf_k, final_limit=final_limit)


def fuse_rrf(
    dense_candidates: Sequence[RetrievalCandidate],
    sparse_candidates: Sequence[RetrievalCandidate],
    *,
    rrf_k: int,
    final_limit: int,
) -> list[RetrievalCandidate]:
    """Fuse component lists by one-based reciprocal rank without combining raw scores."""

    if rrf_k <= 0:
        raise RetrievalError("RRF k must be greater than 0.")
    if final_limit <= 0:
        raise RetrievalError("Hybrid final limit must be greater than 0.")

    merged: dict[str, RetrievalCandidate] = {}
    for candidate in dense_candidates:
        if candidate.dense_rank is None:
            raise RetrievalError("Dense RRF candidate has no one-based dense rank.")
        merged[candidate.chunk_id] = candidate.model_copy(
            update={"rrf_score": 1 / (rrf_k + candidate.dense_rank)}
        )
    for candidate in sparse_candidates:
        if candidate.sparse_rank is None:
            raise RetrievalError("Sparse RRF candidate has no one-based sparse rank.")
        contribution = 1 / (rrf_k + candidate.sparse_rank)
        existing = merged.get(candidate.chunk_id)
        if existing is None:
            merged[candidate.chunk_id] = candidate.model_copy(update={"rrf_score": contribution})
        else:
            merged[candidate.chunk_id] = existing.model_copy(
                update={
                    "sparse_score": candidate.sparse_score,
                    "sparse_rank": candidate.sparse_rank,
                    "rrf_score": (existing.rrf_score or 0.0) + contribution,
                }
            )

    ordered = sorted(
        merged.values(),
        key=lambda candidate: (
            -(candidate.rrf_score or 0.0),
            min(rank for rank in (candidate.dense_rank, candidate.sparse_rank) if rank is not None),
            candidate.chunk_id,
        ),
    )
    return [
        candidate.model_copy(update={"score": candidate.rrf_score, "rrf_rank": rank})
        for rank, candidate in enumerate(ordered[:final_limit], start=1)
    ]


def write_hybrid_index_manifest(
    path: Path,
    *,
    settings: Settings,
    dense_dimension: int,
    bm25_avg_len: float,
    frozen_chunk_set: dict[str, Any],
    ingestion_profile: dict[str, Any],
) -> None:
    """Atomically write the independent runtime contract for collection v2."""

    payload = {
        "schema_version": HYBRID_SCHEMA_VERSION,
        "collection": settings.qdrant_hybrid_collection,
        "dense_vector_name": settings.dense_vector_name,
        "dense_model": settings.embedding_model,
        "dense_dimension": dense_dimension,
        "dense_distance": "cosine",
        "sparse_vector_name": settings.sparse_vector_name,
        "sparse_model": settings.sparse_model,
        "sparse_modifier": SPARSE_MODIFIER,
        "bm25_k": settings.bm25_k,
        "bm25_b": settings.bm25_b,
        "bm25_avg_len": bm25_avg_len,
        "disable_stemmer": settings.bm25_disable_stemmer,
        "normalization_profile": (
            "FastEmbed 0.8.0 Bm25: remove_non_alphanumeric + SimpleTokenizer + disabled stemmer"
        ),
        "frozen_chunk_set": frozen_chunk_set,
        "ingestion_profile": ingestion_profile,
        "dense_candidate_limit": settings.dense_candidate_limit,
        "sparse_candidate_limit": settings.sparse_candidate_limit,
        "rrf_k": settings.rrf_k,
        "hybrid_final_limit": settings.hybrid_final_limit,
        "runtime_versions": _runtime_versions(),
    }
    _write_json_atomic(path, payload)


def validate_hybrid_index_manifest(
    path: Path,
    *,
    settings: Settings,
    dense_dimension: int,
    frozen_chunk_set: dict[str, Any],
) -> dict[str, Any]:
    """Reject a missing or incompatible hybrid manifest before search/evaluation."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RetrievalError(f"Hybrid index manifest is missing at {path}; re-index v2.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalError(f"Hybrid index manifest is invalid at {path}: {exc}") from exc

    expected = {
        "schema_version": HYBRID_SCHEMA_VERSION,
        "collection": settings.qdrant_hybrid_collection,
        "dense_vector_name": settings.dense_vector_name,
        "dense_model": settings.embedding_model,
        "dense_dimension": dense_dimension,
        "dense_distance": "cosine",
        "sparse_vector_name": settings.sparse_vector_name,
        "sparse_model": settings.sparse_model,
        "sparse_modifier": SPARSE_MODIFIER,
        "bm25_k": settings.bm25_k,
        "bm25_b": settings.bm25_b,
        "disable_stemmer": settings.bm25_disable_stemmer,
        "dense_candidate_limit": settings.dense_candidate_limit,
        "sparse_candidate_limit": settings.sparse_candidate_limit,
        "rrf_k": settings.rrf_k,
        "hybrid_final_limit": settings.hybrid_final_limit,
        "frozen_chunk_set": frozen_chunk_set,
    }
    if settings.bm25_avg_len is not None:
        expected["bm25_avg_len"] = settings.bm25_avg_len
    mismatches = [
        f"{key}={payload.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    if not isinstance(payload.get("bm25_avg_len"), (int, float)) or payload["bm25_avg_len"] <= 0:
        mismatches.append("bm25_avg_len must be a positive number")
    if mismatches:
        raise RetrievalError(
            "Hybrid index manifest does not match the current configuration; re-index v2. "
            f"Mismatches: {'; '.join(mismatches)}"
        )
    return payload


def _embed_dense_chunks(
    chunks: Sequence[DocumentChunk], model: Any, batch_size: int, vector_size: int
) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    try:
        for batch in _batched(chunks, batch_size):
            raw_vectors = list(
                model.passage_embed([build_embedding_text(chunk) for chunk in batch])
            )
            if len(raw_vectors) != len(batch):
                raise RetrievalError(
                    "Dense model returned a different number of vectors than chunks."
                )
            for chunk, raw_vector in zip(batch, raw_vectors, strict=True):
                vector = _to_float_vector(raw_vector)
                if len(vector) != vector_size:
                    raise RetrievalError(
                        f"Chunk {chunk.chunk_id} produced dense vector size {len(vector)}, "
                        f"expected {vector_size}."
                    )
                vectors[chunk.chunk_id] = vector
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(f"Failed to embed dense hybrid passages: {exc}") from exc
    return vectors


def _embed_sparse_chunks(
    chunks: Sequence[DocumentChunk], model: Any, batch_size: int
) -> dict[str, models.SparseVector]:
    vectors: dict[str, models.SparseVector] = {}
    try:
        for batch in _batched(chunks, batch_size):
            raw_vectors = list(
                model.passage_embed([build_embedding_text(chunk) for chunk in batch])
            )
            if len(raw_vectors) != len(batch):
                raise RetrievalError(
                    "Sparse model returned a different number of vectors than chunks."
                )
            for chunk, raw_vector in zip(batch, raw_vectors, strict=True):
                vectors[chunk.chunk_id] = _to_sparse_vector(raw_vector)
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(f"Failed to embed sparse hybrid passages: {exc}") from exc
    return vectors


def _to_sparse_vector(raw_vector: Any) -> models.SparseVector:
    indices = [int(value) for value in getattr(raw_vector, "indices", [])]
    values = [float(value) for value in getattr(raw_vector, "values", [])]
    if not indices or not values:
        raise RetrievalError("Sparse embedding model produced an empty sparse vector.")
    if len(indices) != len(values):
        raise RetrievalError("Sparse embedding indices and values have different lengths.")
    return models.SparseVector(indices=indices, values=values)


def _candidate_from_dense(result: RetrievedChunk) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        filename=result.filename,
        text=result.text,
        page_numbers=result.page_numbers,
        headings=result.headings,
        content_type=result.content_type,
        score=result.score,
        dense_score=result.score,
    )


def _candidate_from_payload(payload: Any, *, sparse_score: float) -> RetrievalCandidate:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a JSON object")
    metadata = {key: payload[key] for key in ("source_path", "character_count") if key in payload}
    return RetrievalCandidate(
        chunk_id=payload["chunk_id"],
        document_id=payload["document_id"],
        filename=payload["filename"],
        text=payload["text"],
        page_numbers=payload["page_numbers"],
        headings=payload["headings"],
        content_type=payload["content_type"],
        metadata=metadata,
        score=sparse_score,
        sparse_score=sparse_score,
    )


def _assign_component_ranks(
    candidates: Sequence[RetrievalCandidate], *, component: str
) -> list[RetrievalCandidate]:
    if component not in {"dense", "sparse"}:
        raise ValueError(f"Unknown retrieval component: {component}")
    score_field = f"{component}_score"
    rank_field = f"{component}_rank"
    ordered = sorted(
        candidates,
        key=lambda candidate: (-(getattr(candidate, score_field) or 0.0), candidate.chunk_id),
    )
    return [
        candidate.model_copy(update={rank_field: rank})
        for rank, candidate in enumerate(ordered, start=1)
    ]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        temporary_path.replace(manifest_path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RetrievalError(
            f"Failed to write hybrid index manifest {manifest_path}: {exc}"
        ) from exc


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("fastembed", "qdrant-client"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions

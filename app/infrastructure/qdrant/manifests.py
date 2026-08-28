"""Atomic dense-index manifest persistence and validation."""

from __future__ import annotations

import json
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from app.config import Settings
from app.errors import RetrievalError

INDEX_MANIFEST_PATH = Path("artifacts/metrics/dense-index-manifest.json")
HYBRID_INDEX_MANIFEST_PATH = Path("artifacts/metrics/hybrid-index-manifest.json")
HYBRID_SCHEMA_VERSION = 2
SPARSE_MODIFIER = "idf"


def write_index_manifest(
    path: Path,
    *,
    collection_name: str,
    vector_name: str,
    embedding_model: str,
    embedding_dimension: int,
    ingestion_profile: dict[str, Any] | None = None,
) -> None:
    """Atomically write the runtime contract for a dense index."""

    payload = {
        "collection_name": collection_name,
        "vector_name": vector_name,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "distance": "cosine",
        "ingestion_profile": ingestion_profile or {},
    }
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
            f"Failed to write dense index manifest {manifest_path}: {exc}"
        ) from exc


def validate_index_manifest(
    path: Path,
    *,
    collection_name: str,
    vector_name: str,
    embedding_model: str,
    embedding_dimension: int,
) -> None:
    """Reject a missing or incompatible dense-index manifest before search."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RetrievalError(
            f"Dense index manifest is missing at {manifest_path}; re-index the collection."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalError(f"Dense index manifest is invalid at {manifest_path}: {exc}") from exc

    expected = {
        "collection_name": collection_name,
        "vector_name": vector_name,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "distance": "cosine",
    }
    mismatches = [
        f"{key}={payload.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    if mismatches:
        raise RetrievalError(
            "Dense index manifest does not match the current model/config; "
            f"re-index the collection. Mismatches: {'; '.join(mismatches)}"
        )


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
        "runtime_versions": runtime_versions(),
    }
    _write_json_atomic(path, payload, manifest_kind="hybrid index")


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


def runtime_versions() -> dict[str, str | None]:
    """Return bounded package-version metadata for hybrid manifests."""

    versions: dict[str, str | None] = {}
    for package in ("fastembed", "qdrant-client"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _write_json_atomic(path: Path, payload: dict[str, Any], *, manifest_kind: str) -> None:
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
            f"Failed to write {manifest_kind} manifest {manifest_path}: {exc}"
        ) from exc

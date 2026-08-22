"""Atomic dense-index manifest persistence and validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from app.errors import RetrievalError

INDEX_MANIFEST_PATH = Path("artifacts/metrics/dense-index-manifest.json")


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

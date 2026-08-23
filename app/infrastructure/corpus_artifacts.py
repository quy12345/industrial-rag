"""Active Phase 7 corpus identity and local artifact file operations."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.domain.documents import DocumentChunk
from app.domain.retrieval_contracts import PHASE7_RETRIEVAL_CONTRACT

PHASE7_DENSE_COLLECTION = PHASE7_RETRIEVAL_CONTRACT.dense_collection
PHASE7_HYBRID_COLLECTION = PHASE7_RETRIEVAL_CONTRACT.hybrid_collection
PROTECTED_COLLECTIONS = {"industrial_manual_chunks", "industrial_manual_chunks_v2"}
PHASE7_CORPUS_VERSION = "atv320-2025-04-v1"


class CorpusArtifactError(ValueError):
    """Raised when a local frozen-corpus artifact is invalid."""


def load_frozen_chunks(path: Path) -> list[DocumentChunk]:
    """Load a strict UTF-8 JSONL chunk artifact while preserving record order."""

    chunks: list[DocumentChunk] = []
    seen_ids: set[str] = set()
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CorpusArtifactError(f"Unable to read frozen chunk set {path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise CorpusArtifactError(f"Blank frozen chunk on line {line_number}.")
        try:
            chunk = DocumentChunk.model_validate_json(line)
        except ValidationError as exc:
            raise CorpusArtifactError(
                f"Invalid frozen chunk on line {line_number}: {exc}"
            ) from exc
        if chunk.chunk_id in seen_ids:
            raise CorpusArtifactError(
                f"Duplicate chunk ID on line {line_number}: {chunk.chunk_id}"
            )
        seen_ids.add(chunk.chunk_id)
        chunks.append(chunk)

    if not chunks:
        raise CorpusArtifactError(f"Frozen chunk set is empty: {path}")
    return chunks


def chunk_set_metadata(chunks: Iterable[DocumentChunk]) -> dict[str, Any]:
    """Return stable corpus identity without hashing manual content."""

    chunk_list = list(chunks)
    chunk_ids = sorted(chunk.chunk_id for chunk in chunk_list)
    document_ids = sorted({chunk.document_id for chunk in chunk_list})
    return {
        "chunk_count": len(chunk_list),
        "document_ids": document_ids,
        "chunk_ids_sha256": hashlib.sha256("\n".join(chunk_ids).encode("utf-8")).hexdigest(),
    }


def file_sha256(path: Path) -> str:
    """Hash a source file without loading the complete manual into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON atomically for manifests and audit records."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_jsonl_atomic(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Atomically replace a UTF-8 JSONL dataset while preserving record order."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

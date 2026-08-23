"""Active Phase 7 corpus identity and local artifact file operations."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.domain.retrieval_contracts import PHASE7_RETRIEVAL_CONTRACT

PHASE7_DENSE_COLLECTION = PHASE7_RETRIEVAL_CONTRACT.dense_collection
PHASE7_HYBRID_COLLECTION = PHASE7_RETRIEVAL_CONTRACT.hybrid_collection
PROTECTED_COLLECTIONS = {"industrial_manual_chunks", "industrial_manual_chunks_v2"}
PHASE7_CORPUS_VERSION = "atv320-2025-04-v1"


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

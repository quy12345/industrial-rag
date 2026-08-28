"""Atomic JSONL output for normalized document chunks."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path

from app.domain.documents import DocumentChunk


def write_chunks_jsonl(output_path: Path, chunks: Iterable[DocumentChunk]) -> None:
    """Atomically replace an output path with UTF-8 JSON objects."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            for chunk in chunks:
                output_file.write(chunk.model_dump_json())
                output_file.write("\n")
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

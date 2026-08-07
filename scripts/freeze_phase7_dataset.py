"""Freeze an approved Phase 7 dataset and write its immutable evaluation manifest."""

from __future__ import annotations

import argparse
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation import chunk_set_metadata, load_frozen_chunks
from app.phase7 import (
    PHASE7_DENSE_COLLECTION,
    PHASE7_HYBRID_COLLECTION,
    Phase7Error,
    dataset_sha256,
    read_phase7_dataset,
    validate_phase7_datasets,
    write_json_atomic,
    write_jsonl_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration", type=Path, default=Path("data/eval/phase7/calibration.jsonl")
    )
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/metrics/phase-7-evaluation-manifest.json")
    )
    args = parser.parse_args()
    try:
        calibration = read_phase7_dataset(args.calibration)
        test = read_phase7_dataset(args.test)
        chunks = load_frozen_chunks(args.chunks)
        validate_phase7_datasets(calibration, test, chunks)
    except Phase7Error as exc:
        parser.error(str(exc))
    approved_calibration = [
        item.model_dump() | {"review_status": "approved"} for item in calibration
    ]
    approved_test = [item.model_dump() | {"review_status": "approved"} for item in test]
    write_jsonl_atomic(args.calibration, approved_calibration)
    write_jsonl_atomic(args.test, approved_test)
    approved_calibration_models = read_phase7_dataset(args.calibration)
    approved_test_models = read_phase7_dataset(args.test)
    if any(
        item.review_status != "approved"
        for item in [*approved_calibration_models, *approved_test_models]
    ):
        parser.error("Dataset approval update did not persist.")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "corpus": chunk_set_metadata(chunks),
        "calibration_dataset_sha256": dataset_sha256(approved_calibration_models),
        "test_dataset_sha256": dataset_sha256(approved_test_models),
        "collections": {"dense": PHASE7_DENSE_COLLECTION, "hybrid": PHASE7_HYBRID_COLLECTION},
        "runtime_configuration": {
            "strategy": "union",
            "dense_candidate_limit": 20,
            "sparse_candidate_limit": 20,
            "reranker": "jinaai/jina-reranker-v2-base-multilingual",
            "final_top_k": 5,
        },
    }
    write_json_atomic(args.output, manifest)
    print(f"Phase 7 dataset frozen: {args.output}")
    return 0


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())

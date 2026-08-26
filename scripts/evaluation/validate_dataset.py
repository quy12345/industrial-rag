"""Validate active calibration and held-out annotations against frozen chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.infrastructure.corpus_artifacts import load_frozen_chunks, write_json_atomic
from evaluation.dataset import (
    DatasetValidationError,
    read_dataset,
    validate_dataset_splits,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration", type=Path, default=Path("data/eval/phase7/calibration.jsonl")
    )
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/metrics/phase-7-dataset-validation.json")
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        result = validate_dataset_splits(
            read_dataset(args.calibration),
            read_dataset(args.test),
            load_frozen_chunks(args.chunks),
        )
    except DatasetValidationError as exc:
        parser.error(str(exc))
    write_json_atomic(args.output, result)
    print(f"ATV320 dataset validation PASS: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

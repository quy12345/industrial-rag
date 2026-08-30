"""Validate legacy or optimization evaluation splits against frozen chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.infrastructure.corpus_artifacts import load_frozen_chunks, write_json_atomic
from evaluation.dataset import (
    DatasetValidationError,
    read_dataset,
    validate_dataset_splits,
    validate_development_dataset_splits,
)

LEGACY_OUTPUT = Path("artifacts/metrics/phase-7-dataset-validation.json")
OPTIMIZATION_OUTPUT = Path(
    "artifacts/metrics/atv320-optimization-dataset-validation.json"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("legacy", "optimization"),
        default="legacy",
        help="Choose the pair of dataset contracts to validate.",
    )
    parser.add_argument(
        "--calibration", type=Path, default=Path("data/eval/phase7/calibration.jsonl")
    )
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("data/eval/atv320/development-v1.jsonl"),
    )
    parser.add_argument(
        "--confirmation",
        type=Path,
        default=Path(
            "data/eval/atv320/private-confirmation-v1/confirmation-v1.jsonl"
        ),
    )
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.suite == "optimization":
            result = validate_development_dataset_splits(
                read_dataset(args.development),
                read_dataset(args.confirmation),
                load_frozen_chunks(args.chunks),
            )
        else:
            result = validate_dataset_splits(
                read_dataset(args.calibration),
                read_dataset(args.test),
                load_frozen_chunks(args.chunks),
            )
    except DatasetValidationError as exc:
        parser.error(str(exc))
    output = _output_path(args)
    write_json_atomic(output, result)
    print(f"ATV320 dataset validation PASS: {output}")
    return 0


def _output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    return OPTIMIZATION_OUTPUT if args.suite == "optimization" else LEGACY_OUTPUT


if __name__ == "__main__":
    raise SystemExit(main())

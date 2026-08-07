"""Validate Phase 7 calibration and held-out annotations against frozen chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.evaluation import load_frozen_chunks
from app.phase7 import (
    Phase7Error,
    read_phase7_dataset,
    validate_phase7_datasets,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=Path("data/eval/phase7/calibration.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics/phase-7-dataset-validation.json"))
    args = parser.parse_args()
    try:
        result = validate_phase7_datasets(
            read_phase7_dataset(args.calibration), read_phase7_dataset(args.test), load_frozen_chunks(args.chunks)
        )
    except Phase7Error as exc:
        parser.error(str(exc))
    write_json_atomic(args.output, result)
    print(f"Phase 7 dataset validation PASS: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

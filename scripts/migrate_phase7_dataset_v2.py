"""Create a review-required Phase 7 dataset v2 without provider/model calls.

The migration separates answer facts from evidence phrases and expands qrels only
for same-document chunks whose normalized raw text is exactly identical.  It never
uses broad phrase matches as qrels and never approves the migrated dataset.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.evaluation import load_frozen_chunks
from app.phase7 import (
    Phase7DatasetItem,
    Phase7Error,
    build_exact_content_equivalence,
    dataset_sha256,
    expand_exact_equivalent_qrels,
    file_sha256,
    validate_phase7_datasets,
    write_json_atomic,
    write_jsonl_atomic,
)

MIGRATION_NOTE = (
    "Phase 7 dataset v2 draft: exact-normalized duplicate qrel closure applied; "
    "expected_answer_facts require human review."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration", type=Path, default=Path("data/eval/phase7/calibration.jsonl")
    )
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument(
        "--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/phase-7-dataset-v2-migration.json"),
    )
    args = parser.parse_args()

    chunks = load_frozen_chunks(args.chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    equivalence = build_exact_content_equivalence(chunks)
    old_file_hashes = {
        "calibration": file_sha256(args.calibration),
        "test": file_sha256(args.test),
    }
    calibration, calibration_added = _migrate_file(
        args.calibration, chunks_by_id=chunks_by_id, equivalence=equivalence
    )
    test, test_added = _migrate_file(
        args.test, chunks_by_id=chunks_by_id, equivalence=equivalence
    )
    validation = validate_phase7_datasets(calibration, test, chunks)

    write_jsonl_atomic(
        args.calibration, [item.model_dump(mode="json") for item in calibration]
    )
    write_jsonl_atomic(args.test, [item.model_dump(mode="json") for item in test])
    report = {
        "schema_version": 2,
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "needs_human_review",
        "old_file_sha256": old_file_hashes,
        "new_dataset_sha256": {
            "calibration": dataset_sha256(calibration),
            "test": dataset_sha256(test),
        },
        "qrels_added": {"calibration": calibration_added, "test": test_added},
        "exact_equivalence_groups": len(
            {members for members in equivalence.values() if len(members) > 1}
        ),
        "validation": validation,
        "safety": {
            "phrase_matches_added_as_qrels": False,
            "provider_calls": False,
            "dataset_approved": False,
        },
    }
    write_json_atomic(args.output, report)
    print(f"Phase 7 dataset v2 draft written; human review required: {args.output}")
    return 0


def _migrate_file(
    path: Path,
    *,
    chunks_by_id: dict[str, Any],
    equivalence: dict[str, tuple[str, ...]],
) -> tuple[list[Phase7DatasetItem], int]:
    records = _read_raw_jsonl(path)
    migrated: list[Phase7DatasetItem] = []
    added = 0
    for line_number, record in enumerate(records, start=1):
        record["review_status"] = "needs_human_review"
        record.setdefault("expected_answer_facts", [])
        notes = str(record.get("annotation_notes") or "").strip()
        if MIGRATION_NOTE not in notes:
            record["annotation_notes"] = f"{notes} {MIGRATION_NOTE}".strip()
        try:
            item = Phase7DatasetItem.model_validate(record)
        except ValidationError as exc:
            raise Phase7Error(f"Invalid dataset record on line {line_number}: {exc}") from exc
        expanded = expand_exact_equivalent_qrels(
            item, chunks_by_id=chunks_by_id, equivalence=equivalence
        )
        added += len(expanded.relevant_chunk_ids) - len(item.relevant_chunk_ids)
        migrated.append(expanded)
    return migrated, added


def _read_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise Phase7Error(f"Unable to read Phase 7 dataset {path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Phase7Error(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise Phase7Error(f"Dataset record on line {line_number} must be an object.")
        records.append(value)
    return records


if __name__ == "__main__":
    raise SystemExit(main())

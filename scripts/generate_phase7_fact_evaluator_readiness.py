"""Validate the review-required Phase 7 typed-fact calibration draft.

This command does not activate the draft, mutate any dataset, read provider
output, connect to Qdrant, or execute held-out queries.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evaluation import load_frozen_chunks
from app.phase7 import (
    dataset_sha256,
    read_phase7_dataset,
    validate_phase7_datasets,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/eval/phase7/calibration.jsonl"))
    parser.add_argument(
        "--draft", type=Path, default=Path("data/eval/phase7/calibration-v3-draft.jsonl")
    )
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/phase-7-fact-evaluator-readiness-v1.json"),
    )
    args = parser.parse_args()

    base = read_phase7_dataset(args.base)
    draft = read_phase7_dataset(args.draft)
    held_out = read_phase7_dataset(args.test)
    chunks = load_frozen_chunks(args.chunks)
    base_validation = validate_phase7_datasets(base, held_out, chunks)
    draft_validation = validate_phase7_datasets(draft, held_out, chunks)
    _validate_preserved_ground_truth(base, draft)
    types = Counter(
        fact.type for item in draft if item.answerable for fact in item.expected_answer_facts
    )
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "HUMAN_REVIEW_REQUIRED",
        "activation": "not_active",
        "provider_calls": 0,
        "held_out_queries_executed": 0,
        "base_calibration_sha256": dataset_sha256(base),
        "draft_calibration_sha256": dataset_sha256(draft),
        "held_out_dataset_sha256": dataset_sha256(held_out),
        "corpus": base_validation["corpus"],
        "ground_truth_preserved": True,
        "base_dataset_valid": True,
        "draft_dataset_valid": True,
        "typed_fact_counts": dict(sorted(types.items())),
        "answerable_rows": sum(item.answerable for item in draft),
        "review_status": dict(sorted(Counter(item.review_status for item in draft).items())),
        "sanitization": {
            "question": "excluded",
            "evidence": "excluded",
            "answer": "not applicable",
        },
    }
    if base_validation["corpus"] != draft_validation["corpus"]:
        raise ValueError("Typed-fact draft validated against a different frozen corpus.")
    write_json_atomic(args.output, payload)
    print(f"Phase 7 fact evaluator readiness HUMAN_REVIEW_REQUIRED: {args.output}")
    return 0


def _validate_preserved_ground_truth(base: list[Any], draft: list[Any]) -> None:
    by_id = {item.id: item for item in base}
    if set(by_id) != {item.id for item in draft}:
        raise ValueError("Typed-fact draft IDs differ from the approved calibration dataset.")
    fields = (
        "question",
        "language",
        "answerable",
        "citation_required",
        "relevant_chunk_ids",
        "expected_document_ids",
        "expected_pages",
        "expected_phrases",
        "phrase_match_mode",
        "question_type",
        "scenario",
        "unanswerable_reason",
    )
    for item in draft:
        source = by_id[item.id]
        for field in fields:
            if getattr(source, field) != getattr(item, field):
                raise ValueError(
                    f"Typed-fact draft changed frozen ground-truth field {field} for {item.id}."
                )
        if source.answerable and item.review_status != "needs_human_review":
            raise ValueError(f"Typed-fact draft row {item.id} is not marked for human review.")


if __name__ == "__main__":
    raise SystemExit(main())

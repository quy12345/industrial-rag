"""Freeze the reviewed typed-fact calibration-v3 dataset without touching held-out.

This is deliberately separate from the historic v2 freezer.  It copies the
review-required draft to a new immutable calibration input, marks only that
copy approved after an exact human token, and writes a matching manifest.  The
sealed held-out JSONL is read for validation but is never rewritten.
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evaluation import chunk_set_metadata, load_frozen_chunks
from app.phase7 import (
    PHASE7_DENSE_COLLECTION,
    PHASE7_HYBRID_COLLECTION,
    Phase7DatasetItem,
    Phase7Error,
    dataset_sha256,
    read_phase7_dataset,
    validate_phase7_datasets,
    write_json_atomic,
    write_jsonl_atomic,
)
from app.retrieval_runtime import PHASE7_RETRIEVAL_CONTRACT

APPROVAL_TOKEN = "APPROVE PHASE 7 CALIBRATION V3 AND PROVIDER EGRESS"


def main() -> int:
    args = _parse_args()
    if args.approval_token != APPROVAL_TOKEN:
        args.parser.error(f"Approval token must exactly equal: {APPROVAL_TOKEN}")
    try:
        draft = read_phase7_dataset(args.draft)
        test = read_phase7_dataset(args.test)
        chunks = load_frozen_chunks(args.chunks)
        approved = _approve_draft(draft)
        validate_phase7_datasets(approved, test, chunks)
    except Phase7Error as exc:
        args.parser.error(str(exc))
    _validate_ground_truth_preserved(draft, approved)
    write_jsonl_atomic(args.calibration_output, [item.model_dump(mode="json") for item in approved])
    manifest = _manifest(approved, test, chunks, draft)
    write_json_atomic(args.output, manifest)
    print(
        "Phase 7 calibration-v3 frozen: "
        f"{args.calibration_output}; manifest: {args.output}"
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draft", type=Path, default=Path("data/eval/phase7/calibration-v3-draft.jsonl")
    )
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument(
        "--calibration-output", type=Path, default=Path("data/eval/phase7/calibration-v3.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/metrics/phase-7-evaluation-manifest-v3.json")
    )
    parser.add_argument("--approval-token", required=True)
    args = parser.parse_args()
    args.parser = parser
    return args


def _approve_draft(draft: list[Phase7DatasetItem]) -> list[Phase7DatasetItem]:
    if any(item.review_status != "needs_human_review" for item in draft if item.answerable):
        raise Phase7Error("Calibration-v3 draft must retain needs_human_review before approval.")
    return [
        Phase7DatasetItem.model_validate(
            item.model_dump(mode="json") | {"review_status": "approved"}
        )
        for item in draft
    ]


def _validate_ground_truth_preserved(
    draft: list[Phase7DatasetItem], approved: list[Phase7DatasetItem]
) -> None:
    fields = (
        "id",
        "question",
        "language",
        "answerable",
        "scenario",
        "question_type",
        "expected_document_ids",
        "relevant_chunk_ids",
        "expected_pages",
        "expected_phrases",
        "phrase_match_mode",
        "expected_answer_facts",
        "citation_required",
        "annotation_notes",
        "unanswerable_reason",
    )
    if len(draft) != len(approved):
        raise Phase7Error("Calibration-v3 approval changed record count.")
    for source, frozen in zip(draft, approved, strict=True):
        if any(getattr(source, field) != getattr(frozen, field) for field in fields):
            raise Phase7Error(f"Calibration-v3 approval changed ground truth for {source.id}.")


def _manifest(
    calibration: list[Phase7DatasetItem],
    test: list[Phase7DatasetItem],
    chunks: list[Any],
    draft: list[Phase7DatasetItem],
) -> dict[str, Any]:
    contract = PHASE7_RETRIEVAL_CONTRACT
    return {
        "schema_version": 4,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "approval": {"dataset": "calibration-v3", "token": "verified", "held_out_modified": False},
        "corpus": chunk_set_metadata(chunks),
        "calibration_dataset_sha256": dataset_sha256(calibration),
        "calibration_draft_sha256": dataset_sha256(draft),
        "test_dataset_sha256": dataset_sha256(test),
        "collections": {"dense": PHASE7_DENSE_COLLECTION, "hybrid": PHASE7_HYBRID_COLLECTION},
        "runtime_configuration": {
            "strategy": "union",
            "dense_candidate_limit": contract.dense_candidate_limit,
            "sparse_candidate_limit": contract.sparse_candidate_limit,
            "query_expansion_profile": contract.query_expansion_profile,
            "rrf_k": contract.rrf_k,
            "rrf_prune_limit": contract.union_rrf_prune_limit,
            "reranker": contract.rerank_model,
            "rerank_batch_size": contract.frozen_rerank_batch_size,
            "rerank_threads": contract.frozen_rerank_threads,
            "phase7_fusion_profile": contract.phase7_fusion_profile.name,
            "deduplicate_exact_content": True,
            "final_top_k": 5,
        },
        "answer_scoring": {
            "headline": "phase7_deterministic_typed_facts_v1",
            "strict_phrase": "diagnostic_only",
            "token_coverage": "diagnostic_only",
        },
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())

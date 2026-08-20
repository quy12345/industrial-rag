"""Freeze the private Phase 7 replacement held-out v2 dataset.

The old held-out data is governance-blocked. This command intentionally reads
only a locally ignored v2 draft, the frozen corpus, and an explicit approval
token. It never opens the historic held-out JSONL or calibration annotations.
"""

from __future__ import annotations

import argparse
import os
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
    validate_phase7_dataset,
    write_json_atomic,
    write_jsonl_atomic,
)
from app.retrieval_runtime import PHASE7_RETRIEVAL_CONTRACT

APPROVAL_TOKEN = "APPROVE PHASE 7 HELDOUT V2 DATASET"
PRIVATE_ROOT = Path("data/eval/phase7/private-heldout-v2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=PRIVATE_ROOT / "heldout-v2-draft.jsonl")
    parser.add_argument("--output", type=Path, default=PRIVATE_ROOT / "heldout-v2.jsonl")
    parser.add_argument("--manifest", type=Path, default=PRIVATE_ROOT / "heldout-v2-manifest.json")
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument("--approval-token", required=True)
    args = parser.parse_args()
    if args.approval_token != APPROVAL_TOKEN:
        parser.error(f"Approval token must exactly equal: {APPROVAL_TOKEN}")
    try:
        _require_private_path(args.draft)
        _require_private_path(args.output)
        _require_private_path(args.manifest)
        draft = read_phase7_dataset(args.draft)
        approved = _approve_draft(draft)
        chunks = load_frozen_chunks(args.chunks)
        validation = validate_phase7_dataset(approved, chunks, kind="test")
        _validate_ground_truth_preserved(draft, approved)
    except Phase7Error as exc:
        parser.error(str(exc))

    write_jsonl_atomic(args.output, [item.model_dump(mode="json") for item in approved])
    write_json_atomic(args.manifest, _manifest(approved, chunks, validation, args.draft))
    print(f"Phase 7 private held-out v2 frozen: {args.output}; manifest: {args.manifest}")
    return 0


def _require_private_path(path: Path) -> None:
    try:
        path.resolve().relative_to(PRIVATE_ROOT.resolve())
    except ValueError as exc:
        raise Phase7Error(f"Held-out v2 path must remain under {PRIVATE_ROOT}.") from exc


def _approve_draft(draft: list[Phase7DatasetItem]) -> list[Phase7DatasetItem]:
    if any(item.review_status != "needs_human_review" for item in draft):
        raise Phase7Error("Held-out v2 draft must retain needs_human_review before approval.")
    return [
        Phase7DatasetItem.model_validate(
            item.model_dump(mode="json") | {"review_status": "approved"}
        )
        for item in draft
    ]


def _validate_ground_truth_preserved(
    draft: list[Phase7DatasetItem], approved: list[Phase7DatasetItem]
) -> None:
    if len(draft) != len(approved):
        raise Phase7Error("Held-out v2 approval changed record count.")
    for source, frozen in zip(draft, approved, strict=True):
        source_payload = source.model_dump(mode="json")
        frozen_payload = frozen.model_dump(mode="json")
        source_payload.pop("review_status")
        frozen_payload.pop("review_status")
        if source_payload != frozen_payload:
            raise Phase7Error(f"Held-out v2 approval changed ground truth for {source.id}.")


def _manifest(
    held_out: list[Phase7DatasetItem],
    chunks: list[Any],
    validation: dict[str, Any],
    draft_path: Path,
) -> dict[str, Any]:
    contract = PHASE7_RETRIEVAL_CONTRACT
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "approval": {"dataset": "phase7-heldout-v2", "token": "verified"},
        "privacy": {
            "storage": "gitignored_private_workspace_path",
            "historic_heldout_opened": False,
            "calibration_opened": False,
            "note": "Git ignore prevents repository exposure; it is not an operating-system ACL.",
        },
        "draft_filename": draft_path.name,
        "heldout_dataset_sha256": dataset_sha256(held_out),
        "dataset": validation["dataset"],
        "corpus": chunk_set_metadata(chunks),
        "collections": {"dense": PHASE7_DENSE_COLLECTION, "hybrid": PHASE7_HYBRID_COLLECTION},
        "runtime_configuration": {
            "strategy": "union",
            "dense_candidate_limit": contract.dense_candidate_limit,
            "sparse_candidate_limit": contract.sparse_candidate_limit,
            "rrf_k": contract.rrf_k,
            "reranker": contract.rerank_model,
            "phase7_fusion_profile": contract.phase7_fusion_profile.name,
            "final_top_k": 5,
        },
    }


def _git_commit() -> str | None:
    environment_commit = os.environ.get("PHASE7_GIT_COMMIT", "").strip()
    if environment_commit:
        return environment_commit
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())

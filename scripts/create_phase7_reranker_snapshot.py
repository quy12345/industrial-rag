"""Create one sanitized, calibration-only Jina ranking snapshot for Phase 7.4.1.

The snapshot intentionally excludes questions, raw chunks, prompts and provider
data.  It permits deterministic role-prior ablations without rerunning Jina.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.evaluation import load_frozen_chunks
from app.phase7 import (
    dataset_sha256,
    read_phase7_dataset,
    validate_phase7_datasets,
    write_json_atomic,
)
from app.phase7_optimization import QUERY_ROLE_PROFILE, infer_query_role
from app.reranking import PHASE7_CANDIDATE_TEXT_FORMAT, execute_rerank
from app.retrieval_runtime import PHASE7_RETRIEVAL_CONTRACT, build_union_rerank_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration", type=Path, default=Path("data/eval/phase7/calibration.jsonl")
    )
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/phase-7-reranker-snapshot-v1.json"),
    )
    args = parser.parse_args()

    calibration = read_phase7_dataset(args.calibration)
    held_out = read_phase7_dataset(args.test)
    validation = validate_phase7_datasets(calibration, held_out, load_frozen_chunks(args.chunks))
    selected = [item for item in calibration if item.answerable]
    settings = _phase7_settings(Settings())
    pipeline, runtime = build_union_rerank_runtime(settings, contract=PHASE7_RETRIEVAL_CONTRACT)
    rows: list[dict[str, Any]] = []
    for item in selected:
        pool = pipeline.prepare_pool(item.question, strategy="union", document_id=None)
        execution = execute_rerank(
            item.question,
            pool=pool,
            cross_encoder=pipeline.cross_encoder,
            strategy="union",
            batch_size=pipeline.rerank_batch_size,
        )
        inference = infer_query_role(item.question)
        candidates = [
            _candidate_payload(candidate) for candidate in execution.candidates_after_rerank
        ]
        rows.append(
            {
                "id": item.id,
                "language": item.language,
                "candidate_count": len(candidates),
                "query_role": inference.role,
                "query_role_confidence": inference.confidence,
                "query_role_cue_ids": list(inference.cue_ids),
                "candidates": candidates,
            }
        )
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": "approved answerable calibration rows only",
        "provider_calls": 0,
        "held_out_queries_executed": 0,
        "calibration_dataset_sha256": dataset_sha256(calibration),
        "held_out_dataset_sha256": dataset_sha256(held_out),
        "corpus": validation["corpus"],
        "runtime": runtime,
        "runtime_source_sha256": _file_sha256(Path("app/retrieval_runtime.py")),
        "candidate_text_format": PHASE7_CANDIDATE_TEXT_FORMAT,
        "query_role_profile": QUERY_ROLE_PROFILE,
        "libraries": _libraries(),
        "per_query": rows,
        "sanitization": {
            "question": "excluded",
            "evidence": "excluded",
            "answer": "not applicable",
        },
    }
    write_json_atomic(args.output, payload)
    print(f"Phase 7 reranker snapshot PASS: {args.output}")
    return 0


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    score = candidate.rerank_score
    rank = candidate.rerank_rank
    document_role = candidate.metadata.get("document_role")
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        raise RuntimeError("Snapshot requires finite cross-encoder scores.")
    if not isinstance(rank, int) or rank <= 0:
        raise RuntimeError("Snapshot requires one-based cross-encoder ranks.")
    if document_role not in {"installation", "programming"}:
        raise RuntimeError("Snapshot requires trusted document roles.")
    return {
        "chunk_id": candidate.chunk_id,
        "document_id": candidate.document_id,
        "document_role": document_role,
        "dense_rank": candidate.dense_rank,
        "sparse_rank": candidate.sparse_rank,
        "rrf_rank": candidate.rrf_rank,
        "cross_encoder_rank": rank,
        "rerank_score": score,
    }


def _phase7_settings(settings: Settings) -> Settings:
    contract = PHASE7_RETRIEVAL_CONTRACT
    return settings.model_copy(
        update={
            "qdrant_collection": contract.dense_collection,
            "qdrant_hybrid_collection": contract.hybrid_collection,
            "bm25_avg_len": contract.bm25_avg_len,
            "dense_candidate_limit": contract.dense_candidate_limit,
            "sparse_candidate_limit": contract.sparse_candidate_limit,
            "rrf_k": contract.rrf_k,
            "retrieval_strategy": "union",
            "rerank_enabled": True,
            "rerank_deduplicate_content": True,
        }
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _libraries() -> dict[str, str]:
    names = ("fastembed", "onnxruntime", "qdrant-client")
    return {name: importlib.metadata.version(name) for name in names}


if __name__ == "__main__":
    raise SystemExit(main())

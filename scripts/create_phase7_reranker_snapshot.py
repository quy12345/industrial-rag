"""Create one sanitized, calibration-only Jina ranking snapshot for Phase 7.4.1.

The snapshot intentionally excludes questions, raw chunks, prompts and provider
data.  It permits deterministic role-prior ablations without rerunning Jina.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.content_identity import evidence_content_fingerprint
from app.evaluation import load_frozen_chunks
from app.phase7 import (
    dataset_sha256,
    read_phase7_dataset,
    validate_phase7_dataset,
    write_json_atomic,
)
from app.phase7_optimization import (
    LIST_COMPLETENESS_PROFILE,
    QUERY_ROLE_PROFILE,
    RELATION_LIST_COMPLETENESS_PROFILE,
    infer_list_intent,
    infer_query_role,
    infer_relation_list_intent,
    list_completeness_features,
    relation_list_completeness_features,
)
from app.reranking import PHASE7_CANDIDATE_TEXT_FORMAT, execute_rerank
from app.retrieval_runtime import PHASE7_RETRIEVAL_CONTRACT, build_union_rerank_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration", type=Path, default=Path("data/eval/phase7/calibration-v3.jsonl")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/metrics/phase-7-evaluation-manifest-v3.json"),
    )
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/phase-7-reranker-snapshot-v3.json"),
    )
    args = parser.parse_args()

    calibration = read_phase7_dataset(args.calibration)
    validation = validate_phase7_dataset(
        calibration, load_frozen_chunks(args.chunks), kind="calibration"
    )
    held_out_hash = _sealed_held_out_hash(args.manifest)
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
        list_intent = infer_list_intent(item.question)
        relation_list_intent = infer_relation_list_intent(item.question)
        candidates = [
            _candidate_payload(
                candidate,
                technical_identifiers=list_intent.technical_identifiers,
            )
            for candidate in execution.candidates_after_rerank
        ]
        rows.append(
            {
                "id": item.id,
                "language": item.language,
                "candidate_count": len(candidates),
                "query_role": inference.role,
                "query_role_confidence": inference.confidence,
                "query_role_cue_ids": list(inference.cue_ids),
                "list_intent_enabled": list_intent.enabled,
                "list_intent_cue_ids": list(list_intent.cue_ids),
                "query_technical_identifiers": list(list_intent.technical_identifiers),
                "relation_list_intent_enabled": relation_list_intent.enabled,
                "relation_list_intent_cue_ids": list(relation_list_intent.cue_ids),
                "candidates": candidates,
            }
        )
    payload = {
        "schema_version": 3,
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": "approved answerable calibration rows only",
        "provider_calls": 0,
        "held_out_queries_executed": 0,
        "calibration_dataset_sha256": dataset_sha256(calibration),
        "held_out_dataset_sha256": held_out_hash,
        "corpus": validation["corpus"],
        "runtime": runtime,
        "runtime_source_sha256": _file_sha256(Path("app/retrieval_runtime.py")),
        "candidate_text_format": PHASE7_CANDIDATE_TEXT_FORMAT,
        "query_role_profile": QUERY_ROLE_PROFILE,
        "list_completeness_profile": LIST_COMPLETENESS_PROFILE,
        "relation_list_completeness_profile": RELATION_LIST_COMPLETENESS_PROFILE,
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


def _candidate_payload(
    candidate: Any, *, technical_identifiers: tuple[str, ...]
) -> dict[str, Any]:
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
        "content_fingerprint_sha256": evidence_content_fingerprint(candidate.text),
        "dense_rank": candidate.dense_rank,
        "sparse_rank": candidate.sparse_rank,
        "rrf_rank": candidate.rrf_rank,
        "cross_encoder_rank": rank,
        "rerank_score": score,
        **list_completeness_features(
            candidate.text,
            technical_identifiers=technical_identifiers,
        ),
        **relation_list_completeness_features(
            candidate.text,
            technical_identifiers=technical_identifiers,
        ),
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


def _sealed_held_out_hash(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Unable to read the frozen Phase 7 evaluation manifest.") from exc
    value = payload.get("test_dataset_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError("Phase 7 manifest has no valid sealed held-out hash.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

"""Run provider-free Phase 7.4 retrieval and local reranker calibration closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.application.reranking_service import PHASE7_CANDIDATE_TEXT_FORMAT
from app.composition.retrieval import build_query_retriever
from app.config import Settings
from app.domain.retrieval_contracts import PHASE7_RETRIEVAL_CONTRACT
from app.evaluation_e2e import FACT_EVALUATOR_ID
from app.infrastructure.corpus_artifacts import load_frozen_chunks, write_json_atomic
from evaluation.phase7_dataset import (
    dataset_sha256,
    read_phase7_dataset,
    validate_phase7_datasets,
)
from evaluation.retrieval import direct_evidence_rank
from evaluation.retrieval_closure import (
    aggregate_closure_rows,
    aggregate_closure_rows_by_language,
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
        default=Path("artifacts/metrics/phase-7-retrieval-closure-v1.json"),
    )
    args = parser.parse_args()

    calibration = read_phase7_dataset(args.calibration)
    test = read_phase7_dataset(args.test)
    chunks = load_frozen_chunks(args.chunks)
    validation = validate_phase7_datasets(calibration, test, chunks)
    selected = [item for item in calibration if item.answerable]
    settings = _phase7_settings(Settings())
    retriever = build_query_retriever(settings, contract=PHASE7_RETRIEVAL_CONTRACT)

    rows: list[dict[str, Any]] = []
    for item in selected:
        execution = retriever.retrieve(item.question, document_id=None)
        candidate_pool = execution.candidate_pool or execution.candidates
        relevant = set(item.relevant_chunk_ids)
        candidate_rank = direct_evidence_rank(candidate_pool, relevant)
        final_rank = direct_evidence_rank(execution.candidates, relevant)
        rows.append(
            {
                "id": item.id,
                "language": item.language,
                "candidate_count": len(candidate_pool),
                "final_candidate_count": len(execution.candidates),
                "candidate_direct_evidence_rank": candidate_rank,
                "final_direct_evidence_rank": final_rank,
                "failure_class": _failure_class(candidate_rank, final_rank),
                "wrong_document_top1": bool(execution.candidates)
                and execution.candidates[0].document_id not in item.expected_document_ids,
                "wrong_document_candidate_count_at_5": sum(
                    candidate.document_id not in item.expected_document_ids
                    for candidate in execution.candidates[:5]
                ),
                "document_context_complete": all(
                    candidate.metadata.get("document_role")
                    in {"installation", "programming"}
                    and bool(candidate.metadata.get("document_title"))
                    for candidate in execution.candidates
                ),
                "retrieval_ms": execution.retrieval_ms,
                "rerank_ms": execution.rerank_ms,
            }
        )

    overall = aggregate_closure_rows(rows)
    per_language = aggregate_closure_rows_by_language(rows)
    runtime_identity = {
        "dense_candidate_limit": PHASE7_RETRIEVAL_CONTRACT.dense_candidate_limit,
        "sparse_candidate_limit": PHASE7_RETRIEVAL_CONTRACT.sparse_candidate_limit,
        "rrf_k": PHASE7_RETRIEVAL_CONTRACT.rrf_k,
        "rrf_prune_limit": PHASE7_RETRIEVAL_CONTRACT.union_rrf_prune_limit,
        "phase7_fusion_profile": (
            None
            if PHASE7_RETRIEVAL_CONTRACT.phase7_fusion_profile is None
            else PHASE7_RETRIEVAL_CONTRACT.phase7_fusion_profile.name
        ),
        "query_expansion_profile": PHASE7_RETRIEVAL_CONTRACT.query_expansion_profile,
        "reranker": PHASE7_RETRIEVAL_CONTRACT.rerank_model,
        "candidate_text_format": PHASE7_CANDIDATE_TEXT_FORMAT,
    }
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": "approved answerable calibration rows only",
        "calibration_dataset_sha256": dataset_sha256(calibration),
        "held_out_dataset_sha256": dataset_sha256(test),
        "corpus": validation["corpus"],
        "runtime": runtime_identity,
        "frozen_identity": {
            "runtime_configuration_sha256": _json_sha256(runtime_identity),
            "fact_evaluator_id": FACT_EVALUATOR_ID,
            "evaluator_source_sha256": _file_sha256(Path("app/evaluation_e2e.py")),
            "query_expansion_source_sha256": _file_sha256(Path("app/query_expansion.py")),
            "retrieval_runtime_source_sha256": _file_sha256(Path("app/retrieval_runtime.py")),
        },
        "overall": overall,
        "per_language": per_language,
        "quality_gates": {
            "candidate_recall_12_of_12": overall["candidate_recall"] == 1.0,
            "hit_rate_at_5_at_least_11_of_12": overall["hit_rate_at_5"] >= 11 / 12,
            "mrr_at_5_at_least_0_875": overall["mrr_at_5"] >= 0.875,
            "reranker_input_at_most_30": overall["candidate_count_maximum"] <= 30,
            "wrong_document_top1_zero": overall["wrong_document_top1_rate"] == 0.0,
            "wrong_document_candidate_rate_at_5_at_most_0_15": (
                overall["wrong_document_candidate_rate_at_5"] <= 0.15
            ),
            "document_context_complete": overall["document_context_complete_rate"] == 1.0,
            "english_hit_rate_at_5_6_of_6": per_language["en"]["hit_rate_at_5"] == 1.0,
            "vietnamese_hit_rate_at_5_at_least_5_of_6": per_language["vi"]["hit_rate_at_5"]
            >= 5 / 6,
            "english_wrong_document_top5_at_most_5_of_30": (
                per_language["en"]["wrong_document_candidate_rate_at_5"] <= 5 / 30
            ),
            "vietnamese_wrong_document_top5_at_most_5_of_30": (
                per_language["vi"]["wrong_document_candidate_rate_at_5"] <= 5 / 30
            ),
            "calibration_010_rank_at_most_5": _rank(rows, "phase7_calibration_010") <= 5,
        },
        "per_query": rows,
        "provider_calls": 0,
        "held_out_queries_executed": 0,
        "sanitization": {
            "question": "excluded",
            "answer": "not applicable",
            "evidence": "excluded",
        },
    }
    write_json_atomic(args.output, payload)
    passed = all(payload["quality_gates"].values())
    status = "PASS" if passed else "FAIL"
    print(f"Phase 7.4 provider-free retrieval closure {status}: {args.output}")
    return 0 if passed else 2


def _rank(rows: list[dict[str, Any]], identifier: str) -> int:
    for row in rows:
        if row["id"] == identifier:
            return int(row["final_direct_evidence_rank"] or 2**31)
    raise ValueError(f"Required calibration row missing: {identifier}")


def _json_sha256(value: dict[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _failure_class(candidate_rank: int | None, final_rank: int | None) -> str:
    if candidate_rank is None:
        return "candidate_miss"
    if final_rank is None or final_rank > 20:
        return "reranker_miss_top20"
    if final_rank > 5:
        return "reranker_miss_top5"
    return "hit"


def _phase7_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "qdrant_collection": PHASE7_RETRIEVAL_CONTRACT.dense_collection,
            "qdrant_hybrid_collection": PHASE7_RETRIEVAL_CONTRACT.hybrid_collection,
            "bm25_avg_len": PHASE7_RETRIEVAL_CONTRACT.bm25_avg_len,
            "dense_candidate_limit": PHASE7_RETRIEVAL_CONTRACT.dense_candidate_limit,
            "sparse_candidate_limit": PHASE7_RETRIEVAL_CONTRACT.sparse_candidate_limit,
            "rrf_k": PHASE7_RETRIEVAL_CONTRACT.rrf_k,
            "retrieval_strategy": "union",
            "rerank_enabled": True,
            "rerank_deduplicate_content": True,
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())

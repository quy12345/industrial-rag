"""Rerank a bounded set of selected Phase 7 weighted-fusion profiles.

This calibration-only command never calls a generation provider and never
executes held-out questions. It consumes the sanitized retrieval ablation,
reranks at most six profiles, and emits only IDs, ranks, metrics and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.evaluation import direct_evidence_rank, load_frozen_chunks
from app.phase7 import (
    dataset_sha256,
    read_phase7_dataset,
    validate_phase7_datasets,
    write_json_atomic,
)
from app.phase7_optimization import Phase7FusionProfile, phase7_profile_from_mapping
from app.retrieval_runtime import (
    PHASE7_RETRIEVAL_CONTRACT,
    build_query_retriever,
)
from scripts.evaluate_phase7_retrieval_closure import aggregate_closure_rows

MAX_PARETO_PROFILES = 6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation",
        type=Path,
        default=Path("artifacts/metrics/phase-7-weighted-fusion-ablation-v1.json"),
    )
    parser.add_argument(
        "--calibration", type=Path, default=Path("data/eval/phase7/calibration.jsonl")
    )
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument("--max-profiles", type=int, default=MAX_PARETO_PROFILES)
    parser.add_argument(
        "--profile",
        help="Run one named profile from the ablation instead of its leading Pareto rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/phase-7-weighted-rerank-calibration-v1.json"),
    )
    args = parser.parse_args()
    if not 1 <= args.max_profiles <= MAX_PARETO_PROFILES:
        parser.error(f"--max-profiles must be between 1 and {MAX_PARETO_PROFILES}.")

    calibration = read_phase7_dataset(args.calibration)
    test = read_phase7_dataset(args.test)
    validation = validate_phase7_datasets(calibration, test, load_frozen_chunks(args.chunks))
    selected = [item for item in calibration if item.answerable]
    ablation = _read_ablation(args.ablation)
    profiles = _profiles_from_ablation(
        ablation, maximum=args.max_profiles, selected_name=args.profile
    )
    summaries: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        summaries[profile.name] = _evaluate_profile(profile, selected)
    winner = select_runtime_profile(summaries)
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": "approved answerable calibration rows only",
        "provider_calls": 0,
        "held_out_queries_executed": 0,
        "cross_encoder": PHASE7_RETRIEVAL_CONTRACT.rerank_model,
        "calibration_dataset_sha256": dataset_sha256(calibration),
        "held_out_dataset_sha256": dataset_sha256(test),
        "corpus": validation["corpus"],
        "ablation_source": {
            "path": str(args.ablation).replace("\\", "/"),
            "sha256": _file_sha256(args.ablation),
        },
        "profiles": summaries,
        "recommended_profile": winner,
        "quality": _quality(summaries[winner]),
        "sanitization": {
            "question": "excluded",
            "evidence": "excluded",
            "answer": "not applicable",
        },
    }
    write_json_atomic(args.output, payload)
    status = "PASS" if payload["quality"]["overall_pass"] else "PARTIAL"
    print(f"Phase 7 weighted rerank calibration {status}: {args.output}")
    return 0 if status == "PASS" else 2


def _evaluate_profile(profile: Phase7FusionProfile, items: list[Any]) -> dict[str, Any]:
    contract = replace(
        PHASE7_RETRIEVAL_CONTRACT,
        rrf_k=profile.rrf_k,
        phase7_fusion_profile=profile,
    )
    settings = _phase7_settings(Settings(), contract)
    retriever = build_query_retriever(settings, contract=contract)
    rows: list[dict[str, Any]] = []
    for item in items:
        execution = retriever.retrieve(item.question, document_id=None)
        pool = execution.candidate_pool or execution.candidates
        relevant = set(item.relevant_chunk_ids)
        candidate_rank = direct_evidence_rank(pool, relevant)
        final_rank = direct_evidence_rank(execution.candidates, relevant)
        rows.append(
            {
                "id": item.id,
                "language": item.language,
                "candidate_count": len(pool),
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
                    candidate.metadata.get("document_role") in {"installation", "programming"}
                    and bool(candidate.metadata.get("document_title"))
                    for candidate in execution.candidates
                ),
                "retrieval_ms": execution.retrieval_ms,
                "rerank_ms": execution.rerank_ms,
            }
        )
    metrics = aggregate_closure_rows(rows)
    return {
        "profile": _profile_payload(profile),
        "overall": metrics,
        "per_language": _per_language(rows),
        "per_query": rows,
    }


def select_runtime_profile(summaries: dict[str, dict[str, Any]]) -> str:
    """Select a stable winner after quality metrics, never raw model scores."""

    if not summaries:
        raise ValueError("Weighted rerank calibration requires at least one profile.")
    return min(
        summaries,
        key=lambda name: (
            -float(summaries[name]["overall"]["candidate_recall"]),
            -float(summaries[name]["overall"]["hit_rate_at_5"]),
            -float(summaries[name]["overall"]["mrr_at_5"]),
            float(summaries[name]["overall"]["wrong_document_top1_rate"]),
            float(summaries[name]["overall"]["wrong_document_candidate_rate_at_5"]),
            float(summaries[name]["overall"]["rerank_latency_ms"]["p95"]),
            name,
        ),
    )


def _quality(summary: dict[str, Any]) -> dict[str, Any]:
    overall = summary["overall"]
    gates = {
        "candidate_recall_at_least_11_of_12": overall["candidate_recall"] >= 11 / 12,
        "hit_rate_at_5_at_least_11_of_12": overall["hit_rate_at_5"] >= 11 / 12,
        "candidate_budget_at_most_30": overall["candidate_count_maximum"] <= 30,
        "wrong_document_top1_at_most_1_of_12": overall["wrong_document_top1_rate"] <= 1 / 12,
        "wrong_document_candidate_rate_at_5_at_most_0_15": (
            overall["wrong_document_candidate_rate_at_5"] <= 0.15
        ),
    }
    return {"overall_pass": all(gates.values()), "gates": gates}


def _profiles_from_ablation(
    payload: dict[str, Any], *, maximum: int, selected_name: str | None = None
) -> list[Phase7FusionProfile]:
    rows = payload.get("pareto_profiles_for_rerank")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Ablation artifact has no Pareto profiles for reranking.")
    profiles: list[Phase7FusionProfile] = []
    selected_rows = rows[:maximum]
    if selected_name is not None:
        selected_rows = [row for row in rows if row.get("name") == selected_name]
        if not selected_rows:
            raise ValueError(f"Ablation does not contain selected profile: {selected_name}")
    for row in selected_rows:
        profile = row.get("summary", {}).get("profile")
        if not isinstance(profile, dict):
            raise ValueError("Ablation profile is malformed.")
        profiles.append(phase7_profile_from_mapping(profile))
    return profiles


def _read_ablation(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read weighted-fusion ablation {path}.") from exc
    if payload.get("provider_calls") != 0 or payload.get("held_out_queries_executed") != 0:
        raise ValueError("Weighted-fusion ablation identity is not calibration-only.")
    return payload


def _phase7_settings(settings: Settings, contract: Any) -> Settings:
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


def _per_language(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["language"])].append(row)
    return {
        language: {
            "query_count": len(group),
            "candidate_recall": sum(
                row["candidate_direct_evidence_rank"] is not None for row in group
            )
            / len(group),
            "hit_rate_at_5": sum(
                row["final_direct_evidence_rank"] is not None
                and row["final_direct_evidence_rank"] <= 5
                for row in group
            )
            / len(group),
        }
        for language, group in sorted(groups.items())
    }


def _failure_class(candidate_rank: int | None, final_rank: int | None) -> str:
    if candidate_rank is None:
        return "candidate_miss"
    if final_rank is None or final_rank > 20:
        return "reranker_miss_top20"
    if final_rank > 5:
        return "reranker_miss_top5"
    return "hit"


def _profile_payload(profile: Phase7FusionProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "rrf_k": profile.rrf_k,
        "dense_weight": profile.dense_weight,
        "sparse_weight": profile.sparse_weight,
        "fusion_role_multiplier": profile.fusion_role_multiplier,
        "post_rerank_role_multiplier": profile.post_rerank_role_multiplier,
        "post_rerank_rank_offset": profile.post_rerank_rank_offset,
        "post_rerank_confidence_mode": profile.post_rerank_confidence_mode,
        "dense_reserve": profile.dense_reserve,
        "sparse_reserve": profile.sparse_reserve,
        "max_candidates": profile.max_candidates,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

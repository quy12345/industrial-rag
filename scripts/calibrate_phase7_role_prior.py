"""Replay a sanitized Phase 7 Jina snapshot through bounded role-prior profiles.

This calibration-only command never constructs a model, connects to Qdrant,
calls a provider, or executes held-out questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evaluation import direct_evidence_rank, load_frozen_chunks
from app.phase7 import (
    dataset_sha256,
    read_phase7_dataset,
    validate_phase7_datasets,
    write_json_atomic,
)
from app.phase7_optimization import PHASE7_CALIBRATION_FUSION_PROFILE, Phase7FusionProfile
from app.phase7_replay import Phase7ReplayError, replay_role_prior, snapshot_candidates_to_retrieval
from scripts.evaluate_phase7_retrieval_closure import aggregate_closure_rows

FOLDS: tuple[tuple[str, str], ...] = (
    ("phase7_calibration_001", "phase7_calibration_002"),
    ("phase7_calibration_003", "phase7_calibration_004"),
    ("phase7_calibration_005", "phase7_calibration_006"),
    ("phase7_calibration_007", "phase7_calibration_008"),
    ("phase7_calibration_009", "phase7_calibration_010"),
    ("phase7_calibration_011", "phase7_calibration_012"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("artifacts/metrics/phase-7-reranker-snapshot-v1.json"),
    )
    parser.add_argument(
        "--calibration", type=Path, default=Path("data/eval/phase7/calibration.jsonl")
    )
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/phase-7-role-prior-ablation-v1.json"),
    )
    args = parser.parse_args()

    calibration = read_phase7_dataset(args.calibration)
    held_out = read_phase7_dataset(args.test)
    validation = validate_phase7_datasets(calibration, held_out, load_frozen_chunks(args.chunks))
    selected = [item for item in calibration if item.answerable]
    snapshot = _read_snapshot(args.snapshot, calibration_sha=dataset_sha256(calibration))
    snapshot_rows = {str(row["id"]): row for row in snapshot["per_query"]}
    if set(snapshot_rows) != {item.id for item in selected}:
        raise ValueError(
            "Snapshot query IDs do not exactly match approved answerable calibration rows."
        )
    _validate_folds({item.id for item in selected})

    summaries = {
        profile.name: _evaluate_profile(profile, selected, snapshot_rows)
        for profile in _role_prior_profiles()
    }
    folds = _cross_validate(summaries)
    recommended_profile = folds["consensus_profile"]
    quality = _quality(summaries[recommended_profile], stable=bool(folds["stable_consensus"]))
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": "approved answerable calibration rows only",
        "provider_calls": 0,
        "held_out_queries_executed": 0,
        "calibration_dataset_sha256": dataset_sha256(calibration),
        "held_out_dataset_sha256": dataset_sha256(held_out),
        "corpus": validation["corpus"],
        "snapshot_source": {
            "path": str(args.snapshot).replace("\\", "/"),
            "sha256": _file_sha256(args.snapshot),
            "candidate_text_format": snapshot["candidate_text_format"],
            "query_role_profile": snapshot["query_role_profile"],
        },
        "profiles": summaries,
        "cross_validation": folds,
        "recommended_profile": recommended_profile,
        "quality": quality,
        "latency_methodology": "rank-only replay; Jina and Qdrant timings are not measured",
        "sanitization": {
            "question": "excluded",
            "evidence": "excluded",
            "answer": "not applicable",
        },
    }
    write_json_atomic(args.output, payload)
    status = "PASS" if quality["overall_pass"] else "PARTIAL"
    print(f"Phase 7 role-prior calibration {status}: {args.output}")
    return 0 if quality["overall_pass"] else 2


def _role_prior_profiles() -> tuple[Phase7FusionProfile, ...]:
    profiles: list[Phase7FusionProfile] = []
    for multiplier in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
        for offset in (5, 10, 20):
            for confidence_mode in ("strong_only", "strong_and_weak"):
                profiles.append(
                    replace(
                        PHASE7_CALIBRATION_FUSION_PROFILE,
                        name=(
                            f"phase741_role_prior_m{multiplier:g}_offset{offset}_{confidence_mode}"
                        ),
                        post_rerank_role_multiplier=multiplier,
                        post_rerank_rank_offset=offset,
                        post_rerank_confidence_mode=confidence_mode,
                    )
                )
    return tuple(profiles)


def _evaluate_profile(
    profile: Phase7FusionProfile, items: list[Any], snapshot_rows: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in items:
        snapshot = snapshot_rows[item.id]
        raw_candidates = snapshot_candidates_to_retrieval(snapshot["candidates"])
        replayed = replay_role_prior(
            snapshot["candidates"],
            query_role=snapshot["query_role"],
            confidence=snapshot["query_role_confidence"],
            role_multiplier=profile.post_rerank_role_multiplier,
            rank_offset=profile.post_rerank_rank_offset,
            confidence_mode=profile.post_rerank_confidence_mode,
        )
        relevant = set(item.relevant_chunk_ids)
        rows.append(
            {
                "id": item.id,
                "language": item.language,
                "candidate_count": len(raw_candidates),
                "final_candidate_count": len(replayed),
                "candidate_direct_evidence_rank": direct_evidence_rank(raw_candidates, relevant),
                "final_direct_evidence_rank": direct_evidence_rank(replayed, relevant),
                "failure_class": _failure_class(
                    direct_evidence_rank(raw_candidates, relevant),
                    direct_evidence_rank(replayed, relevant),
                ),
                "wrong_document_top1": bool(replayed)
                and replayed[0].document_id not in item.expected_document_ids,
                "wrong_document_candidate_count_at_5": sum(
                    candidate.document_id not in item.expected_document_ids
                    for candidate in replayed[:5]
                ),
                "query_role": snapshot["query_role"],
                "query_role_confidence": snapshot["query_role_confidence"],
                # Snapshot replay has no live model/Qdrant work.  These placeholders only satisfy
                # the shared metric aggregator and are excluded from this artifact's methodology.
                "retrieval_ms": 0.0,
                "rerank_ms": 0.0,
                "document_context_complete": True,
            }
        )
    return {
        "profile": _profile_payload(profile),
        "overall": aggregate_closure_rows(rows),
        "per_language": _per_language(rows),
        "per_query": rows,
    }


def _cross_validate(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fold_winners: list[dict[str, Any]] = []
    for fold in FOLDS:
        held_ids = set(fold)
        training = {
            name: _summary_for_rows(
                summary, [row for row in summary["per_query"] if row["id"] not in held_ids]
            )
            for name, summary in summaries.items()
        }
        winner = _select_profile(training)
        held = _summary_for_rows(
            summaries[winner],
            [row for row in summaries[winner]["per_query"] if row["id"] in held_ids],
        )
        fold_winners.append(
            {"held_query_ids": list(fold), "profile": winner, "held_metrics": held["overall"]}
        )
    counts = Counter(item["profile"] for item in fold_winners)
    consensus, votes = min(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "folds": fold_winners,
        "consensus_profile": consensus,
        "consensus_votes": votes,
        "stable_consensus": votes >= 4,
    }


def _summary_for_rows(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "profile": summary["profile"],
        "overall": aggregate_closure_rows(rows),
        "per_query": rows,
    }


def _select_profile(summaries: dict[str, dict[str, Any]]) -> str:
    return min(
        summaries,
        key=lambda name: (
            -float(summaries[name]["overall"]["candidate_recall"]),
            -float(summaries[name]["overall"]["hit_rate_at_5"]),
            float(summaries[name]["overall"]["wrong_document_candidate_rate_at_5"]),
            float(summaries[name]["overall"]["wrong_document_top1_rate"]),
            -float(summaries[name]["overall"]["mrr_at_5"]),
            float(summaries[name]["profile"]["post_rerank_role_multiplier"]),
            int(summaries[name]["profile"]["post_rerank_rank_offset"]),
            str(summaries[name]["profile"]["post_rerank_confidence_mode"]),
            name,
        ),
    )


def _quality(summary: dict[str, Any], *, stable: bool) -> dict[str, Any]:
    overall = summary["overall"]
    by_language = summary["per_language"]
    gates = {
        "candidate_recall_12_of_12": overall["candidate_recall"] == 1.0,
        "hit_rate_at_5_at_least_11_of_12": overall["hit_rate_at_5"] >= 11 / 12,
        "mrr_at_5_at_least_0_875": overall["mrr_at_5"] >= 0.875,
        "english_hit_rate_at_5_6_of_6": by_language["en"]["hit_rate_at_5"] == 1.0,
        "vietnamese_hit_rate_at_5_at_least_5_of_6": by_language["vi"]["hit_rate_at_5"] >= 5 / 6,
        "wrong_document_top1_zero": overall["wrong_document_top1_rate"] == 0.0,
        "wrong_document_top5_at_most_9_of_60": overall["wrong_document_candidate_rate_at_5"]
        <= 0.15,
        "english_wrong_document_top5_at_most_5_of_30": by_language["en"][
            "wrong_document_candidate_rate_at_5"
        ]
        <= 5 / 30,
        "vietnamese_wrong_document_top5_at_most_5_of_30": by_language["vi"][
            "wrong_document_candidate_rate_at_5"
        ]
        <= 5 / 30,
        "candidate_budget_at_most_30": overall["candidate_count_maximum"] <= 30,
        "calibration_010_rank_at_most_6": _rank(summary["per_query"], "phase7_calibration_010")
        <= 6,
        "cross_validation_consensus_stable": stable,
    }
    return {"overall_pass": all(gates.values()), "gates": gates}


def _per_language(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["language"])].append(row)
    result: dict[str, dict[str, float | int]] = {}
    for language, group in sorted(groups.items()):
        total_top5 = sum(min(int(row["final_candidate_count"]), 5) for row in group)
        result[language] = {
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
            "wrong_document_candidate_rate_at_5": sum(
                int(row["wrong_document_candidate_count_at_5"]) for row in group
            )
            / total_top5,
        }
    return result


def _rank(rows: list[dict[str, Any]], identifier: str) -> int:
    for row in rows:
        if row["id"] == identifier:
            return int(row["final_direct_evidence_rank"] or 2**31)
    raise ValueError(f"Required calibration row missing: {identifier}")


def _failure_class(candidate_rank: int | None, final_rank: int | None) -> str:
    if candidate_rank is None:
        return "candidate_miss"
    if final_rank is None or final_rank > 20:
        return "reranker_miss_top20"
    if final_rank > 5:
        return "reranker_miss_top5"
    return "hit"


def _validate_folds(ids: set[str]) -> None:
    mapped = {identifier for fold in FOLDS for identifier in fold}
    if mapped != ids or len(mapped) != 12:
        raise ValueError("Evaluation-only fold mapping must cover exactly the 12 answerable rows.")


def _read_snapshot(path: Path, *, calibration_sha: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Unable to read Phase 7 reranker snapshot.") from exc
    if payload.get("provider_calls") != 0 or payload.get("held_out_queries_executed") != 0:
        raise ValueError("Snapshot must be calibration-only with zero provider/held-out calls.")
    if payload.get("calibration_dataset_sha256") != calibration_sha:
        raise ValueError("Snapshot calibration dataset hash does not match current frozen dataset.")
    if payload.get("candidate_text_format") != "document_context_heading_content_v2":
        raise ValueError("Snapshot candidate text format is not the frozen Phase 7.4.1 format.")
    rows = payload.get("per_query")
    if not isinstance(rows, list):
        raise ValueError("Snapshot per-query rows are malformed.")
    try:
        for row in rows:
            snapshot_candidates_to_retrieval(row["candidates"])
    except (KeyError, TypeError, Phase7ReplayError) as exc:
        raise ValueError("Snapshot candidate data is invalid.") from exc
    return payload


def _profile_payload(profile: Phase7FusionProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "rrf_k": profile.rrf_k,
        "dense_weight": profile.dense_weight,
        "sparse_weight": profile.sparse_weight,
        "fusion_role_multiplier": profile.fusion_role_multiplier,
        "dense_reserve": profile.dense_reserve,
        "sparse_reserve": profile.sparse_reserve,
        "max_candidates": profile.max_candidates,
        "post_rerank_role_multiplier": profile.post_rerank_role_multiplier,
        "post_rerank_rank_offset": profile.post_rerank_rank_offset,
        "post_rerank_confidence_mode": profile.post_rerank_confidence_mode,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

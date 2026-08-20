"""Run provider-free Phase 7 candidate-pool ablations on calibration only.

The script retrieves dense@60 and sparse@60 once for each approved answerable
calibration query, then derives smaller union/RRF profiles deterministically.  It
never loads the cross-encoder, calls a generation provider, or evaluates held-out
queries.  Output contains IDs/ranks/counts/timings but no question or evidence text.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from app.candidate_audit import dense_results_to_candidates, union_dense_sparse_candidates
from app.config import Settings
from app.evaluation import direct_evidence_rank, load_frozen_chunks
from app.hybrid_retrieval import create_sparse_embedding_model, fuse_rrf, sparse_search
from app.models import RetrievalCandidate
from app.phase7 import (
    Phase7DatasetItem,
    dataset_sha256,
    read_phase7_dataset,
    validate_phase7_datasets,
    write_json_atomic,
)
from app.query_expansion import augment_vietnamese_technical_query
from app.reranking import deduplicate_candidates_by_content
from app.retrieval import (
    create_embedding_model,
    create_qdrant_client,
    dense_search,
    get_embedding_dimension,
)
from app.retrieval_runtime import PHASE7_RETRIEVAL_CONTRACT, validate_frozen_runtime

MAX_DENSE_LIMIT = 60
MAX_SPARSE_LIMIT = 60
PoolKind = Literal["union", "rrf"]
SparseQueryKind = Literal["original", "expanded"]


@dataclass(frozen=True)
class CandidateProfile:
    name: str
    kind: PoolKind
    dense_limit: int
    sparse_limit: int
    final_limit: int | None = None
    sparse_query: SparseQueryKind = "original"


PROFILES = (
    CandidateProfile("union_d20_s20", "union", 20, 20),
    CandidateProfile("union_d20_s40", "union", 20, 40),
    CandidateProfile("union_d60_s20", "union", 60, 20),
    CandidateProfile("union_d40_s40", "union", 40, 40),
    CandidateProfile("union_d60_s40", "union", 60, 40),
    CandidateProfile("union_d60_s60", "union", 60, 60),
    CandidateProfile("rrf_d40_s40_top20", "rrf", 40, 40, 20),
    CandidateProfile("rrf_d60_s40_top20", "rrf", 60, 40, 20),
    CandidateProfile("rrf_d60_s60_top20", "rrf", 60, 60, 20),
    CandidateProfile("rrf_d60_s40_top30", "rrf", 60, 40, 30),
    CandidateProfile(
        "expanded_rrf_d60_s40_top30", "rrf", 60, 40, 30, sparse_query="expanded"
    ),
    CandidateProfile("rrf_d40_s40_top40", "rrf", 40, 40, 40),
    CandidateProfile("rrf_d60_s40_top40", "rrf", 60, 40, 40),
    CandidateProfile("rrf_d60_s60_top40", "rrf", 60, 60, 40),
    CandidateProfile("rrf_d60_s40_top60", "rrf", 60, 40, 60),
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
        default=Path("artifacts/metrics/phase-7-calibration-retrieval-ablation.json"),
    )
    args = parser.parse_args()

    calibration = read_phase7_dataset(args.calibration)
    test = read_phase7_dataset(args.test)
    chunks = load_frozen_chunks(args.chunks)
    validation = validate_phase7_datasets(calibration, test, chunks)
    if any(item.review_status != "approved" for item in [*calibration, *test]):
        parser.error("Phase 7 retrieval calibration requires approved datasets.")
    selected = [item for item in calibration if item.answerable]

    settings = _phase7_settings(Settings())
    client = create_qdrant_client(settings)
    validate_frozen_runtime(
        client,
        collection_names=(settings.qdrant_collection, settings.qdrant_hybrid_collection),
        contract=PHASE7_RETRIEVAL_CONTRACT,
    )
    dense_model = create_embedding_model(
        settings.embedding_model, cache_dir=settings.embedding_cache_dir
    )
    dimension = get_embedding_dimension(dense_model)
    if dimension != PHASE7_RETRIEVAL_CONTRACT.dense_dimension:
        raise RuntimeError(
            f"Dense dimension {dimension} differs from frozen contract "
            f"{PHASE7_RETRIEVAL_CONTRACT.dense_dimension}."
        )
    sparse_model = create_sparse_embedding_model(
        settings.sparse_model,
        settings.embedding_cache_dir,
        disable_stemmer=settings.bm25_disable_stemmer,
        k=settings.bm25_k,
        b=settings.bm25_b,
        avg_len=PHASE7_RETRIEVAL_CONTRACT.bm25_avg_len,
    )

    profile_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retrieval_rows: list[dict[str, Any]] = []
    for item in selected:
        dense_started = perf_counter()
        dense_results = dense_search(
            client,
            item.question,
            collection_name=settings.qdrant_collection,
            vector_name=settings.dense_vector_name,
            embedding_model=dense_model,
            limit=MAX_DENSE_LIMIT,
        )
        dense_ms = (perf_counter() - dense_started) * 1000
        sparse_started = perf_counter()
        sparse_candidates = sparse_search(
            client,
            item.question,
            collection_name=settings.qdrant_hybrid_collection,
            sparse_vector_name=settings.sparse_vector_name,
            sparse_embedding_model=sparse_model,
            limit=MAX_SPARSE_LIMIT,
        )
        sparse_ms = (perf_counter() - sparse_started) * 1000
        dense_candidates = dense_results_to_candidates(dense_results)
        expanded_sparse_candidates = sparse_candidates
        expanded_sparse_ms = 0.0
        expansion_rule_count = 0
        if item.language == "vi":
            expanded_query, matched_rules = augment_vietnamese_technical_query(item.question)
            expansion_rule_count = len(matched_rules)
            if expanded_query != item.question:
                expanded_started = perf_counter()
                expanded_sparse_candidates = sparse_search(
                    client,
                    expanded_query,
                    collection_name=settings.qdrant_hybrid_collection,
                    sparse_vector_name=settings.sparse_vector_name,
                    sparse_embedding_model=sparse_model,
                    limit=MAX_SPARSE_LIMIT,
                )
                expanded_sparse_ms = (perf_counter() - expanded_started) * 1000
        relevant = set(item.relevant_chunk_ids)
        retrieval_rows.append(
            {
                "id": item.id,
                "language": item.language,
                "dense_qrel_rank_at_60": direct_evidence_rank(dense_candidates, relevant),
                "sparse_qrel_rank_at_60": direct_evidence_rank(sparse_candidates, relevant),
                "dense_ms_at_60": dense_ms,
                "sparse_ms_at_60": sparse_ms,
                "expanded_sparse_qrel_rank_at_60": direct_evidence_rank(
                    expanded_sparse_candidates, relevant
                ),
                "expanded_sparse_ms_at_60": expanded_sparse_ms,
                "query_expansion_rule_count": expansion_rule_count,
            }
        )
        for profile in PROFILES:
            candidates = _build_profile(
                profile,
                dense_candidates,
                sparse_candidates,
                expanded_sparse_candidates=expanded_sparse_candidates,
            )
            profile_rows[profile.name].append(_score_profile(item, profile, candidates))

    summaries = {
        profile.name: _summarize_profile(profile, profile_rows[profile.name])
        for profile in PROFILES
    }
    recommendation = min(
        summaries,
        key=lambda name: (
            -float(summaries[name]["candidate_recall"]),
            float(summaries[name]["candidate_count"]["average"]),
            name,
        ),
    )
    baseline_name = "union_d20_s20"
    baseline_average = float(summaries[baseline_name]["candidate_count"]["average"])
    recommended_average = float(
        summaries[recommendation]["candidate_count"]["average"]
    )
    fixed_budget_profiles = {
        name: summary
        for name, summary in summaries.items()
        if int(summary["candidate_count"]["maximum"]) <= 30
    }
    fixed_budget_winner = min(
        fixed_budget_profiles,
        key=lambda name: (
            -float(fixed_budget_profiles[name]["candidate_recall"]),
            float(fixed_budget_profiles[name]["candidate_count"]["average"]),
            name,
        ),
    )
    fixed_budget_missing = list(
        fixed_budget_profiles[fixed_budget_winner]["missing_query_ids"]
    )
    fixed_budget_passed = (
        float(fixed_budget_profiles[fixed_budget_winner]["candidate_recall"]) >= 11 / 12
    )
    runtime_contract_matches = (
        fixed_budget_winner == "expanded_rrf_d60_s40_top30"
        and PHASE7_RETRIEVAL_CONTRACT.dense_candidate_limit == 60
        and PHASE7_RETRIEVAL_CONTRACT.sparse_candidate_limit == 40
        and PHASE7_RETRIEVAL_CONTRACT.union_rrf_prune_limit == 30
        and PHASE7_RETRIEVAL_CONTRACT.query_expansion_profile is not None
    )
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": "approved answerable calibration rows only",
        "provider_calls": 0,
        "cross_encoder_loaded": False,
        "held_out_queries_executed": 0,
        "dataset_sha256": dataset_sha256(calibration),
        "corpus": validation["corpus"],
        "retrieval_methodology": {
            "dense_max_limit": MAX_DENSE_LIMIT,
            "sparse_max_limit": MAX_SPARSE_LIMIT,
            "component_queries_per_item": (
                "2 plus one sparse query when Vietnamese glossary expansion changes the query"
            ),
            "query_expansion": "deterministic technical glossary; query terms only",
            "profile_latency": (
                "not estimated; profiles are deterministic slices of one max-limit retrieval"
            ),
            "direct_hit": "stable relevant_chunk_ids only",
            "content_deduplication": "same-document exact-normalized text",
        },
        "profiles": summaries,
        "recommended_profile_by_candidate_recall_then_pool_size": recommendation,
        "fixed_reranker_budget": {
            "maximum_candidates": 30,
            "required_candidate_recall": 11 / 12,
            "recommended_profile": fixed_budget_winner,
            "actual_candidate_recall": fixed_budget_profiles[fixed_budget_winner][
                "candidate_recall"
            ],
            "passed": fixed_budget_passed,
        },
        "runtime_recommendation": {
            "profile": fixed_budget_winner if fixed_budget_passed else baseline_name,
            "change_runtime": fixed_budget_passed and not runtime_contract_matches,
            "reason": (
                "The frozen Phase 7.4 runtime matches the best fixed-budget profile."
                if fixed_budget_passed and runtime_contract_matches
                else "No fixed-budget profile is ready for the frozen runtime."
            ),
            "runtime_contract_matches_recommendation": runtime_contract_matches,
            "best_fixed_budget_profile": fixed_budget_winner,
            "best_fixed_budget_missing_query_ids": fixed_budget_missing,
            "candidate_only_winner_pool_size_multiplier_vs_baseline": (
                recommended_average / baseline_average
            ),
        },
        "per_query_component_ranks": retrieval_rows,
        "per_profile_query_results": dict(sorted(profile_rows.items())),
        "sanitization": {
            "raw_question": "excluded",
            "evidence_text": "excluded",
            "provider_response": "not applicable",
        },
    }
    write_json_atomic(args.output, payload)
    print(f"Phase 7 retrieval calibration ablation PASS: {args.output}")
    return 0


def _phase7_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "qdrant_collection": PHASE7_RETRIEVAL_CONTRACT.dense_collection,
            "qdrant_hybrid_collection": PHASE7_RETRIEVAL_CONTRACT.hybrid_collection,
            "bm25_avg_len": PHASE7_RETRIEVAL_CONTRACT.bm25_avg_len,
        }
    )


def _build_profile(
    profile: CandidateProfile,
    dense_candidates: list[RetrievalCandidate],
    sparse_candidates: list[RetrievalCandidate],
    *,
    expanded_sparse_candidates: list[RetrievalCandidate] | None = None,
) -> list[RetrievalCandidate]:
    dense = dense_candidates[: profile.dense_limit]
    sparse_source = (
        expanded_sparse_candidates
        if profile.sparse_query == "expanded" and expanded_sparse_candidates is not None
        else sparse_candidates
    )
    sparse = sparse_source[: profile.sparse_limit]
    if profile.kind == "union":
        candidates = union_dense_sparse_candidates(dense, sparse)
    else:
        candidates = fuse_rrf(
            dense,
            sparse,
            rrf_k=PHASE7_RETRIEVAL_CONTRACT.rrf_k,
            final_limit=len(dense) + len(sparse),
        )
    candidates = deduplicate_candidates_by_content(candidates)
    return candidates[: profile.final_limit] if profile.final_limit is not None else candidates


def _score_profile(
    item: Phase7DatasetItem,
    profile: CandidateProfile,
    candidates: list[RetrievalCandidate],
) -> dict[str, Any]:
    relevant = set(item.relevant_chunk_ids)
    return {
        "id": item.id,
        "language": item.language,
        "profile": profile.name,
        "candidate_count": len(candidates),
        "direct_evidence_rank": direct_evidence_rank(candidates, relevant),
        "qrel_component_ranks": [
            {
                "chunk_id": candidate.chunk_id,
                "dense_rank": candidate.dense_rank,
                "sparse_rank": candidate.sparse_rank,
                "rrf_rank": candidate.rrf_rank,
            }
            for candidate in candidates
            if candidate.chunk_id in relevant
        ],
    }


def _summarize_profile(
    profile: CandidateProfile, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    counts = [int(row["candidate_count"]) for row in rows]
    return {
        "kind": profile.kind,
        "dense_limit": profile.dense_limit,
        "sparse_limit": profile.sparse_limit,
        "final_limit": profile.final_limit,
        "sparse_query": profile.sparse_query,
        "query_count": len(rows),
        "candidate_recall": sum(row["direct_evidence_rank"] is not None for row in rows)
        / len(rows),
        "candidate_count": {
            "minimum": min(counts),
            "maximum": max(counts),
            "average": sum(counts) / len(counts),
        },
        "missing_query_ids": [
            row["id"] for row in rows if row["direct_evidence_rank"] is None
        ],
        "per_language": {
            language: {
                "query_count": len(group),
                "candidate_recall": sum(
                    row["direct_evidence_rank"] is not None for row in group
                )
                / len(group),
            }
            for language, group in _group_by_language(rows).items()
        },
    }


def _group_by_language(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["language"])].append(row)
    return dict(sorted(groups.items()))


if __name__ == "__main__":
    raise SystemExit(main())

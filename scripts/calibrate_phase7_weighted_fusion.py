"""Ablate bounded weighted-RRF profiles on approved Phase 7 calibration only.

This command is retrieval-only: it never initializes the cross-encoder or a
generation provider, and it never loads held-out questions into the retriever.
Artifacts contain identifiers, ranks, counts, roles and hashes; raw questions
and manual text are deliberately excluded.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from app.candidate_audit import dense_results_to_candidates
from app.config import Settings
from app.evaluation import direct_evidence_rank, load_frozen_chunks
from app.hybrid_retrieval import create_sparse_embedding_model, sparse_search
from app.models import RetrievalCandidate
from app.phase7 import (
    Phase7DatasetItem,
    dataset_sha256,
    read_phase7_dataset,
    validate_phase7_datasets,
    write_json_atomic,
)
from app.phase7_optimization import (
    Phase7FusionProfile,
    Phase7OptimizationError,
    infer_query_role,
    phase7_fusion_profile_grid,
    select_coverage_preserving_candidates,
)
from app.query_expansion import QUERY_EXPANSION_PROFILE, augment_vietnamese_technical_query
from app.retrieval import (
    create_embedding_model,
    create_qdrant_client,
    dense_search,
    get_embedding_dimension,
)
from app.retrieval_runtime import PHASE7_RETRIEVAL_CONTRACT, validate_frozen_runtime

MAX_DENSE_LIMIT = 60
MAX_SPARSE_LIMIT = 40
MAX_PARETO_PROFILES = 6


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
        default=Path("artifacts/metrics/phase-7-weighted-fusion-ablation-v1.json"),
    )
    args = parser.parse_args()

    calibration = read_phase7_dataset(args.calibration)
    test = read_phase7_dataset(args.test)
    chunks = load_frozen_chunks(args.chunks)
    validation = validate_phase7_datasets(calibration, test, chunks)
    selected = [item for item in calibration if item.answerable]
    if any(item.review_status != "approved" for item in selected):
        parser.error("Weighted fusion calibration requires approved answerable rows.")

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
    if get_embedding_dimension(dense_model) != PHASE7_RETRIEVAL_CONTRACT.dense_dimension:
        raise RuntimeError("Dense model dimension differs from the frozen Phase 7 contract.")
    sparse_model = create_sparse_embedding_model(
        settings.sparse_model,
        settings.embedding_cache_dir,
        disable_stemmer=settings.bm25_disable_stemmer,
        k=settings.bm25_k,
        b=settings.bm25_b,
        avg_len=PHASE7_RETRIEVAL_CONTRACT.bm25_avg_len,
    )

    profiles = phase7_fusion_profile_grid()
    rows_by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    component_rows: list[dict[str, Any]] = []
    for item in selected:
        dense_started = perf_counter()
        dense = dense_results_to_candidates(
            dense_search(
                client,
                item.question,
                collection_name=settings.qdrant_collection,
                vector_name=settings.dense_vector_name,
                embedding_model=dense_model,
                limit=MAX_DENSE_LIMIT,
            )
        )
        dense_ms = (perf_counter() - dense_started) * 1000
        expanded_question, expansion_rules = augment_vietnamese_technical_query(item.question)
        sparse_started = perf_counter()
        sparse = sparse_search(
            client,
            expanded_question,
            collection_name=settings.qdrant_hybrid_collection,
            sparse_vector_name=settings.sparse_vector_name,
            sparse_embedding_model=sparse_model,
            limit=MAX_SPARSE_LIMIT,
        )
        sparse_ms = (perf_counter() - sparse_started) * 1000
        dense = [_with_document_context(candidate) for candidate in dense]
        sparse = [_with_document_context(candidate) for candidate in sparse]
        inference = infer_query_role(item.question)
        relevant = set(item.relevant_chunk_ids)
        component_rows.append(
            {
                "id": item.id,
                "language": item.language,
                "query_role": inference.role,
                "installation_cue_count": len(inference.installation_cues),
                "programming_cue_count": len(inference.programming_cues),
                "query_expansion_rule_count": len(expansion_rules),
                "dense_qrel_rank_at_60": direct_evidence_rank(dense, relevant),
                "sparse_qrel_rank_at_40": direct_evidence_rank(sparse, relevant),
                "dense_ms": dense_ms,
                "sparse_ms": sparse_ms,
            }
        )
        for profile in profiles:
            rows_by_profile[profile.name].append(
                _score_profile(item, profile, dense, sparse, inference.role)
            )

    summaries = {
        profile.name: _summarize_profile(profile, rows_by_profile[profile.name])
        for profile in profiles
    }
    pareto = _select_pareto_profiles(summaries)
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": "approved answerable calibration rows only",
        "provider_calls": 0,
        "cross_encoder_loaded": False,
        "held_out_queries_executed": 0,
        "calibration_dataset_sha256": dataset_sha256(calibration),
        "held_out_dataset_sha256": dataset_sha256(test),
        "corpus": validation["corpus"],
        "methodology": {
            "dense_limit": MAX_DENSE_LIMIT,
            "sparse_limit": MAX_SPARSE_LIMIT,
            "sparse_query_profile": QUERY_EXPANSION_PROFILE,
            "candidate_budget": 30,
            "direct_hit": "stable relevant_chunk_ids only",
            "role_source": "query text only; no expected document, qrel, page, or answer facts",
            "selection": "mandatory component reserves then weighted RRF fill",
        },
        "frozen_identity": {
            "weighted_fusion_source_sha256": _file_sha256(Path("app/phase7_optimization.py")),
            "runtime_source_sha256": _file_sha256(Path("app/retrieval_runtime.py")),
        },
        "component_rows": component_rows,
        "profiles": dict(sorted(summaries.items())),
        "pareto_profiles_for_rerank": pareto,
        "sanitization": {
            "question": "excluded",
            "evidence": "excluded",
            "answer": "not applicable",
        },
    }
    write_json_atomic(args.output, payload)
    print(f"Phase 7 weighted fusion ablation PASS: {args.output}")
    return 0


def _score_profile(
    item: Phase7DatasetItem,
    profile: Phase7FusionProfile,
    dense: list[RetrievalCandidate],
    sparse: list[RetrievalCandidate],
    query_role: str,
) -> dict[str, Any]:
    try:
        candidates = select_coverage_preserving_candidates(
            dense,
            sparse,
            profile=profile,
            query_role=query_role,  # type: ignore[arg-type]
        )
    except Phase7OptimizationError as exc:
        return {"id": item.id, "language": item.language, "invalid": str(exc)}
    relevant = set(item.relevant_chunk_ids)
    return {
        "id": item.id,
        "language": item.language,
        "invalid": None,
        "candidate_count": len(candidates),
        "direct_evidence_rank": direct_evidence_rank(candidates, relevant),
        "wrong_document_top1": bool(candidates)
        and candidates[0].document_id not in item.expected_document_ids,
        "wrong_document_candidate_count_at_5": sum(
            candidate.document_id not in item.expected_document_ids for candidate in candidates[:5]
        ),
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


def _summarize_profile(profile: Phase7FusionProfile, rows: list[dict[str, Any]]) -> dict[str, Any]:
    invalid = [row for row in rows if row["invalid"] is not None]
    valid = [row for row in rows if row["invalid"] is None]
    if invalid:
        return {
            "valid": False,
            "profile": _profile_payload(profile),
            "invalid_query_count": len(invalid),
            "invalid_reasons": sorted({str(row["invalid"]) for row in invalid}),
        }
    counts = [int(row["candidate_count"]) for row in valid]
    return {
        "valid": True,
        "profile": _profile_payload(profile),
        "candidate_recall": sum(row["direct_evidence_rank"] is not None for row in valid)
        / len(valid),
        "wrong_document_top1_rate": sum(bool(row["wrong_document_top1"]) for row in valid)
        / len(valid),
        "wrong_document_candidate_rate_at_5": sum(
            int(row["wrong_document_candidate_count_at_5"]) for row in valid
        )
        / sum(min(int(row["candidate_count"]), 5) for row in valid),
        "candidate_count": {
            "minimum": min(counts),
            "maximum": max(counts),
            "average": sum(counts) / len(counts),
        },
        "missing_query_ids": [row["id"] for row in valid if row["direct_evidence_rank"] is None],
        "per_language": {
            language: _language_summary([row for row in valid if row["language"] == language])
            for language in sorted({str(row["language"]) for row in valid})
        },
        "per_query": valid,
    }


def _language_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "query_count": len(rows),
        "candidate_recall": sum(row["direct_evidence_rank"] is not None for row in rows)
        / len(rows),
    }


def _select_pareto_profiles(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [(name, summary) for name, summary in summaries.items() if summary["valid"]]
    ordered = sorted(
        valid,
        key=lambda pair: (
            -float(pair[1]["candidate_recall"]),
            float(pair[1]["wrong_document_top1_rate"]),
            float(pair[1]["wrong_document_candidate_rate_at_5"]),
            float(pair[1]["profile"]["sparse_weight"] - 1.0),
            float(pair[1]["profile"]["fusion_role_multiplier"]),
            int(pair[1]["profile"]["rrf_k"]),
            pair[0],
        ),
    )
    return [{"name": name, "summary": summary} for name, summary in ordered[:MAX_PARETO_PROFILES]]


def _with_document_context(candidate: RetrievalCandidate) -> RetrievalCandidate:
    context = PHASE7_RETRIEVAL_CONTRACT.document_context_by_id.get(candidate.document_id)
    if context is None:
        raise RuntimeError(f"Unrecognized Phase 7 document ID: {candidate.document_id}")
    return candidate.model_copy(update={"metadata": dict(candidate.metadata) | context})


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


def _phase7_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "qdrant_collection": PHASE7_RETRIEVAL_CONTRACT.dense_collection,
            "qdrant_hybrid_collection": PHASE7_RETRIEVAL_CONTRACT.hybrid_collection,
            "bm25_avg_len": PHASE7_RETRIEVAL_CONTRACT.bm25_avg_len,
        }
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

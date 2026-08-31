"""Deterministic metrics for the active frozen-corpus retrieval evaluation."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any, Literal, Protocol

from app.domain.content_identity import evidence_content_fingerprint
from app.domain.policies.fusion import ATV320_FUSION_PROFILE, FusionProfile
from app.domain.policies.query_analysis import (
    QUERY_EXPANSION_PROFILE,
    QUERY_EXPANSION_PROFILE_V2,
)
from app.domain.retrieval import (
    QueryRetrievalResult,
    RetrievalCandidate,
    RetrievalTrace,
)
from app.domain.retrieval_contracts import FrozenRetrievalContract

DevelopmentRetrievalVariantId = Literal[
    "baseline",
    "refill",
    "query-expansion-v2",
    "refill-query-expansion-v2",
    "dense-reserve-d20-s5",
    "sparse-reserve-d5-s20",
]


@dataclass(frozen=True)
class DevelopmentRetrievalVariant:
    """One preregistered provider-free development retrieval experiment."""

    identifier: DevelopmentRetrievalVariantId
    query_expansion_profile: str
    fusion_profile: FusionProfile
    refill_after_content_dedup: bool = False
    diagnostic_component_limit: int | None = 100

    def apply(self, contract: FrozenRetrievalContract) -> FrozenRetrievalContract:
        """Return an isolated contract copy; never mutate the production constant."""

        return replace(
            contract,
            query_expansion_profile=self.query_expansion_profile,
            fusion_profile=self.fusion_profile,
            refill_after_content_dedup=self.refill_after_content_dedup,
            diagnostic_component_limit=self.diagnostic_component_limit,
        )

    def identity(self) -> dict[str, Any]:
        """Return the full behavior identity stored in artifacts and checkpoints."""

        return {
            "id": self.identifier,
            "query_expansion_profile": self.query_expansion_profile,
            "refill_after_content_dedup": self.refill_after_content_dedup,
            "diagnostic_component_limit": self.diagnostic_component_limit,
            "fusion_profile": asdict(self.fusion_profile),
        }


_DENSE_RESERVE_PROFILE = replace(
    ATV320_FUSION_PROFILE,
    name="atv320_development_dense_reserve_d20_s5_v1",
    dense_reserve=20,
    sparse_reserve=5,
)
_SPARSE_RESERVE_PROFILE = replace(
    ATV320_FUSION_PROFILE,
    name="atv320_development_sparse_reserve_d5_s20_v1",
    dense_reserve=5,
    sparse_reserve=20,
)
DEVELOPMENT_RETRIEVAL_VARIANTS: Mapping[
    DevelopmentRetrievalVariantId, DevelopmentRetrievalVariant
] = MappingProxyType(
    {
        "baseline": DevelopmentRetrievalVariant(
            "baseline",
            QUERY_EXPANSION_PROFILE,
            ATV320_FUSION_PROFILE,
        ),
        "refill": DevelopmentRetrievalVariant(
            "refill",
            QUERY_EXPANSION_PROFILE,
            ATV320_FUSION_PROFILE,
            refill_after_content_dedup=True,
        ),
        "query-expansion-v2": DevelopmentRetrievalVariant(
            "query-expansion-v2",
            QUERY_EXPANSION_PROFILE_V2,
            ATV320_FUSION_PROFILE,
        ),
        "refill-query-expansion-v2": DevelopmentRetrievalVariant(
            "refill-query-expansion-v2",
            QUERY_EXPANSION_PROFILE_V2,
            ATV320_FUSION_PROFILE,
            refill_after_content_dedup=True,
        ),
        "dense-reserve-d20-s5": DevelopmentRetrievalVariant(
            "dense-reserve-d20-s5",
            QUERY_EXPANSION_PROFILE,
            _DENSE_RESERVE_PROFILE,
        ),
        "sparse-reserve-d5-s20": DevelopmentRetrievalVariant(
            "sparse-reserve-d5-s20",
            QUERY_EXPANSION_PROFILE,
            _SPARSE_RESERVE_PROFILE,
        ),
    }
)


def development_retrieval_variant(
    identifier: str,
) -> DevelopmentRetrievalVariant:
    """Resolve only preregistered variants; arbitrary tuning values are forbidden."""

    try:
        return DEVELOPMENT_RETRIEVAL_VARIANTS[identifier]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"Unknown development retrieval variant: {identifier}") from exc


class RetrievedLike(Protocol):
    """Minimum result identity required by direct-evidence metrics."""

    chunk_id: str


class ContentLike(Protocol):
    """Minimum corpus fields required to derive exact-content equivalence."""

    chunk_id: str
    text: str


class EvaluationItemLike(Protocol):
    """Sanitization-safe annotation fields required by retrieval scoring."""

    id: str
    fact_family_id: str | None
    language: str
    question_type: str
    relevant_chunk_ids: Sequence[str]
    expected_document_ids: Sequence[str]


def build_corpus_exact_content_equivalence(
    chunks: Sequence[ContentLike],
) -> dict[str, tuple[str, ...]]:
    """Map each chunk ID to corpus-wide exact-normalized content alternatives."""

    groups: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        groups[evidence_content_fingerprint(chunk.text)].append(chunk.chunk_id)
    result: dict[str, tuple[str, ...]] = {}
    for members in groups.values():
        alternatives = tuple(sorted(set(members)))
        if len(alternatives) < 2:
            continue
        for chunk_id in alternatives:
            result[chunk_id] = alternatives
    return result


def phrase_matches(text: str, expected_phrase: str) -> bool:
    """Compare evidence text with Unicode and whitespace normalization."""

    return _normalize_text(expected_phrase) in _normalize_text(text)


def direct_evidence_rank(
    results: Sequence[RetrievedLike], relevant_chunk_ids: set[str]
) -> int | None:
    """Return the first one-based result rank whose stable chunk ID is a qrel."""

    return evidence_rank(
        results,
        relevant_chunk_ids,
        include_exact_equivalents=False,
    )


def exact_equivalent_evidence_rank(
    results: Sequence[RetrievedLike],
    relevant_chunk_ids: set[str],
    *,
    equivalence: Mapping[str, Sequence[str]] | None = None,
) -> int | None:
    """Return the first rank matching a qrel or its declared exact-content equivalent."""

    return evidence_rank(
        results,
        expand_exact_content_ids(relevant_chunk_ids, equivalence=equivalence),
        include_exact_equivalents=True,
    )


def expand_exact_content_ids(
    chunk_ids: set[str],
    *,
    equivalence: Mapping[str, Sequence[str]] | None,
) -> set[str]:
    """Expand qrel IDs only through a precomputed exact-content map."""

    expanded = set(chunk_ids)
    if equivalence is None:
        return expanded
    for chunk_id in chunk_ids:
        expanded.update(equivalence.get(chunk_id, ()))
    return expanded


def evidence_rank(
    results: Sequence[RetrievedLike],
    relevant_chunk_ids: set[str],
    *,
    include_exact_equivalents: bool,
) -> int | None:
    """Return the first qrel rank under an explicit strict or exact-equivalent policy."""

    return next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if relevant_chunk_ids.intersection(
                _result_chunk_ids(
                    result,
                    include_exact_equivalents=include_exact_equivalents,
                )
            )
        ),
        None,
    )


def retrieval_trace_payload(
    trace: RetrievalTrace | None,
    *,
    relevant_chunk_ids: set[str],
    exact_relevant_chunk_ids: set[str],
) -> dict[str, Any]:
    """Serialize only allowlisted IDs and rank signals from an opt-in trace."""

    if trace is None:
        return {"status": "disabled"}
    stages = {
        "dense": trace.dense_candidates,
        "sparse": trace.sparse_candidates,
        "raw_fusion": trace.fused_candidates,
        "selected_pre_dedup": trace.selected_candidates_pre_dedup,
        "candidate_pool_post_dedup": trace.candidate_pool_post_dedup,
        "candidates_before_rerank": trace.candidates_before_rerank,
        "raw_cross_encoder": trace.raw_cross_encoder_candidates,
        "post_policy": trace.final_candidates,
    }
    fused_ids = {candidate.chunk_id for candidate in trace.fused_candidates}
    selected_ids = {
        candidate.chunk_id for candidate in trace.selected_candidates_pre_dedup
    }
    post_dedup_ids = {
        candidate.chunk_id for candidate in trace.candidate_pool_post_dedup
    }
    before_rerank_ids = {
        candidate.chunk_id for candidate in trace.candidates_before_rerank
    }
    return {
        "status": "captured",
        "candidate_count_pre_dedup": trace.candidate_count_pre_dedup,
        "candidate_count_post_dedup": trace.candidate_count_post_dedup,
        "candidate_count_before_rerank": trace.candidate_count_before_rerank,
        "removed_by_candidate_selection_ids": sorted(fused_ids - selected_ids),
        "removed_by_same_document_dedup_ids": sorted(selected_ids - post_dedup_ids),
        "removed_by_prune_ids": sorted(post_dedup_ids - before_rerank_ids),
        "refilled_candidate_ids": list(trace.refilled_candidate_ids),
        "stages": {
            name: _retrieval_trace_stage_payload(
                candidates,
                relevant_chunk_ids=relevant_chunk_ids,
                exact_relevant_chunk_ids=exact_relevant_chunk_ids,
            )
            for name, candidates in stages.items()
        },
    }


def _retrieval_trace_stage_payload(
    candidates: Sequence[RetrievalCandidate],
    *,
    relevant_chunk_ids: set[str],
    exact_relevant_chunk_ids: set[str],
) -> dict[str, Any]:
    return {
        "count": len(candidates),
        "strict_direct_evidence_rank": direct_evidence_rank(
            candidates, relevant_chunk_ids
        ),
        "exact_equivalent_evidence_rank": exact_equivalent_evidence_rank(
            candidates, exact_relevant_chunk_ids
        ),
        "candidates": [
            {
                "chunk_id": candidate.chunk_id,
                "document_id": candidate.document_id,
                "ordinal_rank": ordinal,
                "dense_rank": candidate.dense_rank,
                "sparse_rank": candidate.sparse_rank,
                "rrf_rank": candidate.rrf_rank,
                "rerank_rank": candidate.rerank_rank,
                "document_role": _trace_metadata_value(
                    candidate, "document_role"
                ),
                "query_role": _trace_metadata_value(candidate, "query_role"),
                "query_role_confidence": _trace_metadata_value(
                    candidate, "query_role_confidence"
                ),
                "cross_encoder_rank": _trace_metadata_value(
                    candidate, "cross_encoder_rank"
                ),
                "role_rank": _trace_metadata_value(candidate, "role_rank"),
            }
            for ordinal, candidate in enumerate(candidates, start=1)
        ],
    }


def _trace_metadata_value(
    candidate: RetrievalCandidate,
    key: str,
) -> str | int | None:
    value = candidate.metadata.get(key)
    if key in {"document_role", "query_role"}:
        return (
            value
            if isinstance(value, str)
            and value in {"installation", "programming", "neutral"}
            else None
        )
    if key == "query_role_confidence":
        return (
            value
            if isinstance(value, str) and value in {"strong", "weak", "neutral"}
            else None
        )
    if key in {"cross_encoder_rank", "role_rank"}:
        return value if type(value) is int and value >= 1 else None
    return None


def _result_chunk_ids(
    result: RetrievedLike, *, include_exact_equivalents: bool
) -> set[str]:
    chunk_ids = {result.chunk_id}
    if not include_exact_equivalents:
        return chunk_ids
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, dict):
        return chunk_ids
    equivalent_ids = metadata.get("equivalent_chunk_ids", ())
    if isinstance(equivalent_ids, (list, tuple, set, frozenset)):
        chunk_ids.update(
            chunk_id for chunk_id in equivalent_ids if isinstance(chunk_id, str)
        )
    return chunk_ids


def percentile_nearest_rank(values: Sequence[float], percentile: int) -> float:
    """Calculate a deterministic nearest-rank percentile for a non-empty sample."""

    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sample.")
    if not 0 < percentile <= 100:
        raise ValueError("Percentile must be in the range 1..100.")
    ordered = sorted(float(value) for value in values)
    index = math.ceil((percentile / 100) * len(ordered)) - 1
    return ordered[index]


def score_retrieval_execution(
    item: EvaluationItemLike,
    execution: QueryRetrievalResult,
    *,
    corpus_equivalence: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Score one retrieval execution without retaining question or chunk content."""

    candidate_pool = execution.candidate_pool or execution.candidates
    relevant = set(item.relevant_chunk_ids)
    exact_relevant = expand_exact_content_ids(relevant, equivalence=corpus_equivalence)
    candidate_rank = direct_evidence_rank(candidate_pool, relevant)
    final_rank = direct_evidence_rank(execution.candidates, relevant)
    candidate_exact_rank = exact_equivalent_evidence_rank(
        candidate_pool,
        relevant,
        equivalence=corpus_equivalence,
    )
    final_exact_rank = exact_equivalent_evidence_rank(
        execution.candidates,
        relevant,
        equivalence=corpus_equivalence,
    )
    trace = execution.trace
    dense_rank = (
        direct_evidence_rank(trace.dense_candidates, relevant) if trace is not None else None
    )
    sparse_rank = (
        direct_evidence_rank(trace.sparse_candidates, relevant) if trace is not None else None
    )
    return {
        "id": item.id,
        "fact_family_id": item.fact_family_id,
        "language": item.language,
        "question_type": item.question_type,
        "candidate_count": len(candidate_pool),
        "final_candidate_count": len(execution.candidates),
        "raw_dense_direct_evidence_rank": dense_rank,
        "raw_sparse_direct_evidence_rank": sparse_rank,
        "raw_component_direct_evidence": dense_rank is not None or sparse_rank is not None,
        "candidate_direct_evidence_rank": candidate_rank,
        "final_direct_evidence_rank": final_rank,
        "candidate_exact_equivalent_evidence_rank": candidate_exact_rank,
        "final_exact_equivalent_evidence_rank": final_exact_rank,
        "failure_class": retrieval_failure_class(candidate_rank, final_rank),
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
        "retrieval_trace": retrieval_trace_payload(
            trace,
            relevant_chunk_ids=relevant,
            exact_relevant_chunk_ids=exact_relevant,
        ),
        "retrieval_ms": execution.retrieval_ms,
        "rerank_ms": execution.rerank_ms,
    }


def evaluate_retrieval_quality_gates(
    dataset: Literal["legacy", "development"],
    *,
    rows: list[dict[str, Any]],
    overall: Mapping[str, Any],
    per_language: Mapping[str, Mapping[str, float | int]],
) -> dict[str, bool]:
    """Evaluate legacy compatibility or preregistered development thresholds."""

    if dataset == "development":
        return {
            "raw_component_recall_at_least_candidate_recall": (
                overall["raw_component_recall"] >= overall["candidate_recall"]
            ),
            "candidate_recall_at_least_0_9167": (
                overall["candidate_recall"] >= 11 / 12
            ),
            "hit_rate_at_5_at_least_0_85": overall["hit_rate_at_5"] >= 0.85,
            "reranker_input_at_most_30": overall["candidate_count_maximum"] <= 30,
            "wrong_document_top1_zero": overall["wrong_document_top1_rate"] == 0.0,
            "document_context_complete": (
                overall["document_context_complete_rate"] == 1.0
            ),
        }
    return {
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
        "vietnamese_hit_rate_at_5_at_least_5_of_6": (
            per_language["vi"]["hit_rate_at_5"] >= 5 / 6
        ),
        "english_wrong_document_top5_at_most_5_of_30": (
            per_language["en"]["wrong_document_candidate_rate_at_5"] <= 5 / 30
        ),
        "vietnamese_wrong_document_top5_at_most_5_of_30": (
            per_language["vi"]["wrong_document_candidate_rate_at_5"] <= 5 / 30
        ),
        "calibration_010_rank_at_most_5": (
            _required_rank(rows, "phase7_calibration_010") <= 5
        ),
    }


def retrieval_failure_class(candidate_rank: int | None, final_rank: int | None) -> str:
    """Classify the first retrieval boundary that lost direct evidence."""

    if candidate_rank is None:
        return "candidate_miss"
    if final_rank is None or final_rank > 20:
        return "reranker_miss_top20"
    if final_rank > 5:
        return "reranker_miss_top5"
    return "hit"


def _required_rank(rows: list[dict[str, Any]], identifier: str) -> int:
    for row in rows:
        if row["id"] == identifier:
            return int(row["final_direct_evidence_rank"] or 2**31)
    raise ValueError(f"Required calibration row missing: {identifier}")


def aggregate_retrieval_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval, contamination, context, and latency metrics."""

    if not rows:
        raise ValueError("Retrieval evaluation requires at least one row.")
    candidate_ranks = [row["candidate_direct_evidence_rank"] for row in rows]
    final_ranks = [row["final_direct_evidence_rank"] for row in rows]
    exact_candidate_ranks = [
        row.get("candidate_exact_equivalent_evidence_rank", rank)
        for row, rank in zip(rows, candidate_ranks, strict=True)
    ]
    exact_final_ranks = [
        row.get("final_exact_equivalent_evidence_rank", rank)
        for row, rank in zip(rows, final_ranks, strict=True)
    ]
    raw_component_hits = [
        bool(row.get("raw_component_direct_evidence", rank is not None))
        for row, rank in zip(rows, candidate_ranks, strict=True)
    ]
    candidate_counts = [int(row["candidate_count"]) for row in rows]
    rerank_values = [float(row["rerank_ms"]) for row in rows]
    retrieval_values = [float(row["retrieval_ms"]) for row in rows]
    return {
        "query_count": len(rows),
        "raw_component_recall": sum(raw_component_hits) / len(rows),
        "candidate_recall": sum(rank is not None for rank in candidate_ranks) / len(rows),
        "candidate_exact_equivalent_recall": sum(
            rank is not None for rank in exact_candidate_ranks
        )
        / len(rows),
        "hit_rate_at_5": sum(rank is not None and rank <= 5 for rank in final_ranks)
        / len(rows),
        "exact_equivalent_hit_rate_at_5": sum(
            rank is not None and rank <= 5 for rank in exact_final_ranks
        )
        / len(rows),
        "mrr_at_5": sum(1 / rank if rank is not None and rank <= 5 else 0 for rank in final_ranks)
        / len(rows),
        "candidate_count_minimum": min(candidate_counts),
        "candidate_count_average": sum(candidate_counts) / len(candidate_counts),
        "candidate_count_maximum": max(candidate_counts),
        "wrong_document_top1_rate": sum(bool(row["wrong_document_top1"]) for row in rows)
        / len(rows),
        "wrong_document_candidate_rate_at_5": sum(
            int(row["wrong_document_candidate_count_at_5"]) for row in rows
        )
        / sum(min(int(row["final_candidate_count"]), 5) for row in rows),
        "document_context_complete_rate": sum(
            bool(row["document_context_complete"]) for row in rows
        )
        / len(rows),
        "failure_classes": dict(sorted(Counter(row["failure_class"] for row in rows).items())),
        "retrieval_latency_ms": _latency_summary(retrieval_values),
        "rerank_latency_ms": _latency_summary(rerank_values),
    }


def aggregate_retrieval_rows_by_language(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate closure coverage and contamination for each language."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["language"])].append(row)
    result: dict[str, dict[str, float | int]] = {}
    for language, group in sorted(groups.items()):
        total_top5 = sum(min(int(row["final_candidate_count"]), 5) for row in group)
        result[language] = {
            "query_count": len(group),
            "raw_component_recall": sum(
                bool(
                    row.get(
                        "raw_component_direct_evidence",
                        row["candidate_direct_evidence_rank"] is not None,
                    )
                )
                for row in group
            )
            / len(group),
            "candidate_recall": sum(
                row["candidate_direct_evidence_rank"] is not None for row in group
            )
            / len(group),
            "candidate_exact_equivalent_recall": sum(
                row.get(
                    "candidate_exact_equivalent_evidence_rank",
                    row["candidate_direct_evidence_rank"],
                )
                is not None
                for row in group
            )
            / len(group),
            "hit_rate_at_5": sum(
                row["final_direct_evidence_rank"] is not None
                and row["final_direct_evidence_rank"] <= 5
                for row in group
            )
            / len(group),
            "exact_equivalent_hit_rate_at_5": sum(
                row.get(
                    "final_exact_equivalent_evidence_rank",
                    row["final_direct_evidence_rank"],
                )
                is not None
                and row.get(
                    "final_exact_equivalent_evidence_rank",
                    row["final_direct_evidence_rank"],
                )
                <= 5
                for row in group
            )
            / len(group),
            "wrong_document_candidate_rate_at_5": sum(
                int(row["wrong_document_candidate_count_at_5"]) for row in group
            )
            / total_top5,
        }
    return result


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "average": sum(values) / len(values),
        "p50": percentile_nearest_rank(values, 50),
        "p95": percentile_nearest_rank(values, 95),
        "maximum": max(values),
    }


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

"""Application orchestration for retrieval candidate preparation and reranking."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal

from app.domain.content_identity import evidence_content_fingerprint
from app.domain.policies.fusion import (
    FusionPolicyError,
    FusionProfile,
    apply_role_aware_rank_fusion,
    fuse_rrf,
    fuse_weighted_rrf,
    select_coverage_preserving_candidates,
)
from app.domain.policies.list_completeness import (
    apply_list_completeness_fallback,
    apply_relation_list_completeness_fallback,
)
from app.domain.policies.query_roles import QueryRoleInference, infer_query_role
from app.domain.reranking import CrossEncoder, CrossEncoderScore, RerankingError
from app.domain.retrieval import (
    DenseSearchPort,
    RetrievalCandidate,
    RetrievalTrace,
    RetrievedChunk,
    SparseSearchPort,
    dense_results_to_candidates,
    union_dense_sparse_candidates,
)

RerankStrategy = Literal["sparse", "hybrid", "union"]
CANDIDATE_TEXT_FORMAT = "document_context_heading_content_v2"


@dataclass(frozen=True)
class CandidatePool:
    """Pre-rerank pool plus independently measured retrieval-stage latencies."""

    candidates: list[RetrievalCandidate]
    stage_latency_ms: dict[str, float]
    trace: RetrievalTrace | None = None


@dataclass(frozen=True)
class RerankExecution:
    """One full query execution before and after cross-encoder ordering."""

    candidates_before_rerank: list[RetrievalCandidate]
    candidates_after_rerank: list[RetrievalCandidate]
    stage_latency_ms: dict[str, float]
    trace: RetrievalTrace | None = None


class RerankPipeline:
    """Retrieve one configured pool, preserve stage timings, then rerank it."""

    def __init__(
        self,
        *,
        dense_searcher: DenseSearchPort,
        sparse_searcher: SparseSearchPort,
        cross_encoder: CrossEncoder,
        dense_collection: str,
        hybrid_collection: str,
        dense_candidate_limit: int = 20,
        sparse_candidate_limit: int = 20,
        rrf_k: int = 60,
        rerank_batch_size: int = 16,
        deduplicate_content: bool = False,
        refill_after_content_dedup: bool = False,
        document_contexts: Mapping[str, Mapping[str, str]] | None = None,
        sparse_query_transform: Callable[[str], str] | None = None,
        union_rrf_prune_limit: int | None = None,
        fusion_profile: FusionProfile | None = None,
        query_role_inferer: Callable[[str], QueryRoleInference] = infer_query_role,
        capture_trace: bool = False,
        diagnostic_component_limit: int | None = None,
    ) -> None:
        self.dense_searcher = dense_searcher
        self.sparse_searcher = sparse_searcher
        self.cross_encoder = cross_encoder
        self.dense_collection = dense_collection
        self.hybrid_collection = hybrid_collection
        self.dense_candidate_limit = dense_candidate_limit
        self.sparse_candidate_limit = sparse_candidate_limit
        self.rrf_k = rrf_k
        self.rerank_batch_size = rerank_batch_size
        self.deduplicate_content = deduplicate_content
        if refill_after_content_dedup and not deduplicate_content:
            raise RerankingError(
                "Candidate refill requires exact-content deduplication to be enabled."
            )
        self.refill_after_content_dedup = refill_after_content_dedup
        self.document_contexts = dict(document_contexts or {})
        self.sparse_query_transform = sparse_query_transform
        self.union_rrf_prune_limit = union_rrf_prune_limit
        self.fusion_profile = fusion_profile
        self.query_role_inferer = query_role_inferer
        self.capture_trace = capture_trace
        if diagnostic_component_limit is not None:
            if not capture_trace:
                raise RerankingError(
                    "Diagnostic component retrieval requires trace capture to be enabled."
                )
            if diagnostic_component_limit <= max(
                dense_candidate_limit, sparse_candidate_limit
            ):
                raise RerankingError(
                    "Diagnostic component limit must exceed both normal component limits."
                )
        self.diagnostic_component_limit = diagnostic_component_limit

    def search(
        self, question: str, *, strategy: RerankStrategy, document_id: str | None = None
    ) -> RerankExecution:
        pool = self.prepare_pool(question, strategy=strategy, document_id=document_id)
        execution = execute_rerank(
            question,
            pool=pool,
            cross_encoder=self.cross_encoder,
            strategy=strategy,
            batch_size=self.rerank_batch_size,
        )
        if strategy != "union" or self.fusion_profile is None:
            return execution
        inference_started = perf_counter()
        inference = self.query_role_inferer(question)
        try:
            candidates = apply_role_aware_rank_fusion(
                execution.candidates_after_rerank,
                query_role=inference.role,
                role_multiplier=self.fusion_profile.post_rerank_role_multiplier,
                rrf_rank_multiplier=self.fusion_profile.post_rerank_rrf_multiplier,
                rank_offset=self.fusion_profile.post_rerank_rank_offset,
                confidence=inference.confidence,
                confidence_mode=self.fusion_profile.post_rerank_confidence_mode,
            )
            if self.fusion_profile.list_completeness_enabled:
                candidates = apply_list_completeness_fallback(candidates, query=question)
            if self.fusion_profile.relation_list_completeness_enabled:
                candidates = apply_relation_list_completeness_fallback(
                    candidates,
                    query=question,
                )
        except FusionPolicyError as exc:
            raise RerankingError(str(exc)) from exc
        stages = dict(execution.stage_latency_ms)
        stages["role_aware_rank_fusion"] = (perf_counter() - inference_started) * 1000
        trace = (
            replace(execution.trace, final_candidates=tuple(candidates))
            if execution.trace is not None
            else None
        )
        return RerankExecution(
            execution.candidates_before_rerank,
            candidates,
            stages,
            trace,
        )

    def prepare_pool(
        self, question: str, *, strategy: RerankStrategy, document_id: str | None = None
    ) -> CandidatePool:
        """Retrieve exactly the inputs required by a strategy with document filtering intact."""

        if strategy not in ("sparse", "hybrid", "union"):
            raise RerankingError(f"Unsupported rerank candidate strategy: {strategy}")
        stages: dict[str, float] = {}
        dense_results: list[RetrievedChunk] = []
        if strategy in ("hybrid", "union"):
            dense_started = perf_counter()
            dense_results = self.dense_searcher.search(
                question,
                collection_name=(
                    self.hybrid_collection if strategy == "hybrid" else self.dense_collection
                ),
                limit=self.dense_candidate_limit,
                document_id=document_id,
            )
            stages["dense_retrieval"] = (perf_counter() - dense_started) * 1000

        sparse_query = question
        if self.sparse_query_transform is not None:
            expansion_started = perf_counter()
            sparse_query = self.sparse_query_transform(question)
            stages["query_expansion"] = (perf_counter() - expansion_started) * 1000
        sparse_started = perf_counter()
        sparse_candidates = self.sparse_searcher.search(
            sparse_query,
            collection_name=self.hybrid_collection,
            limit=self.sparse_candidate_limit,
            document_id=document_id,
        )
        stages["sparse_retrieval"] = (perf_counter() - sparse_started) * 1000

        diagnostic_dense_results = dense_results
        diagnostic_sparse_candidates = sparse_candidates
        if self.diagnostic_component_limit is not None:
            if strategy in ("hybrid", "union"):
                diagnostic_dense_started = perf_counter()
                diagnostic_dense_results = self.dense_searcher.search(
                    question,
                    collection_name=(
                        self.hybrid_collection
                        if strategy == "hybrid"
                        else self.dense_collection
                    ),
                    limit=self.diagnostic_component_limit,
                    document_id=document_id,
                )
                stages["diagnostic_dense_retrieval"] = (
                    perf_counter() - diagnostic_dense_started
                ) * 1000
            diagnostic_sparse_started = perf_counter()
            diagnostic_sparse_candidates = self.sparse_searcher.search(
                sparse_query,
                collection_name=self.hybrid_collection,
                limit=self.diagnostic_component_limit,
                document_id=document_id,
            )
            stages["diagnostic_sparse_retrieval"] = (
                perf_counter() - diagnostic_sparse_started
            ) * 1000

        preparation_started = perf_counter()
        pending_prune_limit: int | None = None
        dense_candidates = [
            _with_document_context(candidate, self.document_contexts.get(candidate.document_id))
            for candidate in dense_results_to_candidates(dense_results)
        ]
        contextual_sparse_candidates = [
            _with_document_context(candidate, self.document_contexts.get(candidate.document_id))
            for candidate in sparse_candidates
        ]
        diagnostic_dense_candidates = dense_candidates
        contextual_diagnostic_sparse_candidates = contextual_sparse_candidates
        fused_candidates: list[RetrievalCandidate] = []
        if strategy == "union" and self.fusion_profile is not None:
            role_started = perf_counter()
            query_role = self.query_role_inferer(question)
            stages["query_role_inference"] = (perf_counter() - role_started) * 1000
            try:
                if self.capture_trace or self.refill_after_content_dedup:
                    fused_candidates = fuse_weighted_rrf(
                        dense_candidates,
                        contextual_sparse_candidates,
                        profile=self.fusion_profile,
                        query_role=query_role.role,
                    )
                candidates = select_coverage_preserving_candidates(
                    dense_candidates,
                    contextual_sparse_candidates,
                    profile=self.fusion_profile,
                    query_role=query_role.role,
                    fused_candidates=(
                        fused_candidates
                        if self.capture_trace or self.refill_after_content_dedup
                        else None
                    ),
                )
            except FusionPolicyError as exc:
                raise RerankingError(str(exc)) from exc
            preparation_stage = "coverage_preserving_weighted_rrf"
        elif strategy == "union" and self.union_rrf_prune_limit is not None:
            if self.union_rrf_prune_limit <= 0:
                raise RerankingError("Union RRF prune limit must be greater than 0.")
            candidates = fuse_rrf(
                dense_candidates,
                contextual_sparse_candidates,
                rrf_k=self.rrf_k,
                final_limit=len(dense_results) + len(sparse_candidates),
            )
            if self.capture_trace:
                fused_candidates = list(candidates)
            pending_prune_limit = self.union_rrf_prune_limit
            preparation_stage = "rrf_pruning"
        else:
            candidates = build_candidate_pool(
                strategy,
                dense_results=dense_results,
                sparse_candidates=contextual_sparse_candidates,
                rrf_k=self.rrf_k,
                hybrid_limit=max(self.dense_candidate_limit, self.sparse_candidate_limit),
            )
            if self.capture_trace and strategy != "sparse":
                fused_candidates = list(candidates)
            preparation_stage = "fusion" if strategy == "hybrid" else "union_preparation"
        if not (strategy == "union" and self.fusion_profile is not None):
            candidates = [
                _with_document_context(candidate, self.document_contexts.get(candidate.document_id))
                for candidate in candidates
            ]
        stages[preparation_stage] = (perf_counter() - preparation_started) * 1000
        if self.diagnostic_component_limit is not None:
            diagnostic_context_started = perf_counter()
            diagnostic_dense_candidates = [
                _with_document_context(
                    candidate,
                    self.document_contexts.get(candidate.document_id),
                )
                for candidate in dense_results_to_candidates(diagnostic_dense_results)
            ]
            contextual_diagnostic_sparse_candidates = [
                _with_document_context(
                    candidate,
                    self.document_contexts.get(candidate.document_id),
                )
                for candidate in diagnostic_sparse_candidates
            ]
            stages["diagnostic_candidate_context"] = (
                perf_counter() - diagnostic_context_started
            ) * 1000
        selected_candidate_ids = {candidate.chunk_id for candidate in candidates}
        selected_candidates_pre_dedup = list(candidates) if self.capture_trace else []
        refilled_candidate_ids: tuple[str, ...] = ()
        if self.deduplicate_content:
            deduplication_started = perf_counter()
            candidates = deduplicate_candidates_by_content(candidates)
            stages["content_deduplication"] = (
                perf_counter() - deduplication_started
            ) * 1000
            candidates_post_dedup = list(candidates) if self.capture_trace else []
            if (
                self.refill_after_content_dedup
                and strategy == "union"
                and self.fusion_profile is not None
            ):
                refill_started = perf_counter()
                candidates, refilled_candidate_ids = refill_candidates_after_content_dedup(
                    candidates,
                    selected_candidate_ids=selected_candidate_ids,
                    fused_candidates=fused_candidates,
                    max_candidates=self.fusion_profile.max_candidates,
                )
                stages["candidate_refill"] = (perf_counter() - refill_started) * 1000
        else:
            candidates_post_dedup = list(candidates) if self.capture_trace else []
        if pending_prune_limit is not None:
            candidates = candidates[:pending_prune_limit]
        if strategy == "sparse" and "union_preparation" in stages:
            stages.pop("union_preparation")
        trace = (
            RetrievalTrace(
                dense_candidates=tuple(diagnostic_dense_candidates),
                sparse_candidates=tuple(contextual_diagnostic_sparse_candidates),
                fused_candidates=tuple(fused_candidates),
                selected_candidates_pre_dedup=tuple(selected_candidates_pre_dedup),
                candidate_pool_post_dedup=tuple(candidates_post_dedup),
                candidates_before_rerank=tuple(candidates),
                refilled_candidate_ids=refilled_candidate_ids,
                candidate_count_pre_dedup=len(selected_candidates_pre_dedup),
                candidate_count_post_dedup=len(candidates_post_dedup),
                candidate_count_before_rerank=len(candidates),
            )
            if self.capture_trace
            else None
        )
        return CandidatePool(candidates, stages, trace)


def build_candidate_text(candidate: RetrievalCandidate) -> str:
    """Build trusted document context, heading breadcrumb, and raw-content input."""

    heading = " > ".join(value.strip() for value in candidate.headings if value.strip())
    context: list[str] = []
    title = candidate.metadata.get("document_title")
    role = candidate.metadata.get("document_role")
    if isinstance(title, str) and title.strip():
        context.append(f"Document title: {title.strip()}")
    if isinstance(role, str) and role.strip():
        context.append(f"Document role: {role.strip()}")
    sections = ["\n".join(context)] if context else []
    if heading:
        sections.append(heading)
    sections.append(candidate.text)
    return "\n\n".join(sections)


def _with_document_context(
    candidate: RetrievalCandidate, context: Mapping[str, str] | None
) -> RetrievalCandidate:
    if context is None:
        return candidate
    title = str(context.get("document_title", "")).strip()
    role = str(context.get("document_role", "")).strip()
    if not title or role not in {"installation", "programming"}:
        raise RerankingError(
            f"Invalid trusted document context for candidate {candidate.chunk_id}."
        )
    metadata = dict(candidate.metadata)
    metadata.update({"document_title": title, "document_role": role})
    return candidate.model_copy(update={"metadata": metadata})


def build_candidate_pool(
    strategy: RerankStrategy,
    *,
    dense_results: Sequence[RetrievedChunk] = (),
    sparse_candidates: Sequence[RetrievalCandidate] = (),
    rrf_k: int = 60,
    hybrid_limit: int = 20,
) -> list[RetrievalCandidate]:
    """Construct one deterministic Phase 5 pool from existing retrieval outputs."""

    dense_candidates = dense_results_to_candidates(dense_results)
    if strategy == "sparse":
        return list(sparse_candidates)
    if strategy == "hybrid":
        return fuse_rrf(
            dense_candidates,
            sparse_candidates,
            rrf_k=rrf_k,
            final_limit=hybrid_limit,
        )
    if strategy == "union":
        return union_dense_sparse_candidates(dense_candidates, sparse_candidates)
    raise RerankingError(f"Unsupported rerank candidate strategy: {strategy}")


def deduplicate_candidates_by_content(
    candidates: Sequence[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    """Collapse exact-normalized text duplicates while preserving provenance IDs."""

    groups: dict[tuple[str, str], list[RetrievalCandidate]] = {}
    order: list[tuple[str, str]] = []
    for candidate in candidates:
        fingerprint = evidence_content_fingerprint(candidate.text)
        key = (candidate.document_id, fingerprint)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(candidate)

    deduplicated: list[RetrievalCandidate] = []
    for document_id, fingerprint in order:
        equivalents = groups[(document_id, fingerprint)]
        representative = equivalents[0]
        equivalent_ids = sorted({candidate.chunk_id for candidate in equivalents})
        metadata = dict(representative.metadata)
        metadata.update(
            {
                "content_fingerprint": fingerprint,
                "equivalent_chunk_ids": equivalent_ids,
                "equivalent_chunk_count": len(equivalent_ids),
            }
        )
        deduplicated.append(
            representative.model_copy(
                update={
                    "metadata": metadata,
                    "dense_rank": _minimum_optional(
                        candidate.dense_rank for candidate in equivalents
                    ),
                    "dense_score": _maximum_optional(
                        candidate.dense_score for candidate in equivalents
                    ),
                    "sparse_rank": _minimum_optional(
                        candidate.sparse_rank for candidate in equivalents
                    ),
                    "sparse_score": _maximum_optional(
                        candidate.sparse_score for candidate in equivalents
                    ),
                    "rrf_rank": _minimum_optional(
                        candidate.rrf_rank for candidate in equivalents
                    ),
                    "rrf_score": _maximum_optional(
                        candidate.rrf_score for candidate in equivalents
                    ),
                }
            )
        )
    return deduplicated


def refill_candidates_after_content_dedup(
    candidates: Sequence[RetrievalCandidate],
    *,
    selected_candidate_ids: set[str],
    fused_candidates: Sequence[RetrievalCandidate],
    max_candidates: int,
) -> tuple[list[RetrievalCandidate], tuple[str, ...]]:
    """Fill deduplication vacancies from the deterministic weighted-RRF tail.

    Content identity remains document-scoped: equal text from different manuals
    preserves both provenance choices, while an equal-text candidate from the same
    manual cannot consume another reranker slot.
    """

    if max_candidates <= 0:
        raise RerankingError("Refill candidate budget must be greater than 0.")
    if len(candidates) > max_candidates:
        raise RerankingError("Deduplicated candidate pool exceeds the refill budget.")
    if any(candidate.rrf_rank is None for candidate in fused_candidates):
        raise RerankingError("Refill candidates require one-based RRF ranks.")

    refilled = list(candidates)
    content_keys = {
        (candidate.document_id, evidence_content_fingerprint(candidate.text))
        for candidate in refilled
    }
    refilled_candidate_ids: list[str] = []
    fused_tail = sorted(
        fused_candidates,
        key=lambda candidate: (candidate.rrf_rank or 2**31, candidate.chunk_id),
    )
    for candidate in fused_tail:
        if len(refilled) >= max_candidates:
            break
        if candidate.chunk_id in selected_candidate_ids:
            continue
        content_key = (
            candidate.document_id,
            evidence_content_fingerprint(candidate.text),
        )
        if content_key in content_keys:
            continue
        normalized_candidate = deduplicate_candidates_by_content((candidate,))[0]
        refilled.append(normalized_candidate)
        content_keys.add(content_key)
        refilled_candidate_ids.append(candidate.chunk_id)

    refilled.sort(
        key=lambda candidate: (candidate.rrf_rank or 2**31, candidate.chunk_id)
    )
    return refilled, tuple(refilled_candidate_ids)


def rerank_candidates(
    query: str,
    candidates: Sequence[RetrievalCandidate],
    cross_encoder: CrossEncoder,
    *,
    strategy: RerankStrategy,
    batch_size: int,
) -> list[RetrievalCandidate]:
    """Return the full candidate pool in deterministic cross-encoder order."""

    normalized_query = query.strip()
    if not normalized_query:
        raise RerankingError("Rerank query must not be empty.")
    if batch_size <= 0:
        raise RerankingError("Rerank batch size must be greater than 0.")
    if not candidates:
        return []
    ids = [candidate.chunk_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise RerankingError("Rerank candidate chunk IDs must be unique.")

    previous_ranks = [
        _previous_rank(candidate, strategy=strategy, union_rank=index)
        for index, candidate in enumerate(candidates, start=1)
    ]
    documents = [build_candidate_text(candidate) for candidate in candidates]
    try:
        outputs = list(cross_encoder.score(normalized_query, documents, batch_size=batch_size))
    except RerankingError:
        raise
    except Exception as exc:
        raise RerankingError(f"Cross-encoder inference failed: {exc}") from exc
    scores = _validate_model_output(outputs, candidate_count=len(candidates))

    ordered_indices = sorted(
        range(len(candidates)),
        key=lambda index: (-scores[index], previous_ranks[index], candidates[index].chunk_id),
    )
    return [
        candidates[index].model_copy(
            update={
                "score": scores[index],
                "rerank_score": scores[index],
                "rerank_rank": rank,
            }
        )
        for rank, index in enumerate(ordered_indices, start=1)
    ]


def execute_rerank(
    question: str,
    *,
    pool: CandidatePool,
    cross_encoder: CrossEncoder,
    strategy: RerankStrategy,
    batch_size: int,
) -> RerankExecution:
    """Rerank a prepared pool while retaining stage timing and all candidates."""

    started = perf_counter()
    reranked = rerank_candidates(
        question,
        pool.candidates,
        cross_encoder,
        strategy=strategy,
        batch_size=batch_size,
    )
    rerank_ms = (perf_counter() - started) * 1000
    stages = {**pool.stage_latency_ms, "rerank": rerank_ms}
    stages["total"] = sum(stages.values())
    trace = (
        replace(
            pool.trace,
            raw_cross_encoder_candidates=tuple(reranked),
            final_candidates=tuple(reranked),
        )
        if pool.trace is not None
        else None
    )
    return RerankExecution(list(pool.candidates), reranked, stages, trace)


def _previous_rank(
    candidate: RetrievalCandidate, *, strategy: RerankStrategy, union_rank: int
) -> int:
    if strategy == "sparse":
        if candidate.sparse_rank is None:
            raise RerankingError("Sparse rerank candidate is missing sparse_rank.")
        return candidate.sparse_rank
    if strategy == "hybrid":
        if candidate.rrf_rank is None:
            raise RerankingError("Hybrid rerank candidate is missing rrf_rank.")
        return candidate.rrf_rank
    if strategy == "union":
        return union_rank
    raise RerankingError(f"Unsupported rerank candidate strategy: {strategy}")


def _validate_model_output(
    outputs: Sequence[CrossEncoderScore], *, candidate_count: int
) -> list[float]:
    if len(outputs) != candidate_count:
        raise RerankingError(
            f"Cross-encoder returned {len(outputs)} scores for {candidate_count} candidates."
        )
    scores: list[float | None] = [None] * candidate_count
    for output in outputs:
        index = output.candidate_index
        if not isinstance(index, int) or not 0 <= index < candidate_count:
            raise RerankingError(f"Cross-encoder returned invalid candidate index: {index}.")
        if scores[index] is not None:
            raise RerankingError(f"Cross-encoder returned duplicate candidate index: {index}.")
        score = float(output.score)
        if not math.isfinite(score):
            raise RerankingError("Cross-encoder scores must be finite.")
        scores[index] = score
    if any(score is None for score in scores):
        raise RerankingError("Cross-encoder output is missing one or more candidate indices.")
    return [float(score) for score in scores]


def _minimum_optional(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _maximum_optional(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None

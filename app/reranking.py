"""Multilingual cross-encoder reranking with explicit candidate-pool diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Protocol

from app.candidate_audit import dense_results_to_candidates, union_dense_sparse_candidates
from app.content_identity import evidence_content_fingerprint
from app.evaluation import (
    EvaluationCase,
    EvaluationError,
    diagnostic_page_rank,
    diagnostic_phrase_rank,
    direct_evidence_rank,
    percentile_nearest_rank,
)
from app.hybrid_retrieval import fuse_rrf, sparse_search
from app.models import RetrievalCandidate, RetrievedChunk
from app.retrieval import dense_search

RerankStrategy = Literal["sparse", "hybrid", "union"]
FailureClass = Literal["candidate_miss", "reranker_miss_top5", "reranker_miss_top20", "hit"]
CANDIDATE_TEXT_FORMAT = "heading_content_v1"


class RerankingError(ValueError):
    """Raised when candidate construction or cross-encoder output is invalid."""


@dataclass(frozen=True)
class CrossEncoderScore:
    """One model score mapped to the corresponding input candidate index."""

    candidate_index: int
    score: float


class CrossEncoder(Protocol):
    """Dependency-injection boundary used by real and fake cross-encoders."""

    def score(
        self, query: str, documents: Sequence[str], *, batch_size: int
    ) -> Iterable[CrossEncoderScore]: ...


class FastEmbedCrossEncoder:
    """Lazy adapter for FastEmbed 0.8 TextCrossEncoder's input-ordered scores."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._model: Any | None = None

    def score(
        self, query: str, documents: Sequence[str], *, batch_size: int
    ) -> Iterable[CrossEncoderScore]:
        try:
            model = self._get_model()
            scores = model.rerank(query, documents, batch_size=batch_size)
            return [
                CrossEncoderScore(candidate_index=index, score=float(score))
                for index, score in enumerate(scores)
            ]
        except RerankingError:
            raise
        except Exception as exc:
            raise RerankingError(f"Cross-encoder inference failed: {exc}") from exc

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._model = TextCrossEncoder(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir,
                    threads=self.threads,
                    cuda=False,
                    lazy_load=True,
                )
            except Exception as exc:
                raise RerankingError(f"Unable to initialize cross-encoder: {exc}") from exc
        return self._model


@dataclass(frozen=True)
class CandidatePool:
    """Pre-rerank pool plus independently measured retrieval-stage latencies."""

    candidates: list[RetrievalCandidate]
    stage_latency_ms: dict[str, float]


@dataclass(frozen=True)
class RerankExecution:
    """One full query execution before and after cross-encoder ordering."""

    candidates_before_rerank: list[RetrievalCandidate]
    candidates_after_rerank: list[RetrievalCandidate]
    stage_latency_ms: dict[str, float]


class RerankPipeline:
    """Retrieve one configured pool, preserve stage timings, then rerank it."""

    def __init__(
        self,
        *,
        client: Any,
        dense_embedding_model: Any,
        sparse_embedding_model: Any,
        cross_encoder: CrossEncoder,
        dense_collection: str,
        hybrid_collection: str,
        dense_vector_name: str,
        sparse_vector_name: str,
        dense_candidate_limit: int = 20,
        sparse_candidate_limit: int = 20,
        rrf_k: int = 60,
        rerank_batch_size: int = 16,
        deduplicate_content: bool = False,
        dense_search_fn: Callable[..., list[RetrievedChunk]] = dense_search,
        sparse_search_fn: Callable[..., list[RetrievalCandidate]] = sparse_search,
    ) -> None:
        self.client = client
        self.dense_embedding_model = dense_embedding_model
        self.sparse_embedding_model = sparse_embedding_model
        self.cross_encoder = cross_encoder
        self.dense_collection = dense_collection
        self.hybrid_collection = hybrid_collection
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.dense_candidate_limit = dense_candidate_limit
        self.sparse_candidate_limit = sparse_candidate_limit
        self.rrf_k = rrf_k
        self.rerank_batch_size = rerank_batch_size
        self.deduplicate_content = deduplicate_content
        self.dense_search_fn = dense_search_fn
        self.sparse_search_fn = sparse_search_fn

    def search(
        self, question: str, *, strategy: RerankStrategy, document_id: str | None = None
    ) -> RerankExecution:
        pool = self.prepare_pool(question, strategy=strategy, document_id=document_id)
        return execute_rerank(
            question,
            pool=pool,
            cross_encoder=self.cross_encoder,
            strategy=strategy,
            batch_size=self.rerank_batch_size,
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
            dense_results = self.dense_search_fn(
                self.client,
                question,
                collection_name=(
                    self.hybrid_collection if strategy == "hybrid" else self.dense_collection
                ),
                vector_name=self.dense_vector_name,
                embedding_model=self.dense_embedding_model,
                limit=self.dense_candidate_limit,
                document_id=document_id,
            )
            stages["dense_retrieval"] = (perf_counter() - dense_started) * 1000

        sparse_started = perf_counter()
        sparse_candidates = self.sparse_search_fn(
            self.client,
            question,
            collection_name=self.hybrid_collection,
            sparse_vector_name=self.sparse_vector_name,
            sparse_embedding_model=self.sparse_embedding_model,
            limit=self.sparse_candidate_limit,
            document_id=document_id,
        )
        stages["sparse_retrieval"] = (perf_counter() - sparse_started) * 1000

        preparation_started = perf_counter()
        candidates = build_candidate_pool(
            strategy,
            dense_results=dense_results,
            sparse_candidates=sparse_candidates,
            rrf_k=self.rrf_k,
            hybrid_limit=max(self.dense_candidate_limit, self.sparse_candidate_limit),
        )
        stages["fusion" if strategy == "hybrid" else "union_preparation"] = (
            perf_counter() - preparation_started
        ) * 1000
        if self.deduplicate_content:
            deduplication_started = perf_counter()
            candidates = deduplicate_candidates_by_content(candidates)
            stages["content_deduplication"] = (
                perf_counter() - deduplication_started
            ) * 1000
        if strategy == "sparse":
            stages.pop("union_preparation")
        return CandidatePool(candidates, stages)


def fastembed_model_metadata(model_name: str) -> dict[str, Any]:
    """Return FastEmbed's local registry metadata without constructing/downloading a model."""

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        metadata = next(
            (
                item
                for item in TextCrossEncoder.list_supported_models()
                if item.get("model", "").casefold() == model_name.casefold()
            ),
            None,
        )
    except Exception as exc:
        raise RerankingError(f"Unable to inspect FastEmbed cross-encoder registry: {exc}") from exc
    if metadata is None:
        raise RerankingError(f"FastEmbed does not support rerank model: {model_name}")
    return metadata


def build_candidate_text(candidate: RetrievalCandidate) -> str:
    """Build the frozen heading-breadcrumb plus raw-content reranker input."""

    heading = " > ".join(value.strip() for value in candidate.headings if value.strip())
    return f"{heading}\n\n{candidate.text}" if heading else candidate.text


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
                    "rrf_rank": _minimum_optional(candidate.rrf_rank for candidate in equivalents),
                    "rrf_score": _maximum_optional(
                        candidate.rrf_score for candidate in equivalents
                    ),
                }
            )
        )
    return deduplicated


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
    return RerankExecution(list(pool.candidates), reranked, stages)


def evaluate_reranked_cases(
    cases: Sequence[EvaluationCase],
    search: Callable[[str, str], RerankExecution],
    *,
    cutoff: int = 20,
) -> dict[str, Any]:
    """Evaluate final ranks separately from pre-rerank candidate availability."""

    if not cases:
        raise EvaluationError("Cannot evaluate an empty case list.")
    if cutoff < 5:
        raise EvaluationError("Rerank evaluation cutoff must be at least 5.")
    rows = [_evaluate_rerank_case(case, search(case.question, case.document_id)) for case in cases]
    return {
        "candidate_limit": cutoff,
        "metric_definitions": {
            "candidate_recall": "direct evidence present anywhere in the full pre-rerank pool",
            "hit_mrr_at_5": "computed on final reranked top 5",
            "hit_mrr_at_20": "computed on final reranked top 20",
        },
        "overall": aggregate_rerank_rows(rows, cutoff=cutoff),
        "per_language": _aggregate_rerank_groups(rows, "language", cutoff),
        "per_retrieval_scenario": _aggregate_rerank_groups(rows, "retrieval_scenario", cutoff),
        "critical_questions": [row for row in rows if row["critical"]],
        "critical_metrics": aggregate_rerank_rows(
            [row for row in rows if row["critical"]], cutoff=cutoff
        ),
        "failure_cases": [row for row in rows if row["failure_class"] != "hit"],
        "per_query": rows,
    }


def classify_rerank_failure(*, candidate_rank: int | None, final_rank: int | None) -> FailureClass:
    """Distinguish candidate absence from cross-encoder ordering failures."""

    if candidate_rank is None:
        return "candidate_miss"
    if final_rank is not None and final_rank <= 5:
        return "hit"
    if final_rank is not None and final_rank <= 20:
        return "reranker_miss_top5"
    return "reranker_miss_top20"


def aggregate_rerank_rows(rows: Sequence[dict[str, Any]], *, cutoff: int) -> dict[str, Any]:
    """Aggregate final quality, candidate coverage, failure classes, and stage latency."""

    if not rows:
        raise EvaluationError("Cannot aggregate an empty rerank result set.")
    ranks = [row["direct_evidence_rank"] for row in rows]
    candidate_ranks = [row["candidate_evidence_rank"] for row in rows]
    result: dict[str, Any] = {
        "query_count": len(rows),
        "hit_rate_at_1": _hit_rate(ranks, 1),
        "hit_rate_at_3": _hit_rate(ranks, 3),
        "hit_rate_at_5": _hit_rate(ranks, 5),
        "hit_rate_at_20": _hit_rate(ranks, cutoff),
        "candidate_recall": _hit_rate(candidate_ranks, max(row["candidate_count"] for row in rows)),
        "mrr_at_5": _mrr(ranks, 5),
        "mrr_at_20": _mrr(ranks, cutoff),
        "failure_counts": {
            name: sum(row["failure_class"] == name for row in rows)
            for name in ("candidate_miss", "reranker_miss_top5", "reranker_miss_top20", "hit")
        },
        "candidate_count": {
            "minimum": min(row["candidate_count"] for row in rows),
            "maximum": max(row["candidate_count"] for row in rows),
            "average": sum(row["candidate_count"] for row in rows) / len(rows),
        },
    }
    stage_names = sorted({name for row in rows for name in row["stage_latency_ms"]})
    result["stage_latency_ms"] = {
        name: _latency_summary([row["stage_latency_ms"].get(name, 0.0) for row in rows])
        for name in stage_names
    }
    return result


def _evaluate_rerank_case(case: EvaluationCase, execution: RerankExecution) -> dict[str, Any]:
    relevant = set(case.relevant_chunk_ids)
    candidate_rank = direct_evidence_rank(execution.candidates_before_rerank, relevant)
    final_rank = direct_evidence_rank(execution.candidates_after_rerank, relevant)
    return {
        "id": case.id,
        "language": case.language,
        "document_language": case.document_language,
        "retrieval_scenario": case.retrieval_scenario,
        "category": case.category,
        "critical": case.critical,
        "question": case.question,
        "document_id": case.document_id,
        "relevant_chunk_ids": case.relevant_chunk_ids,
        "expected_pages": case.expected_pages,
        "candidate_count": len(execution.candidates_before_rerank),
        "candidate_chunk_ids": [item.chunk_id for item in execution.candidates_before_rerank],
        "candidate_evidence_rank": candidate_rank,
        "direct_evidence_rank": final_rank,
        "failure_class": classify_rerank_failure(
            candidate_rank=candidate_rank, final_rank=final_rank
        ),
        "diagnostic_phrase_rank": diagnostic_phrase_rank(
            execution.candidates_after_rerank, case.expected_phrases
        ),
        "diagnostic_page_rank": diagnostic_page_rank(
            execution.candidates_after_rerank, set(case.expected_pages)
        ),
        "stage_latency_ms": execution.stage_latency_ms,
        "retrieved": [
            _candidate_summary(candidate) for candidate in execution.candidates_after_rerank
        ],
    }


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


def _candidate_summary(candidate: RetrievalCandidate) -> dict[str, Any]:
    return {
        "rerank_rank": candidate.rerank_rank,
        "chunk_id": candidate.chunk_id,
        "document_id": candidate.document_id,
        "page_numbers": candidate.page_numbers,
        "headings": candidate.headings,
        "rerank_score": candidate.rerank_score,
        "dense_rank": candidate.dense_rank,
        "dense_score": candidate.dense_score,
        "sparse_rank": candidate.sparse_rank,
        "sparse_score": candidate.sparse_score,
        "rrf_rank": candidate.rrf_rank,
        "rrf_score": candidate.rrf_score,
    }


def _aggregate_rerank_groups(
    rows: Sequence[dict[str, Any]], key: str, cutoff: int
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        name: aggregate_rerank_rows(group, cutoff=cutoff) for name, group in sorted(groups.items())
    }


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "average": sum(values) / len(values),
        "p50": percentile_nearest_rank(values, 50),
        "p95": percentile_nearest_rank(values, 95),
    }


def _hit_rate(ranks: Sequence[int | None], cutoff: int) -> float:
    return sum(rank is not None and rank <= cutoff for rank in ranks) / len(ranks)


def _mrr(ranks: Sequence[int | None], cutoff: int) -> float:
    return sum(1 / rank if rank is not None and rank <= cutoff else 0.0 for rank in ranks) / len(
        ranks
    )


def _minimum_optional(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _maximum_optional(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None

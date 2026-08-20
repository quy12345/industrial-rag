"""Orchestration for retrieval, evidence gating, generation, and citations."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Literal

from app.citations import build_citations, validate_generated_answer
from app.config import Settings, get_settings
from app.errors import CitationValidationError, GenerationValidationError, LLMRefusalError
from app.evidence_selection import (
    EvidenceDuplicateGroup,
    EvidenceSelectionError,
    select_evidence_candidates,
)
from app.generation import (
    AnswerGenerator,
    LangChainOpenAIGenerator,
    TokenUsage,
    format_evidence,
)
from app.models import QueryResponse, RetrievalCandidate
from app.request_context import request_id
from app.retrieval_runtime import (
    LazyQueryRetriever,
    QueryRetriever,
    build_query_retriever,
    resolve_retrieval_runtime,
)

logger = logging.getLogger(__name__)

AbstentionReason = Literal[
    "no_candidates",
    "invalid_candidate_metadata",
    "configured_score_gate_failed",
    "llm_insufficient_evidence",
    "llm_refusal",
    "citation_validation_failed",
]

ABSTENTION_MESSAGE = (
    "Không đủ bằng chứng trong tài liệu / Insufficient evidence in the supplied document."
)


@dataclass(frozen=True)
class EvidenceGateDecision:
    """Deterministic pre-generation decision over final retrieved evidence."""

    passed: bool
    reason: AbstentionReason | None = None


@dataclass(frozen=True)
class QueryTimings:
    """Internal stage timings; these are not exposed by the public API."""

    retrieval_ms: float
    rerank_ms: float
    evidence_gate_ms: float
    generation_ms: float
    citation_validation_ms: float
    total_ms: float


@dataclass(frozen=True)
class QueryExecution:
    """Public response plus sanitized internal diagnostics for smoke validation."""

    response: QueryResponse
    timings: QueryTimings
    usage: TokenUsage | None
    candidates: tuple[RetrievalCandidate, ...] = ()
    candidate_pool: tuple[RetrievalCandidate, ...] = ()
    evidence_candidates: tuple[RetrievalCandidate, ...] = ()
    evidence_duplicate_groups: tuple[EvidenceDuplicateGroup, ...] = ()
    generation_attempts: int = 0


class EvidenceGate:
    """Reject absent or malformed candidate sets before any document leaves the service."""

    def __init__(self, *, score_threshold: float | None = None) -> None:
        self.score_threshold = score_threshold

    def evaluate(
        self,
        candidates: Sequence[RetrievalCandidate],
        *,
        requested_document_id: str | None,
    ) -> EvidenceGateDecision:
        if not candidates:
            return EvidenceGateDecision(False, "no_candidates")
        if not _valid_candidate_metadata(candidates, requested_document_id=requested_document_id):
            return EvidenceGateDecision(False, "invalid_candidate_metadata")
        if self.score_threshold is not None and candidates[0].score < self.score_threshold:
            return EvidenceGateDecision(False, "configured_score_gate_failed")
        return EvidenceGateDecision(True)


class QueryService:
    """Execute one grounded query without coupling business logic to FastAPI."""

    def __init__(
        self,
        *,
        retriever: QueryRetriever,
        evidence_gate: EvidenceGate,
        generator: AnswerGenerator,
        settings: Settings,
    ) -> None:
        self.retriever = retriever
        self.evidence_gate = evidence_gate
        self.generator = generator
        self.settings = settings

    def execute(
        self,
        *,
        question: str,
        document_id: str | None,
        top_k: int,
    ) -> QueryExecution:
        total_started = perf_counter()
        self.generator.ensure_configured()
        retrieved = self.retriever.retrieve(question, document_id=document_id)
        candidate_pool = tuple(retrieved.candidate_pool or retrieved.candidates)
        try:
            selection = select_evidence_candidates(question, retrieved.candidates, top_k=top_k)
        except EvidenceSelectionError:
            return self._abstained_execution(
                reason="invalid_candidate_metadata",
                total_started=total_started,
                retrieval_ms=retrieved.retrieval_ms,
                rerank_ms=retrieved.rerank_ms,
                gate_ms=0.0,
                candidates=tuple(retrieved.candidates),
                candidate_pool=candidate_pool,
            )
        candidates = list(selection.candidates)

        gate_started = perf_counter()
        gate = self.evidence_gate.evaluate(candidates, requested_document_id=document_id)
        gate_ms = (perf_counter() - gate_started) * 1000
        if not gate.passed:
            return self._abstained_execution(
                reason=gate.reason or "invalid_candidate_metadata",
                total_started=total_started,
                retrieval_ms=retrieved.retrieval_ms,
                rerank_ms=retrieved.rerank_ms,
                gate_ms=gate_ms,
                candidates=tuple(retrieved.candidates),
                candidate_pool=candidate_pool,
                evidence_candidates=selection.candidates,
                evidence_duplicate_groups=selection.duplicate_groups,
            )

        try:
            evidence = format_evidence(
                candidates, max_chars=self.settings.generation_max_context_chars
            )
        except GenerationValidationError:
            return self._abstained_execution(
                reason="invalid_candidate_metadata",
                total_started=total_started,
                retrieval_ms=retrieved.retrieval_ms,
                rerank_ms=retrieved.rerank_ms,
                gate_ms=gate_ms,
                candidates=tuple(retrieved.candidates),
                candidate_pool=candidate_pool,
                evidence_candidates=selection.candidates,
                evidence_duplicate_groups=selection.duplicate_groups,
            )

        generation_ms = 0.0
        citation_ms = 0.0
        usage: TokenUsage | None = None
        validation_errors: tuple[str, ...] = ()
        for attempt in range(2):
            generation_started = perf_counter()
            try:
                generated = self.generator.generate(
                    question=question,
                    evidence=evidence,
                    validation_errors=validation_errors,
                )
            except LLMRefusalError:
                generation_ms += (perf_counter() - generation_started) * 1000
                return self._abstained_execution(
                    reason="llm_refusal",
                    total_started=total_started,
                    retrieval_ms=retrieved.retrieval_ms,
                    rerank_ms=retrieved.rerank_ms,
                    gate_ms=gate_ms,
                    generation_ms=generation_ms,
                    citation_ms=citation_ms,
                    usage=usage,
                    candidates=tuple(retrieved.candidates),
                    candidate_pool=candidate_pool,
                    evidence_candidates=selection.candidates,
                    evidence_duplicate_groups=selection.duplicate_groups,
                    generation_attempts=attempt + 1,
                )
            except GenerationValidationError as exc:
                generation_ms += (perf_counter() - generation_started) * 1000
                validation_errors = exc.errors
                if attempt == 0:
                    continue
                return self._abstained_execution(
                    reason="citation_validation_failed",
                    total_started=total_started,
                    retrieval_ms=retrieved.retrieval_ms,
                    rerank_ms=retrieved.rerank_ms,
                    gate_ms=gate_ms,
                    generation_ms=generation_ms,
                    citation_ms=citation_ms,
                    usage=usage,
                    candidates=tuple(retrieved.candidates),
                    candidate_pool=candidate_pool,
                    evidence_candidates=selection.candidates,
                    evidence_duplicate_groups=selection.duplicate_groups,
                    generation_attempts=attempt + 1,
                )
            generation_ms += (perf_counter() - generation_started) * 1000
            usage = _combine_usage(usage, generated.usage)

            citation_started = perf_counter()
            try:
                validated = validate_generated_answer(
                    generated.output, source_map=evidence.source_map
                )
                if validated.insufficient_evidence:
                    citation_ms += (perf_counter() - citation_started) * 1000
                    return self._abstained_execution(
                        reason="llm_insufficient_evidence",
                        total_started=total_started,
                        retrieval_ms=retrieved.retrieval_ms,
                        rerank_ms=retrieved.rerank_ms,
                        gate_ms=gate_ms,
                        generation_ms=generation_ms,
                        citation_ms=citation_ms,
                        usage=usage,
                        candidates=tuple(retrieved.candidates),
                        candidate_pool=candidate_pool,
                        evidence_candidates=selection.candidates,
                        evidence_duplicate_groups=selection.duplicate_groups,
                        generation_attempts=attempt + 1,
                    )
                citations = build_citations(
                    validated.source_ids,
                    source_map=evidence.source_map,
                    requested_document_id=document_id,
                    excerpt_max_chars=self.settings.citation_excerpt_max_chars,
                )
                if not citations:
                    raise CitationValidationError(["grounded answer produced no citations"])
            except CitationValidationError as exc:
                citation_ms += (perf_counter() - citation_started) * 1000
                validation_errors = exc.errors
                if attempt == 0:
                    continue
                return self._abstained_execution(
                    reason="citation_validation_failed",
                    total_started=total_started,
                    retrieval_ms=retrieved.retrieval_ms,
                    rerank_ms=retrieved.rerank_ms,
                    gate_ms=gate_ms,
                    generation_ms=generation_ms,
                    citation_ms=citation_ms,
                    usage=usage,
                    candidates=tuple(retrieved.candidates),
                    candidate_pool=candidate_pool,
                    evidence_candidates=selection.candidates,
                    evidence_duplicate_groups=selection.duplicate_groups,
                    generation_attempts=attempt + 1,
                )
            citation_ms += (perf_counter() - citation_started) * 1000
            response = QueryResponse(
                answer=validated.answer,
                abstained=False,
                abstention_reason=None,
                citations=citations,
            )
            timings = _timings(
                total_started=total_started,
                retrieval_ms=retrieved.retrieval_ms,
                rerank_ms=retrieved.rerank_ms,
                gate_ms=gate_ms,
                generation_ms=generation_ms,
                citation_ms=citation_ms,
            )
            _log_completion(timings, abstention_reason=None)
            return QueryExecution(
                response=response,
                timings=timings,
                usage=usage,
                candidates=tuple(retrieved.candidates),
                candidate_pool=candidate_pool,
                evidence_candidates=selection.candidates,
                evidence_duplicate_groups=selection.duplicate_groups,
                generation_attempts=attempt + 1,
            )

        raise RuntimeError("Correction loop terminated without a query result.")

    def _abstained_execution(
        self,
        *,
        reason: AbstentionReason,
        total_started: float,
        retrieval_ms: float,
        rerank_ms: float,
        gate_ms: float,
        generation_ms: float = 0.0,
        citation_ms: float = 0.0,
        usage: TokenUsage | None = None,
        candidates: tuple[RetrievalCandidate, ...] = (),
        candidate_pool: tuple[RetrievalCandidate, ...] = (),
        evidence_candidates: tuple[RetrievalCandidate, ...] = (),
        evidence_duplicate_groups: tuple[EvidenceDuplicateGroup, ...] = (),
        generation_attempts: int = 0,
    ) -> QueryExecution:
        timings = _timings(
            total_started=total_started,
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            gate_ms=gate_ms,
            generation_ms=generation_ms,
            citation_ms=citation_ms,
        )
        _log_completion(timings, abstention_reason=reason)
        return QueryExecution(
            response=QueryResponse(
                answer=ABSTENTION_MESSAGE,
                abstained=True,
                abstention_reason=reason,
                citations=[],
            ),
            timings=timings,
            usage=usage,
            candidates=candidates,
            candidate_pool=candidate_pool,
            evidence_candidates=evidence_candidates,
            evidence_duplicate_groups=evidence_duplicate_groups,
            generation_attempts=generation_attempts,
        )


@lru_cache
def get_query_service() -> QueryService:
    """Return a lightweight cached service whose heavy retrieval dependencies remain lazy."""

    settings, contract = resolve_retrieval_runtime(get_settings())
    return QueryService(
        retriever=LazyQueryRetriever(
            lambda: build_query_retriever(settings, contract=contract)
        ),
        evidence_gate=EvidenceGate(score_threshold=settings.evidence_score_threshold),
        generator=LangChainOpenAIGenerator(settings),
        settings=settings,
    )


def _valid_candidate_metadata(
    candidates: Sequence[RetrievalCandidate], *, requested_document_id: str | None
) -> bool:
    seen: set[str] = set()
    for candidate in candidates:
        required_strings = (
            candidate.chunk_id,
            candidate.document_id,
            candidate.filename,
            candidate.text,
        )
        if any(not value.strip() for value in required_strings):
            return False
        if candidate.chunk_id in seen or not math.isfinite(candidate.score):
            return False
        seen.add(candidate.chunk_id)
        if requested_document_id is not None and candidate.document_id != requested_document_id:
            return False
        if any(not isinstance(page, int) or page <= 0 for page in candidate.page_numbers):
            return False
        if any(
            not isinstance(heading, str) or not heading.strip() for heading in candidate.headings
        ):
            return False
    return True


def _combine_usage(current: TokenUsage | None, new: TokenUsage | None) -> TokenUsage | None:
    if current is None:
        return new
    if new is None:
        return current
    return TokenUsage(
        input_tokens=_sum_optional(current.input_tokens, new.input_tokens),
        output_tokens=_sum_optional(current.output_tokens, new.output_tokens),
        cached_input_tokens=_sum_optional(current.cached_input_tokens, new.cached_input_tokens),
    )


def _sum_optional(first: int | None, second: int | None) -> int | None:
    if first is None and second is None:
        return None
    return (first or 0) + (second or 0)


def _timings(
    *,
    total_started: float,
    retrieval_ms: float,
    rerank_ms: float,
    gate_ms: float,
    generation_ms: float,
    citation_ms: float,
) -> QueryTimings:
    return QueryTimings(
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        evidence_gate_ms=gate_ms,
        generation_ms=generation_ms,
        citation_validation_ms=citation_ms,
        total_ms=(perf_counter() - total_started) * 1000,
    )


def _log_completion(timings: QueryTimings, *, abstention_reason: AbstentionReason | None) -> None:
    logger.info(
        "query_completed request_id=%s total_ms=%.2f retrieval_ms=%.2f rerank_ms=%.2f "
        "abstention=%s",
        request_id.get() or "none",
        timings.total_ms,
        timings.retrieval_ms,
        timings.rerank_ms,
        abstention_reason or "none",
    )

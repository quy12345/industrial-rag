"""Composition root for the active backend runtime."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from app.config import Settings, get_settings, resolve_retrieval_runtime
from app.generation import LangChainOpenAIGenerator
from app.query_service import EvidenceGate, QueryService
from app.retrieval import create_qdrant_client
from app.retrieval_runtime import (
    LazyQueryRetriever,
    build_query_retriever,
    validate_frozen_runtime,
)

ReadinessChecker = Callable[[], None]


def build_query_service(settings: Settings) -> QueryService:
    """Compose one query service while keeping heavy retrieval dependencies lazy."""

    resolved, contract = resolve_retrieval_runtime(settings)
    return QueryService(
        retriever=LazyQueryRetriever(
            lambda: build_query_retriever(resolved, contract=contract)
        ),
        evidence_gate=EvidenceGate(score_threshold=resolved.evidence_score_threshold),
        generator=LangChainOpenAIGenerator(resolved),
        settings=resolved,
    )


@lru_cache
def get_query_service() -> QueryService:
    """Return the cached service composed from canonical backend settings."""

    return build_query_service(get_settings())


def build_readiness_checker(settings: Settings) -> ReadinessChecker:
    """Build a read-only readiness check without opening clients during app creation."""

    def check() -> None:
        resolved, contract = resolve_retrieval_runtime(settings)
        client = create_qdrant_client(resolved)
        validate_frozen_runtime(
            client,
            collection_names=(
                resolved.qdrant_collection,
                resolved.qdrant_hybrid_collection,
            ),
            contract=contract,
        )

    return check

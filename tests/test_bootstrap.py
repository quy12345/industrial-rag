"""Offline composition tests for query-service construction and caching."""

from __future__ import annotations

from app import bootstrap
from app.config import Settings
from app.domain.retrieval_contracts import PHASE7_RETRIEVAL_CONTRACT


class FakeLazyRetriever:
    def __init__(self, factory) -> None:
        self.factory = factory

    def retrieve(self, question: str, *, document_id: str | None):
        return self.factory().retrieve(question, document_id=document_id)


def test_build_query_service_uses_one_resolved_graph_and_keeps_retrieval_lazy(monkeypatch) -> None:
    source = Settings(evidence_score_threshold=-0.25)
    resolved = source.model_copy(update={"qdrant_url": "http://resolved-qdrant"})
    events: list[object] = []
    generator = object()
    retriever = object()

    def resolve(settings):
        events.append(("resolve", settings))
        return resolved, PHASE7_RETRIEVAL_CONTRACT

    def build_retriever(settings, *, contract):
        events.append(("build_retriever", settings, contract))
        return retriever

    def build_generator(settings):
        events.append(("build_generator", settings))
        return generator

    monkeypatch.setattr(bootstrap, "resolve_retrieval_runtime", resolve)
    monkeypatch.setattr(bootstrap, "build_query_retriever", build_retriever)
    monkeypatch.setattr(bootstrap, "LangChainOpenAIGenerator", build_generator)
    monkeypatch.setattr(bootstrap, "LazyQueryRetriever", FakeLazyRetriever)

    service = bootstrap.build_query_service(source)

    assert events == [
        ("resolve", source),
        ("build_generator", resolved),
    ]
    assert service.settings is resolved
    assert service.generator is generator
    assert service.evidence_gate.score_threshold == -0.25
    assert isinstance(service.retriever, FakeLazyRetriever)

    assert service.retriever.factory() is retriever
    assert events[-1] == (
        "build_retriever",
        resolved,
        PHASE7_RETRIEVAL_CONTRACT,
    )


def test_get_query_service_caches_one_service_for_cached_settings(monkeypatch) -> None:
    settings = Settings()
    service = object()
    calls: list[Settings] = []
    monkeypatch.setattr(bootstrap, "get_settings", lambda: settings)
    monkeypatch.setattr(
        bootstrap,
        "build_query_service",
        lambda value: calls.append(value) or service,
    )

    bootstrap.get_query_service.cache_clear()
    try:
        assert bootstrap.get_query_service() is service
        assert bootstrap.get_query_service() is service
        assert calls == [settings]
    finally:
        bootstrap.get_query_service.cache_clear()

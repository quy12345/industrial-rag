"""Offline HTTP contract and error-mapping tests for POST /api/v1/query."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import get_settings
from app.errors import (
    LLMNotConfiguredError,
    LLMTimeoutError,
    LLMUnavailableError,
    RerankerUnavailableError,
    RetrievalUnavailableError,
)
from app.main import app
from app.models import Citation, QueryResponse
from app.query_service import QueryExecution, QueryTimings, get_query_service


class FakeService:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response or QueryResponse(
            answer="Grounded.",
            abstained=False,
            citations=[
                Citation(
                    chunk_id="chunk-a",
                    document_id="manual-a",
                    filename="manual.pdf",
                    page_numbers=[1],
                    headings=["Limits"],
                    excerpt="Evidence.",
                )
            ],
        )
        self.error = error
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return QueryExecution(
            self.response,
            QueryTimings(1, 2, 0, 3, 0, 6),
            None,
        )


@pytest.fixture
def client_and_service():
    service = FakeService()
    app.dependency_overrides[get_query_service] = lambda: service
    with TestClient(app) as client:
        yield client, service
    app.dependency_overrides.clear()


def test_valid_unicode_request_default_top_k_and_optional_document(client_and_service) -> None:
    client, service = client_and_service
    response = client.post(
        "/api/v1/query",
        json={"question": "  Điện áp là bao nhiêu?  ", "document_id": "manual-a"},
    )
    assert response.status_code == 200
    assert response.json()["citations"][0]["chunk_id"] == "chunk-a"
    assert service.calls == [
        {"question": "Điện áp là bao nhiêu?", "document_id": "manual-a", "top_k": 5}
    ]


def test_success_response_declares_utf8_and_preserves_full_unicode(client_and_service) -> None:
    client, service = client_and_service
    answer = "Thuật toán phát hiện dữ liệu cảm biến bất thường."
    service.response = QueryResponse(answer=answer, abstained=False, citations=[])
    response = client.post("/api/v1/query", json={"question": "Câu hỏi"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert response.json()["answer"] == answer
    assert answer in response.content.decode("utf-8")


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "q", "top_k": 0},
        {"question": "q", "top_k": 11},
        {"question": "q", "document_id": " "},
        {"question": "q", "unexpected": True},
    ],
)
def test_invalid_requests_return_422_without_calling_service(client_and_service, body) -> None:
    client, service = client_and_service
    response = client.post("/api/v1/query", json=body)
    assert response.status_code == 422
    assert service.calls == []


def test_valid_abstention_is_http_200_with_empty_citations(client_and_service) -> None:
    client, service = client_and_service
    service.response = QueryResponse(
        answer="Insufficient evidence.",
        abstained=True,
        abstention_reason="no_candidates",
        citations=[],
    )
    response = client.post("/api/v1/query", json={"question": "unknown"})
    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["citations"] == []


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (RetrievalUnavailableError("qdrant secret"), 503, "retrieval_unavailable"),
        (RerankerUnavailableError("model secret"), 503, "reranker_unavailable"),
        (LLMNotConfiguredError("key secret"), 503, "llm_not_configured"),
        (LLMUnavailableError("provider secret"), 503, "llm_unavailable"),
        (LLMTimeoutError("timeout secret"), 504, "llm_timeout"),
    ],
)
def test_dependency_errors_are_sanitized(client_and_service, error, status, code) -> None:
    client, service = client_and_service
    service.error = error
    response = client.post("/api/v1/query", json={"question": "q"})
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert "secret" not in response.text


def test_unexpected_error_returns_sanitized_500(client_and_service, caplog) -> None:
    client, service = client_and_service
    service.error = RuntimeError("raw evidence secret")
    response = client.post("/api/v1/query", json={"question": "secret question"})
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "internal_error"
    assert "raw evidence" not in response.text
    assert "secret question" not in caplog.text


def test_optional_query_auth_rejects_missing_or_wrong_bearer_token(
    client_and_service, monkeypatch
) -> None:
    client, service = client_and_service
    monkeypatch.setattr(get_settings(), "api_auth_enabled", True)
    monkeypatch.setattr(get_settings(), "api_auth_key", SecretStr("expected"))
    missing = client.post("/api/v1/query", json={"question": "q"})
    wrong = client.post(
        "/api/v1/query", json={"question": "q"}, headers={"Authorization": "Bearer wrong"}
    )
    accepted = client.post(
        "/api/v1/query", json={"question": "q"}, headers={"Authorization": "Bearer expected"}
    )
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert accepted.status_code == 200
    assert len(service.calls) == 1

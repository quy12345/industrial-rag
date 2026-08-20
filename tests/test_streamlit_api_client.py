"""Offline HTTP contract tests for the lightweight Streamlit API client."""

from __future__ import annotations

import json

import httpx
import pytest

from ui.api_client import RAGAPIClient, RAGAPIError
from ui.config import UISettings


def _client(handler, *, auth_token=None) -> RAGAPIClient:
    return RAGAPIClient(
        base_url="http://api:8000/api/v1",
        timeout_seconds=10,
        auth_token=auth_token,
        transport=httpx.MockTransport(handler),
    )


def test_client_calls_health_ready_and_query_with_document_filter() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith(("/health", "/ready")):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(
            200,
            json={
                "answer": "Điện áp được nêu trong tài liệu.",
                "abstained": False,
                "abstention_reason": None,
                "citations": [
                    {
                        "chunk_id": "chunk-1",
                        "document_id": "doc-1",
                        "filename": "manual.pdf",
                        "page_numbers": [2, 1],
                        "headings": ["Electrical"],
                        "excerpt": "24 VDC",
                    }
                ],
            },
        )

    client = _client(handler, auth_token="ui-secret")
    assert client.health() is True
    assert client.ready() is True
    result = client.query(question="Điện áp là bao nhiêu?", document_id="doc-1", top_k=4)
    assert result.answer.startswith("Điện áp")
    assert result.citations[0].chunk_id == "chunk-1"
    query_request = requests[-1]
    assert json.loads(query_request.content) == {
        "question": "Điện áp là bao nhiêu?",
        "document_id": "doc-1",
        "top_k": 4,
    }
    assert query_request.headers["authorization"] == "Bearer ui-secret"
    assert len(requests) == 3


def test_client_omits_optional_document_and_authorization() -> None:
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={"answer": "No evidence.", "abstained": True, "citations": []},
        )

    result = _client(handler).query(question="unknown", document_id=None, top_k=5)
    assert result.abstained is True
    assert "document_id" not in json.loads(captured[0].content)
    assert "authorization" not in captured[0].headers


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (401, "unauthorized", "authentication"),
        (422, "validation_error", "invalid"),
        (503, "retrieval_unavailable", "retrieval"),
        (503, "reranker_unavailable", "reranker"),
        (503, "llm_unavailable", "Gemini"),
        (504, "llm_timeout", "timed out"),
    ],
)
def test_client_maps_api_errors_without_exposing_response(
    status: int, code: str, expected: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"detail": {"code": code, "message": "private provider response"}},
        )

    with pytest.raises(RAGAPIError, match=expected) as captured:
        _client(handler).query(question="secret question", document_id=None, top_k=5)
    assert "private provider response" not in captured.value.message


def test_client_rejects_malformed_json_and_schema() -> None:
    malformed = _client(lambda request: httpx.Response(200, content=b"not-json"))
    with pytest.raises(RAGAPIError, match="malformed JSON"):
        malformed.health()

    invalid = _client(lambda request: httpx.Response(200, json={"answer": 123}))
    with pytest.raises(RAGAPIError, match="invalid response schema"):
        invalid.query(question="q", document_id=None, top_k=5)


def test_client_timeout_is_sanitized_and_post_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("raw timeout detail", request=request)

    with pytest.raises(RAGAPIError, match="before timeout"):
        _client(handler).query(question="q", document_id=None, top_k=5)
    assert calls == 1


def test_ui_settings_validate_environment(monkeypatch) -> None:
    monkeypatch.setenv("RAG_API_URL", "http://localhost:8000/api/v1/")
    monkeypatch.setenv("RAG_API_TIMEOUT_SECONDS", "180")
    monkeypatch.setenv("RAG_API_AUTH_TOKEN", " token ")
    settings = UISettings.from_environment()
    assert settings.api_url == "http://localhost:8000/api/v1"
    assert settings.timeout_seconds == 180
    assert settings.auth_token == "token"

    monkeypatch.setenv("RAG_API_URL", "file:///private")
    with pytest.raises(ValueError, match="HTTP"):
        UISettings.from_environment()

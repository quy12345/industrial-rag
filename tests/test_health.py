"""Tests for the health endpoint."""

from fastapi.testclient import TestClient

import app.main as app_main
from app.main import app
from app.retrieval import RetrievalError

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "industrial-rag",
        "version": "0.1.0",
    }


def test_health_has_safe_request_correlation_id() -> None:
    response = client.get("/api/v1/health", headers={"X-Request-ID": "request.123"})
    assert response.headers["x-request-id"] == "request.123"

    generated = client.get("/api/v1/health", headers={"X-Request-ID": "bad value"})
    assert generated.headers["x-request-id"] != "bad value"


def test_readiness_checks_frozen_qdrant_contract_without_loading_models(monkeypatch) -> None:
    fake_client = object()
    calls = []
    monkeypatch.setattr(app_main, "create_qdrant_client", lambda settings: fake_client)
    monkeypatch.setattr(
        app_main,
        "validate_frozen_runtime",
        lambda client, *, collection_names, contract: calls.append(
            (client, collection_names, contract)
        ),
    )
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    assert calls[0][0] is fake_client


def test_readiness_uses_selected_phase7_contract(monkeypatch) -> None:
    fake_client = object()
    calls = []
    monkeypatch.setattr(app_main.settings, "retrieval_profile", "phase7")
    monkeypatch.setattr(app_main, "create_qdrant_client", lambda settings: fake_client)
    monkeypatch.setattr(
        app_main,
        "validate_frozen_runtime",
        lambda client, *, collection_names, contract: calls.append(
            (client, collection_names, contract)
        ),
    )
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    assert calls[0][1] == (
        "industrial_manual_phase7_dense_v1",
        "industrial_manual_phase7_hybrid_v1",
    )
    assert calls[0][2].chunk_count == 2753


def test_readiness_returns_sanitized_503_when_qdrant_is_unavailable(monkeypatch) -> None:
    def unavailable(settings):
        raise RetrievalError("private endpoint")

    monkeypatch.setattr(app_main, "create_qdrant_client", unavailable)
    response = client.get("/api/v1/ready")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "retrieval_not_ready"
    assert "private" not in response.text

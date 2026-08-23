"""Tests for the health endpoint."""

from fastapi.testclient import TestClient

from app.api import dependencies
from app.api.app import create_app
from app.application.query_service import QueryService
from app.bootstrap import get_query_service
from app.config import Settings, get_settings
from app.main import app
from app.retrieval import RetrievalError

client = TestClient(app)


def test_api_dependency_seams_preserve_composition_identity() -> None:
    assert dependencies.QueryService is QueryService
    assert dependencies.get_query_service is get_query_service
    assert dependencies.get_settings is get_settings


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


def test_readiness_calls_the_injected_checker() -> None:
    calls = []
    test_app = create_app(readiness_checker=lambda: calls.append("checked"))
    with TestClient(test_app) as test_client:
        response = test_client.get("/api/v1/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "industrial-rag",
        "version": "0.1.0",
    }
    assert calls == ["checked"]


def test_create_app_preserves_custom_metadata_routes_and_request_id() -> None:
    settings = Settings(
        app_name="Test Industrial RAG",
        app_version="9.8.7",
        api_prefix="/custom/v1",
    )
    test_app = create_app(settings, readiness_checker=lambda: None)
    paths = set(test_app.openapi()["paths"])

    assert test_app.title == "Test Industrial RAG"
    assert test_app.version == "9.8.7"
    assert {"/custom/v1/health", "/custom/v1/ready", "/custom/v1/query"} <= paths
    with TestClient(test_app) as test_client:
        response = test_client.get(
            "/custom/v1/health",
            headers={"X-Request-ID": "factory.request"},
        )

    assert response.status_code == 200
    assert response.json()["version"] == "9.8.7"
    assert response.headers["x-request-id"] == "factory.request"


def test_readiness_returns_sanitized_503_when_qdrant_is_unavailable() -> None:
    def unavailable() -> None:
        raise RetrievalError("private endpoint")

    test_app = create_app(readiness_checker=unavailable)
    with TestClient(test_app) as test_client:
        response = test_client.get("/api/v1/ready")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "retrieval_not_ready"
    assert "private" not in response.text

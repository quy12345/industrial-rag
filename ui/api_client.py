"""Small fail-closed HTTP client for the public Industrial RAG API."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import ValidationError

from app.models import QueryResponse

_ERROR_MESSAGES = {
    "unauthorized": "API authentication failed.",
    "auth_not_configured": "API authentication is not configured.",
    "retrieval_not_ready": "Qdrant retrieval is not ready.",
    "retrieval_unavailable": "Qdrant retrieval is unavailable.",
    "reranker_unavailable": "The multilingual reranker is unavailable.",
    "llm_not_configured": "Gemini is not configured on the API service.",
    "llm_unavailable": "Gemini is temporarily unavailable.",
    "llm_timeout": "Gemini timed out.",
    "internal_error": "The API returned an internal error.",
}


class RAGAPIError(RuntimeError):
    """Sanitized API/client failure safe to display in the demo UI."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class RAGAPIClient:
    """Call health, readiness, and query endpoints without automatic retries."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        auth_token: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.auth_token = auth_token
        self.transport = transport

    def health(self) -> bool:
        payload = self._request("GET", "/health")
        return payload.get("status") == "ok"

    def ready(self) -> bool:
        payload = self._request("GET", "/ready")
        return payload.get("status") == "ok"

    def query(
        self,
        *,
        question: str,
        document_id: str | None,
        top_k: int,
    ) -> QueryResponse:
        body: dict[str, Any] = {"question": question, "top_k": top_k}
        if document_id is not None:
            body["document_id"] = document_id
        payload = self._request("POST", "/query", json=body)
        try:
            return QueryResponse.model_validate(payload)
        except ValidationError as exc:
            raise RAGAPIError(
                "invalid_response", "The API returned an invalid response schema."
            ) from exc

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.auth_token is not None:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                headers=headers,
                transport=self.transport,
            ) as client:
                response = client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            raise RAGAPIError("api_timeout", "The API did not respond before timeout.") from exc
        except httpx.HTTPError as exc:
            raise RAGAPIError("api_unreachable", "The API is unreachable.") from exc

        if response.is_error:
            code = _error_code(response)
            if response.status_code == 422:
                raise RAGAPIError("invalid_request", "The query request is invalid.")
            message = _ERROR_MESSAGES.get(code, "The API request failed.")
            raise RAGAPIError(code, message)
        try:
            payload = response.json()
        except ValueError as exc:
            raise RAGAPIError(
                "invalid_response", "The API returned malformed JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise RAGAPIError("invalid_response", "The API returned an invalid response.")
        return payload


def _error_code(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"http_{response.status_code}"
    if not isinstance(payload, dict):
        return f"http_{response.status_code}"
    detail = payload.get("detail")
    if not isinstance(detail, dict):
        return f"http_{response.status_code}"
    code = detail.get("code")
    return code if isinstance(code, str) and code else f"http_{response.status_code}"

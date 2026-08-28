"""Qdrant client construction."""

from __future__ import annotations

from qdrant_client import QdrantClient

from app.config import Settings
from app.errors import RetrievalError


def create_qdrant_client(settings: Settings) -> QdrantClient:
    """Create a Qdrant client and confirm the configured endpoint is reachable."""

    endpoint = f"{settings.qdrant_url}:{settings.qdrant_port}"
    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            port=settings.qdrant_port,
            timeout=settings.qdrant_timeout_seconds,
        )
        client.get_collections()
        return client
    except Exception as exc:
        raise RetrievalError(f"Qdrant is unavailable at {endpoint}") from exc

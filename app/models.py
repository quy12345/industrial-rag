"""Compatibility exports for Pydantic contracts with canonical package owners."""

from app.contracts.health import HealthResponse as HealthResponse
from app.contracts.health import ReadinessResponse as ReadinessResponse
from app.contracts.query import Citation as Citation
from app.contracts.query import QueryRequest as QueryRequest
from app.contracts.query import QueryResponse as QueryResponse
from app.domain.documents import DocumentChunk as DocumentChunk
from app.domain.retrieval import RetrievalCandidate as RetrievalCandidate
from app.domain.retrieval import RetrievedChunk as RetrievedChunk

__all__ = [
    "Citation",
    "DocumentChunk",
    "HealthResponse",
    "QueryRequest",
    "QueryResponse",
    "ReadinessResponse",
    "RetrievalCandidate",
    "RetrievedChunk",
]

"""Health and readiness response contracts shared by HTTP adapters."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    status: Literal["ok"]
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Response from the Qdrant-backed readiness check."""

    status: Literal["ok"]
    service: str
    version: str

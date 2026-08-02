"""FastAPI application entry point."""

from fastapi import FastAPI

from app.config import get_settings
from app.models import HealthResponse

settings = get_settings()

app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.get(f"{settings.api_prefix}/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return the service health status."""

    return HealthResponse(
        status="ok",
        service="industrial-rag",
        version=settings.app_version,
    )

"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.query import router as query_router
from app.config import get_settings
from app.models import HealthResponse

settings = get_settings()


class UTF8JSONResponse(JSONResponse):
    """Declare UTF-8 explicitly for legacy clients such as Windows PowerShell 5.1."""

    media_type = "application/json; charset=utf-8"


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    default_response_class=UTF8JSONResponse,
)
app.include_router(query_router, prefix=settings.api_prefix)


@app.get(f"{settings.api_prefix}/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return the service health status."""

    return HealthResponse(
        status="ok",
        service="industrial-rag",
        version=settings.app_version,
    )

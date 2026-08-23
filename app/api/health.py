"""Health and readiness HTTP mappings."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.dependencies import ReadinessChecker, Settings
from app.models import HealthResponse, ReadinessResponse
from app.retrieval import RetrievalError


def create_health_router(
    settings: Settings,
    *,
    readiness_checker: ReadinessChecker,
) -> APIRouter:
    """Create status routes around one injected read-only readiness check."""

    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Return the service health status."""

        return HealthResponse(
            status="ok",
            service="industrial-rag",
            version=settings.app_version,
        )

    @router.get("/ready", response_model=ReadinessResponse)
    def ready() -> ReadinessResponse:
        """Check frozen Qdrant identity through the injected readiness seam."""

        try:
            readiness_checker()
        except RetrievalError:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": {
                        "code": "retrieval_not_ready",
                        "message": "Retrieval is unavailable.",
                    }
                },
            )
        return ReadinessResponse(
            status="ok",
            service="industrial-rag",
            version=settings.app_version,
        )

    return router

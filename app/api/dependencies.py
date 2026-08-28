"""FastAPI dependency seams backed by the production composition root."""

from collections.abc import Callable

from app.application.query_service import QueryService
from app.bootstrap import ReadinessChecker as ReadinessChecker
from app.bootstrap import build_readiness_checker as build_readiness_checker
from app.bootstrap import get_query_service as get_query_service
from app.config import Settings as Settings
from app.config import get_settings as get_settings

QueryServiceProvider = Callable[[], QueryService]

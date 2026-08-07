"""HTTP mapping for the grounded query service."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.api.auth import require_query_auth
from app.errors import (
    LLMNotConfiguredError,
    LLMTimeoutError,
    LLMUnavailableError,
    RerankerUnavailableError,
    RetrievalUnavailableError,
)
from app.models import QueryRequest, QueryResponse
from app.query_service import QueryService, get_query_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    service: Annotated[QueryService, Depends(get_query_service)],
    _: Annotated[None, Depends(require_query_auth)],
) -> QueryResponse:
    """Return a grounded answer, a valid abstention, or a sanitized dependency error."""

    try:
        execution = await run_in_threadpool(
            service.execute,
            question=request.question,
            document_id=request.document_id,
            top_k=request.top_k,
        )
    except RetrievalUnavailableError:
        raise _http_error(
            503, "retrieval_unavailable", "Retrieval service is unavailable."
        ) from None
    except RerankerUnavailableError:
        raise _http_error(
            503, "reranker_unavailable", "Reranking service is unavailable."
        ) from None
    except LLMNotConfiguredError:
        raise _http_error(
            503, "llm_not_configured", "Generation provider is not configured."
        ) from None
    except LLMTimeoutError:
        raise _http_error(504, "llm_timeout", "Generation provider timed out.") from None
    except LLMUnavailableError:
        raise _http_error(
            503, "llm_unavailable", "Generation provider is unavailable."
        ) from None
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("query_failed exception_class=%s", type(exc).__name__)
        raise _http_error(500, "internal_error", "Internal query error.") from None
    return execution.response


def _http_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})

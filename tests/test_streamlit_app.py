"""Offline tests for Streamlit presentation helpers and bounded session history."""

from app.models import Citation, QueryResponse
from ui.api_client import RAGAPIError
from ui.config import (
    INSTALLATION_DOCUMENT_ID,
    PROGRAMMING_DOCUMENT_ID,
)
from ui.state import MAX_HISTORY_TURNS, append_history, page_label


def _response(index: int) -> QueryResponse:
    return QueryResponse(
        answer=f"Trả lời {index}",
        abstained=False,
        citations=[
            Citation(
                chunk_id=f"chunk-{index}",
                document_id=INSTALLATION_DOCUMENT_ID,
                filename="installation.pdf",
                page_numbers=[3, 1, 3],
                headings=["Safety", "Mounting"],
                excerpt="Bằng chứng Unicode: điện áp.",
            )
        ],
    )


def test_history_is_bounded_and_preserves_unicode_response() -> None:
    history = []
    for index in range(MAX_HISTORY_TURNS + 3):
        history = append_history(
            history,
            question=f"Câu hỏi {index}",
            response=_response(index),
        )
    assert len(history) == MAX_HISTORY_TURNS
    assert history[0]["question"] == "Câu hỏi 3"
    assert history[-1]["response"]["citations"][0]["excerpt"].endswith("điện áp.")


def test_history_stores_only_sanitized_display_error() -> None:
    error = RAGAPIError("api_unreachable", "The API is unreachable.")
    history = append_history([], question="q", error=error)
    assert history == [
        {
            "question": "q",
            "response": None,
            "error": {"code": "api_unreachable", "message": "The API is unreachable."},
        }
    ]


def test_page_label_is_unique_sorted_and_document_ids_are_distinct() -> None:
    assert page_label([9, 2, 9, 3]) == "2, 3, 9"
    assert page_label([]) == "unknown"
    assert INSTALLATION_DOCUMENT_ID != PROGRAMMING_DOCUMENT_ID

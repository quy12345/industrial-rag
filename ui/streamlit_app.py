"""Interactive Streamlit demo for the frozen Phase 7 ATV320 RAG runtime."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.models import QueryResponse
from ui.api_client import RAGAPIClient, RAGAPIError
from ui.config import DEMO_QUESTIONS, DOCUMENT_OPTIONS, UISettings
from ui.state import append_history, page_label


def render() -> None:
    """Render the demo without exposing provider credentials or internal scores."""

    st.set_page_config(page_title="ATV320 Manual Assistant", page_icon="⚙️", layout="wide")
    st.title("⚙️ ATV320 Technical Manual Assistant")
    st.caption(
        "Demo RAG: Phase 7 dense + BM25 retrieval, multilingual reranking, "
        "grounded generation and validated citations."
    )
    try:
        settings = UISettings.from_environment()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    client = RAGAPIClient(
        base_url=settings.api_url,
        timeout_seconds=settings.timeout_seconds,
        auth_token=settings.auth_token,
    )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Demo settings")
        _render_service_status(client)
        selected_document = st.selectbox("Tài liệu", tuple(DOCUMENT_OPTIONS))
        top_k = st.slider("Số evidence chunks", min_value=1, max_value=10, value=5)
        if st.button("Xóa lịch sử chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        st.warning("Demo cục bộ, chưa phải giao diện production.")
        st.caption("Mỗi câu hỏi được xử lý độc lập; chat history chỉ dùng để hiển thị.")
        st.subheader("Câu hỏi gợi ý")
        for index, example in enumerate(DEMO_QUESTIONS):
            if st.button(example, key=f"example-{index}", use_container_width=True):
                st.session_state.pending_question = example

    for turn in st.session_state.chat_history:
        _render_turn(turn)

    prompt = st.session_state.pop("pending_question", None)
    submitted = st.chat_input(
        "Hỏi về lắp đặt, đấu dây, tham số hoặc chẩn đoán ATV320…",
        max_chars=2_000,
        submit_mode="disable",
    )
    question = submitted or prompt
    if not question:
        return
    normalized_question = question.strip()
    if not normalized_question:
        return
    with st.chat_message("user"):
        st.write(normalized_question)
    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất tài liệu, rerank và tạo câu trả lời…"):
            try:
                response = client.query(
                    question=normalized_question,
                    document_id=DOCUMENT_OPTIONS[selected_document],
                    top_k=top_k,
                )
                _render_response(response)
                st.session_state.chat_history = append_history(
                    st.session_state.chat_history,
                    question=normalized_question,
                    response=response,
                )
            except RAGAPIError as exc:
                st.error(exc.message)
                st.session_state.chat_history = append_history(
                    st.session_state.chat_history,
                    question=normalized_question,
                    error=exc,
                )


def _render_service_status(client: RAGAPIClient) -> None:
    try:
        healthy = client.health()
    except RAGAPIError:
        healthy = False
    try:
        ready = client.ready()
    except RAGAPIError:
        ready = False
    st.write("API:", "🟢 online" if healthy else "🔴 offline")
    st.write("Phase 7 corpus:", "🟢 ready" if ready else "🟠 not ready")


def _render_turn(turn: dict[str, Any]) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        if turn["response"] is not None:
            _render_response(QueryResponse.model_validate(turn["response"]))
        else:
            st.error(turn["error"]["message"])


def _render_response(response: QueryResponse) -> None:
    if response.abstained:
        st.warning(response.answer)
        if response.abstention_reason:
            st.caption(f"Reason: {response.abstention_reason}")
        return
    st.markdown(response.answer)
    if not response.citations:
        return
    st.caption(f"Nguồn tham khảo: {len(response.citations)}")
    for index, citation in enumerate(response.citations, start=1):
        pages = page_label(citation.page_numbers)
        with st.expander(f"[{index}] {citation.filename} — page {pages}"):
            if citation.headings:
                st.write(" > ".join(citation.headings))
            st.text(citation.excerpt)
            st.caption(f"chunk_id: {citation.chunk_id}")
            st.caption(f"document_id: {citation.document_id}")


if __name__ == "__main__":
    render()

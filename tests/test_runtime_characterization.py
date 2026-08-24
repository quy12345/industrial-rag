"""Cross-stage characterization of the active Phase 7 query contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.application.query_service import QueryService
from app.config import Settings
from app.contracts.query import QueryRequest, QueryResponse
from app.domain.evidence import EvidenceDuplicateGroup, EvidenceGate
from app.domain.generation import GeneratedAnswer, GenerationResult, TokenUsage
from app.domain.retrieval import QueryRetrievalResult, RetrievalCandidate
from app.domain.retrieval_contracts import PHASE7_RETRIEVAL_CONTRACT
from app.main import app

INSTALLATION_DOCUMENT_ID = PHASE7_RETRIEVAL_CONTRACT.document_ids[0]
PROGRAMMING_DOCUMENT_ID = PHASE7_RETRIEVAL_CONTRACT.document_ids[1]


def _candidate(
    chunk_id: str,
    *,
    document_id: str,
    document_role: str,
    text: str,
    page_numbers: list[int],
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"ATV320_{document_role}.pdf",
        text=text,
        page_numbers=page_numbers,
        headings=["Wiring"],
        content_type="text",
        metadata={
            "document_role": document_role,
            "document_title": f"ATV320 {document_role.title()} Manual",
        },
        score=1.0,
        rerank_score=1.0,
        rerank_rank=1,
    )


class RecordingRetriever:
    def __init__(
        self,
        *,
        events: list[str],
        final_candidates: list[RetrievalCandidate],
        candidate_pool: list[RetrievalCandidate],
    ) -> None:
        self.events = events
        self.final_candidates = final_candidates
        self.candidate_pool = candidate_pool
        self.calls: list[tuple[str, str | None]] = []

    def retrieve(self, question: str, *, document_id: str | None) -> QueryRetrievalResult:
        self.events.append("retriever.retrieve")
        self.calls.append((question, document_id))
        return QueryRetrievalResult(
            candidates=self.final_candidates,
            candidate_pool=self.candidate_pool,
            retrieval_ms=2.5,
            rerank_ms=4.5,
        )


class RecordingGenerator:
    def __init__(self, *, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[object, tuple[str, ...]]] = []
        self.outputs = [
            GenerationResult(
                output=GeneratedAnswer(
                    answer="Invalid citation.",
                    source_ids=["S9"],
                    insufficient_evidence=False,
                ),
                usage=TokenUsage(input_tokens=10, output_tokens=2),
            ),
            GenerationResult(
                output=GeneratedAnswer(
                    answer="Use terminal X1 after disconnecting power.",
                    source_ids=["S1", "S2"],
                    insufficient_evidence=False,
                ),
                usage=TokenUsage(input_tokens=11, output_tokens=3, cached_input_tokens=1),
            ),
        ]

    def ensure_configured(self) -> None:
        self.events.append("generator.ensure_configured")

    def generate(self, *, question, evidence, validation_errors=()):
        self.events.append("generator.generate")
        self.calls.append((evidence, tuple(validation_errors)))
        return self.outputs.pop(0)


def test_public_query_schema_matches_the_active_http_contract() -> None:
    request_schema = QueryRequest.model_json_schema()
    response_schema = QueryResponse.model_json_schema()
    operation = app.openapi()["paths"]["/api/v1/query"]["post"]

    assert request_schema["additionalProperties"] is False
    assert request_schema["required"] == ["question"]
    assert request_schema["properties"]["top_k"] == {
        "default": 5,
        "maximum": 10,
        "minimum": 1,
        "title": "Top K",
        "type": "integer",
    }
    assert response_schema["additionalProperties"] is False
    assert response_schema["required"] == ["answer", "abstained"]
    assert response_schema["$defs"]["Citation"]["required"] == [
        "chunk_id",
        "document_id",
        "filename",
        "page_numbers",
        "headings",
        "excerpt",
    ]
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/QueryRequest"
    }
    assert set(operation["responses"]) == {"200", "422"}
    assert set(operation["responses"]["200"]["content"]) == {
        "application/json; charset=utf-8"
    }
    assert operation["parameters"][0]["name"] == "authorization"
    assert operation["parameters"][0]["in"] == "header"
    assert operation["parameters"][0]["required"] is False


def test_retrieval_candidate_schema_and_copy_contract_are_stable() -> None:
    schema = RetrievalCandidate.model_json_schema()

    assert schema["required"] == [
        "chunk_id",
        "document_id",
        "filename",
        "text",
        "page_numbers",
        "headings",
        "content_type",
        "score",
    ]
    for field_name in ("dense_rank", "sparse_rank", "rrf_rank", "rerank_rank"):
        assert schema["properties"][field_name]["anyOf"][0] == {
            "minimum": 1,
            "type": "integer",
        }

    candidate = RetrievalCandidate(
        chunk_id="candidate-1",
        document_id=INSTALLATION_DOCUMENT_ID,
        filename="ATV320_installation.pdf",
        text="Disconnect all power before servicing the drive.",
        page_numbers=[42],
        headings=["Safety"],
        content_type="text",
        score=0.75,
    )
    copied = candidate.model_copy(update={"dense_rank": 2, "dense_score": 0.75})

    assert candidate.dense_rank is None
    assert candidate.dense_score is None
    assert copied.dense_rank == 2
    assert copied.dense_score == 0.75
    assert copied.metadata == {}


@pytest.mark.parametrize("field_name", ["dense_rank", "sparse_rank", "rrf_rank", "rerank_rank"])
def test_retrieval_candidate_rejects_non_positive_one_based_ranks(field_name: str) -> None:
    values = {
        "chunk_id": "candidate-1",
        "document_id": INSTALLATION_DOCUMENT_ID,
        "filename": "ATV320_installation.pdf",
        "text": "Disconnect all power before servicing the drive.",
        "page_numbers": [42],
        "headings": ["Safety"],
        "content_type": "text",
        "score": 0.75,
        field_name: 0,
    }

    with pytest.raises(ValidationError) as caught:
        RetrievalCandidate(**values)

    assert caught.value.errors()[0]["loc"] == (field_name,)
    assert caught.value.errors()[0]["type"] == "greater_than_equal"


def test_golden_phase7_query_execution_preserves_all_stage_boundaries() -> None:
    events: list[str] = []
    duplicate_text = "Disconnect all power before wiring terminal X1."
    programming_copy = _candidate(
        "programming-copy",
        document_id=PROGRAMMING_DOCUMENT_ID,
        document_role="programming",
        text=duplicate_text,
        page_numbers=[210],
    )
    installation_copy = _candidate(
        "installation-copy",
        document_id=INSTALLATION_DOCUMENT_ID,
        document_role="installation",
        text=duplicate_text,
        page_numbers=[42, 42],
    )
    unique_installation = _candidate(
        "installation-unique",
        document_id=INSTALLATION_DOCUMENT_ID,
        document_role="installation",
        text="Connect the control conductor to terminal X1.",
        page_numbers=[43],
    )
    pre_rerank = _candidate(
        "pre-rerank-only",
        document_id=PROGRAMMING_DOCUMENT_ID,
        document_role="programming",
        text="A candidate removed by reranking.",
        page_numbers=[99],
    )
    final_candidates = [programming_copy, installation_copy, unique_installation]
    candidate_pool = [pre_rerank, *final_candidates]
    retriever = RecordingRetriever(
        events=events,
        final_candidates=final_candidates,
        candidate_pool=candidate_pool,
    )
    generator = RecordingGenerator(events=events)
    service = QueryService(
        retriever=retriever,
        evidence_gate=EvidenceGate(),
        generator=generator,
        settings=Settings(),
    )

    execution = service.execute(
        question="Which wiring terminal must be used?",
        document_id=None,
        top_k=2,
    )

    assert events == [
        "generator.ensure_configured",
        "retriever.retrieve",
        "generator.generate",
        "generator.generate",
    ]
    assert retriever.calls == [("Which wiring terminal must be used?", None)]
    assert generator.calls[0][0] is generator.calls[1][0]
    assert generator.calls[0][1] == ()
    assert generator.calls[1][1] == ("unknown source ID: S9",)
    assert [candidate.chunk_id for candidate in execution.candidate_pool] == [
        "pre-rerank-only",
        "programming-copy",
        "installation-copy",
        "installation-unique",
    ]
    assert [candidate.chunk_id for candidate in execution.candidates] == [
        "programming-copy",
        "installation-copy",
        "installation-unique",
    ]
    assert [candidate.chunk_id for candidate in execution.evidence_candidates] == [
        "installation-copy",
        "installation-unique",
    ]
    assert execution.evidence_duplicate_groups == (
        EvidenceDuplicateGroup(
            representative_chunk_id="installation-copy",
            equivalent_chunk_ids=("installation-copy", "programming-copy"),
            equivalent_document_ids=(INSTALLATION_DOCUMENT_ID, PROGRAMMING_DOCUMENT_ID),
        ),
    )
    assert execution.response.model_dump() == {
        "answer": "Use terminal X1 after disconnecting power.",
        "abstained": False,
        "abstention_reason": None,
        "citations": [
            {
                "chunk_id": "installation-copy",
                "document_id": INSTALLATION_DOCUMENT_ID,
                "filename": "ATV320_installation.pdf",
                "page_numbers": [42],
                "headings": ["Wiring"],
                "excerpt": duplicate_text,
            },
            {
                "chunk_id": "installation-unique",
                "document_id": INSTALLATION_DOCUMENT_ID,
                "filename": "ATV320_installation.pdf",
                "page_numbers": [43],
                "headings": ["Wiring"],
                "excerpt": "Connect the control conductor to terminal X1.",
            },
        ],
    }
    assert execution.usage == TokenUsage(
        input_tokens=21,
        output_tokens=5,
        cached_input_tokens=1,
    )
    assert execution.generation_attempts == 2
    assert execution.timings.retrieval_ms == 2.5
    assert execution.timings.rerank_ms == 4.5

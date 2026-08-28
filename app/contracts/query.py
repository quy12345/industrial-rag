"""Public request, citation, and response contracts for grounded queries."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    """Validated request body for the grounded query endpoint."""

    model_config = ConfigDict(extra="forbid")

    question: str
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be empty")
        return normalized

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("document_id must not be empty")
        return normalized


class Citation(BaseModel):
    """Trusted citation metadata built from a retrieved Qdrant payload."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    filename: str
    page_numbers: list[int]
    headings: list[str]
    excerpt: str


class QueryResponse(BaseModel):
    """Public grounded-answer response."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    abstained: bool
    abstention_reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)

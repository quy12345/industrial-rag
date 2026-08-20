"""Application settings loaded from environment variables and .env."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the application, dense retrieval, and hybrid retrieval."""

    app_name: str = "Industrial Technical Manual RAG"
    app_version: str = "0.1.0"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    qdrant_url: str = "http://localhost"
    qdrant_port: int = 6333
    qdrant_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    qdrant_collection: str = "industrial_manual_chunks"
    dense_vector_name: str = "dense"
    qdrant_hybrid_collection: str = "industrial_manual_chunks_v2"
    sparse_vector_name: str = "sparse"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_cache_dir: str | None = None
    embedding_batch_size: int = Field(default=16, gt=0)
    sparse_model: str = "Qdrant/bm25"
    sparse_embedding_batch_size: int = Field(default=64, gt=0)
    bm25_disable_stemmer: bool = True
    bm25_k: float = Field(default=1.2, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    bm25_avg_len: float | None = Field(default=None, gt=0)
    dense_candidate_limit: int = Field(default=20, ge=5)
    sparse_candidate_limit: int = Field(default=20, ge=5)
    hybrid_final_limit: int = Field(default=5, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    retrieval_top_k: int = Field(default=5, gt=0)
    retrieval_score_threshold: float | None = None
    rerank_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    rerank_cache_dir: str | None = None
    rerank_batch_size: int = Field(default=16, gt=0)
    rerank_threads: int | None = Field(default=None, ge=1, le=4)
    rerank_deduplicate_content: bool = False
    rerank_candidate_strategy: Literal["sparse", "hybrid", "union"] | None = None
    rerank_final_limit: int = Field(default=5, gt=0)
    retrieval_strategy: Literal["union", "sparse"] = "union"
    retrieval_profile: Literal["phase6", "phase7"] = "phase6"
    rerank_enabled: bool = True
    evidence_score_threshold: float | None = None
    generation_max_context_chars: int = Field(default=24_000, ge=4_000)
    citation_excerpt_max_chars: int = Field(default=400, ge=50, le=4_000)
    generation_provider: Literal["openai", "gemini"] = "openai"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-terra"
    openai_reasoning_effort: Literal["low", "medium", "high"] = "low"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_reasoning_effort: Literal["minimal", "low", "medium", "high"] = "minimal"
    gemini_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    openai_max_output_tokens: int = Field(default=800, gt=0, le=4_096)
    openai_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    openai_max_retries: int = Field(default=1, ge=0, le=2)
    openai_store: bool = False
    api_auth_enabled: bool = False
    api_auth_key: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    @field_validator(
        "qdrant_collection",
        "dense_vector_name",
        "qdrant_hybrid_collection",
        "sparse_vector_name",
        "embedding_model",
        "sparse_model",
        "rerank_model",
        "openai_model",
        "gemini_model",
    )
    @classmethod
    def validate_non_empty_name(cls, value: str) -> str:
        """Reject empty names while normalizing surrounding whitespace."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("gemini_base_url")
    @classmethod
    def validate_gemini_base_url(cls, value: str) -> str:
        """Normalize the explicit Gemini OpenAI-compatibility endpoint."""

        normalized = value.strip()
        if not normalized.startswith("https://"):
            raise ValueError("must be an HTTPS URL")
        return normalized.rstrip("/") + "/"

    @field_validator("embedding_cache_dir", "rerank_cache_dir")
    @classmethod
    def normalize_embedding_cache_dir(cls, value: str | None) -> str | None:
        """Treat blank cache-directory configuration as the library default."""

        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @property
    def generation_api_key(self) -> SecretStr | None:
        """Return the credential for the selected generation provider."""

        if self.generation_provider == "gemini":
            return self.gemini_api_key
        return self.openai_api_key

    @property
    def generation_model(self) -> str:
        """Return the model ID for the selected generation provider."""

        if self.generation_provider == "gemini":
            return self.gemini_model
        return self.openai_model


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()

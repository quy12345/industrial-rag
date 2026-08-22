"""Application settings loaded from environment variables and .env."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.domain.retrieval_contracts import FrozenRetrievalContract, retrieval_contract_for
from app.errors import RetrievalUnavailableError


class Settings(BaseSettings):
    """Runtime settings for the application, dense retrieval, and hybrid retrieval."""

    app_name: str = "Industrial Technical Manual RAG"
    app_version: str = "0.1.0"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    qdrant_url: str = "http://localhost"
    qdrant_port: int = 6333
    qdrant_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    qdrant_collection: str = "industrial_manual_phase7_dense_v1"
    dense_vector_name: str = "dense"
    qdrant_hybrid_collection: str = "industrial_manual_phase7_hybrid_v1"
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
    dense_candidate_limit: int = Field(default=60, ge=5)
    sparse_candidate_limit: int = Field(default=40, ge=5)
    hybrid_final_limit: int = Field(default=5, gt=0)
    rrf_k: int = Field(default=40, gt=0)
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
    retrieval_profile: Literal["phase7"] = "phase7"
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


def resolve_retrieval_runtime(
    settings: Settings,
) -> tuple[Settings, FrozenRetrievalContract]:
    """Apply one complete frozen profile without mixing mutable overrides into it."""

    # Settings validation makes Phase 7 the only constructible profile. Resolve the
    # canonical value rather than treating later mutation as a profile-selection path.
    contract = retrieval_contract_for("phase7")
    resolved = settings.model_copy(
        update={
            "qdrant_collection": contract.dense_collection,
            "qdrant_hybrid_collection": contract.hybrid_collection,
            "dense_vector_name": contract.dense_vector_name,
            "sparse_vector_name": contract.sparse_vector_name,
            "embedding_model": contract.dense_model,
            "sparse_model": contract.sparse_model,
            "rerank_model": contract.rerank_model,
            "dense_candidate_limit": contract.dense_candidate_limit,
            "sparse_candidate_limit": contract.sparse_candidate_limit,
            "rrf_k": contract.rrf_k,
            "bm25_k": contract.bm25_k,
            "bm25_b": contract.bm25_b,
            "bm25_avg_len": contract.bm25_avg_len,
            "bm25_disable_stemmer": contract.bm25_disable_stemmer,
            "rerank_deduplicate_content": True,
        }
    )
    validate_retrieval_settings(resolved, contract)
    return resolved, contract


def validate_retrieval_settings(
    settings: Settings,
    contract: FrozenRetrievalContract,
) -> None:
    """Reject partial profile overrides and unsupported strategy combinations."""

    if contract.phase7_fusion_profile is not None:
        profile = contract.phase7_fusion_profile
        if profile.rrf_k != contract.rrf_k:
            raise RetrievalUnavailableError(
                "Frozen Phase 7 fusion profile RRF k differs from the retrieval contract."
            )
        if profile.max_candidates != contract.union_rrf_prune_limit:
            raise RetrievalUnavailableError(
                "Frozen Phase 7 fusion profile candidate budget differs from the "
                "retrieval contract."
            )
    if contract.frozen_rerank_batch_size is not None and contract.frozen_rerank_batch_size <= 0:
        raise RetrievalUnavailableError("Frozen rerank batch size must be greater than zero.")
    expected = {
        "qdrant_collection": contract.dense_collection,
        "qdrant_hybrid_collection": contract.hybrid_collection,
        "dense_vector_name": contract.dense_vector_name,
        "sparse_vector_name": contract.sparse_vector_name,
        "embedding_model": contract.dense_model,
        "sparse_model": contract.sparse_model,
        "rerank_model": contract.rerank_model,
        "dense_candidate_limit": contract.dense_candidate_limit,
        "sparse_candidate_limit": contract.sparse_candidate_limit,
        "rrf_k": contract.rrf_k,
        "bm25_k": contract.bm25_k,
        "bm25_b": contract.bm25_b,
        "bm25_disable_stemmer": contract.bm25_disable_stemmer,
    }
    mismatches = [
        name
        for name, expected_value in expected.items()
        if getattr(settings, name) != expected_value
    ]
    if mismatches:
        raise RetrievalUnavailableError(
            "Runtime settings differ from the frozen retrieval contract: "
            + ", ".join(sorted(mismatches))
        )
    combination = (settings.retrieval_strategy, settings.rerank_enabled)
    if combination not in {("union", True), ("sparse", False)}:
        raise RetrievalUnavailableError(
            "Supported runtime combinations are union+rerank or sparse without reranking."
        )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()

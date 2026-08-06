"""Application settings loaded from environment variables and .env."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the application and dense retrieval."""

    app_name: str = "Industrial Technical Manual RAG"
    app_version: str = "0.1.0"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    qdrant_url: str = "http://localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "industrial_manual_chunks"
    dense_vector_name: str = "dense"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_cache_dir: str | None = None
    embedding_batch_size: int = Field(default=16, gt=0)
    retrieval_top_k: int = Field(default=5, gt=0)
    retrieval_score_threshold: float | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    @field_validator("qdrant_collection", "dense_vector_name", "embedding_model")
    @classmethod
    def validate_non_empty_name(cls, value: str) -> str:
        """Reject empty names while normalizing surrounding whitespace."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("embedding_cache_dir")
    @classmethod
    def normalize_embedding_cache_dir(cls, value: str | None) -> str | None:
        """Treat blank cache-directory configuration as the library default."""

        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()

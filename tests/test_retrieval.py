"""Unit tests for dense indexing and retrieval with in-memory Qdrant."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from app.config import Settings
from app.models import DocumentChunk
from app.retrieval import (
    RetrievalError,
    build_embedding_text,
    build_point_id,
    create_embedding_model,
    dense_search,
    ensure_dense_collection,
    get_indexed_chunk_ids,
    index_chunks,
    validate_index_manifest,
    write_index_manifest,
)

COLLECTION = "test_chunks"
VECTOR_NAME = "dense"


class FakeEmbeddingModel:
    """Deterministic keyword embedding model with three dimensions."""

    def passage_embed(self, texts):
        return (self._vector(text) for text in texts)

    def query_embed(self, query):
        texts = [query] if isinstance(query, str) else query
        return (self._vector(text) for text in texts)

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.casefold()
        if "sensor" in normalized or "cảm biến" in normalized:
            return [1.0, 0.0, 0.0]
        if "database" in normalized or "cơ sở dữ liệu" in normalized:
            return [0.0, 1.0, 0.0]
        if "voltage" in normalized or "điện áp" in normalized:
            return [0.0, 0.0, 1.0]
        return [0.1, 0.1, 0.1]


class FailingEmbeddingModel(FakeEmbeddingModel):
    """Fail only when embedding document passages, after dimension probing."""

    def passage_embed(self, texts):
        texts = list(texts)
        if texts == ["dimension probe"]:
            return iter([[0.1, 0.1, 0.1]])
        raise RuntimeError("embedding failed")


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "manual-a",
    page: int = 1,
    headings: list[str] | None = None,
) -> DocumentChunk:
    """Create one normalized test chunk."""

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"{document_id}.pdf",
        text=text,
        page_numbers=[page],
        headings=headings or [],
        content_type="text",
        metadata={
            "source_path": rf"C:\private\{document_id}.pdf",
            "character_count": len(text),
        },
    )


def test_embedding_text_includes_heading_without_mutating_chunk() -> None:
    chunk = make_chunk(
        "chunk-1",
        "Sensor voltage is monitored.",
        headings=["Safety", "Monitoring"],
    )
    original_text = chunk.text

    first = build_embedding_text(chunk)
    second = build_embedding_text(chunk)

    assert first == "Section: Safety > Monitoring\nContent:\nSensor voltage is monitored."
    assert second == first
    assert chunk.text == original_text


def test_embedding_text_without_heading_contains_raw_content() -> None:
    chunk = make_chunk("chunk-1", "Database maintenance procedure.")

    assert build_embedding_text(chunk) == "Content:\nDatabase maintenance procedure."


def test_point_id_is_deterministic_valid_uuid() -> None:
    first = build_point_id("manual_p1_c0000")
    second = build_point_id("manual_p1_c0000")

    assert first == second
    assert str(UUID(first)) == first
    assert build_point_id("manual_p1_c0001") != first


@pytest.mark.parametrize("field_name", ["embedding_batch_size", "retrieval_top_k"])
def test_retrieval_settings_require_positive_integers(field_name: str) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field_name: 0})


@pytest.mark.parametrize(
    "field_name",
    ["qdrant_collection", "dense_vector_name", "embedding_model"],
)
def test_retrieval_settings_reject_empty_names(field_name: str) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field_name: "   "})


def test_retrieval_settings_reject_invalid_score_threshold() -> None:
    with pytest.raises(ValidationError):
        Settings(retrieval_score_threshold="not-a-number")


def test_retrieval_settings_normalize_optional_embedding_cache_directory() -> None:
    assert Settings(embedding_cache_dir="  /models/cache  ").embedding_cache_dir == "/models/cache"
    assert Settings(embedding_cache_dir="   ").embedding_cache_dir is None


def test_embedding_model_receives_optional_cache_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, str | None] = {}

    class FakeTextEmbedding:
        def __init__(self, *, model_name: str, cache_dir: str | None) -> None:
            created.update(model_name=model_name, cache_dir=cache_dir)

        @staticmethod
        def list_supported_models():
            return [{"model": "test-model"}]

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeTextEmbedding))

    create_embedding_model("test-model", cache_dir="/models/cache")

    assert created == {"model_name": "test-model", "cache_dir": "/models/cache"}


def test_ensure_dense_collection_creates_named_cosine_vector() -> None:
    client = QdrantClient(":memory:")

    ensure_dense_collection(
        client,
        collection_name=COLLECTION,
        vector_name=VECTOR_NAME,
        vector_size=3,
    )

    vectors = client.get_collection(COLLECTION).config.params.vectors
    assert isinstance(vectors, dict)
    assert vectors[VECTOR_NAME].size == 3
    assert vectors[VECTOR_NAME].distance == models.Distance.COSINE


def test_ensure_dense_collection_does_not_recreate_compatible_collection() -> None:
    client = QdrantClient(":memory:")
    ensure_dense_collection(
        client,
        collection_name=COLLECTION,
        vector_name=VECTOR_NAME,
        vector_size=3,
    )
    client.upsert(
        COLLECTION,
        [models.PointStruct(id=1, vector={VECTOR_NAME: [1.0, 0.0, 0.0]})],
    )

    ensure_dense_collection(
        client,
        collection_name=COLLECTION,
        vector_name=VECTOR_NAME,
        vector_size=3,
    )

    assert client.count(COLLECTION).count == 1


def test_ensure_dense_collection_rejects_dimension_mismatch() -> None:
    client = QdrantClient(":memory:")
    ensure_dense_collection(
        client,
        collection_name=COLLECTION,
        vector_name=VECTOR_NAME,
        vector_size=3,
    )

    with pytest.raises(RetrievalError, match="uses vector size 3, but model produces 4"):
        ensure_dense_collection(
            client,
            collection_name=COLLECTION,
            vector_name=VECTOR_NAME,
            vector_size=4,
        )


def test_indexing_payload_and_reindex_behavior() -> None:
    client = QdrantClient(":memory:")
    model = FakeEmbeddingModel()
    document_a = [
        make_chunk("a-1", "Sensor monitoring", page=1),
        make_chunk("a-2", "Voltage limits", page=2),
    ]
    document_b = [
        make_chunk("b-1", "Database storage", document_id="manual-b", page=4),
    ]

    assert _index(client, model, document_a) == 2
    assert _index(client, model, document_b) == 1
    assert client.count(COLLECTION).count == 3

    assert _index(client, model, document_a[:1]) == 1
    assert client.count(COLLECTION).count == 2

    points, _ = client.scroll(COLLECTION, limit=10, with_payload=True)
    payloads = {point.payload["chunk_id"]: point.payload for point in points}
    assert set(payloads) == {"a-1", "b-1"}
    assert payloads["a-1"]["page_numbers"] == [1]
    assert payloads["a-1"]["headings"] == []
    assert payloads["a-1"]["source_path"] == "manual-a.pdf"
    assert payloads["a-1"]["character_count"] == len("Sensor monitoring")
    assert "embedding_text" not in payloads["a-1"]


def test_get_indexed_chunk_ids_returns_document_payload_ids() -> None:
    client = QdrantClient(":memory:")
    model = FakeEmbeddingModel()
    _index(client, model, [make_chunk("a-1", "Sensor monitoring")])
    _index(client, model, [make_chunk("b-1", "Database", document_id="manual-b")])

    assert get_indexed_chunk_ids(
        client,
        collection_name=COLLECTION,
        document_id="manual-a",
    ) == {"a-1"}


def test_indexing_rejects_empty_chunks() -> None:
    with pytest.raises(RetrievalError, match="empty chunk list"):
        _index(QdrantClient(":memory:"), FakeEmbeddingModel(), [])


def test_embedding_failure_does_not_delete_existing_points() -> None:
    client = QdrantClient(":memory:")
    original_chunks = [make_chunk("a-1", "Sensor monitoring"), make_chunk("a-2", "Voltage limits")]
    _index(client, FakeEmbeddingModel(), original_chunks)

    with pytest.raises(RetrievalError, match="Failed to embed document chunks"):
        _index(client, FailingEmbeddingModel(), [make_chunk("a-new", "New chunk")])

    assert client.count(COLLECTION).count == 2


def test_upsert_failure_does_not_delete_existing_points(monkeypatch: pytest.MonkeyPatch) -> None:
    client = QdrantClient(":memory:")
    original_chunks = [make_chunk("a-1", "Sensor monitoring"), make_chunk("a-2", "Voltage limits")]
    _index(client, FakeEmbeddingModel(), original_chunks)
    original_upsert = client.upsert
    calls = 0

    def fail_on_second_upsert(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("upsert failed")
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(client, "upsert", fail_on_second_upsert)
    with pytest.raises(RetrievalError, match="Failed to update Qdrant collection"):
        _index(
            client,
            FakeEmbeddingModel(),
            [
                make_chunk("new-1", "Sensor one"),
                make_chunk("new-2", "Sensor two"),
                make_chunk("new-3", "Sensor three"),
            ],
        )

    points, _ = client.scroll(COLLECTION, limit=10, with_payload=True)
    chunk_ids = {point.payload["chunk_id"] for point in points}
    assert {"a-1", "a-2"}.issubset(chunk_ids)


def test_dense_search_returns_ranked_payload_and_respects_limit() -> None:
    client = QdrantClient(":memory:")
    model = FakeEmbeddingModel()
    chunks = [
        make_chunk("sensor", "Sensor anomaly detection", page=3, headings=["Detection"]),
        make_chunk("database", "Database indexing", page=5),
        make_chunk("voltage", "Voltage threshold", page=7),
    ]
    _index(client, model, chunks)

    results = _search(client, model, "Which sensor detects anomalies?", limit=2)

    assert len(results) == 2
    assert results[0].chunk_id == "sensor"
    assert results[0].page_numbers == [3]
    assert results[0].headings == ["Detection"]
    assert results[0].score == pytest.approx(1.0)
    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )


def test_dense_search_uses_qdrant_document_filter() -> None:
    client = QdrantClient(":memory:")
    model = FakeEmbeddingModel()
    _index(client, model, [make_chunk("a-sensor", "Sensor A")])
    _index(
        client,
        model,
        [make_chunk("b-sensor", "Sensor B", document_id="manual-b")],
    )

    results = _search(
        client,
        model,
        "sensor",
        limit=5,
        document_id="manual-b",
    )

    assert [result.chunk_id for result in results] == ["b-sensor"]
    assert {result.document_id for result in results} == {"manual-b"}


def test_dense_search_rejects_empty_question() -> None:
    with pytest.raises(RetrievalError, match="must not be empty"):
        _search(QdrantClient(":memory:"), FakeEmbeddingModel(), "   ", limit=5)


def test_dense_search_rejects_incomplete_payload() -> None:
    client = QdrantClient(":memory:")
    ensure_dense_collection(
        client,
        collection_name=COLLECTION,
        vector_name=VECTOR_NAME,
        vector_size=3,
    )
    client.upsert(
        COLLECTION,
        [
            models.PointStruct(
                id=1,
                vector={VECTOR_NAME: [1.0, 0.0, 0.0]},
                payload={"chunk_id": "missing-fields"},
            )
        ],
    )

    with pytest.raises(RetrievalError, match="Invalid payload"):
        _search(client, FakeEmbeddingModel(), "sensor", limit=1)


def test_index_manifest_round_trip_and_mismatch(tmp_path) -> None:
    manifest = tmp_path / "dense-index-manifest.json"
    write_index_manifest(
        manifest,
        collection_name=COLLECTION,
        vector_name=VECTOR_NAME,
        embedding_model="model-a",
        embedding_dimension=3,
        ingestion_profile={"chunker": "hierarchical"},
    )

    validate_index_manifest(
        manifest,
        collection_name=COLLECTION,
        vector_name=VECTOR_NAME,
        embedding_model="model-a",
        embedding_dimension=3,
    )
    with pytest.raises(RetrievalError, match="re-index"):
        validate_index_manifest(
            manifest,
            collection_name=COLLECTION,
            vector_name=VECTOR_NAME,
            embedding_model="model-b",
            embedding_dimension=3,
        )


def _index(
    client: QdrantClient,
    model: FakeEmbeddingModel,
    chunks: list[DocumentChunk],
) -> int:
    """Index test chunks with shared settings."""

    return index_chunks(
        client,
        chunks,
        collection_name=COLLECTION,
        vector_name=VECTOR_NAME,
        embedding_model=model,
        embedding_batch_size=2,
    )


def _search(
    client: QdrantClient,
    model: FakeEmbeddingModel,
    question: str,
    *,
    limit: int,
    document_id: str | None = None,
):
    """Search test chunks with shared settings."""

    return dense_search(
        client,
        question,
        collection_name=COLLECTION,
        vector_name=VECTOR_NAME,
        embedding_model=model,
        limit=limit,
        document_id=document_id,
    )

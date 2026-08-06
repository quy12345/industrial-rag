"""Offline unit tests for sparse BM25 indexing/search and client-side RRF."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from app.config import Settings
from app.hybrid_retrieval import (
    HYBRID_SCHEMA_VERSION,
    compute_bm25_average_length,
    ensure_hybrid_collection,
    fuse_rrf,
    hybrid_search,
    index_hybrid_chunks,
    sparse_search,
    validate_hybrid_index_manifest,
    write_hybrid_index_manifest,
)
from app.models import DocumentChunk, RetrievalCandidate
from app.retrieval import RetrievalError, build_point_id, ensure_dense_collection

V1 = "dense-v1"
V2 = "hybrid-v2"
DENSE = "dense"
SPARSE = "sparse"


class FakeDenseModel:
    """Small deterministic model that never downloads a real embedding model."""

    def passage_embed(self, texts):
        return (self._vector(text) for text in texts)

    def query_embed(self, query):
        values = [query] if isinstance(query, str) else query
        return (self._vector(value) for value in values)

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.casefold()
        if "sensor" in normalized:
            return [1.0, 0.0, 0.0]
        if "plc" in normalized:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class FakeSparseModel:
    """Sparse keyword embeddings with explicit passage/query call tracking."""

    def __init__(self) -> None:
        self.passage_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def passage_embed(self, texts):
        values = list(texts)
        self.passage_calls.append(values)
        return (self._vector(text) for text in values)

    def query_embed(self, question):
        self.query_calls.append(question)
        values = [question] if isinstance(question, str) else question
        return (self._vector(value) for value in values)

    @staticmethod
    def _vector(text: str) -> SimpleNamespace:
        indices = []
        if "sensor" in text.casefold():
            indices.append(1)
        if "plc" in text.casefold():
            indices.append(2)
        if not indices:
            indices.append(3)
        return SimpleNamespace(indices=indices, values=[1.0] * len(indices))


class FailingSparseModel(FakeSparseModel):
    def passage_embed(self, texts):
        raise RuntimeError("sparse failed")


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "manual-a",
    page: int = 1,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"{document_id}.pdf",
        text=text,
        page_numbers=[page],
        headings=["Safety"],
        content_type="text",
        metadata={"source_path": f"C:/private/{document_id}.pdf", "character_count": len(text)},
    )


def candidate(
    chunk_id: str,
    *,
    dense_score: float | None = None,
    dense_rank: int | None = None,
    sparse_score: float | None = None,
    sparse_rank: int | None = None,
) -> RetrievalCandidate:
    score = dense_score if dense_score is not None else sparse_score
    assert score is not None
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="manual-a",
        filename="manual-a.pdf",
        text=chunk_id,
        page_numbers=[1],
        headings=[],
        content_type="text",
        score=score,
        dense_score=dense_score,
        dense_rank=dense_rank,
        sparse_score=sparse_score,
        sparse_rank=sparse_rank,
    )


def test_hybrid_settings_default_to_bm25_with_stemming_disabled() -> None:
    settings = Settings()

    assert settings.qdrant_hybrid_collection == "industrial_manual_chunks_v2"
    assert settings.sparse_model == "Qdrant/bm25"
    assert settings.bm25_disable_stemmer is True
    with pytest.raises(ValidationError):
        Settings(bm25_b=1.1)
    with pytest.raises(ValidationError):
        Settings(rrf_k=0)


def test_compute_average_length_uses_active_bm25_preprocessing() -> None:
    class FakeTokenizer:
        @staticmethod
        def tokenize(text: str) -> list[str]:
            return text.split()

    class FakeBackend:
        tokenizer = FakeTokenizer()

        @staticmethod
        def _stem(tokens: list[str]) -> list[str]:
            return [token for token in tokens if token != "Content:"]

    model = SimpleNamespace(model=FakeBackend())
    average = compute_bm25_average_length(
        model,
        [make_chunk("a", "24 VDC IP65 PLC"), make_chunk("b", "S7-1200 E-Stop 3.5 bar")],
    )

    assert average > 0


def test_hybrid_schema_creates_dense_sparse_idf_without_touching_v1() -> None:
    client = QdrantClient(":memory:")
    ensure_dense_collection(client, collection_name=V1, vector_name=DENSE, vector_size=3)
    client.upsert(V1, [models.PointStruct(id=1, vector={DENSE: [1.0, 0.0, 0.0]})])

    with pytest.warns(UserWarning, match="Payload indexes have no effect"):
        ensure_hybrid_collection(
            client,
            collection_name=V2,
            dense_vector_name=DENSE,
            dense_vector_size=3,
            sparse_vector_name=SPARSE,
        )
    collection = client.get_collection(V2)
    assert collection.config.params.vectors[DENSE].size == 3
    assert collection.config.params.sparse_vectors[SPARSE].modifier == models.Modifier.IDF
    assert client.count(V1).count == 1


def test_hybrid_schema_mismatch_fails_without_deleting_existing_points() -> None:
    client = QdrantClient(":memory:")
    ensure_hybrid_collection(
        client,
        collection_name=V2,
        dense_vector_name=DENSE,
        dense_vector_size=3,
        sparse_vector_name=SPARSE,
    )
    client.upsert(
        V2,
        [
            models.PointStruct(
                id=1,
                vector={
                    DENSE: [1.0, 0.0, 0.0],
                    SPARSE: models.SparseVector(indices=[1], values=[1.0]),
                },
                payload={"document_id": "manual-a"},
            )
        ],
    )

    with pytest.raises(RetrievalError, match="dense vector size"):
        ensure_hybrid_collection(
            client,
            collection_name=V2,
            dense_vector_name=DENSE,
            dense_vector_size=4,
            sparse_vector_name=SPARSE,
        )
    assert client.count(V2).count == 1


def test_hybrid_index_reindex_and_sparse_failure_safety() -> None:
    client = QdrantClient(":memory:")
    dense_model = FakeDenseModel()
    sparse_model = FakeSparseModel()
    original = [make_chunk("a-1", "Sensor 24 VDC"), make_chunk("a-2", "PLC IP65", page=2)]
    other = [make_chunk("b-1", "Sensor other", document_id="manual-b")]

    assert _index(client, dense_model, sparse_model, original) == 2
    assert _index(client, dense_model, sparse_model, other) == 1
    assert _index(client, dense_model, sparse_model, original[:1]) == 1
    assert client.count(V2).count == 2

    with pytest.raises(RetrievalError, match="Failed to embed sparse hybrid passages"):
        _index(client, dense_model, FailingSparseModel(), [make_chunk("replacement", "PLC")])
    points, _ = client.scroll(V2, limit=10, with_payload=True, with_vectors=False)
    payloads = {point.payload["chunk_id"]: point.payload for point in points}
    assert set(payloads) == {"a-1", "b-1"}
    assert "embedding_text" not in payloads["a-1"]
    assert build_point_id("a-1") in {str(point.id) for point in points}
    assert sparse_model.passage_calls


def test_sparse_search_filters_documents_and_preserves_metadata() -> None:
    client = QdrantClient(":memory:")
    sparse_model = FakeSparseModel()
    _index(client, FakeDenseModel(), sparse_model, [make_chunk("a", "Sensor 24 VDC")])
    _index(
        client,
        FakeDenseModel(),
        sparse_model,
        [make_chunk("b", "Sensor PLC", document_id="manual-b")],
    )

    results = sparse_search(
        client,
        "sensor",
        collection_name=V2,
        sparse_vector_name=SPARSE,
        sparse_embedding_model=sparse_model,
        limit=5,
        document_id="manual-b",
    )

    assert [result.chunk_id for result in results] == ["b"]
    assert results[0].sparse_rank == 1
    assert results[0].metadata["source_path"] == "manual-b.pdf"
    assert sparse_model.query_calls == ["sensor"]
    with pytest.raises(RetrievalError, match="must not be empty"):
        sparse_search(
            client,
            " ",
            collection_name=V2,
            sparse_vector_name=SPARSE,
            sparse_embedding_model=sparse_model,
            limit=5,
        )


def test_rrf_formula_duplicate_collapse_ties_and_empty_components() -> None:
    dense = [
        candidate("a", dense_score=0.9, dense_rank=1),
        candidate("b", dense_score=0.8, dense_rank=2),
    ]
    sparse = [
        candidate("b", sparse_score=20.0, sparse_rank=1),
        candidate("c", sparse_score=10.0, sparse_rank=2),
    ]

    fused = fuse_rrf(dense, sparse, rrf_k=60, final_limit=5)
    assert [result.chunk_id for result in fused] == ["b", "a", "c"]
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].dense_score == 0.8
    assert fused[0].sparse_score == 20.0
    assert fuse_rrf([], [], rrf_k=60, final_limit=5) == []
    with pytest.raises(RetrievalError, match="RRF k"):
        fuse_rrf(dense, sparse, rrf_k=0, final_limit=5)


def test_hybrid_search_uses_component_limits_and_does_not_mix_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, int] = {}

    def fake_dense(*args, **kwargs):
        calls["dense"] = kwargs["limit"]
        return [
            SimpleNamespace(
                chunk_id="dense-only",
                document_id="manual-a",
                filename="manual.pdf",
                text="dense",
                page_numbers=[1],
                headings=[],
                content_type="text",
                score=0.99,
            )
        ]

    def fake_sparse(*args, **kwargs):
        calls["sparse"] = kwargs["limit"]
        return [candidate("sparse-only", sparse_score=999.0, sparse_rank=1)]

    monkeypatch.setattr("app.hybrid_retrieval.dense_search", fake_dense)
    monkeypatch.setattr("app.hybrid_retrieval.sparse_search", fake_sparse)
    results = hybrid_search(
        QdrantClient(":memory:"),
        "question",
        collection_name=V2,
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
        dense_embedding_model=FakeDenseModel(),
        sparse_embedding_model=FakeSparseModel(),
        dense_candidate_limit=20,
        sparse_candidate_limit=15,
        final_limit=1,
        rrf_k=60,
    )

    assert calls == {"dense": 20, "sparse": 15}
    assert results[0].rrf_score == pytest.approx(1 / 61)
    assert results[0].score == results[0].rrf_score


def test_hybrid_manifest_round_trip_and_mismatch(tmp_path: Path) -> None:
    settings = Settings(qdrant_hybrid_collection=V2)
    frozen = {"chunk_count": 2, "document_ids": ["manual-a"], "chunk_ids_sha256": "hash"}
    manifest = tmp_path / "hybrid-index-manifest.json"
    write_hybrid_index_manifest(
        manifest,
        settings=settings,
        dense_dimension=3,
        bm25_avg_len=12.5,
        frozen_chunk_set=frozen,
        ingestion_profile={"page_batch_size": 4},
    )

    payload = validate_hybrid_index_manifest(
        manifest, settings=settings, dense_dimension=3, frozen_chunk_set=frozen
    )
    assert payload["schema_version"] == HYBRID_SCHEMA_VERSION
    assert payload["bm25_avg_len"] == 12.5
    with pytest.raises(RetrievalError, match="does not match"):
        validate_hybrid_index_manifest(
            manifest,
            settings=Settings(qdrant_hybrid_collection=V2, bm25_k=2.0),
            dense_dimension=3,
            frozen_chunk_set=frozen,
        )


def _index(
    client: QdrantClient,
    dense_model: FakeDenseModel,
    sparse_model: FakeSparseModel,
    chunks: list[DocumentChunk],
) -> int:
    return index_hybrid_chunks(
        client,
        chunks,
        collection_name=V2,
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
        dense_embedding_model=dense_model,
        sparse_embedding_model=sparse_model,
        dense_embedding_batch_size=2,
        sparse_embedding_batch_size=2,
        dense_vector_size=3,
    )

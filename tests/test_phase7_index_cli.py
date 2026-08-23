"""Offline characterization of the supported Phase 7 indexing command."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.index_phase7_corpus as compatibility_index_cli
from app.application.indexing_service import (
    IndexingSafetyError,
    Phase7IndexingService,
    validate_chunk_preview,
)
from app.domain.documents import DocumentChunk
from app.infrastructure.corpus_artifacts import (
    PHASE7_DENSE_COLLECTION,
    PHASE7_HYBRID_COLLECTION,
)
from scripts.operations import index_phase7_corpus as index_cli


def test_indexing_shim_and_parser_preserve_supported_contract() -> None:
    assert compatibility_index_cli.main is index_cli.main
    assert compatibility_index_cli._parser is index_cli._parser

    args = index_cli._parser().parse_args([])
    assert args.inputs == list(index_cli.DEFAULT_INPUTS)
    assert args.page_batch_size == 16
    assert args.chunker == "hybrid"
    assert args.dense_collection == PHASE7_DENSE_COLLECTION
    assert args.hybrid_collection == PHASE7_HYBRID_COLLECTION
    assert args.preview_only is False
    assert args.verify_reindex is False


def _chunk(chunk_id: str, document_id: str, index: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"{document_id}.pdf",
        text=f"Complete technical content for {chunk_id}." * 4,
        page_numbers=[index + 1],
        headings=["Section"],
        metadata={"chunk_index": index},
    )


def _chunks_by_document() -> dict[str, list[DocumentChunk]]:
    return {
        "installation": [_chunk("i-1", "installation", 0), _chunk("i-2", "installation", 1)],
        "programming": [_chunk("p-1", "programming", 0)],
    }


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        qdrant_collection="dense",
        qdrant_hybrid_collection="hybrid",
        dense_vector_name="dense-vector",
        sparse_vector_name="sparse-vector",
        embedding_batch_size=8,
        sparse_embedding_batch_size=4,
    )


@pytest.mark.parametrize(
    ("dense_collection", "hybrid_collection", "message"),
    [
        ("industrial_manual_chunks", "phase7-hybrid", "protected Phase 3--6"),
        ("same", "same", "must differ"),
    ],
)
def test_cli_rejects_invalid_targets_before_external_access(
    monkeypatch: pytest.MonkeyPatch,
    dense_collection: str,
    hybrid_collection: str,
    message: str,
) -> None:
    args = SimpleNamespace(
        dense_collection=dense_collection,
        hybrid_collection=hybrid_collection,
    )
    monkeypatch.setattr(index_cli, "_parser", lambda: SimpleNamespace(parse_args=lambda: args))

    def unexpected_external_access(*args, **kwargs):
        raise AssertionError("target validation must run before external access")

    monkeypatch.setattr(index_cli, "ingest_document", unexpected_external_access)
    monkeypatch.setattr(index_cli, "load_frozen_chunks", unexpected_external_access)

    with pytest.raises(SystemExit, match=message):
        index_cli.main()


def test_index_once_preserves_dense_then_hybrid_order_per_document() -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []

    def dense_indexer(client, chunks, **kwargs):
        calls.append(("dense", tuple(chunk.chunk_id for chunk in chunks)))

    def hybrid_indexer(client, chunks, **kwargs):
        calls.append(("hybrid", tuple(chunk.chunk_id for chunk in chunks)))

    service = Phase7IndexingService(
        client=object(),
        settings=_settings(),
        dense_model=object(),
        sparse_model=object(),
        dense_dimension=3,
        dense_indexer=dense_indexer,
        hybrid_indexer=hybrid_indexer,
        indexed_chunk_reader=lambda *args, **kwargs: set(),
    )
    service.index_once(_chunks_by_document())

    assert calls == [
        ("dense", ("i-1", "i-2")),
        ("hybrid", ("i-1", "i-2")),
        ("dense", ("p-1",)),
        ("hybrid", ("p-1",)),
    ]


def test_preview_validation_preserves_chunk_contract() -> None:
    chunks_by_document = _chunks_by_document()

    validate_chunk_preview(chunks_by_document)
    chunks_by_document["installation"][1].metadata["chunk_index"] = 3

    with pytest.raises(IndexingSafetyError, match="non-contiguous chunk indices"):
        validate_chunk_preview(chunks_by_document)


def test_protected_collection_guard_requires_frozen_99_point_archives() -> None:
    class Client:
        counts = {
            "industrial_manual_chunks": 99,
            "industrial_manual_chunks_v2": 99,
        }

        def count(self, collection_name: str, *, exact: bool):
            assert exact is True
            return SimpleNamespace(count=self.counts[collection_name])

    client = Client()
    service = Phase7IndexingService(
        client=client,
        settings=_settings(),
        dense_model=object(),
        sparse_model=object(),
        dense_dimension=3,
        dense_indexer=lambda *args, **kwargs: 0,
        hybrid_indexer=lambda *args, **kwargs: 0,
        indexed_chunk_reader=lambda *args, **kwargs: set(),
    )
    service.verify_protected_collections(
        dense_collection="industrial_manual_chunks",
        hybrid_collection="industrial_manual_chunks_v2",
        expected_count=99,
    )
    client.counts["industrial_manual_chunks_v2"] = 98

    with pytest.raises(IndexingSafetyError, match="not frozen at 99/99: 99/98"):
        service.verify_protected_collections(
            dense_collection="industrial_manual_chunks",
            hybrid_collection="industrial_manual_chunks_v2",
            expected_count=99,
        )


def test_index_verification_checks_total_and_each_document_chunk_set() -> None:
    chunks_by_document = _chunks_by_document()
    calls: list[tuple[str, str]] = []

    class Client:
        @staticmethod
        def count(collection_name: str, *, exact: bool):
            assert exact is True
            return SimpleNamespace(count=3)

    def indexed_ids(client, *, collection_name: str, document_id: str) -> set[str]:
        calls.append((collection_name, document_id))
        return {chunk.chunk_id for chunk in chunks_by_document[document_id]}

    service = Phase7IndexingService(
        client=Client(),
        settings=_settings(),
        dense_model=object(),
        sparse_model=object(),
        dense_dimension=3,
        dense_indexer=lambda *args, **kwargs: 0,
        hybrid_indexer=lambda *args, **kwargs: 0,
        indexed_chunk_reader=indexed_ids,
    )
    service.verify_index(chunks_by_document)

    assert calls == [
        ("dense", "installation"),
        ("dense", "programming"),
        ("hybrid", "installation"),
        ("hybrid", "programming"),
    ]

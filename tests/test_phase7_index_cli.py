"""Offline characterization of the supported Phase 7 indexing command."""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.application.indexing_service import (
    CorpusIndexingService,
    IndexingSafetyError,
    validate_chunk_preview,
)
from app.domain.documents import DocumentChunk
from app.infrastructure.corpus_artifacts import (
    ATV320_DENSE_COLLECTION,
    ATV320_HYBRID_COLLECTION,
)
from scripts.operations import index_phase7_corpus as index_cli

PROJECT_ROOT = Path(__file__).parents[1]


def test_indexing_parser_preserves_supported_contract() -> None:
    args = index_cli._parser().parse_args([])
    assert args.inputs == list(index_cli.DEFAULT_INPUTS)
    assert args.page_batch_size == index_cli.FROZEN_PAGE_BATCH_SIZE == 64
    assert args.chunker == "hybrid"
    assert args.dense_collection == ATV320_DENSE_COLLECTION
    assert args.hybrid_collection == ATV320_HYBRID_COLLECTION
    assert args.preview_only is False
    assert args.verify_reindex is False


def test_runtime_dependencies_preserve_frozen_versions() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    retrieval_dependencies = {
        dependency.replace(" ", "")
        for dependency in project["optional-dependencies"]["retrieval"]
    }
    assert retrieval_dependencies == {
        "qdrant-client[fastembed]>=1.19.0,<1.20.0",
        "fastembed==0.8.0",
        "onnxruntime==1.28.0",
    }
    ingestion_dependencies = {
        "docling==2.117.0",
        "docling-core==2.90.0",
        "docling-ibm-models==3.13.3",
        "docling-parse==7.10.0",
        "huggingface-hub==1.26.0",
        "pypdfium2==5.12.1",
        "semchunk==3.2.5",
        "tokenizers==0.22.2",
        "transformers==5.14.1",
    }
    assert set(project["optional-dependencies"]["ingestion"]) == ingestion_dependencies
    assert ingestion_dependencies | {"onnxruntime==1.28.0"} <= set(
        project["optional-dependencies"]["dev"]
    )


def test_ingestion_compose_mount_matches_default_input_paths() -> None:
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "./data/raw:/app/data/raw:ro" in compose
    assert "./data/raw:/data/raw:ro" not in compose


def test_ingestion_build_uses_resumable_pip_and_shared_cache() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert '"pip==26.2.1"' in dockerfile
    assert dockerfile.count(
        "--mount=type=cache,id=industrial-rag-pip,target=/root/.cache/pip,sharing=locked"
    ) == 3
    assert dockerfile.count("--retries 10") == 3
    assert dockerfile.count("--resume-retries 10") == 2
    assert dockerfile.count("--timeout 120") == 3
    assert 'python -m pip install --no-cache-dir "pip==26.2.1"' not in dockerfile
    assert 'pip install --no-cache-dir ".[retrieval]"' not in dockerfile
    assert 'pip install --no-cache-dir ".[retrieval,ingestion,llm]"' not in dockerfile


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
        ("industrial_manual_chunks", "phase7-hybrid", "protected historical"),
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

    service = CorpusIndexingService(
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
    service = CorpusIndexingService(
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

    service = CorpusIndexingService(
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

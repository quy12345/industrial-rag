"""Unit tests for document ingestion normalization and CLI-independent helpers."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from docling.datamodel.base_models import ConversionStatus

import app.ingestion as ingestion
import scripts.ingest_preview as ingest_preview
from app.models import DocumentChunk


def test_document_id_is_deterministic_and_content_based(tmp_path: Path) -> None:
    first = tmp_path / "Motor Drive Manual!!.PDF"
    same_name = tmp_path / "same" / first.name
    same_content = tmp_path / "other" / first.name
    same_name.parent.mkdir()
    same_content.parent.mkdir()
    first.write_bytes(b"manual contents")
    same_name.write_bytes(b"different contents")
    same_content.write_bytes(b"manual contents")

    assert ingestion.build_document_id(first) == ingestion.build_document_id(first)
    assert ingestion.build_document_id(first) != ingestion.build_document_id(same_name)
    assert ingestion.build_document_id(first) == ingestion.build_document_id(same_content)
    assert " " not in ingestion.build_document_id(first)
    assert ingestion.build_document_id(first).startswith("motor-drive-manual-")


@pytest.mark.parametrize("filename", ["manual.PDF", "manual.DOCX"])
def test_uppercase_supported_extensions(tmp_path: Path, filename: str) -> None:
    document = tmp_path / filename
    document.write_bytes(b"content")

    assert ingestion.validate_input_path(document) == document


def test_input_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(ingestion.IngestionError, match="does not exist"):
        ingestion.validate_input_path(tmp_path / "missing.pdf")

    directory = tmp_path / "manual.pdf"
    directory.mkdir()
    with pytest.raises(ingestion.IngestionError, match="not a file"):
        ingestion.validate_input_path(directory)

    unsupported = tmp_path / "manual.txt"
    unsupported.write_text("text", encoding="utf-8")
    with pytest.raises(ingestion.IngestionError, match=r"Unsupported document type: \.txt"):
        ingestion.validate_input_path(unsupported)


def test_chunk_id_is_content_stable_and_duplicate_safe() -> None:
    first = ingestion.build_chunk_id("manual-id", [19, 18], ["Safety"], "Disconnect power", 0)

    assert first == ingestion.build_chunk_id(
        "manual-id", [18, 19], ["Safety"], "Disconnect power", 0
    )
    assert first != ingestion.build_chunk_id(
        "manual-id", [18, 19], ["Safety"], "Disconnect power first", 0
    )
    assert first != ingestion.build_chunk_id(
        "manual-id", [18, 19], ["Safety"], "Disconnect power", 1
    )
    assert ingestion.build_chunk_id("manual-id", [], [], "Unknown page", 0).startswith(
        "manual-id_punknown_h"
    )


@pytest.mark.parametrize(
    ("start_page", "end_page", "batch_size", "expected"),
    [
        (1, 21, 8, [(1, 8), (9, 16), (17, 21)]),
        (1, 8, 8, [(1, 8)]),
        (1, 5, 8, [(1, 5)]),
        (9, 21, 8, [(9, 16), (17, 21)]),
    ],
)
def test_build_page_batches(
    start_page: int,
    end_page: int,
    batch_size: int,
    expected: list[tuple[int, int]],
) -> None:
    assert ingestion.build_page_batches(start_page, end_page, batch_size) == expected


@pytest.mark.parametrize(
    ("start_page", "end_page", "batch_size", "message"),
    [
        (0, 8, 8, "Page start"),
        (9, 8, 8, "Page end"),
        (1, 8, 0, "Batch size"),
        (1, 8, -1, "Batch size"),
    ],
)
def test_build_page_batches_rejects_invalid_values(
    start_page: int,
    end_page: int,
    batch_size: int,
    message: str,
) -> None:
    with pytest.raises(ingestion.IngestionError, match=message):
        ingestion.build_page_batches(start_page, end_page, batch_size)


def test_conversion_status_success_is_accepted() -> None:
    result = SimpleNamespace(status=ConversionStatus.SUCCESS, errors=[])

    ingestion._validate_conversion_result(result, (1, 8))


@pytest.mark.parametrize(
    ("status", "status_name"),
    [
        (ConversionStatus.PARTIAL_SUCCESS, "PARTIAL_SUCCESS"),
        (ConversionStatus.FAILURE, "FAILURE"),
    ],
)
def test_incomplete_conversion_status_is_rejected(
    status: ConversionStatus,
    status_name: str,
) -> None:
    result = SimpleNamespace(
        status=status,
        errors=[SimpleNamespace(page_no=9, error_message="std::bad_alloc")],
    )

    with pytest.raises(ingestion.IngestionError) as error:
        ingestion._validate_conversion_result(result, (1, 21))

    message = str(error.value)
    assert status_name in message
    assert "pages 1-21" in message
    assert "Failed pages: 9" in message
    assert "std::bad_alloc" in message


def test_jsonl_writer_preserves_unicode(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "chunks.jsonl"
    chunk = DocumentChunk(
        chunk_id="manual_p1_c0000",
        document_id="manual",
        filename="manual.pdf",
        text="Kiểm tra an toàn điện.",
        page_numbers=[1],
        headings=["An toàn"],
    )

    ingestion.write_chunks_jsonl(output, [chunk])

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["text"] == "Kiểm tra an toàn điện."
    assert set(payload) == {
        "chunk_id",
        "document_id",
        "filename",
        "text",
        "page_numbers",
        "headings",
        "content_type",
        "metadata",
    }


def test_jsonl_writer_does_not_replace_output_after_serialization_error(
    tmp_path: Path,
) -> None:
    output = tmp_path / "chunks.jsonl"
    output.write_text("previous complete output\n", encoding="utf-8")
    chunk = DocumentChunk(
        chunk_id="manual_p1_c0000",
        document_id="manual",
        filename="manual.pdf",
        text="Complete chunk",
        page_numbers=[1],
    )

    def interrupted_chunks():
        yield chunk
        raise RuntimeError("batch failed")

    with pytest.raises(RuntimeError, match="batch failed"):
        ingestion.write_chunks_jsonl(output, interrupted_chunks())

    assert output.read_text(encoding="utf-8") == "previous complete output\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_ingest_document_normalizes_docling_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Provenance:
        def __init__(self, page_no: int) -> None:
            self.page_no = page_no

    class Item:
        label = "table"
        prov = [Provenance(4), Provenance(2), Provenance(4)]

    class Metadata:
        doc_items = [Item()]
        headings = ["Safety", "Safety", "Electrical"]

    class RawChunk:
        text = "  Disconnect power before service.  "
        meta = Metadata()

    document = tmp_path / "manual.pdf"
    document.write_bytes(b"content")
    monkeypatch.setattr(ingestion, "get_pdf_page_count", lambda _: 4)
    monkeypatch.setattr(
        ingestion,
        "_convert_document",
        lambda _, *, page_range: [RawChunk()],
    )

    chunks = ingestion.ingest_document(document)

    assert len(chunks) == 1
    assert chunks[0].text == "Disconnect power before service."
    assert chunks[0].page_numbers == [2, 4]
    assert chunks[0].headings == ["Safety", "Electrical"]
    assert chunks[0].content_type == "table"
    assert chunks[0].chunk_id.startswith(f"{chunks[0].document_id}_p2_h")
    assert chunks[0].metadata["character_count"] == len(chunks[0].text)


def test_pdf_batches_are_forwarded_and_chunk_indices_are_global(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int] | None] = []

    def raw_chunk(page_number: int):
        provenance = SimpleNamespace(page_no=page_number)
        item = SimpleNamespace(label="text", prov=[provenance])
        metadata = SimpleNamespace(doc_items=[item], headings=[])
        return SimpleNamespace(text=f"Page {page_number}", meta=metadata)

    def convert_batch(
        _: Path,
        *,
        page_range: tuple[int, int] | None,
    ) -> list[SimpleNamespace]:
        calls.append(page_range)
        assert page_range is not None
        return [raw_chunk(page_range[0])]

    document = tmp_path / "manual.pdf"
    document.write_bytes(b"complete document")
    monkeypatch.setattr(ingestion, "get_pdf_page_count", lambda _: 21)
    monkeypatch.setattr(ingestion, "_convert_document", convert_batch)

    chunks = ingestion.ingest_document(document, batch_size=8)

    assert calls == [(1, 8), (9, 16), (17, 21)]
    assert [chunk.page_numbers for chunk in chunks] == [[1], [9], [17]]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert len({chunk.document_id for chunk in chunks}) == 1
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == 3


def test_pdf_page_range_must_not_exceed_page_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "manual.pdf"
    document.write_bytes(b"content")
    monkeypatch.setattr(ingestion, "get_pdf_page_count", lambda _: 21)

    with pytest.raises(ingestion.IngestionError, match="exceeds the PDF page count"):
        ingestion.ingest_document(document, page_range=(1, 22), batch_size=8)


def test_batch_size_is_rejected_for_docx(tmp_path: Path) -> None:
    document = tmp_path / "manual.docx"
    document.write_bytes(b"content")

    with pytest.raises(ingestion.IngestionError, match="only for PDF"):
        ingestion.ingest_document(document, batch_size=8)


def test_cli_requires_both_page_bounds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = tmp_path / "manual.pdf"
    document.write_bytes(b"content")

    exit_code = ingest_preview.main([str(document), "--page-start", "1"])

    assert exit_code == 1
    assert "must be provided together" in capsys.readouterr().err


def test_cli_does_not_write_jsonl_when_ingestion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "manual.pdf"
    output = tmp_path / "chunks.jsonl"
    document.write_bytes(b"content")

    monkeypatch.setattr(ingest_preview, "get_pdf_page_count", lambda _: 21)

    def fail_ingestion(*args, **kwargs):
        raise ingestion.IngestionError(
            "Docling returned PARTIAL_SUCCESS for pages 9-16; refusing incomplete output."
        )

    monkeypatch.setattr(ingest_preview, "ingest_document", fail_ingestion)

    exit_code = ingest_preview.main(
        [str(document), "--batch-size", "8", "--output", str(output)]
    )

    assert exit_code == 1
    assert not output.exists()

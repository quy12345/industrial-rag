"""Offline validation of Phase 7 corpus and held-out annotation contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.phase7 import (
    Phase7Error,
    Phase7DatasetItem,
    Phase7Source,
    dataset_sha256,
    file_sha256,
    read_phase7_dataset,
    validate_phase7_datasets,
    validate_source_records,
)
from app.models import DocumentChunk


def _chunk(identifier: str, document_id: str, text: str, page: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        document_id=document_id,
        filename=f"{document_id}.pdf",
        text=text,
        page_numbers=[page],
        headings=["Heading"],
        content_type="text",
    )


def _answerable(index: int, *, language: str = "vi", question_type: str = "installation") -> Phase7DatasetItem:
    document_id = "installation" if index % 2 else "programming"
    return Phase7DatasetItem(
        id=f"item-{index}", question=f"Question {index}", language=language,
        answerable=True, scenario="vi_to_en" if language == "vi" else "en_to_en",
        question_type=question_type, expected_document_ids=[document_id],
        relevant_chunk_ids=[f"chunk-{index}"], expected_pages=[index + 1],
        expected_phrases=[f"phrase {index}"], citation_required=True,
    )


def _unanswerable(index: int, *, language: str = "en") -> Phase7DatasetItem:
    return Phase7DatasetItem(
        id=f"item-{index}", question=f"Unsupported question {index}", language=language,
        answerable=False, scenario="vi_to_en" if language == "vi" else "en_to_en",
        question_type="unanswerable", expected_document_ids=[], relevant_chunk_ids=[],
        expected_pages=[], expected_phrases=[], citation_required=False,
        unanswerable_reason="Verified absent from both manuals.",
    )


def _valid_sets() -> tuple[list[Phase7DatasetItem], list[Phase7DatasetItem], list[DocumentChunk]]:
    calibration = [_answerable(index) for index in range(1, 13)] + [
        _unanswerable(index) for index in range(13, 21)
    ]
    types = [
        "installation", "safety", "wiring", "parameter_code", "fault_diagnosis", "cross_document",
    ]
    test = [
        _answerable(index, language="vi" if index <= 16 else "en", question_type=types[(index - 21) % len(types)])
        for index in range(21, 51)
    ] + [_unanswerable(index, language="vi" if index % 2 else "en") for index in range(51, 66)]
    all_items = [*calibration, *test]
    chunks = [
        _chunk(
            item.relevant_chunk_ids[0], item.expected_document_ids[0], item.expected_phrases[0], item.expected_pages[0]
        )
        for item in all_items if item.answerable
    ]
    return calibration, test, chunks


def test_valid_phase7_sets_have_deterministic_hashes() -> None:
    calibration, test, chunks = _valid_sets()
    result = validate_phase7_datasets(calibration, test, chunks)
    assert result["calibration"]["answerable"] == 12
    assert result["test"]["unanswerable"] == 15
    assert result["test"]["by_scenario"]["vi_to_en"] >= 10
    assert dataset_sha256(test) == dataset_sha256(list(test))


def test_phase7_rejects_missing_qrel_wrong_document_and_absent_phrase() -> None:
    calibration, test, chunks = _valid_sets()
    calibration[0] = calibration[0].model_copy(update={"relevant_chunk_ids": ["missing"]})
    with pytest.raises(Phase7Error, match="missing frozen chunk"):
        validate_phase7_datasets(calibration, test, chunks)

    calibration, test, chunks = _valid_sets()
    calibration[0] = calibration[0].model_copy(update={"expected_document_ids": ["wrong"]})
    with pytest.raises(Phase7Error, match="unexpected document"):
        validate_phase7_datasets(calibration, test, chunks)

    calibration, test, chunks = _valid_sets()
    calibration[0] = calibration[0].model_copy(update={"expected_phrases": ["not in chunk"]})
    with pytest.raises(Phase7Error, match="expected phrase"):
        validate_phase7_datasets(calibration, test, chunks)


def test_phase7_rejects_overlap_and_invalid_unanswerable_qrels() -> None:
    calibration, test, chunks = _valid_sets()
    test[0] = test[0].model_copy(update={"id": calibration[0].id})
    with pytest.raises(Phase7Error, match="IDs overlap"):
        validate_phase7_datasets(calibration, test, chunks)

    invalid = _unanswerable(99).model_dump()
    invalid["relevant_chunk_ids"] = ["chunk"]
    with pytest.raises(ValueError, match="unanswerable items cannot contain qrels"):
        Phase7DatasetItem.model_validate(invalid)


def test_dataset_loader_rejects_bad_json_and_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text('{"id": "bad"}\n', encoding="utf-8")
    with pytest.raises(Phase7Error, match="Invalid dataset record"):
        read_phase7_dataset(path)

    item = _unanswerable(1).model_dump_json()
    path.write_text(f"{item}\n{item}\n", encoding="utf-8")
    with pytest.raises(Phase7Error, match="Duplicate dataset ID"):
        read_phase7_dataset(path)


def test_file_hash_is_streamed_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"atv320")
    assert file_sha256(source) == file_sha256(source)


def test_source_manifest_contract_requires_unique_installation_and_programming() -> None:
    installation = Phase7Source(
        filename="installation.pdf", manufacturer="Schneider Electric", title="Installation",
        document_reference="NVE41289.09", version="04/2025", language="en",
        document_role="installation", official_url="https://example.test/installation",
        sha256="a" * 64, file_size_bytes=1, page_count=1, retrieved_at="2026-08-07",
        redistribution_note="Do not redistribute.",
    )
    programming = installation.model_copy(update={"filename": "programming.pdf", "document_role": "programming"})
    validate_source_records([installation, programming])
    with pytest.raises(Phase7Error, match="duplicate filenames"):
        validate_source_records([installation, installation])

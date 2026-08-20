"""Offline validation of Phase 7 corpus and held-out annotation contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models import DocumentChunk
from app.phase7 import (
    ExpectedAnswerFact,
    Phase7DatasetItem,
    Phase7Error,
    Phase7Source,
    build_exact_content_equivalence,
    dataset_sha256,
    expand_exact_equivalent_qrels,
    file_sha256,
    read_phase7_dataset,
    validate_phase7_datasets,
    validate_source_records,
)
from scripts.apply_phase7_answer_facts import (
    QREL_CORRECTIONS,
    REFERENCE_MODE_QREL,
    REVIEWED_ANSWER_FACTS,
)


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


def _answerable(
    index: int, *, language: str = "vi", question_type: str = "installation"
) -> Phase7DatasetItem:
    document_id = "installation" if index % 2 else "programming"
    return Phase7DatasetItem(
        id=f"item-{index}",
        question=f"Question {index}",
        language=language,
        answerable=True,
        scenario="vi_to_en" if language == "vi" else "en_to_en",
        question_type=question_type,
        expected_document_ids=[document_id],
        relevant_chunk_ids=[f"chunk-{index}"],
        expected_pages=[index + 1],
        expected_phrases=[f"phrase {index}"],
        expected_answer_facts=[
            {"id": f"fact-{index}", "aliases": [f"phrase {index}"]}
        ],
        citation_required=True,
    )


def _unanswerable(index: int, *, language: str = "en") -> Phase7DatasetItem:
    return Phase7DatasetItem(
        id=f"item-{index}",
        question=f"Unsupported question {index}",
        language=language,
        answerable=False,
        scenario="vi_to_en" if language == "vi" else "en_to_en",
        question_type="unanswerable",
        expected_document_ids=[],
        relevant_chunk_ids=[],
        expected_pages=[],
        expected_phrases=[],
        citation_required=False,
        unanswerable_reason="Verified absent from both manuals.",
    )


def _valid_sets() -> tuple[list[Phase7DatasetItem], list[Phase7DatasetItem], list[DocumentChunk]]:
    calibration = [_answerable(index) for index in range(1, 13)] + [
        _unanswerable(index) for index in range(13, 21)
    ]
    types = [
        "installation",
        "safety",
        "wiring",
        "parameter_code",
        "fault_diagnosis",
        "cross_document",
    ]
    test = [
        _answerable(
            index,
            language="vi" if index <= 36 else "en",
            question_type=types[(index - 21) % len(types)],
        )
        for index in range(21, 51)
    ] + [_unanswerable(index, language="vi" if index % 2 else "en") for index in range(51, 66)]
    all_items = [*calibration, *test]
    chunks = [
        _chunk(
            item.relevant_chunk_ids[0],
            item.expected_document_ids[0],
            item.expected_phrases[0],
            item.expected_pages[0],
        )
        for item in all_items
        if item.answerable
    ]
    return calibration, test, chunks


def test_valid_phase7_sets_have_deterministic_hashes() -> None:
    calibration, test, chunks = _valid_sets()
    result = validate_phase7_datasets(calibration, test, chunks)
    assert result["calibration"]["answerable"] == 12
    assert result["test"]["unanswerable"] == 15
    assert result["test"]["by_scenario"]["vi_to_en"] >= 10
    assert result["review"]["answerable_missing_answer_facts"] == 0
    assert dataset_sha256(test) == dataset_sha256(list(test))


def test_approved_answerable_requires_reviewed_answer_facts() -> None:
    draft = _answerable(1).model_copy(
        update={"expected_answer_facts": [], "review_status": "needs_human_review"}
    )
    assert draft.expected_answer_facts == []
    with pytest.raises(ValueError, match="approved answerable items require"):
        Phase7DatasetItem.model_validate(
            draft.model_dump() | {"review_status": "approved"}
        )


def test_answer_facts_reject_duplicate_ids_and_aliases() -> None:
    item = _answerable(1).model_dump()
    item["expected_answer_facts"] = [
        {"id": "range", "aliases": ["a"]},
        {"id": "range", "aliases": ["b"]},
    ]
    with pytest.raises(ValueError, match="unique IDs"):
        Phase7DatasetItem.model_validate(item)

    item["expected_answer_facts"] = [{"id": "range", "aliases": ["a", "a"]}]
    with pytest.raises(ValueError, match="duplicate aliases"):
        Phase7DatasetItem.model_validate(item)


def test_typed_answer_fact_schema_rejects_incomplete_or_mixed_contracts() -> None:
    numeric = ExpectedAnswerFact.model_validate(
        {
            "id": "supply",
            "aliases": ["24 VDC"],
            "type": "numeric_unit",
            "value": "24",
            "unit": "VDC",
        }
    )
    assert numeric.type == "numeric_unit"
    with pytest.raises(ValueError, match="require value and unit"):
        ExpectedAnswerFact.model_validate(
            {"id": "supply", "aliases": ["24 VDC"], "type": "numeric_unit"}
        )
    with pytest.raises(ValueError, match="require acceptable_values"):
        ExpectedAnswerFact.model_validate(
            {"id": "rating", "aliases": ["IP65"], "type": "identifier"}
        )
    with pytest.raises(ValueError, match="non-empty alternative groups"):
        ExpectedAnswerFact.model_validate(
            {
                "id": "mounting",
                "aliases": ["vertical mounting"],
                "required_token_groups": [[]],
            }
        )


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


def test_qrel_closure_adds_only_same_document_exact_content() -> None:
    item = _answerable(1)
    chunks = [
        _chunk("chunk-1", "installation", " Exact\ncontent ", 2),
        _chunk("same", "installation", "exact content", 3),
        _chunk("similar", "installation", "exact contents", 4),
        _chunk("other-document", "programming", "exact content", 5),
    ]
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    expanded = expand_exact_equivalent_qrels(
        item,
        chunks_by_id=by_id,
        equivalence=build_exact_content_equivalence(chunks),
    )
    assert expanded.relevant_chunk_ids == ["chunk-1", "same"]
    assert expanded.expected_pages == [2, 3]
    assert "similar" not in expanded.relevant_chunk_ids
    assert "other-document" not in expanded.relevant_chunk_ids


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
        filename="installation.pdf",
        manufacturer="Schneider Electric",
        title="Installation",
        document_reference="NVE41289.09",
        version="04/2025",
        language="en",
        document_role="installation",
        official_url="https://example.test/installation",
        sha256="a" * 64,
        file_size_bytes=1,
        page_count=1,
        retrieved_at="2026-08-07",
        redistribution_note="Do not redistribute.",
    )
    programming = installation.model_copy(
        update={"filename": "programming.pdf", "document_role": "programming"}
    )
    validate_source_records([installation, programming])
    with pytest.raises(Phase7Error, match="duplicate filenames"):
        validate_source_records([installation, installation])


def test_source_reviewed_answer_fact_mapping_is_complete_and_strict() -> None:
    assert len(REVIEWED_ANSWER_FACTS) == 42
    assert set(QREL_CORRECTIONS) == {
        "phase7_calibration_011",
        "phase7_calibration_012",
    }
    assert all(
        correction["relevant_chunk_ids"] == [REFERENCE_MODE_QREL]
        and correction["expected_pages"] == [45]
        and correction["expected_phrases"] == ["actual reference value"]
        for correction in QREL_CORRECTIONS.values()
    )
    for facts in REVIEWED_ANSWER_FACTS.values():
        assert facts
        assert len({fact["id"] for fact in facts}) == len(facts)
        assert all(fact["aliases"] for fact in facts)

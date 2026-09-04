"""Offline validation of active corpus and annotation contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.documents import DocumentChunk
from app.infrastructure import corpus_artifacts
from evaluation.dataset import (
    CorpusSource,
    DatasetValidationError,
    EvaluationItem,
    EvidenceRequirement,
    ExpectedAnswerFact,
    build_exact_content_equivalence,
    dataset_sha256,
    expand_exact_equivalent_qrels,
    read_dataset,
    validate_dataset,
    validate_dataset_splits,
    validate_development_dataset_splits,
    validate_source_records,
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
) -> EvaluationItem:
    document_id = "installation" if index % 2 else "programming"
    return EvaluationItem(
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


def _unanswerable(index: int, *, language: str = "en") -> EvaluationItem:
    return EvaluationItem(
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


def _valid_sets() -> tuple[list[EvaluationItem], list[EvaluationItem], list[DocumentChunk]]:
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


def _optimization_split(
    *,
    prefix: str,
    first_index: int,
    answerable_count: int,
    unanswerable_count: int,
) -> tuple[list[EvaluationItem], list[DocumentChunk]]:
    required_types = [
        "mounting",
        "safety",
        "menu_navigation",
        "parameter_code",
        "fault_diagnosis",
        "wiring",
        "table_lookup",
        "ordered_procedure",
        "cross_document",
    ]
    items: list[EvaluationItem] = []
    answerable_pairs = answerable_count // 2
    for offset in range(answerable_count):
        pair_index = offset % answerable_pairs
        language = "en" if offset < answerable_pairs else "vi"
        items.append(
            _answerable(
                first_index + pair_index,
                language=language,
                question_type=required_types[pair_index % len(required_types)],
            ).model_copy(
                update={
                    "id": f"{prefix}-item-{offset}",
                    "question": f"{language} question {prefix} {offset}",
                    "fact_family_id": f"{prefix}-family-{pair_index}",
                    "review_status": "approved",
                }
            )
        )
    unanswerable_pairs = unanswerable_count // 2
    for offset in range(unanswerable_count):
        pair_index = offset % unanswerable_pairs
        language = "en" if offset < unanswerable_pairs else "vi"
        items.append(
            _unanswerable(
                first_index + answerable_count + pair_index,
                language=language,
            ).model_copy(
                update={
                    "id": f"{prefix}-unsupported-item-{offset}",
                    "question": f"{language} unsupported question {prefix} {offset}",
                    "fact_family_id": f"{prefix}-unsupported-family-{pair_index}",
                    "review_status": "approved",
                }
            )
        )
    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in [
        _chunk(
            item.relevant_chunk_ids[0],
            item.expected_document_ids[0],
            item.expected_phrases[0],
            item.expected_pages[0],
        )
        for item in items
        if item.answerable
        ]
    }
    return items, list(chunks_by_id.values())


def test_valid_dataset_splits_have_deterministic_hashes() -> None:
    calibration, test, chunks = _valid_sets()
    result = validate_dataset_splits(calibration, test, chunks)
    assert result["calibration"]["answerable"] == 12
    assert result["test"]["unanswerable"] == 15
    assert result["test"]["by_scenario"]["vi_to_en"] >= 10
    assert result["review"]["answerable_missing_answer_facts"] == 0
    assert dataset_sha256(test) == dataset_sha256(list(test))


def test_development_splits_are_balanced_covered_and_leakage_free() -> None:
    development, development_chunks = _optimization_split(
        prefix="development",
        first_index=100,
        answerable_count=80,
        unanswerable_count=40,
    )
    confirmation, confirmation_chunks = _optimization_split(
        prefix="confirmation",
        first_index=300,
        answerable_count=40,
        unanswerable_count=20,
    )

    result = validate_development_dataset_splits(
        development,
        confirmation,
        [*development_chunks, *confirmation_chunks],
    )

    assert result["development"]["by_language"] == {"en": 60, "vi": 60}
    assert result["confirmation"]["by_language"] == {"en": 30, "vi": 30}
    assert result["development_dataset_sha256"] == dataset_sha256(development)
    assert result["confirmation_dataset_sha256"] == dataset_sha256(confirmation)


def test_optimization_split_sealing_rejects_unreviewed_items() -> None:
    development, development_chunks = _optimization_split(
        prefix="development",
        first_index=100,
        answerable_count=80,
        unanswerable_count=40,
    )
    confirmation, confirmation_chunks = _optimization_split(
        prefix="confirmation",
        first_index=300,
        answerable_count=40,
        unanswerable_count=20,
    )
    confirmation[0] = confirmation[0].model_copy(
        update={"review_status": "needs_human_review"}
    )

    with pytest.raises(DatasetValidationError, match="every item to be approved"):
        validate_development_dataset_splits(
            development,
            confirmation,
            [*development_chunks, *confirmation_chunks],
        )


def test_development_contract_requires_fact_family_balance_and_type_coverage() -> None:
    development, chunks = _optimization_split(
        prefix="development",
        first_index=100,
        answerable_count=80,
        unanswerable_count=40,
    )
    missing_family = development.copy()
    missing_family[0] = missing_family[0].model_copy(update={"fact_family_id": None})
    with pytest.raises(DatasetValidationError, match="require fact_family_id"):
        validate_dataset(missing_family, chunks, kind="development")

    unbalanced = development.copy()
    unbalanced[0] = unbalanced[0].model_copy(
        update={"language": "vi", "scenario": "vi_to_en"}
    )
    with pytest.raises(DatasetValidationError, match="exactly 40 EN and 40 VI"):
        validate_dataset(unbalanced, chunks, kind="development")

    missing_type = [
        item.model_copy(update={"question_type": "installation"})
        if item.answerable and item.question_type == "cross_document"
        else item
        for item in development
    ]
    with pytest.raises(DatasetValidationError, match="cross_document"):
        validate_dataset(missing_type, chunks, kind="development")

    below_quota = development.copy()
    cross_document_index = next(
        index
        for index, item in enumerate(below_quota)
        if item.question_type == "cross_document"
    )
    below_quota[cross_document_index] = below_quota[cross_document_index].model_copy(
        update={"question_type": "safety"}
    )
    with pytest.raises(DatasetValidationError, match="at least 8 per type"):
        validate_dataset(below_quota, chunks, kind="development")


def test_optimization_fact_families_require_bilingual_structural_parity() -> None:
    development, chunks = _optimization_split(
        prefix="development",
        first_index=100,
        answerable_count=80,
        unanswerable_count=40,
    )
    paired_index = next(
        index
        for index, item in enumerate(development)
        if item.fact_family_id == development[0].fact_family_id and item.language == "vi"
    )
    mismatch = development.copy()
    mismatch[paired_index] = mismatch[paired_index].model_copy(
        update={
            "expected_answer_facts": [
                mismatch[paired_index].expected_answer_facts[0].model_copy(
                    update={"id": "different-fact"}
                )
            ]
        }
    )

    with pytest.raises(DatasetValidationError, match="mismatched expected-fact structure"):
        validate_dataset(mismatch, chunks, kind="development")

    orphaned = development.copy()
    orphaned[paired_index] = orphaned[paired_index].model_copy(
        update={"fact_family_id": "orphan-family"}
    )
    with pytest.raises(DatasetValidationError, match="invalid family sizes"):
        validate_dataset(orphaned, chunks, kind="development")


def test_development_split_rejects_fact_family_and_question_leakage() -> None:
    development, development_chunks = _optimization_split(
        prefix="development",
        first_index=100,
        answerable_count=80,
        unanswerable_count=40,
    )
    confirmation, confirmation_chunks = _optimization_split(
        prefix="confirmation",
        first_index=300,
        answerable_count=40,
        unanswerable_count=20,
    )
    chunks = [*development_chunks, *confirmation_chunks]

    family_overlap = confirmation.copy()
    overlapping_family = development[0].fact_family_id
    for index, item in enumerate(family_overlap):
        if item.fact_family_id == confirmation[0].fact_family_id:
            family_overlap[index] = item.model_copy(
                update={"fact_family_id": overlapping_family}
            )
    with pytest.raises(DatasetValidationError, match="fact families overlap"):
        validate_development_dataset_splits(development, family_overlap, chunks)

    question_overlap = confirmation.copy()
    question_overlap[0] = question_overlap[0].model_copy(
        update={"question": f"  {development[0].question.upper()}!!!"}
    )
    with pytest.raises(DatasetValidationError, match="normalized exact duplicates"):
        validate_development_dataset_splits(development, question_overlap, chunks)


def test_unset_fact_family_preserves_legacy_canonical_hash_shape() -> None:
    legacy_item = _answerable(1)

    assert "fact_family_id" not in legacy_item.model_dump(mode="json", exclude_unset=True)
    assert dataset_sha256([legacy_item]) == dataset_sha256([legacy_item.model_copy()])



@pytest.mark.parametrize("field", ["item", "family", "fact"])
def test_optimization_contract_rejects_content_bearing_persisted_ids(field: str) -> None:
    development, chunks = _optimization_split(
        prefix="development",
        first_index=100,
        answerable_count=80,
        unanswerable_count=40,
    )
    invalid = development.copy()
    item = invalid[0]
    if field == "item":
        invalid[0] = item.model_copy(update={"id": "Raw question used as ID"})
    elif field == "family":
        invalid[0] = item.model_copy(
            update={"fact_family_id": "Raw fact family sentence"}
        )
    else:
        invalid[0] = item.model_copy(
            update={
                "expected_answer_facts": [
                    item.expected_answer_facts[0].model_copy(
                        update={"id": "Raw expected fact used as ID"}
                    )
                ]
            }
        )

    with pytest.raises(DatasetValidationError, match="opaque lowercase"):
        validate_dataset(invalid, chunks, kind="development")


def test_approved_answerable_requires_reviewed_answer_facts() -> None:
    draft = _answerable(1).model_copy(
        update={"expected_answer_facts": [], "review_status": "needs_human_review"}
    )
    assert draft.expected_answer_facts == []
    with pytest.raises(ValueError, match="approved answerable items require"):
        EvaluationItem.model_validate(
            draft.model_dump() | {"review_status": "approved"}
        )


def test_answer_facts_reject_duplicate_ids_and_aliases() -> None:
    item = _answerable(1).model_dump()
    item["expected_answer_facts"] = [
        {"id": "range", "aliases": ["a"]},
        {"id": "range", "aliases": ["b"]},
    ]
    with pytest.raises(ValueError, match="unique IDs"):
        EvaluationItem.model_validate(item)

    item["expected_answer_facts"] = [{"id": "range", "aliases": ["a", "a"]}]
    with pytest.raises(ValueError, match="duplicate aliases"):
        EvaluationItem.model_validate(item)


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


def test_evidence_requirement_requires_disjoint_reviewed_chunk_ids() -> None:
    requirement = EvidenceRequirement(
        id="safety",
        canonical_chunk_ids=["canonical"],
        reviewed_alternative_chunk_ids=["reviewed-alternative"],
    )
    assert requirement.canonical_chunk_ids == ["canonical"]

    with pytest.raises(ValueError, match="must be disjoint"):
        EvidenceRequirement(
            id="safety",
            canonical_chunk_ids=["canonical"],
            reviewed_alternative_chunk_ids=["canonical"],
        )


def test_optimization_multi_qrel_requires_explicit_evidence_requirements() -> None:
    development, chunks = _optimization_split(
        prefix="development",
        first_index=100,
        answerable_count=80,
        unanswerable_count=40,
    )
    item = development[0]
    extra = _chunk("extra-qrel", item.expected_document_ids[0], item.expected_phrases[0], 99)
    ambiguous = development.copy()
    ambiguous[0] = item.model_copy(
        update={
            "relevant_chunk_ids": [item.relevant_chunk_ids[0], "extra-qrel"],
            "expected_pages": [item.expected_pages[0], 99],
        }
    )

    with pytest.raises(DatasetValidationError, match="require explicit evidence_requirements"):
        validate_dataset(ambiguous, [*chunks, extra], kind="development")


def test_rejects_missing_qrel_wrong_document_and_absent_phrase() -> None:
    calibration, test, chunks = _valid_sets()
    calibration[0] = calibration[0].model_copy(update={"relevant_chunk_ids": ["missing"]})
    with pytest.raises(DatasetValidationError, match="missing frozen chunk"):
        validate_dataset_splits(calibration, test, chunks)

    calibration, test, chunks = _valid_sets()
    calibration[0] = calibration[0].model_copy(update={"expected_document_ids": ["wrong"]})
    with pytest.raises(DatasetValidationError, match="unexpected document"):
        validate_dataset_splits(calibration, test, chunks)

    calibration, test, chunks = _valid_sets()
    calibration[0] = calibration[0].model_copy(update={"expected_phrases": ["not in chunk"]})
    with pytest.raises(DatasetValidationError, match="expected phrase"):
        validate_dataset_splits(calibration, test, chunks)


def test_rejects_overlap_and_invalid_unanswerable_qrels() -> None:
    calibration, test, chunks = _valid_sets()
    test[0] = test[0].model_copy(update={"id": calibration[0].id})
    with pytest.raises(DatasetValidationError, match="IDs overlap"):
        validate_dataset_splits(calibration, test, chunks)

    invalid = _unanswerable(99).model_dump()
    invalid["relevant_chunk_ids"] = ["chunk"]
    with pytest.raises(ValueError, match="unanswerable items cannot contain qrels"):
        EvaluationItem.model_validate(invalid)


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
    with pytest.raises(DatasetValidationError, match="Invalid dataset record"):
        read_dataset(path)

    item = _unanswerable(1).model_dump_json()
    path.write_text(f"{item}\n{item}\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Duplicate dataset ID"):
        read_dataset(path)


def test_file_hash_is_streamed_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"atv320")
    assert corpus_artifacts.file_sha256(source) == corpus_artifacts.file_sha256(source)


def test_artifact_writers_preserve_utf8_order_newlines_and_cleanup(tmp_path: Path) -> None:
    json_path = tmp_path / "manifest.json"
    jsonl_path = tmp_path / "records.jsonl"

    corpus_artifacts.write_json_atomic(json_path, {"z": "điện", "a": 1})
    corpus_artifacts.write_jsonl_atomic(
        jsonl_path,
        [{"z": "điện", "a": 1}, {"id": "second"}],
    )

    assert json_path.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "z": "điện"\n}\n'
    assert jsonl_path.read_text(encoding="utf-8") == (
        '{"a": 1, "z": "điện"}\n{"id": "second"}\n'
    )
    assert list(tmp_path.glob(".*.tmp")) == []


def test_source_manifest_contract_requires_unique_installation_and_programming() -> None:
    installation = CorpusSource(
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
    with pytest.raises(DatasetValidationError, match="duplicate filenames"):
        validate_source_records([installation, installation])

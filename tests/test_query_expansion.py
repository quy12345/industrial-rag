"""Offline tests for deterministic Phase 7 lexical query augmentation."""

from __future__ import annotations

import pytest

from app.query_expansion import augment_vietnamese_technical_query


def test_query_expansion_adds_only_matching_technical_terms_deterministically() -> None:
    expanded, rules = augment_vietnamese_technical_query(
        "Tránh để trục động cơ quay ngoài ý muốn"
    )
    assert expanded.endswith("avoid prevent shaft motor rotate rotation")
    assert rules == ("tránh", "trục", "động cơ", "quay")
    assert augment_vietnamese_technical_query(
        "Tránh để trục động cơ quay ngoài ý muốn"
    ) == (expanded, rules)


def test_query_expansion_preserves_identifiers_and_does_not_invent_unmatched_terms() -> None:
    expanded, rules = augment_vietnamese_technical_query(
        "Phím MODE chuyển giữa nhóm menu nào?"
    )
    assert expanded.startswith("Phím MODE")
    assert expanded.endswith("key switch navigate group")
    assert "reference" not in expanded.casefold()
    assert rules == ("phím", "chuyển", "nhóm")


def test_query_expansion_is_noop_without_glossary_match_and_rejects_blank() -> None:
    assert augment_vietnamese_technical_query("MODE menu") == ("MODE menu", ())
    with pytest.raises(ValueError, match="non-empty"):
        augment_vietnamese_technical_query("  ")


def test_query_expansion_translates_protective_equipment_terms_present_in_query() -> None:
    expanded, rules = augment_vietnamese_technical_query(
        "Thiết bị bảo vệ phải được lắp như thế nào?"
    )
    assert expanded.endswith("protective equipment install installed")
    assert rules == ("thiết bị bảo vệ", "lắp")

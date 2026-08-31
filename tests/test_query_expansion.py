"""Offline tests for deterministic ATV320 lexical query augmentation."""

from __future__ import annotations

import unicodedata

import pytest

from app.domain.policies.query_analysis import (
    QUERY_EXPANSION_PROFILE,
    QUERY_EXPANSION_PROFILE_V2,
    augment_vietnamese_technical_query,
    augment_vietnamese_technical_query_v2,
    expand_query_for_profile,
)


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


@pytest.mark.parametrize(
    "query",
    (
        "Điện áp tại đầu cực là bao nhiêu?",
        unicodedata.normalize("NFD", "Điện áp tại đầu cực là bao nhiêu?"),
        "Dien ap tai dau cuc la bao nhieu?",
        "Dien áp tại dau cuc là bao nhiêu?",
    ),
)
def test_v2_matches_nfc_nfd_accentless_and_mixed_vietnamese(query: str) -> None:
    expanded, rules = augment_vietnamese_technical_query_v2(query)

    assert expanded.startswith(query)
    assert expanded.endswith("terminal voltage")
    assert rules == ("terminal", "voltage")


def test_v2_matches_all_caps_accentless_multiword_vietnamese_phrases() -> None:
    voltage_query = "DIEN AP TAI DAU CUC LA BAO NHIEU?"
    voltage_expanded, voltage_rules = augment_vietnamese_technical_query_v2(voltage_query)
    assert voltage_expanded == f"{voltage_query} terminal voltage"
    assert voltage_rules == ("terminal", "voltage")

    power_query = "NGAT NGUON DIEN TRUOC KHI LAM VIEC"
    power_expanded, power_rules = augment_vietnamese_technical_query_v2(power_query)
    assert power_expanded == f"{power_query} disconnect power supply before"
    assert power_rules == ("disconnect_power", "power_supply", "before")


def test_v2_uses_complete_phrase_boundaries_without_collapsing_vietnamese_tones() -> None:
    repeated = "Lặp lại thao tác này."
    assert augment_vietnamese_technical_query_v2(repeated) == (repeated, ())

    partial = "Kiểm tra điện ápx."
    assert augment_vietnamese_technical_query_v2(partial) == (partial, ())

    punctuated = "Kiểm tra điện-áp."
    expanded, rules = augment_vietnamese_technical_query_v2(punctuated)
    assert expanded == f"{punctuated} voltage"
    assert rules == ("voltage",)


def test_v2_target_suppression_uses_boundaries_instead_of_substrings() -> None:
    query = "Phím keypad chuyển nhóm nào?"
    expanded, rules = augment_vietnamese_technical_query_v2(query)

    assert expanded.endswith("key switch group")
    assert rules == ("navigation_key", "switch", "group")


@pytest.mark.parametrize(
    "query",
    (
        "  What is the terminal voltage at 24 VDC?  ",
        "Is quay clearance specified for this installation?",
        "Does the café near the quay have clearance?",
        "What does the LOI code mean?",
        "MODE keypad switches between which menu groups?",
    ),
)
def test_v2_is_an_exact_noop_for_english_queries(query: str) -> None:
    expanded, rules = augment_vietnamese_technical_query_v2(query)

    assert expanded == query
    assert rules == ()


@pytest.mark.parametrize("identifier", ("LOI", "ACC", "LI1"))
def test_v2_does_not_accent_fold_single_token_identifiers(identifier: str) -> None:
    query = f"What does {identifier} mean?"
    assert augment_vietnamese_technical_query_v2(query) == (query, ())


def test_v2_protects_all_caps_identifier_inside_an_accentless_vietnamese_query() -> None:
    query = "DIEN AP LOI"
    expanded, rules = augment_vietnamese_technical_query_v2(query)

    assert expanded == f"{query} voltage"
    assert rules == ("voltage",)


def test_v2_preserves_identifiers_numbers_units_punctuation_and_original_prefix() -> None:
    query = "Cần cấu hình Fr1, ACC ở 0.5 s; kiểm tra LI1 với 4–20 mA và 24 VDC.  "
    expanded, rules = augment_vietnamese_technical_query_v2(query)

    assert expanded.startswith(query)
    assert expanded == f"{query}setting configuration"
    assert rules == ("settings",)
    for literal in ("Fr1", "ACC", "0.5 s", "LI1", "4–20 mA", "24 VDC"):
        assert literal in expanded


def test_v2_overlapping_phrases_produce_stable_rule_ids_and_unique_terms() -> None:
    query = "Phải ngắt nguồn điện và kiểm tra cài đặt gốc."
    expanded, rules = augment_vietnamese_technical_query_v2(query)

    assert rules == ("disconnect_power", "power_supply", "settings", "factory_default")
    assert expanded.endswith(
        "disconnect power supply setting configuration factory default"
    )
    assert expanded.split().count("power") == 1
    assert expanded.split().count("setting") == 1
    assert augment_vietnamese_technical_query_v2(query) == (expanded, rules)


def test_explicit_profile_resolution_preserves_v1_and_rejects_unknown_profiles() -> None:
    query = "Phím MODE chuyển nhóm menu nào?"

    assert expand_query_for_profile(query, profile=QUERY_EXPANSION_PROFILE) == (
        augment_vietnamese_technical_query(query)
    )
    assert expand_query_for_profile(query, profile=QUERY_EXPANSION_PROFILE_V2) == (
        augment_vietnamese_technical_query_v2(query)
    )
    with pytest.raises(ValueError, match="Unsupported query expansion profile"):
        expand_query_for_profile(query, profile="unknown")


def test_v2_rejects_blank_queries() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        augment_vietnamese_technical_query_v2(" \t ")

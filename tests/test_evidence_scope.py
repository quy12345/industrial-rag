"""Offline tests for the deterministic protected-concept evidence guard."""

from __future__ import annotations

import pytest

from app.domain.policies.evidence_scope import (
    EVIDENCE_SCOPE_PROFILE,
    evaluate_evidence_scope,
)


@pytest.mark.parametrize(
    ("question", "sources", "status", "concepts", "missing"),
    [
        (
            "What Bluetooth pairing PIN is required?",
            ("Pairing password and public unlock code are documented.",),
            "missing_scope",
            ("bluetooth", "credential", "pairing"),
            ("bluetooth",),
        ),
        (
            "Ứng dụng di động ghép nối bằng mã QR ở đâu?",
            ("Quét mã QR để xem bảng thông số sản phẩm.",),
            "missing_scope",
            ("qr", "pairing", "mobile_application"),
            ("pairing", "mobile_application"),
        ),
        (
            "Ứng dụng điện thoại ghép nối bằng mã QR?",
            ("Mobile application pairing uses the QR code.",),
            "covered",
            ("qr", "pairing", "mobile_application"),
            (),
        ),
        (
            "Cấu hình nào dùng được?",
            ("Configuration is described in this section.",),
            "not_applicable",
            ("configuration",),
            (),
        ),
    ],
)
def test_scope_guard_is_bilingual_boundary_safe_and_conservative(
    question, sources, status, concepts, missing
) -> None:
    decision = evaluate_evidence_scope(question, sources)
    assert EVIDENCE_SCOPE_PROFILE == "protected_technical_concepts_v1"
    assert decision.status == status
    assert decision.query_concept_ids == concepts
    assert decision.missing_concept_ids == missing


def test_scope_guard_handles_nfd_and_accentless_phrases_without_generic_aliases() -> None:
    decision = evaluate_evidence_scope(
        "Mã QR ghép nối với ủng dụng diẹn thoại.",
        ("ma qr ghep noi voi ung dung dien thoai",),
    )
    assert decision.status == "covered"
    assert (
        evaluate_evidence_scope("application code", ("application code",)).status
        == "not_applicable"
    )
    assert evaluate_evidence_scope("Bluetooth pairing", ("Bluetooth pairing",)).status == "covered"


def test_scope_guard_does_not_claim_to_verify_relations() -> None:
    decision = evaluate_evidence_scope(
        "Can the mobile application pair with Bluetooth using a PIN?",
        ("Bluetooth settings, mobile application, pairing, and PIN code are separate topics.",),
    )
    assert decision.status == "covered"
    assert decision.missing_concept_ids == ()

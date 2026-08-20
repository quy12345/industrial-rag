"""Apply source-reviewed Phase 7 answer facts without model or provider calls.

Facts were reviewed against the frozen direct-evidence chunks derived from the two
ATV320 source PDFs.  This script leaves every row in ``needs_human_review`` so the
dataset owner must still provide the explicit v2 freeze token after inspecting the
diff.  It does not read retrieval results, provider outputs, or historical answers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from app.evaluation import load_frozen_chunks
from app.phase7 import (
    Phase7DatasetItem,
    Phase7Error,
    build_exact_content_equivalence,
    expand_exact_equivalent_qrels,
    read_phase7_dataset,
    validate_phase7_datasets,
    write_jsonl_atomic,
)

REVIEW_NOTE = (
    "Answer facts source-reviewed against frozen direct-evidence chunks on 2026-08-09; "
    "no retrieval result, generated answer, or provider output was used."
)
QREL_CORRECTION_NOTE = (
    "Dataset-v2 source review corrected the reference-mode qrel from unrelated page 355 "
    "load-variation text to page 45 direct evidence."
)


def _fact(identifier: str, *aliases: str) -> dict[str, Any]:
    return {"id": identifier, "aliases": list(aliases)}


REVIEWED_ANSWER_FACTS: dict[str, list[dict[str, Any]]] = {
    "phase7_calibration_001": [
        _fact("power-to-disconnect", "disconnect all power", "all power must be disconnected")
    ],
    "phase7_calibration_002": [
        _fact(
            "power-to-disconnect",
            "ngắt toàn bộ nguồn điện",
            "ngắt tất cả nguồn điện",
            "phải ngắt toàn bộ nguồn điện",
        )
    ],
    "phase7_calibration_003": [
        _fact("prevent-rotation", "block the motor shaft", "the motor shaft must be blocked")
    ],
    "phase7_calibration_004": [
        _fact(
            "prevent-rotation",
            "chặn trục động cơ",
            "khóa trục động cơ",
            "cố định trục động cơ",
        )
    ],
    "phase7_calibration_005": [
        _fact(
            "protective-equipment-state",
            "protective equipment is installed and/or closed",
            "protective equipment is installed or closed",
            "covers, doors, and grids are installed and/or closed",
        )
    ],
    "phase7_calibration_006": [
        _fact(
            "protective-equipment-state",
            "thiết bị bảo vệ đã được lắp đặt và/hoặc đóng kín",
            "các nắp, cửa và lưới bảo vệ đã được lắp hoặc đóng",
            "thiết bị bảo vệ được lắp đặt hoặc đóng lại",
        )
    ],
    "phase7_calibration_007": [
        _fact(
            "contact-timing",
            "closed before a Run command is executed",
            "contacts must be closed before the Run command",
        )
    ],
    "phase7_calibration_008": [
        _fact(
            "contact-timing",
            "phải đóng trước khi thực hiện lệnh Run",
            "đóng trước lệnh Run",
            "tiếp điểm phải được đóng trước khi lệnh Run được thực thi",
        )
    ],
    "phase7_calibration_009": [
        _fact("reference-menu", "Reference speed rEF", "[Reference speed] rEF"),
        _fact("monitoring-menu", "Monitoring MON", "[MONITORING] MON"),
        _fact("configuration-menu", "Configuration CONF", "[Configuration] CONF"),
    ],
    "phase7_calibration_010": [
        _fact("reference-menu", "Reference speed rEF", "nhóm Reference speed rEF"),
        _fact("monitoring-menu", "Monitoring MON", "nhóm Monitoring MON"),
        _fact("configuration-menu", "Configuration CONF", "nhóm Configuration CONF"),
    ],
    "phase7_calibration_011": [
        _fact("monitored-value", "actual reference value", "the reference value")
    ],
    "phase7_calibration_012": [
        _fact(
            "monitored-value",
            "giá trị tham chiếu thực tế",
            "giá trị reference thực tế",
            "giá trị tham chiếu",
        )
    ],
    "phase7_test_001": [
        _fact(
            "safety-control",
            "functioning emergency stop push-button",
            "emergency stop push-button",
        )
    ],
    "phase7_test_002": [
        _fact(
            "safety-control",
            "nút dừng khẩn cấp hoạt động tốt",
            "nút nhấn dừng khẩn cấp",
            "nút dừng khẩn cấp",
        )
    ],
    "phase7_test_003": [
        _fact("required-assessment", "risk assessment", "a risk assessment")
    ],
    "phase7_test_004": [
        _fact("required-assessment", "đánh giá rủi ro", "thực hiện đánh giá rủi ro")
    ],
    "phase7_test_005": [
        _fact(
            "energized-parts",
            "unshielded components or terminals with voltage present",
            "unshielded components and energized terminals",
        )
    ],
    "phase7_test_006": [
        _fact(
            "energized-parts",
            "các bộ phận không được che chắn hoặc đầu cực đang có điện",
            "bộ phận không che chắn và đầu cực có điện",
            "linh kiện không che chắn hoặc đầu nối đang mang điện",
        )
    ],
    "phase7_test_007": [
        _fact(
            "items-to-remove",
            "remove the ground and the short circuits",
            "ground and short circuits must be removed",
        )
    ],
    "phase7_test_008": [
        _fact(
            "items-to-remove",
            "tháo nối đất và các mạch ngắn mạch",
            "loại bỏ nối đất và ngắn mạch",
            "tháo dây nối đất và các cầu nối ngắn mạch",
        )
    ],
    "phase7_test_009": [
        _fact(
            "factory-setting-definition",
            "machine status in factory settings when the product was shipped",
            "the machine status when the product was shipped",
        )
    ],
    "phase7_test_010": [
        _fact(
            "factory-setting-definition",
            "trạng thái máy theo cài đặt nhà máy khi sản phẩm được xuất xưởng",
            "trạng thái của máy khi sản phẩm được giao từ nhà máy",
        )
    ],
    "phase7_test_011": [
        _fact(
            "fault-reset-purpose",
            "restore the drive to an operational state",
            "return the drive to an operational state",
        )
    ],
    "phase7_test_012": [
        _fact(
            "fault-reset-purpose",
            "khôi phục biến tần về trạng thái hoạt động",
            "đưa drive trở lại trạng thái vận hành",
            "đưa biến tần trở lại trạng thái hoạt động",
        )
    ],
    "phase7_test_013": [
        _fact("jog-dial-control", "jog dial"),
        _fact("navigation-key-control", "Up/Down navigation keys", "up and down navigation keys"),
    ],
    "phase7_test_014": [
        _fact("jog-dial-control", "núm xoay jog", "jog dial"),
        _fact("navigation-key-control", "phím điều hướng Up/Down", "các phím điều hướng lên/xuống"),
    ],
    "phase7_test_015": [
        _fact("ent-confirmation", "no need to press the ENT key", "ENT is not required")
    ],
    "phase7_test_016": [
        _fact(
            "ent-confirmation",
            "không cần nhấn phím ENT",
            "không cần nhấn ENT",
            "phím ENT không bắt buộc",
        )
    ],
    "phase7_test_017": [
        _fact("display-determinant", "drive settings", "the drive settings")
    ],
    "phase7_test_018": [
        _fact(
            "display-determinant",
            "cài đặt của biến tần",
            "các thiết lập của drive",
            "cấu hình của biến tần",
        )
    ],
    "phase7_test_019": [
        _fact("reference-frequency-label", "[Ref Frequency] LFR", "Ref Frequency LFR")
    ],
    "phase7_test_020": [
        _fact("reference-frequency-label", "[Ref Frequency] LFR", "Ref Frequency LFR")
    ],
    "phase7_test_021": [
        _fact("reference-frequency-range", "-599 to +599 Hz")
    ],
    "phase7_test_022": [
        _fact(
            "reference-frequency-range",
            "-599 đến +599 Hz",
            "từ -599 đến +599 Hz",
            "-599 to +599 Hz",
        )
    ],
    "phase7_test_023": [
        _fact("virtual-input-name", "AIV1 Image input", "[AIV1 Image input] AIV1")
    ],
    "phase7_test_024": [
        _fact("virtual-input-name", "AIV1 Image input", "[AIV1 Image input] AIV1")
    ],
    "phase7_test_025": [
        _fact(
            "installation-manual-description",
            "Fault reset is required to exit this operating state",
            "fault reset is required after the cause of the detected error has been removed",
        ),
        _fact(
            "programming-manual-description",
            "[Fault reset] RST",
            "[Fault Reset Assign] RSF",
        ),
    ],
    "phase7_test_026": [
        _fact(
            "installation-manual-description",
            "cần fault reset để thoát khỏi trạng thái Fault",
            "cần reset lỗi sau khi nguyên nhân lỗi đã được loại bỏ",
        ),
        _fact(
            "programming-manual-description",
            "[Fault reset] RST",
            "[Fault Reset Assign] RSF",
        ),
    ],
    "phase7_test_027": [
        _fact("installation-manual", "installation manual", "ATV320 installation manual"),
        _fact("programming-manual", "programming manual", "ATV320 programming manual"),
    ],
    "phase7_test_028": [
        _fact("installation-manual", "manual lắp đặt", "installation manual"),
        _fact("programming-manual", "manual lập trình", "programming manual"),
    ],
    "phase7_test_029": [
        _fact("installation-manual", "installation manual", "ATV320 installation manual"),
        _fact("programming-manual", "programming manual", "ATV320 programming manual"),
    ],
    "phase7_test_030": [
        _fact("installation-manual", "manual lắp đặt", "installation manual"),
        _fact("programming-manual", "manual lập trình", "programming manual"),
    ],
}

REFERENCE_MODE_QREL = (
    "atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p45_ha25328d0c34430b2"
)
QREL_CORRECTIONS = {
    "phase7_calibration_011": {
        "relevant_chunk_ids": [REFERENCE_MODE_QREL],
        "expected_pages": [45],
        "expected_phrases": ["actual reference value"],
    },
    "phase7_calibration_012": {
        "relevant_chunk_ids": [REFERENCE_MODE_QREL],
        "expected_pages": [45],
        "expected_phrases": ["actual reference value"],
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration", type=Path, default=Path("data/eval/phase7/calibration.jsonl")
    )
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument(
        "--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl")
    )
    args = parser.parse_args()

    chunks = load_frozen_chunks(args.chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    equivalence = build_exact_content_equivalence(chunks)
    calibration = read_phase7_dataset(args.calibration)
    test = read_phase7_dataset(args.test)
    answerable_ids = {item.id for item in [*calibration, *test] if item.answerable}
    if answerable_ids != set(REVIEWED_ANSWER_FACTS):
        missing = sorted(answerable_ids - set(REVIEWED_ANSWER_FACTS))
        extra = sorted(set(REVIEWED_ANSWER_FACTS) - answerable_ids)
        raise Phase7Error(f"Answer-fact mapping mismatch; missing={missing}, extra={extra}")

    reviewed = [
        _review_item(item, chunks_by_id=chunks_by_id, equivalence=equivalence)
        for item in [*calibration, *test]
    ]
    reviewed_calibration = reviewed[: len(calibration)]
    reviewed_test = reviewed[len(calibration) :]
    validate_phase7_datasets(reviewed_calibration, reviewed_test, chunks)
    write_jsonl_atomic(
        args.calibration, [item.model_dump(mode="json") for item in reviewed_calibration]
    )
    write_jsonl_atomic(args.test, [item.model_dump(mode="json") for item in reviewed_test])
    print("Phase 7 answer facts applied: 42 answerable rows; dataset remains review-required.")
    return 0


def _review_item(
    item: Phase7DatasetItem,
    *,
    chunks_by_id: dict[str, Any],
    equivalence: dict[str, tuple[str, ...]],
) -> Phase7DatasetItem:
    update: dict[str, Any] = {"review_status": "needs_human_review"}
    notes = [item.annotation_notes or "", REVIEW_NOTE]
    if item.id in QREL_CORRECTIONS:
        update.update(QREL_CORRECTIONS[item.id])
        notes.append(QREL_CORRECTION_NOTE)
    if item.answerable:
        update["expected_answer_facts"] = REVIEWED_ANSWER_FACTS[item.id]
    update["annotation_notes"] = " ".join(note.strip() for note in notes if note.strip())
    reviewed = Phase7DatasetItem.model_validate(item.model_dump(mode="json") | update)
    return expand_exact_equivalent_qrels(
        reviewed, chunks_by_id=chunks_by_id, equivalence=equivalence
    )


if __name__ == "__main__":
    raise SystemExit(main())

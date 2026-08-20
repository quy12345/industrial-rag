"""Generate a review-only Phase 7 annotation draft from verified frozen chunks.

The output is deliberately ``needs_human_review``.  It never approves, calibrates,
or benchmarks the resulting data.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation import load_frozen_chunks

INSTALLATION = "atv320-installation-manual-en-nve41289-09-c181b4d7f11b"
PROGRAMMING = "atv320-programming-manual-en-nve41295-06-f5e9bb48167a"

# phrase, English question, Vietnamese question, document, question type
CALIBRATION = (
    (
        "Disconnect all power",
        "What must be disconnected before electrical work?",
        "Trước khi làm việc điện phải ngắt nguồn nào?",
        INSTALLATION,
        "safety",
    ),
    (
        "block the motor shaft",
        "What must be done to prevent motor rotation before work?",
        "Cần làm gì để tránh trục động cơ quay trước khi thao tác?",
        INSTALLATION,
        "safety",
    ),
    (
        "protective equipment",
        "What must be verified after completing installation work?",
        "Sau khi hoàn tất lắp đặt phải xác minh điều gì về thiết bị bảo vệ?",
        INSTALLATION,
        "installation",
    ),
    (
        "contacts between the motor and the drive",
        "When must motor-drive contacts be closed relative to a Run command?",
        "Tiếp điểm giữa động cơ và drive phải đóng khi nào so với lệnh Run?",
        INSTALLATION,
        "wiring",
    ),
    (
        "MODE key",
        "Which menu groups can the MODE key switch between?",
        "Phím MODE có thể chuyển giữa những nhóm menu nào?",
        PROGRAMMING,
        "menu_navigation",
    ),
    (
        "reference mode",
        "What is reference mode used to monitor?",
        "Reference mode được dùng để giám sát nội dung gì?",
        PROGRAMMING,
        "parameter_code",
    ),
)

TEST = (
    (
        "emergency stop push-button",
        "What safety control must be within reach during operation?",
        "Khi vận hành, nút điều khiển an toàn nào phải trong tầm với?",
        INSTALLATION,
        "safety",
    ),
    (
        "risk assessment",
        "What assessment is required when designing the machine application?",
        "Khi thiết kế ứng dụng máy cần thực hiện đánh giá nào?",
        INSTALLATION,
        "safety",
    ),
    (
        "Do not touch unshielded components",
        "Which energized parts must not be touched?",
        "Không được chạm vào bộ phận nào khi còn điện?",
        INSTALLATION,
        "safety",
    ),
    (
        "remove the ground and the short circuits",
        "What must be removed from input and motor terminals after work?",
        "Sau khi thao tác cần tháo gì khỏi đầu vào và đầu cực động cơ?",
        INSTALLATION,
        "wiring",
    ),
    (
        "Machine status in factory settings",
        "What does the installation manual mean by factory setting?",
        "Installation manual định nghĩa factory setting là gì?",
        INSTALLATION,
        "installation",
    ),
    (
        "restore the drive to an operational state",
        "What is the purpose of Fault Reset?",
        "Fault Reset có mục đích gì?",
        INSTALLATION,
        "fault_diagnosis",
    ),
    (
        "local control is enabled",
        "Which controls change the reference when local control is enabled?",
        "Khi local control bật, điều khiển nào thay đổi reference?",
        PROGRAMMING,
        "menu_navigation",
    ),
    (
        "no need to press the ENT key",
        "Is ENT required to confirm a reference change?",
        "Có cần nhấn ENT để xác nhận thay đổi reference không?",
        PROGRAMMING,
        "menu_navigation",
    ),
    (
        "Displayed parameters depend on drive settings",
        "What determines which parameters are displayed?",
        "Yếu tố nào quyết định các parameter được hiển thị?",
        PROGRAMMING,
        "parameter_code",
    ),
    (
        "[Ref Frequency] LFR",
        "What is the HMI label for the reference-frequency parameter?",
        "Nhãn HMI của parameter reference frequency là gì?",
        PROGRAMMING,
        "parameter_code",
    ),
    (
        "-599 to +599 Hz",
        "What is the documented range for the shown reference-frequency parameter?",
        "Dải giá trị được ghi cho parameter reference frequency là bao nhiêu?",
        PROGRAMMING,
        "parameter_code",
    ),
    (
        "AIV1 Image input",
        "Which parameter name is shown for the first virtual analog-input image?",
        "Tên parameter nào được hiển thị cho virtual analog-input image thứ nhất?",
        PROGRAMMING,
        "parameter_code",
    ),
)

CROSS_DOCUMENT = (
    (
        "fault reset",
        "How is fault reset described across the two ATV320 manuals?",
        "Hai manual ATV320 mô tả fault reset như thế nào?",
    ),
    (
        "emergency stop push-button",
        "Which manual evidence addresses an emergency-stop push-button?",
        "Bằng chứng ở manual nào đề cập emergency-stop push-button?",
    ),
    (
        "factory setting",
        "Which documents define or expose factory-setting information?",
        "Những document nào định nghĩa hoặc thể hiện thông tin factory setting?",
    ),
)

CALIBRATION_UNANSWERABLE = (
    "What is the drive's Ethernet MAC address?",
    "What is the factory GPS coordinate?",
    "What hydraulic oil grade is required?",
    "What is the warranty claim telephone number?",
    "What is the excavator bucket capacity?",
    "What is the latest firmware download URL?",
    "What color is the optional external cabinet?",
    "What is the serial number of this specific drive?",
)
TEST_UNANSWERABLE = (
    "What is the maximum hydraulic pressure of the excavator?",
    "Which employee approved this installation?",
    "What is the building fire-escape route?",
    "What is the current electricity tariff?",
    "What is the drive's Wi-Fi password?",
    "Which truck delivered the ATV320?",
    "What is the annual weather forecast for the installation site?",
    "What is the operator's personal phone number?",
    "What is the chemical composition of the room floor?",
    "Which cloud region stores the drive data?",
    "What is the purchase order number?",
    "What is the machine's paint supplier?",
    "What is the next scheduled factory shutdown date?",
    "Which PLC program file is installed?",
    "What is the operator's blood type?",
)


def main() -> int:
    chunks = load_frozen_chunks(Path("artifacts/phase7/frozen-chunks.jsonl"))
    calibration = _answerable_rows(CALIBRATION, chunks, "phase7_calibration")
    calibration.extend(_unanswerable_rows(CALIBRATION_UNANSWERABLE, "phase7_calibration", start=13))
    test = _answerable_rows(TEST, chunks, "phase7_test")
    test.extend(_cross_rows(chunks, start=len(test) + 1))
    test.extend(_unanswerable_rows(TEST_UNANSWERABLE, "phase7_test", start=len(test) + 1))
    _write_jsonl(Path("data/eval/phase7/calibration.jsonl"), calibration)
    _write_jsonl(Path("data/eval/phase7/test.jsonl"), test)
    _write_review(calibration, test)
    print("Phase 7 annotation draft generated: 20 calibration rows, 45 held-out rows.")
    return 0


def _answerable_rows(anchors, chunks, prefix: str):
    rows = []
    for phrase, english, vietnamese, document_id, question_type in anchors:
        chunk = _find_one(chunks, phrase, document_id)
        for language, question in (("en", english), ("vi", vietnamese)):
            rows.append(
                _answerable(
                    prefix, len(rows) + 1, question, language, question_type, [chunk], [phrase]
                )
            )
    return rows


def _cross_rows(chunks, *, start: int):
    rows = []
    for _offset, (phrase, english, vietnamese) in enumerate(CROSS_DOCUMENT):
        matching = [_find_one(chunks, phrase, INSTALLATION), _find_one(chunks, phrase, PROGRAMMING)]
        for language, question in (("en", english), ("vi", vietnamese)):
            rows.append(
                _answerable(
                    "phase7_test",
                    start + len(rows),
                    question,
                    language,
                    "cross_document",
                    matching,
                    [phrase],
                )
            )
    return rows


def _answerable(prefix, number, question, language, question_type, chunks, phrases):
    return {
        "id": f"{prefix}_{number:03d}",
        "question": question,
        "language": language,
        "answerable": True,
        "scenario": "vi_to_en" if language == "vi" else "en_to_en",
        "question_type": question_type,
        "expected_document_ids": sorted({chunk.document_id for chunk in chunks}),
        "relevant_chunk_ids": [chunk.chunk_id for chunk in chunks],
        "expected_pages": sorted({page for chunk in chunks for page in chunk.page_numbers}),
        "expected_phrases": phrases,
        "phrase_match_mode": "all",
        "citation_required": True,
        "annotation_notes": (
            "Direct phrase verified in frozen HybridChunker output; human review required."
        ),
        "review_status": "needs_human_review",
    }


def _unanswerable_rows(questions, prefix, *, start):
    return [
        {
            "id": f"{prefix}_{start + index:03d}",
            "question": question,
            "language": "en" if index % 2 == 0 else "vi",
            "answerable": False,
            "scenario": "en_to_en" if index % 2 == 0 else "vi_to_en",
            "question_type": "unanswerable",
            "expected_document_ids": [],
            "relevant_chunk_ids": [],
            "expected_pages": [],
            "expected_phrases": [],
            "phrase_match_mode": "all",
            "citation_required": False,
            "unanswerable_reason": (
                "Plausible domain question; no direct evidence selected from either frozen manual."
            ),
            "review_status": "needs_human_review",
        }
        for index, question in enumerate(questions)
    ]


def _find_one(chunks, phrase, document_id):
    matches = [
        chunk
        for chunk in chunks
        if chunk.document_id == document_id and phrase.casefold() in chunk.text.casefold()
    ]
    if not matches:
        raise ValueError(f"No frozen direct-evidence chunk for {phrase!r} in {document_id}")
    return min(matches, key=lambda chunk: (len(chunk.text), chunk.chunk_id))


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _write_review(calibration, test):
    lines = [
        "# Phase 7 dataset review — pending human approval",
        "",
        "Every row remains `needs_human_review`.",
        "",
    ]
    for label, rows in (("Calibration", calibration), ("Held-out test", test)):
        lines.extend(
            [
                f"## {label}",
                "",
                "| ID | Language / scenario / type | Question | Evidence |",
                "|---|---|---|---|",
            ]
        )
        for row in rows:
            evidence = (
                ", ".join(row["relevant_chunk_ids"])
                if row["answerable"]
                else "unanswerable — no qrels"
            )
            lines.append(
                f"| {row['id']} | {row['language']} / {row['scenario']} / "
                f"{row['question_type']} | {row['question']} | {evidence} |"
            )
        lines.append("")
    lines.extend(
        [
            "Run `python -m scripts.validate_phase7_dataset` before review.",
            "",
            "Approval phrase: `APPROVE PHASE 7 DATASET`",
        ]
    )
    Path("docs/phase-7-dataset-review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

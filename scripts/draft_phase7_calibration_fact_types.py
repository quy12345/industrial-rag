"""Create a review-required typed-fact draft for Phase 7 calibration only.

The command preserves every question, qrel, page and evidence phrase. It does
not read Qdrant, retrieval artifacts, generated answers, provider output, or
held-out rows. The result intentionally requires a separate human approval.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from app.phase7 import Phase7DatasetItem, Phase7Error, read_phase7_dataset, write_jsonl_atomic

REVIEW_NOTE = (
    "Phase 7.4 typed-fact draft derived from source-reviewed v2 aliases; qrels, pages and "
    "expected phrases are unchanged. Requires independent human review before use."
)


def _text(identifier: str, groups: list[list[str]]) -> dict[str, Any]:
    return {"id": identifier, "type": "text", "required_token_groups": groups}


def _identifier(identifier: str, *values: str) -> dict[str, Any]:
    return {"id": identifier, "type": "identifier", "acceptable_values": list(values)}


TYPED_FACT_OVERRIDES: dict[str, list[dict[str, Any]]] = {
    "phase7_calibration_001": [
        _text("power-to-disconnect", [["disconnect", "isolate"], ["power"]])
    ],
    "phase7_calibration_002": [
        _text("power-to-disconnect", [["ngắt", "cắt"], ["nguồn điện"]])
    ],
    "phase7_calibration_003": [
        _text("prevent-rotation", [["block", "lock", "secure"], ["motor shaft"]])
    ],
    "phase7_calibration_004": [
        _text("prevent-rotation", [["chặn", "khóa", "cố định"], ["trục động cơ"]])
    ],
    "phase7_calibration_005": [
        _text("protective-equipment-state", [["protective equipment"], ["installed", "closed"]])
    ],
    "phase7_calibration_006": [
        _text("protective-equipment-state", [["thiết bị bảo vệ"], ["lắp đặt", "đóng"]])
    ],
    "phase7_calibration_007": [
        _text("contact-timing", [["contact"], ["closed"], ["before"], ["run"]])
    ],
    "phase7_calibration_008": [
        _text("contact-timing", [["tiếp điểm"], ["đóng"], ["trước"], ["run"]])
    ],
    "phase7_calibration_009": [
        _identifier("reference-menu", "rEF"),
        _identifier("monitoring-menu", "MON"),
        _identifier("configuration-menu", "CONF"),
    ],
    "phase7_calibration_010": [
        _identifier("reference-menu", "rEF"),
        _identifier("monitoring-menu", "MON"),
        _identifier("configuration-menu", "CONF"),
    ],
    "phase7_calibration_011": [
        _text("monitored-value", [["actual reference value", "reference value"]])
    ],
    "phase7_calibration_012": [
        _text("monitored-value", [["giá trị tham chiếu", "giá trị reference"]])
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/eval/phase7/calibration.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/eval/phase7/calibration-v3-draft.jsonl")
    )
    args = parser.parse_args()
    records = read_phase7_dataset(args.input)
    answerable = [item for item in records if item.answerable]
    if {item.id for item in answerable} != set(TYPED_FACT_OVERRIDES):
        raise Phase7Error(
            "Typed calibration fact override IDs do not match answerable calibration rows."
        )
    drafted = []
    for item in records:
        if not item.answerable:
            drafted.append(item)
            continue
        existing = {fact.id: fact.model_dump(mode="json") for fact in item.expected_answer_facts}
        typed_facts = []
        for override in TYPED_FACT_OVERRIDES[item.id]:
            source = existing[override["id"]]
            typed_facts.append(source | override)
        note = " ".join(part for part in (item.annotation_notes, REVIEW_NOTE) if part)
        drafted.append(
            Phase7DatasetItem.model_validate(
                item.model_dump(mode="json")
                | {
                    "expected_answer_facts": typed_facts,
                    "annotation_notes": note,
                    "review_status": "needs_human_review",
                }
            )
        )
    write_jsonl_atomic(args.output, [item.model_dump(mode="json") for item in drafted])
    print(f"Phase 7 typed calibration draft written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Phase 7 dataset v2 review — approved and frozen

The dataset-v1 approval and tables below are retained as historical annotation
evidence. Dataset v2 was source-reviewed and frozen on 2026-08-09.

> **Current status:** 65/65 records are `approved`; all 42 answerable rows contain
> reviewed `expected_answer_facts`. Sending evidence to a provider remains a
> separate decision from dataset approval.

Dataset v2 rules:

- `relevant_chunk_ids` alone defines retrieval/citation direct evidence.
- `expected_phrases` validates qrels against the English source text only.
- `expected_answer_facts` scores generated answers. Every required fact has an ID
  and aliases appropriate for the question language.
- The automatic migration added only same-document exact-normalized duplicate
  chunks: 2 qrels in calibration and 14 in held-out. It did not add broad phrase
  matches or inspect held-out retrieval outputs.

Example for one Vietnamese answerable row:

```json
"expected_answer_facts": [
  {
    "id": "required-action",
    "aliases": ["ngắt toàn bộ nguồn điện", "ngắt tất cả nguồn điện"]
  }
]
```

All facts listed for a row are required; aliases within one fact are alternatives.
Review the source chunk, preserve technical numbers/units/codes, and avoid aliases
that accept a materially different claim.

Historical dataset-v1 hashes:

| Split | Answerable | Unanswerable | Canonical SHA-256 |
|---|---:|---:|---|
| Calibration | 12 | 8 | `a795b16acfdaed12adff3c5f1c2c9a4fd21ba5a978e024f7f107087a5e235` |
| Held-out test | 30 | 15 | `810956096df88702707b075f2931a269b7d3fb03f6c4f6c3e335fb5fb3a289a6` |

Intermediate migration-draft hashes, retained for audit history:

| Split | Canonical SHA-256 | Exact-equivalent qrels added |
|---|---|---:|
| Calibration | `6260118215e980e86312920569495fd6b68786e4dfb88034a403d79c41f168be` | 2 |
| Held-out test | `09fb9f06852c770f5d7da815a2e4acf076418dd006579f7d899257cc73cc42d5` | 14 |

Final frozen dataset-v2 hashes:

| Split | Canonical SHA-256 | Status |
|---|---|---|
| Calibration | `7ae670a705dcda2ff63f7e16f67bd8c308b5f58079b4a4b3066dd0f15d9f3999` | approved |
| Held-out test | `68c9c52e745a7616a869a2f55024964501dfb0cb537bbe6ff91dac5fbcae3c54` | approved |

Calibration 011/012 originally pointed at page 355 load-variation text, which
did not directly answer the reference-mode question. Source review corrected both
to page 45 chunk
`atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p45_ha25328d0c34430b2`
and changed the qrel-validation phrase to `actual reference value`. This correction
was based only on frozen source content, not benchmark retrieval or provider output.

The ignored `artifacts/metrics/phase-7-evaluation-manifest.json` records the same
hashes with the frozen corpus and runtime contract.

## Calibration

| ID | Language / scenario / type | Question | Evidence |
|---|---|---|---|
| phase7_calibration_001 | en / en_to_en / safety | What must be disconnected before electrical work? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p21_hb3451538371222ef |
| phase7_calibration_002 | vi / vi_to_en / safety | Trước khi làm việc điện phải ngắt nguồn nào? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p21_hb3451538371222ef |
| phase7_calibration_003 | en / en_to_en / safety | What must be done to prevent motor rotation before work? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p10_hb32dcdd44ab56ee4 |
| phase7_calibration_004 | vi / vi_to_en / safety | Cần làm gì để tránh trục động cơ quay trước khi thao tác? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p10_hb32dcdd44ab56ee4 |
| phase7_calibration_005 | en / en_to_en / installation | What must be verified after completing installation work? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p21_hb3451538371222ef |
| phase7_calibration_006 | vi / vi_to_en / installation | Sau khi hoàn tất lắp đặt phải xác minh điều gì về thiết bị bảo vệ? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p21_hb3451538371222ef |
| phase7_calibration_007 | en / en_to_en / wiring | When must motor-drive contacts be closed relative to a Run command? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p118_h460fd7a19fb32759 |
| phase7_calibration_008 | vi / vi_to_en / wiring | Tiếp điểm giữa động cơ và drive phải đóng khi nào so với lệnh Run? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p118_h460fd7a19fb32759 |
| phase7_calibration_009 | en / en_to_en / menu_navigation | Which menu groups can the MODE key switch between? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p27_he552bd94889b78aa |
| phase7_calibration_010 | vi / vi_to_en / menu_navigation | Phím MODE có thể chuyển giữa những nhóm menu nào? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p27_he552bd94889b78aa |
| phase7_calibration_011 | en / en_to_en / parameter_code | What is reference mode used to monitor? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p45_ha25328d0c34430b2 |
| phase7_calibration_012 | vi / vi_to_en / parameter_code | Reference mode được dùng để giám sát nội dung gì? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p45_ha25328d0c34430b2 |
| phase7_calibration_013 | en / en_to_en / unanswerable | What is the drive's Ethernet MAC address? | unanswerable — no qrels |
| phase7_calibration_014 | vi / vi_to_en / unanswerable | What is the factory GPS coordinate? | unanswerable — no qrels |
| phase7_calibration_015 | en / en_to_en / unanswerable | What hydraulic oil grade is required? | unanswerable — no qrels |
| phase7_calibration_016 | vi / vi_to_en / unanswerable | What is the warranty claim telephone number? | unanswerable — no qrels |
| phase7_calibration_017 | en / en_to_en / unanswerable | What is the excavator bucket capacity? | unanswerable — no qrels |
| phase7_calibration_018 | vi / vi_to_en / unanswerable | What is the latest firmware download URL? | unanswerable — no qrels |
| phase7_calibration_019 | en / en_to_en / unanswerable | What color is the optional external cabinet? | unanswerable — no qrels |
| phase7_calibration_020 | vi / vi_to_en / unanswerable | What is the serial number of this specific drive? | unanswerable — no qrels |

## Held-out test

| ID | Language / scenario / type | Question | Evidence |
|---|---|---|---|
| phase7_test_001 | en / en_to_en / safety | What safety control must be within reach during operation? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p182_h1a19646d5ebc64e1 |
| phase7_test_002 | vi / vi_to_en / safety | Khi vận hành, nút điều khiển an toàn nào phải trong tầm với? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p182_h1a19646d5ebc64e1 |
| phase7_test_003 | en / en_to_en / safety | What assessment is required when designing the machine application? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p9_hf6209a74646e2589 |
| phase7_test_004 | vi / vi_to_en / safety | Khi thiết kế ứng dụng máy cần thực hiện đánh giá nào? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p9_hf6209a74646e2589 |
| phase7_test_005 | en / en_to_en / safety | Which energized parts must not be touched? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p10_hb32dcdd44ab56ee4 |
| phase7_test_006 | vi / vi_to_en / safety | Không được chạm vào bộ phận nào khi còn điện? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p10_hb32dcdd44ab56ee4 |
| phase7_test_007 | en / en_to_en / wiring | What must be removed from input and motor terminals after work? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p21_hb3451538371222ef |
| phase7_test_008 | vi / vi_to_en / wiring | Sau khi thao tác cần tháo gì khỏi đầu vào và đầu cực động cơ? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p21_hb3451538371222ef |
| phase7_test_009 | en / en_to_en / installation | What does the installation manual mean by factory setting? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p191_h13f5f76dbb614657 |
| phase7_test_010 | vi / vi_to_en / installation | Installation manual định nghĩa factory setting là gì? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p191_h13f5f76dbb614657 |
| phase7_test_011 | en / en_to_en / fault_diagnosis | What is the purpose of Fault Reset? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p191_h219ac8690dc63c8f |
| phase7_test_012 | vi / vi_to_en / fault_diagnosis | Fault Reset có mục đích gì? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p191_h219ac8690dc63c8f |
| phase7_test_013 | en / en_to_en / menu_navigation | Which controls change the reference when local control is enabled? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p45_ha25328d0c34430b2 |
| phase7_test_014 | vi / vi_to_en / menu_navigation | Khi local control bật, điều khiển nào thay đổi reference? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p45_ha25328d0c34430b2 |
| phase7_test_015 | en / en_to_en / menu_navigation | Is ENT required to confirm a reference change? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p45_ha25328d0c34430b2 |
| phase7_test_016 | vi / vi_to_en / menu_navigation | Có cần nhấn ENT để xác nhận thay đổi reference không? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p45_ha25328d0c34430b2 |
| phase7_test_017 | en / en_to_en / parameter_code | What determines which parameters are displayed? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p389_h4dd5a5f87549cc89 |
| phase7_test_018 | vi / vi_to_en / parameter_code | Yếu tố nào quyết định các parameter được hiển thị? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p389_h4dd5a5f87549cc89 |
| phase7_test_019 | en / en_to_en / parameter_code | What is the HMI label for the reference-frequency parameter? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p373_he0b26588759aed86 |
| phase7_test_020 | vi / vi_to_en / parameter_code | Nhãn HMI của parameter reference frequency là gì? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p373_he0b26588759aed86 |
| phase7_test_021 | en / en_to_en / parameter_code | What is the documented range for the shown reference-frequency parameter? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p46_h32da8bea0b5b7e96 |
| phase7_test_022 | vi / vi_to_en / parameter_code | Dải giá trị được ghi cho parameter reference frequency là bao nhiêu? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p46_h32da8bea0b5b7e96 |
| phase7_test_023 | en / en_to_en / parameter_code | Which parameter name is shown for the first virtual analog-input image? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p46_h32da8bea0b5b7e96 |
| phase7_test_024 | vi / vi_to_en / parameter_code | Tên parameter nào được hiển thị cho virtual analog-input image thứ nhất? | atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p46_h32da8bea0b5b7e96 |
| phase7_test_025 | en / en_to_en / cross_document | How is fault reset described across the two ATV320 manuals? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p191_h1c24f7dadd716dc2, atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p327_h029647154a53f0f3 |
| phase7_test_026 | vi / vi_to_en / cross_document | Hai manual ATV320 mô tả fault reset như thế nào? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p191_h1c24f7dadd716dc2, atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p327_h029647154a53f0f3 |
| phase7_test_027 | en / en_to_en / cross_document | Which manual evidence addresses an emergency-stop push-button? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p182_h1a19646d5ebc64e1, atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p44_heff908e11423127d |
| phase7_test_028 | vi / vi_to_en / cross_document | Bằng chứng ở manual nào đề cập emergency-stop push-button? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p182_h1a19646d5ebc64e1, atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p44_heff908e11423127d |
| phase7_test_029 | en / en_to_en / cross_document | Which documents define or expose factory-setting information? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p191_h13f5f76dbb614657, atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p111_h8c01ec4eeabe1c5f |
| phase7_test_030 | vi / vi_to_en / cross_document | Những document nào định nghĩa hoặc thể hiện thông tin factory setting? | atv320-installation-manual-en-nve41289-09-c181b4d7f11b_p191_h13f5f76dbb614657, atv320-programming-manual-en-nve41295-06-f5e9bb48167a_p111_h8c01ec4eeabe1c5f |
| phase7_test_031 | en / en_to_en / unanswerable | What is the maximum hydraulic pressure of the excavator? | unanswerable — no qrels |
| phase7_test_032 | vi / vi_to_en / unanswerable | Which employee approved this installation? | unanswerable — no qrels |
| phase7_test_033 | en / en_to_en / unanswerable | What is the building fire-escape route? | unanswerable — no qrels |
| phase7_test_034 | vi / vi_to_en / unanswerable | What is the current electricity tariff? | unanswerable — no qrels |
| phase7_test_035 | en / en_to_en / unanswerable | What is the drive's Wi-Fi password? | unanswerable — no qrels |
| phase7_test_036 | vi / vi_to_en / unanswerable | Which truck delivered the ATV320? | unanswerable — no qrels |
| phase7_test_037 | en / en_to_en / unanswerable | What is the annual weather forecast for the installation site? | unanswerable — no qrels |
| phase7_test_038 | vi / vi_to_en / unanswerable | What is the operator's personal phone number? | unanswerable — no qrels |
| phase7_test_039 | en / en_to_en / unanswerable | What is the chemical composition of the room floor? | unanswerable — no qrels |
| phase7_test_040 | vi / vi_to_en / unanswerable | Which cloud region stores the drive data? | unanswerable — no qrels |
| phase7_test_041 | en / en_to_en / unanswerable | What is the purchase order number? | unanswerable — no qrels |
| phase7_test_042 | vi / vi_to_en / unanswerable | What is the machine's paint supplier? | unanswerable — no qrels |
| phase7_test_043 | en / en_to_en / unanswerable | What is the next scheduled factory shutdown date? | unanswerable — no qrels |
| phase7_test_044 | vi / vi_to_en / unanswerable | Which PLC program file is installed? | unanswerable — no qrels |
| phase7_test_045 | en / en_to_en / unanswerable | What is the operator's blood type? | unanswerable — no qrels |

Run `python -m scripts.validate_phase7_dataset` before review.

The following validation/freeze commands have completed successfully and remain
the reproducibility procedure:

```powershell
python -m scripts.validate_phase7_dataset
python -m scripts.freeze_phase7_dataset `
  --approval-token "APPROVE PHASE 7 DATASET V2"
```

Applied approval phrase: `APPROVE PHASE 7 DATASET V2`

# Phase 7 script archive

This package preserves completed Phase 7 construction and migration tools as historical evidence.
They are unsupported, are not active runtime or evaluation entry points, and must not be used to
modify frozen datasets or artifacts during Round 1.

## Archived commands

| Command | Historical purpose | Completion evidence |
| --- | --- | --- |
| `migrate_phase7_dataset_v2` | Created a review-required dataset-v2 draft with exact-content qrel closure. | The migration report exists and current documentation records dataset v2 as source-reviewed, approved, and frozen. |
| `draft_phase7_calibration_fact_types` | Created the review-required typed-fact calibration-v3 draft. | The evaluation manifest records the draft identity and current documentation records calibration-v3 as approved, active, and frozen. |
| `freeze_phase7_calibration_v3` | Approved the reviewed typed-fact draft and wrote the calibration-v3 manifest. | Current documentation and the manifest record calibration-v3 as approved, active, and frozen. |

No compatibility shim is provided at the former top-level path. Read the source for provenance; do
not run it against current data without a separately approved migration plan.

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
| `apply_phase7_answer_facts` | Applied the source-reviewed answer-fact mapping and narrow qrel correction before dataset approval. | The mapping was incorporated before the approved dataset-v2 and calibration-v3 freezes; no current runtime imports this command. |
| `generate_phase7_annotation_draft` | Generated the initial review-only calibration/test annotation files and review document. | The tracked review receipt is now metadata-only and records the approved frozen dataset; no current consumer imports this generator. |
| `freeze_phase7_dataset` | Approved dataset-v2 and wrote its historical evaluation manifest. | The active evaluator now pins calibration-v3 and the v3 manifest; both v2 and v3 receipts already exist. |

No compatibility shim is provided at the former top-level path. Read the source for provenance; do
not run it against current data without a separately approved migration plan.

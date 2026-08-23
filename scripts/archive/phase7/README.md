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
| `freeze_phase7_heldout_v2` | Approved the private replacement held-out v2 dataset and wrote its private manifest. | The private approved files and sanitized one-shot result already exist; current governance forbids treating the set as a fresh tuning benchmark. |
| `evaluate_phase7_heldout_v2` | Performed the explicitly approved one-shot private held-out-v2 provider evaluation. | The sanitized final artifact exists and is a historical measurement, not a rerunnable tuning command. |
| `diagnose_phase7_calibration_005` | Reused one fixed evidence bundle for the approved three-attempt provider diagnostic. | The sanitized result records three positive attempts; readiness consumes that artifact rather than importing this command. |
| `rescore_phase7_calibration_facts` | Reconstructed deterministic text-fact decisions from an earlier sanitized diagnostic artifact. | Both the historical source and derived rescore artifacts exist; no current consumer imports this command. |
| `generate_phase7_fact_evaluator_readiness` | Validated the review-required typed-fact draft and wrote a sanitized review receipt. | Calibration-v3 is approved and active, so the draft-only `HUMAN_REVIEW_REQUIRED` decision is historical. |
| `generate_phase7_runtime_readiness` | Combined provider-free closure receipts into the pre-egress provider-approval decision. | The runtime-readiness v2 receipt exists and the approved calibration provider run has completed. |
| `evaluate_phase7_weighted_rerank` | Reran at most six weighted-fusion profiles through local Jina and selected an intermediate calibration profile. | Phase 7.4.1–7.5 superseded this Phase 7.4 experiment, and the v1–v3 sanitized result artifacts remain as historical evidence. |
| `audit_phase7_retrieval_failures` | Audited three named calibration misses with sanitized dense/sparse ranks and fixed query variants. | The audit artifact exists and the later frozen retrieval closure resolves the tracked misses. |
| `calibrate_phase7_retrieval` | Compared early dense/sparse union and RRF candidate-pool profiles. | The ablation artifact exists and the Phase 7.4.1–7.5 runtime profile is frozen. |
| `calibrate_phase7_weighted_fusion` | Searched the bounded weighted-RRF, role and reserve grid before reranking. | The weighted-fusion artifact exists and the selected frozen profile was validated by later closure. |
| `create_phase7_reranker_snapshot` | Created sanitized Jina snapshots for deterministic rank-policy replay. | Snapshot v3 and its downstream relation-list ablation already exist. |
| `calibrate_phase7_role_prior` | Replayed the sanitized snapshot through the finite role/list policy grid. | The relation-list ablation and real runtime closure both record the released profile. |
| `benchmark_phase7_reranker_cpu` | Measured the bounded micro/full CPU configuration grid. | Both sanitized benchmark artifacts exist; further latency work is deferred to Round 2. |
| `aggregate_phase7_calibration_stability` | Applied worst-run gates to three approved calibration runs. | The three run artifacts and the stability receipt exist and record a technical pass. |
| `generate_phase7_calibration_closure_readiness` | Combined calibration closure receipts into the historical technical/governance decision. | The readiness artifact exists and the replacement held-out-v2 one-shot workflow has completed. |

No compatibility shim is provided at the former top-level path. Read the source for provenance; do
not run it against current data without a separately approved migration plan.

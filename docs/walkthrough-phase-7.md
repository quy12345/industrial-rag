# Phase 7 walkthrough: isolated corpus, E2E evaluation, and hardening

## What Phase 7 changes

Phase 7 is intentionally separate from the earlier 99-chunk development corpus.
It introduces two Schneider ATV320 manuals, frozen as 2,753 chunks in dedicated
Qdrant collections:

```text
industrial_manual_phase7_dense_v1
industrial_manual_phase7_hybrid_v1
```

The two legacy collections remain unchanged at 99 points each. The Phase 7
stable-ID hash is:

```text
2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d
```

## Evaluation discipline

`data/eval/phase7/calibration.jsonl` has 12 answerable and 8 unanswerable
records. `data/eval/phase7/test.jsonl` has a separate 30 answerable and 15
unanswerable records. Dataset v1 was approved for the historical calibration.
Dataset v2 is now source-reviewed and frozen: all 42 answerable rows contain
`expected_answer_facts`, all 65 rows are approved, and hashes are locked in the
schema-v2 evaluation manifest.

The E2E evaluator is deliberately strict:

```text
approved dataset-v2 JSONL + frozen manifest + live Qdrant hash
  -> union retrieval + reranking
  -> evidence gate
  -> generation
  -> citation validation
  -> qrel-only/citation/abstention/latency metrics
```

Retrieval hits use only `relevant_chunk_ids`. Same page, matching phrase, or
matching document are diagnostics, not a hit. The output excludes raw questions,
answers, prompts, evidence, and provider responses.

`expected_phrases` remains an evidence/qrel validation field. Generated answers
are scored only against reviewed `expected_answer_facts`; each fact contains one
or more language-appropriate aliases. This prevents a Vietnamese answer from
being marked wrong merely because the evidence manual is English.

The Phase 7 runtime also collapses same-document candidates only when their raw
text is identical after NFKC, case and whitespace normalization. It preserves all
equivalent chunk IDs in metadata and merges the best dense/sparse/RRF signals.
This optimization is disabled by default outside Phase 7.

## Run order

```powershell
python -m scripts.validate_phase7_dataset
# Reproduce the already-completed dataset-v2 freeze if the reviewed files are unchanged.
python -m scripts.freeze_phase7_dataset `
  --approval-token "APPROVE PHASE 7 DATASET V2"
python -m scripts.evaluate_phase7_e2e --dataset calibration
python -m scripts.evaluate_phase7_e2e --dataset test
```

Calibration comes first. If it reveals a correctness issue, fix code and rerun
calibration; do not tune on the held-out test. The last two commands send selected
questions and retrieved excerpts to the configured provider, so they require an
explicit corpus-owner data-egress approval.

## Operational hardening

- `GET /api/v1/health` is a liveness check and needs no Qdrant/model/provider.
- `GET /api/v1/ready` validates Qdrant connectivity and the legacy frozen
  collection contract without loading an embedding model or provider.
- Every HTTP response has an `X-Request-ID`; safe stage logs include this ID and
  latency only, never question/evidence/secret content.
- `QDRANT_TIMEOUT_SECONDS` bounds client calls (default 10 seconds).
- `API_AUTH_ENABLED=true` plus `API_AUTH_KEY` enables a constant-time `Bearer`
  guard for `/api/v1/query`.
- API Docker healthcheck calls only `/api/v1/health`; it does not pull models.

## Status and known limits

Offline Python 3.11.15 validation passed with Ruff and 185 tests (one known
third-party Starlette/TestClient deprecation warning). Live Phase 7 Qdrant hash
validation passed.

The dataset-v1 Gemini calibration ran on 2026-08-09. Referential citation validity
and abstention precision/recall passed, but answer phrase accuracy, qrel candidate
recall, and direct-evidence citation diagnostics did not meet release expectations.
The held-out benchmark therefore remains unrun. See the generated calibration
artifact for the measured values; do not tune against the held-out set.

| Calibration metric | Result | Interpretation |
|---|---:|---|
| Candidate recall | 0.583 | 5/12 answerable qrels absent from the union pool |
| Hit@1 | 0.333 | Direct-evidence stable ID only |
| Hit@5 / Hit@20 | 0.583 / 0.583 | Reranker recovered every qrel present in the pool into top 5 |
| MRR@5 / MRR@20 | 0.444 / 0.444 | One-based reciprocal rank |
| Historical answer phrase score | 0.417 | Invalid bilingual metric; replaced by reviewed answer facts in dataset v2 |
| Referential citation validity | 1.000 | PASS; no unknown/out-of-context source IDs |
| Direct-evidence citation rate | 0.500 | Diagnostic against approved qrels |
| Citation document correctness | 0.583 | Some answers included citations from another manual |
| Abstention precision / recall | 1.000 / 1.000 | All 8 unsupported cases abstained; all 12 answerable cases answered |
| Total p50 / p95 | 8.836 s / 11.585 s | Includes retrieval, reranking, generation and validation |
| Rerank p95 | 10.450 s | Dominant CPU latency stage |

The historical dataset-v1 source-of-truth artifact is
`artifacts/metrics/phase-7-calibration-e2e.json`. Its gate status is `FAIL`, so
the evaluator exits non-zero and the held-out command must not run yet.

The initial v2 artifact remains
`artifacts/metrics/phase-7-calibration-e2e-v2.json`. The improved evaluator writes
`artifacts/metrics/phase-7-calibration-e2e-v2-diagnostics.json` and a separate
checkpoint. It never overwrites historical v1/v2 evidence.

## Dataset-v2 calibration diagnostics

The diagnostics run completed all 20 calibration rows. It still fails the strict
answer-fact gate, so held-out remains sealed.

| Calibration-v2 metric | Result | Interpretation |
|---|---:|---|
| Candidate recall / Hit@5 | 0.667 / 0.667 | 4/12 direct qrels absent from the baseline pool |
| Hit@1 / MRR@5 | 0.500 / 0.583 | Stable direct-evidence IDs only |
| Strict answer-fact accuracy | 0.500 | FAIL against 0.85 release gate |
| All-alias-token coverage | 0.667 | Diagnostic only; order/negation semantics are not guaranteed |
| Referential citation validity | 1.000 | PASS; no unsupported source IDs |
| Direct-evidence citation rate | 0.667 | 8/12 answers cite a direct qrel |
| Citation document correctness | 0.583 | Cross-manual citations remain common |
| Abstention precision / recall | 1.000 / 1.000 | All 8 unsupported cases abstained |
| Total p50 / p95 | 8.830 s / 10.043 s | Reranker remains dominant |
| Rerank p95 | 8.784 s | Still unsuitable for a low-latency CPU target |

Sanitized fact diagnostics show calibration 002 and 008 have direct evidence at
rank 1 and full alias-token coverage, but fail strict contiguous-substring matching.
They are classified as scorer-format mismatches, not silently changed to PASS.
Calibration 004, 005, 006 and 010 remain candidate misses.

The provider-free artifact
`artifacts/metrics/phase-7-calibration-retrieval-ablation.json` compares 13 pools.
The baseline union20/20 has recall 0.667 with 31.25 candidates/query. Union60/40
reaches 0.833 but averages 81.17 candidates; RRF60/40 top60 has the same recall
with 59.5. Both still miss calibration 004/010 and would enlarge the expensive
reranker stage, so no runtime setting was changed.

The v2 migration report is
`artifacts/metrics/phase-7-dataset-v2-migration.json`: corpus identity stayed
unchanged, exact-content closure added 2 calibration and 14 held-out qrels, no
provider was called, and no broad phrase match was converted into a qrel. Those
migration-draft hashes were intentionally not frozen before source review.

Source review then populated all 42 answerable records and corrected calibration
011/012 from unrelated load-variation content on page 355 to direct reference-mode
evidence on page 45. Final hashes are:

```text
calibration: 7ae670a705dcda2ff63f7e16f67bd8c308b5f58079b4a4b3066dd0f15d9f3999
held-out:    68c9c52e745a7616a869a2f55024964501dfb0cb537bbe6ff91dac5fbcae3c54
```

The freeze manifest has `git_commit: null` because Git was unavailable inside
the validation image; dataset/corpus hashes remain the reproducibility contract.

The HybridChunker preview emitted a warning for a 4,244-token source segment
against a 512-token tokenizer limit. The frozen output validated, but the warning
remains a chunking limitation for production review.

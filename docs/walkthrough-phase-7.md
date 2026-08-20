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

`expected_phrases` remains an evidence/qrel validation field. Generated answers use
`phase7_deterministic_typed_facts_v1` as the headline evaluator. Supported contracts are text,
numeric value+unit, identifier, and explicit required-token groups. Existing dataset-v2 aliases
remain compatible through an order-independent token-set matcher. Contiguous alias accuracy and
token coverage are separate diagnostics and never use source-language evidence phrases.

The Phase 7 runtime also collapses same-document candidates only when their raw
text is identical after NFKC, case and whitespace normalization. It preserves all
equivalent chunk IDs in metadata and merges the best dense/sparse/RRF signals.
This optimization is disabled by default outside Phase 7. Phase 7.4 additionally runs dense@60 and
sparse@40 after deterministic Vietnamese technical-term augmentation, prunes with RRF `k=60` to 30,
and sends at most 30 candidates to the unchanged Jina reranker. Trusted document title and role are
included in reranker/evidence inputs; expected document IDs are never used as a retrieval filter.

## Run order

```powershell
python -m scripts.validate_phase7_dataset
# Reproduce the already-completed dataset-v2 freeze if the reviewed files are unchanged.
python -m scripts.freeze_phase7_dataset `
  --approval-token "APPROVE PHASE 7 DATASET V2"
python -m scripts.audit_phase7_retrieval_failures
python -m scripts.calibrate_phase7_retrieval
python -m scripts.evaluate_phase7_retrieval_closure
# Requires explicit approval to send calibration questions/evidence externally:
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

Offline Python 3.11.15 validation passed with Ruff and 208 tests (one known
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
`artifacts/metrics/phase-7-calibration-e2e-v2.json`. The historical diagnostics artifact is
`artifacts/metrics/phase-7-calibration-e2e-v2-diagnostics.json`. Phase 7.4 defaults to a new
`phase-7-calibration-e2e-v3-phase74.json` artifact/checkpoint and never overwrites v1/v2 evidence.

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
`artifacts/metrics/phase-7-calibration-retrieval-ablation.json` now compares 15 pools. The original
13-pool result was:
The historical baseline union20/20 has recall 0.667 with 31.25 candidates/query. Union60/40
reaches 0.833 but averages 81.17 candidates; RRF60/40 top60 has the same recall
with 59.5. Both still miss calibration 004/010 and would enlarge the expensive
reranker stage, so that first ablation did not change runtime.

## Phase 7.4 calibration closure

The evaluator now reports three deliberately different answer metrics:

```text
strict_phrase_accuracy        contiguous legacy alias; diagnostic only
deterministic_fact_accuracy   typed/order-independent headline metric
token_coverage                overlap diagnostic; not a release gate
```

Because the old sanitized artifact intentionally contains no answer text, Phase 7.4 did not pretend
to rerun generation. `phase-7-calibration-fact-rescore-v1.json` reconstructs only legacy text-fact
decisions from stored per-fact token recall. It measures `6/12 = 0.500` strict versus
`8/12 = 0.667` deterministic, changing only 002/008. The deterministic result still fails the
required `11/12` gate and is explicitly a derived rescore.

The canonical top-200 audit now verifies qrels for 004/005/010 exist in both Qdrant collections and
reviewed source phrases exist in frozen chunks. Calibration 005 is a sparse-tail case (original rank
24; its diagnostic technical wording reaches rank 8), while 010 is an ordering/query-formulation case
(original sparse rank 185; diagnostic English/technical variants rank 4). This is evidence against an
ingestion-loss diagnosis. The glossary contains query terms only—no qrel, page, expected answer,
document ID, or held-out-specific rule.

The original closure artifact remains historical. The current canonical Python 3.11 Phase 7.4 runtime
uses dense@60, expanded sparse@40, weighted rank-only RRF `k=40`, sparse weight `1.25`, dense@5 and
sparse@24 coverage reserves, plus a query-only soft document-role prior before and after Jina. It
measured:

| Metric | Result |
|---|---:|
| Candidate recall | 12/12 = 1.000 |
| Maximum reranker input | 30 |
| Hit@5 after unchanged Jina | 11/12 = 0.917 |
| MRR@5 | 0.875 |
| Candidate miss | none |
| Present but outside top 5 | calibration 010 at rank 6 |
| Wrong-document top-1 retrieval | 0.000 |
| Wrong-document candidates in top 5 | 0.267 |
| Document title/role metadata completeness | 1.000 |
| Retrieval p95 | 178.0 ms |
| Reranker p95 | 13.399 s |

The source of truth is `artifacts/metrics/phase-7-retrieval-closure-v2.json`; it records zero provider
calls and zero held-out executions. Its overall gate is `PARTIAL`: candidate recall, Hit@5, budget,
context completeness and wrong-document top-1 pass, but the `<= 0.15` wrong-document top-5 candidate
gate fails. A fresh provider E2E run was not performed because external calibration data egress requires
explicit approval. New answer/citation gates are therefore pending, and opening held-out remains
prohibited.

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

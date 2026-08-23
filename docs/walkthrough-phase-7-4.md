# Phase 7.4 walkthrough — calibration retrieval closure

## Outcome

This document records the Phase 7.4 intermediate checkpoint. Phase 7.4.1--7.5 subsequently closes
the contamination gate and measures the frozen CPU profile; see
[`walkthrough-phase-7-5.md`](walkthrough-phase-7-5.md) for the current contract and metrics.

Phase 7.4 improves the frozen ATV320 calibration runtime without changing chunks, qrels, embedding
model, Jina model, Qdrant schema, collections, or held-out data. The canonical Python 3.11 closure is
`PARTIAL`, not release-ready: retrieval quality improved, but final top-5 still contains too many
chunks from the wrong manual.

| Metric | Historical closure | Current closure |
|---|---:|---:|
| Candidate recall | 11/12 | 12/12 |
| Hit@5 | 10/12 | 11/12 |
| MRR@5 | 0.667 | 0.875 |
| Wrong-document top-1 | 0.250 | 0.000 |
| Wrong-document top-5 candidates | 0.300 | 0.267 |
| Maximum Jina inputs | 30 | 30 |

The current artifact is `artifacts/metrics/phase-7-retrieval-closure-v2.json`. It contains zero
provider calls and zero held-out executions.

## What changed

```text
question
  -> dense top 60
  -> Vietnamese glossary expansion, then sparse top 40
  -> query-only role inference: installation | programming | neutral
  -> weighted RRF k=40 (dense weight 1.0, sparse weight 1.25)
  -> retain dense top 5 and sparse top 24; fill the rest by fused order
  -> exact-content deduplication; never more than 30 candidates
  -> unchanged Jina cross-encoder
  -> small rank-only role prior; final top-k
```

The role classifier uses generic bilingual technical terms only. It never reads qrels, expected pages,
expected document IDs, answer facts, or held-out data. It is a soft boost, never a document filter.
The original cross-encoder score remains in `rerank_score`; the final role-aware score is rank-derived,
not a probability and not a raw-score mixture.

## Why these changes

- Calibration 005 was present only at sparse rank 24. Ordinary RRF pruning discarded it. Sparse@24
  reserve now preserves it inside the same reranker budget.
- Calibration 010 is available but Jina places its qrel at rank 6. It is classified as a reranker
  top-5 miss, not an ingestion or qrel failure.
- The audit proves direct phrases and qrels for 004/005/010 exist in both Phase 7 collections. Chunking
  is therefore not changed in this phase.
- Strict phrase matching is diagnostic only. Deterministic typed fact scoring is the answer-quality
  headline metric and includes numeric/unit, identifier, token-group and negation checks.

## Commands

Run these against Qdrant plus the shared FastEmbed cache. They do not call a generation provider:

```powershell
python -m scripts.calibrate_phase7_weighted_fusion
python -m scripts.audit_phase7_retrieval_failures `
  --output artifacts/metrics/phase-7-calibration-004-005-010-audit-v2.json
python -m scripts.evaluate_phase7_retrieval_closure `
  --output artifacts/metrics/phase-7-retrieval-closure-v2.json
```

The weighted-rerank evaluator produced the historical v1–v3 calibration artifacts and has been
superseded by the Phase 7.4.1–7.5 closure. Its unsupported source is retained under
`scripts/archive/phase7/`; do not rerun it as a current evaluation command.

The completed `draft_phase7_calibration_fact_types` construction command is now retained only under
`scripts/archive/phase7/`. It produced the review-required calibration-v3 draft while preserving every
qrel/page/phrase and without touching the sealed held-out file. Calibration-v3 was subsequently
reviewed, approved, and frozen, so regenerating that draft is not a supported current workflow.

Use the existing ingestion image for canonical Python 3.11 validation; do not rebuild it merely for
these commands. The local `.venv` in this workspace is Python 3.13.5 and is not the canonical Phase
7 validation environment.

## Gates and next step

Passed: candidate recall, Hit@5, reranker budget, trusted context completeness, and wrong-document
top-1. Failed: wrong-document candidate rate in final top-5 must be at most `0.15`; actual is `0.267`.

Do not run provider E2E or held-out yet. First decide whether to accept a more expressive, still
query-only document-role policy, or record the cross-document contamination as a known limitation.
If and only if the offline gate passes, obtain explicit permission before sending calibration questions
and retrieved excerpts to Gemini/OpenAI. Held-out can run once only after fresh calibration also passes
fact, citation, document and abstention gates.

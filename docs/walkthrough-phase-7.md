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
unanswerable records. Both are approved and hash-locked in the ignored
evaluation manifest.

The E2E evaluator is deliberately strict:

```text
approved JSONL + frozen manifest + live Qdrant hash
  -> union retrieval + reranking
  -> evidence gate
  -> generation
  -> citation validation
  -> qrel-only/citation/abstention/latency metrics
```

Retrieval hits use only `relevant_chunk_ids`. Same page, matching phrase, or
matching document are diagnostics, not a hit. The output excludes raw questions,
answers, prompts, evidence, and provider responses.

## Run order

```powershell
python scripts/validate_phase7_dataset.py
python scripts/evaluate_phase7_e2e.py --dataset calibration
python scripts/evaluate_phase7_e2e.py --dataset test
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

Offline Python 3.11 validation passed with Ruff and 177 tests (one known
third-party Starlette/TestClient deprecation warning). Live Phase 7 Qdrant hash
validation passed. Provider calibration and the held-out benchmark are pending
data-egress approval, so no Phase 7 quality metrics are claimed yet.

The HybridChunker preview emitted a warning for a 4,244-token source segment
against a 512-token tokenizer limit. The frozen output validated, but the warning
remains a chunking limitation for production review.

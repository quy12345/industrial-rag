# Phase 7.4.1--7.5 walkthrough: contamination closure and CPU reranking

## Outcome

The Phase 7 calibration runtime now passes its provider-free retrieval and contamination gates while
the held-out dataset remains sealed. Nothing in this work changes the 2,753 frozen chunks, qrels,
MiniLM embedding model, Jina reranker model, Qdrant collections, or public query API.

| Metric | Before | Selected runtime |
|---|---:|---:|
| Candidate recall | 12/12 | 12/12 |
| Hit@5 | 11/12 | 11/12 |
| MRR@5 | 0.875 | 0.875 |
| Wrong-document top-1 | 0/12 | 0/12 |
| Wrong-document top-5 | 16/60 (0.267) | 8/60 (0.133) |
| Jina rerank p95 | 13.399 s | 6.996 s |
| Total p95 | not a Phase 7.4 target | 7.027 s |

The target for wrong-document top-5 is at most `9/60 = 0.15`, so the observed `8/60` passes. The
CPU result is three full repetitions of the 12 answerable calibration items. It improves rerank p95
by 47.8% and keeps all measured retrieval gates unchanged. The separate one-pass closure measures
rerank p95 `9.706 s`; it validates the final runtime configuration but is not statistically
comparable to the repeated CPU benchmark.

## Selected runtime profile

```text
dense top 60
  + expanded sparse top 40 (frozen Vietnamese technical glossary)
  -> query-role inference v2
  -> weighted RRF k=40, dense=1.00, sparse=1.25, fusion role multiplier=0.10
  -> reserve dense 5 + sparse 24, then deterministic fill to 30
  -> same-document exact-content deduplication
  -> Jina multilingual cross-encoder
  -> rank-only document-role prior: multiplier=0.50, offset=20,
     confidence=strong-and-weak
  -> final top-k
```

The post-rerank signal uses ranks only: it does not add cosine, BM25, RRF, and Jina raw scores
together. `rerank_score` remains the original cross-encoder score for diagnosis. A neutral or
conflicting role inference receives no post-rerank role prior. The role classifier only reads generic
bilingual query cues; it cannot read qrels, query IDs, expected documents, pages, or answer facts.
There is no hard document filter.

The runtime freezes the reranker candidate budget at 30 and batch size at 8. ONNX thread configuration
is intentionally the runtime default because it won the measured CPU comparison. These settings are
part of the frozen Phase 7 package contract, not an environment accident.

## Why replay exists

Running Jina for every role-prior configuration would make a small calibration selection expensive
and hard to audit. `create_phase7_reranker_snapshot.py` therefore runs the baseline cross-encoder once
and stores a sanitized snapshot with only IDs, document roles, ranks, finite scores, role cue IDs, and
hashes. It excludes questions, raw chunks, and evidence text.

`calibrate_phase7_role_prior.py` replays that snapshot with a finite grid of post-rerank multipliers,
rank offsets, and confidence modes. It uses six bilingual intent folds for selection. Fold membership
is evaluation-only; the runtime does not import it. This avoids both hundreds of redundant Jina runs
and an evaluation rule leaking into production retrieval.

## Artifacts and their privacy boundary

All generated artifacts are ignored by Git and are machine-readable evidence rather than runtime
dependencies.

| Artifact | Purpose | Excluded data |
|---|---|---|
| `phase-7-reranker-snapshot-v1.json` | One local-Jina replay input | question, raw chunk/evidence |
| `phase-7-role-prior-ablation-v1.json` | Fold and full-profile comparison | question, raw evidence, provider data |
| `phase-7-contamination-closure-v4.json` | Final real-Qdrant/Jina closure | provider output, held-out rows |
| `phase-7-cpu-reranker-micro-v1.json` | Small CPU configuration screen | provider data, held-out rows |
| `phase-7-cpu-reranker-ablation-v1.json` | Full repeated CPU benchmark | provider data, held-out rows |
| `phase-7-fact-evaluator-readiness-v1.json` | Typed-fact draft review state | generated answers and held-out output |
| `phase-7-runtime-readiness-v2.json` | Freeze evidence before provider E2E | questions, evidence, answers |

The runtime validates corpus/index identity directly; it does not import `artifacts/` in the API
image. A changed collection count, corpus hash, model, or frozen profile must fail closed.

## Reproduction commands

Use the existing Python 3.11 ingestion container with the source bind-mounted and shared FastEmbed
cache. Do not build an image, re-index, prune Docker, or delete Qdrant volumes for these commands.

```powershell
python -m scripts.create_phase7_reranker_snapshot
python -m scripts.calibrate_phase7_role_prior
python -m scripts.evaluate_phase7_retrieval_closure `
  --output artifacts/metrics/phase-7-contamination-closure-v4.json
python -m scripts.benchmark_phase7_reranker_cpu --stage micro
python -m scripts.benchmark_phase7_reranker_cpu --stage full
python -m scripts.generate_phase7_fact_evaluator_readiness
python -m scripts.generate_phase7_runtime_readiness `
  --closure artifacts/metrics/phase-7-contamination-closure-v4.json `
  --output artifacts/metrics/phase-7-runtime-readiness-v2.json
```

The first, third, and CPU commands use local Qdrant/Jina and model cache. The replay and readiness
commands are provider-free; they never execute held-out questions. Standard offline validation remains:

```powershell
python -m ruff check .
python -m pytest -q --basetemp /tmp/pytest-phase75
git diff --check
docker compose config --quiet
```

## What is still blocked intentionally

The typed `calibration-v3-draft.jsonl` was approved into a separate `calibration-v3.jsonl`; it did not
change calibration-v2, qrels, pages, phrases, chunks, or held-out. The new freezer requires the exact
calibration/provider approval token and writes a distinct v3 manifest. The held-out file is only read
for schema/hash validation and is never rewritten.

Gemini 3.5 Flash Lite then completed the approved 20-row calibration E2E. Its sanitized artifact
reports candidate recall `1.000`, valid citation IDs `1.000`, direct-evidence citation rate `0.917`,
and abstention precision/recall `1.000`. It still fails release: deterministic typed-fact accuracy is
`7/12 = 0.583` and two answerable outputs carry a wrong-document citation. No raw question, answer,
prompt, evidence, or provider response is stored.

Do not run held-out. A new readiness decision is required only after calibration fixes meet all fact,
citation-document, and abstention gates. The E2E CLI now requires a different held-out provider token,
so the calibration token cannot authorize it accidentally.

## Remaining limitations

- Jina remains CC-BY-NC-4.0 and is not cleared for commercial deployment.
- About seven seconds p95 is still slow for an interactive CPU-only production service.
- No GPU, ONNX quantization/model change, ingestion/chunking change, or cross-encoder replacement is
  part of this closure.
- The corpus is a controlled two-manual evaluation corpus, not a production industrial-document
  collection.

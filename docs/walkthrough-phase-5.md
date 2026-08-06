# Phase 5 walkthrough — multilingual cross-encoder reranking

Phase 5 reranks three frozen Phase 4 candidate pools without changing chunks, qrels, embedding/BM25
settings, RRF, Qdrant schemas, or either 99-point collection. Implementation and real benchmark are
complete. Ranking quality passed, but measured CPU latency did not; the phase is therefore
`PARTIAL` and no reranking strategy is configured as the runtime default.

## 1. Frozen input and model contract

The run used:

- document `manual-77d5dae4c2c5`;
- 30 direct-evidence development qrels: 15 Vietnamese and 15 English-to-Vietnamese;
- 99 frozen chunks with ID hash
  `bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be`;
- v1 `industrial_manual_chunks`: 99 dense points;
- v2 `industrial_manual_chunks_v2`: 99 dense+sparse points;
- Python 3.11.15, FastEmbed 0.8.0, qdrant-client 1.19.0, Qdrant server 1.18.3.

FastEmbed 0.8.0 exposes:

```python
from fastembed.rerank.cross_encoder import TextCrossEncoder

model = TextCrossEncoder(
    model_name="jinaai/jina-reranker-v2-base-multilingual",
    cache_dir="/models/fastembed",
    lazy_load=True,
)
scores = model.rerank(query, documents, batch_size=16)
```

`rerank` yields one float per input document in input order. The project adapter converts these to
explicit `{candidate_index, score}` records, then validates count, index range, uniqueness,
completeness, and finite values before ranking. Model initialization is lazy; importing
`app.reranking` neither initializes nor downloads a model.

FastEmbed metadata reports an approximately 1.11 GB ONNX model with 1K/sliding-window context and
license `CC-BY-NC-4.0`. The official Hugging Face model API independently returned the same license
for revision `9cfeff2df7d40d1b78e75e5e9cebec92a99813c9`. This model is acceptable for the current
non-commercial benchmark/demo, but it is not approved here for commercial deployment.

## 2. Candidate text and pools

Candidate text format ID is `heading_content_v1`:

```text
Heading > Subheading

raw chunk text
```

With no heading, only raw chunk text is sent. Stored text and payloads are unchanged.

The strategies are:

| Strategy | Candidate construction | Previous rank used for ties |
|---|---|---|
| `sparse` | v2 sparse top 20 | `sparse_rank` |
| `hybrid` | v2 dense top 20 + sparse top 20 → RRF k=60 → top 20 | `rrf_rank` |
| `union` | v1 dense top 20 ∪ v2 sparse top 20, stable-ID dedup, no truncation | deterministic union ordinal |

The final sort is reranker score descending, previous rank ascending, then chunk ID ascending.
Reranker scores are ranking signals, not probabilities. The pipeline returns the full reranked pool;
`--limit` only controls CLI display.

## 3. Commands

Start Qdrant and run a search using the existing images:

```powershell
docker compose up -d qdrant
docker compose --profile tools run --rm -v "${PWD}:/app" ingestion `
  python -m scripts.search_reranked `
  "What sensor attributes are used by the algorithm?" `
  --strategy union --document-id manual-77d5dae4c2c5 --limit 5
```

Run the complete benchmark. The first runtime use downloads the model into the named
`fastembed_cache` volume; Dockerfile does not initialize the model and this command does not rebuild
the image:

```powershell
docker compose --profile tools run --rm -v "${PWD}:/app" ingestion `
  python -m scripts.evaluate_reranking --strategy all
```

Rebuild only the comparison JSON from existing measurements, with no model load:

```powershell
python -m scripts.evaluate_reranking --comparison-only
```

Generated runtime artifacts, all ignored by Git, are:

- `artifacts/metrics/phase-5-candidate-audit.json` — reference to the Phase 4.1 audit with source
  SHA-256 `1962af4f37a056c0fcf25aa5716c3786583bcb0b26c34f02139597b0fcd4bfc1`;
- `artifacts/metrics/rerank-sparse.json`;
- `artifacts/metrics/rerank-hybrid.json`;
- `artifacts/metrics/rerank-union.json`;
- `artifacts/metrics/phase-5-comparison.json` — source of truth for gates and selection.

## 4. Measured results

The real 30-query benchmark completed in 681.5 seconds. Warm latency excludes model download,
model initialization, and one warmup query; it includes retrieval, fusion/union where applicable,
cross-encoder inference, and the total pipeline. Percentiles use nearest rank.

| Strategy | Hit@1 | Hit@3 | Hit@5 | Hit@20 | MRR@5 | MRR@20 | Recall | Total p50 | Total p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sparse rerank | 0.367 | 0.700 | 0.733 | 0.867 | 0.529 | 0.544 | 0.867 | 5,843.27 ms | 9,879.69 ms |
| Hybrid rerank | 0.367 | 0.733 | 0.767 | 0.867 | 0.546 | 0.556 | 0.867 | 6,966.92 ms | 8,465.75 ms |
| Union rerank | 0.367 | 0.733 | 0.767 | 0.933 | 0.546 | 0.560 | 0.933 | 8,490.41 ms | 11,889.45 ms |

Per retrieval scenario:

| Strategy | VI Hit@5 | VI MRR@5 | VI recall | EN→VI Hit@5 | EN→VI MRR@5 | EN→VI recall |
|---|---:|---:|---:|---:|---:|---:|
| Sparse rerank | 0.800 | 0.536 | 0.933 | 0.667 | 0.522 | 0.800 |
| Hybrid rerank | 0.800 | 0.536 | 0.933 | 0.733 | 0.556 | 0.800 |
| Union rerank | 0.800 | 0.536 | 0.933 | 0.733 | 0.556 | 0.933 |

Critical query ranks are shown as candidate rank → final rerank rank:

| Query | Sparse | Hybrid | Union |
|---|---:|---:|---:|
| `dense_001` | 4 → 2 | 4 → 2 | 6 → 2 |
| `dense_002` | 5 → 1 | 1 → 1 | 4 → 1 |
| `dense_003` | 11 → 2 | 8 → 2 | 17 → 2 |
| `dense_004` | 8 → 1 | 9 → 1 | 11 → 1 |
| `dense_005` | 1 → 1 | 8 → 1 | 2 → 1 |
| `dense_006` | 8 → 1 | 16 → 1 | 16 → 1 |

All three bilingual intent pairs therefore pass top 5 for every strategy.

Failure diagnosis:

- Sparse candidate misses: `008`, `014`, `017`, `020`; top-5 reranker misses: `007`, `018`,
  `021`, `022`.
- Hybrid candidate misses: `008`, `014`, `017`, `018`; top-5 reranker misses: `007`, `021`,
  `022`.
- Union candidate misses: `014`, `017`; top-5 reranker misses: `007`, `008`, `018`, `021`,
  `022`. Union retains `008` and `018`, but reranks each to 18.

The first VI smoke included runtime model download/model initialization and took about 57.5 seconds.
After cache population, the benchmark recorded first initialization/inference separately at
11,473.85 ms. The cached ONNX blob measured 1,114,040,223 bytes; it resides in the shared runtime
volume, not an image layer.

## 5. Gate result and rollback

| Gate | Target | Result |
|---|---|---|
| Bilingual critical intents top 5 | 3/3 | PASS for all strategies |
| Hit@5 | ≥ 0.633 | PASS for all strategies |
| MRR@5 | ≥ 0.485 | PASS for all strategies |
| Union candidate recall | ≥ 0.933 | PASS: 0.933 |
| Warm total CPU p95 | < 1.5 s | FAIL: 8.466–11.889 s |

Union is the best observed research strategy under the locked selection order, but the comparison
artifact intentionally stores `recommended_default_strategy: null` and `quality_gate: PARTIAL`.
Rollback is immediate because the API has no query/rerank endpoint and settings do not select a
default: continue using the Phase 4 sparse CLI/baseline. Errors also never trigger an implicit
fallback, which keeps failures observable.

Before Phase 3B/Phase 6, choose one of these explicit follow-ups: optimize/quantize/profile the
reranker while keeping the frozen benchmark, choose a commercially compatible model and rerun all
gates, or accept sparse retrieval without a reranker. Do not report this development set as a
held-out final result.

## 6. Validation record and Docker deviation

Commands actually run:

```text
python -m ruff check .                                      PASS
python -m pytest -q (Python 3.11.15 one-shot container)    PASS — 99 tests
docker compose config --quiet                              PASS
real VI and EN→VI search smoke                             PASS
six critical queries / all three strategies               PASS
30 queries × sparse/hybrid/union reranking                 PASS
v1/v2 frozen identity checks                               PASS — 99/99 points
```

One default-suite warning remains: FastAPI imports Starlette's deprecated TestClient/httpx
compatibility path. It is third-party behavior and is not globally suppressed or mass-upgraded.
The local-Qdrant payload-index warning is captured only in the exact test that triggers it.
FastEmbed also emits a mean-pooling compatibility warning during real model runtime; it does not
appear in default offline tests and is documented rather than globally hidden.

No ingestion image was rebuilt in Phase 5. The current source was bind-mounted into the existing
Python 3.11 image as planned. Baked-image validation remains deferred until the user manually runs:

```powershell
docker compose --progress plain --profile tools build ingestion
```

This deferred build is not required to reproduce the bind-mounted Phase 5 benchmark. Do not run
Docker prune commands and do not delete the persistent Qdrant or FastEmbed cache volumes.

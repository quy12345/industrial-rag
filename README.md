# Industrial Technical Manual RAG

## Status

**Phase 5 — Multilingual Cross-Encoder Reranking**

Phase 5 implementation and real-model benchmarking are complete. The project can rerank sparse,
hybrid-RRF, or dense/sparse-union candidate pools with a multilingual cross-encoder while retaining
all dense, sparse, fusion, metadata, and direct-evidence diagnostics. Ranking quality passed, but
the CPU latency gate failed, so Phase 5 is **PARTIAL** and no reranking strategy is the runtime
default.
The FastAPI service currently exposes only `GET /api/v1/health`.

Out of scope for this phase: LangChain, OpenAI, answer generation, citations, abstention, and a
query endpoint.

The canonical runtime is Python 3.11 with `qdrant-client >=1.19.0,<1.20.0`, FastEmbed `0.8.0`,
and Qdrant server `v1.18.3`. The historic dense artifact that records client `1.18.0` is retained
unchanged; it is a historic measurement, not the dependency declaration used after closure.

For a module-by-module architecture explanation, Qdrant mental model, code-redundancy audit, query
debugging guide, and interview preparation, see `docs/project-deep-dive.md`.

## Requirements and installation

The supported development/runtime version is **Python 3.11**. Use one interpreter consistently:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
```

Extras are separated by responsibility:

- `.[retrieval]`: Qdrant client, FastEmbed, dense retrieval, sparse BM25, and RRF.
- `.[ingestion]`: Docling document parsing.
- `.[dev]`: test/lint tooling plus retrieval and ingestion dependencies, so the complete unit suite runs.

`EMBEDDING_CACHE_DIR` is optional locally. In Docker it is set to `/models/fastembed` and backed by
a shared named volume; model weights are never baked into either image.

## Ingestion and dense indexing

Place the document under `data/raw/`. The checked retrieval-development baseline is frozen against
`artifacts/manual-batched.jsonl`: the 21-page manual, batch size 4, document ID
`manual-77d5dae4c2c5`, and 99 chunks.

```powershell
python scripts/index_document.py data/raw/manual.pdf `
  --page-start 1 `
  --page-end 21 `
  --page-batch-size 4
```

The index manifest records collection/model compatibility. Indexing parses and embeds before it
updates Qdrant, upserts new points, then removes stale points for only the indexed document.
Chunk and point IDs are deterministic. Re-indexing the same frozen input must leave 99 points.

Search a specific indexed document:

```powershell
python scripts/search_dense.py `
  "What sensor attributes are used by the algorithm?" `
  --document-id manual-77d5dae4c2c5 `
  --limit 5
```

Dense cosine scores are ranking signals, not probabilities.

## Retrieval development evaluation

`data/eval/dense_smoke.jsonl` is a **30-query retrieval development set**, with 15 Vietnamese and
15 English factual queries. It is not the held-out final evaluation set planned for Phase 6.

Every item has stable `relevant_chunk_ids`. Direct retrieval hits, Hit@k, and MRR use only those
IDs. Expected phrases are validated against the relevant frozen chunks; pages are diagnostics only,
so a same-page unrelated chunk never counts as a hit.

The evaluator supports the same qrels and chunk set for all strategies. English queries are reported
as **cross-lingual** (`en` query → `vi` evidence); Vietnamese queries are monolingual.

Run the immutable dense metric closure after Qdrant contains exactly the frozen chunk IDs:

```powershell
python scripts/evaluate.py --strategy dense `
  --chunks artifacts/manual-batched.jsonl `
  --limit 20 `
  --output artifacts/metrics/dense-baseline-closure.json
```

The JSON artifact is the source of truth. It records the chunk-set hash, dataset hash, model/index
contract, direct-evidence rank per query, Hit@1/3/5/20, Candidate Recall@20, MRR@5, MRR@20,
per-language/scenario/category metrics,
critical-query diagnostics, failure cases, and warm-query average/p50/p95 latency. Primary latency
includes query embedding and Qdrant round trip but excludes model initialization, model download,
and one warmup query.

Old page-or-phrase smoke metrics are not directly comparable with this direct-evidence baseline.

Verified 2026-08-05 baseline on the frozen 99 chunks: Hit@1 `0.167`, Hit@3 `0.367`, Hit@5 `0.400`,
MRR@5 `0.269`, MRR@20 `0.298`, p50 latency `15.14 ms`, p95 latency `37.77 ms`, and 18 failure
cases. The immutable `dense-baseline.json` is not overwritten; this additive closure recorded
Hit@20 `0.767`, p50 `25.69 ms`, and p95 `35.40 ms` on 2026-08-06.

## Hybrid BM25 + RRF retrieval

Dense collection v1, `industrial_manual_chunks`, remains intact for the dense baseline. Hybrid data
uses the independent `industrial_manual_chunks_v2` collection: named `dense` (384-d cosine) and
named `sparse` (Qdrant IDF modifier). Each point keeps the same deterministic UUIDv5 and
citation-ready payload as v1. `document_id` has a keyword payload index.

BM25 uses FastEmbed 0.8.0 `Qdrant/bm25` with `disable_stemmer=True`, `k=1.2`, `b=0.75`, and the
exact FastEmbed BM25 preprocessing/tokenizer result computed from the frozen 99 chunks:
`avg_len=72.838384`. It does not use English stopword removal/stemming for the Vietnamese corpus.
The independent manifest is `artifacts/metrics/hybrid-index-manifest.json`; hybrid search refuses
to run if its schema, models, sparse IDF configuration, chunk hash, or RRF configuration mismatch.

Index v2 only after ingestion reproduces the frozen set:

```powershell
python -m scripts.index_hybrid data/raw/manual.pdf --page-batch-size 4
python -m scripts.index_hybrid data/raw/manual.pdf --page-batch-size 4
```

Search with 20 dense candidates, 20 sparse candidates, RRF `k=60`, and bounded output. Ranks are
one-based. RRF is `sum(1 / (k + rank))`; raw cosine and BM25 scores are never added together and
all scores are ranking signals, not probabilities.

```powershell
python -m scripts.search_hybrid "Thuật toán ODA-MD có mục tiêu gì?" `
  --document-id manual-77d5dae4c2c5 --limit 5
```

```powershell
python -m scripts.evaluate --strategy sparse --limit 20
python -m scripts.evaluate --strategy hybrid --limit 20
```

The 2026-08-06 Python 3.11 container run measured the following development-set results:

| Metric | Dense | Sparse | Hybrid | Hybrid − Dense |
|---|---:|---:|---:|---:|
| Hit@1 | 0.167 | 0.333 | 0.267 | +0.100 |
| Hit@3 | 0.367 | 0.500 | 0.400 | +0.033 |
| Hit@5 | 0.400 | 0.633 | 0.533 | +0.133 |
| Hit@20 | 0.767 | 0.867 | 0.867 | +0.100 |
| MRR@5 | 0.269 | 0.441 | 0.365 | +0.096 |
| MRR@20 | 0.298 | 0.469 | 0.398 | +0.100 |
| p50 latency | 25.69 ms | 2.16 ms | 19.14 ms | -6.55 ms |
| p95 latency | 35.40 ms | 2.78 ms | 26.16 ms | -9.24 ms |

Hybrid clears the Hit@5, MRR@20, Hit@20, and p95 targets, but only one of the three bilingual
critical intents has direct evidence in top 5. See `docs/walkthrough-phase-4.md` for per-language,
scenario, critical, and failure diagnostics. This remains a retrieval-development set, not Phase 6
final evaluation.

## Phase 5 candidate-pool handoff

Candidate recall answers a different question from Hit@k: whether a reranker could see direct
evidence at all. It is measured before any reranking over the same 30 qrels and frozen 99 chunks.

```powershell
python -m scripts.audit_candidate_pools --limit 20
python -m scripts.generate_phase5_readiness
```

The generated JSON artifacts are ignored runtime evidence: `artifacts/metrics/candidate-pool-audit.json`
and `artifacts/metrics/phase-5-readiness.json`. The 2026-08-06 audit found query-level candidate
coverage of `0.767` (dense top 20), `0.867` (sparse top 20), `0.867` (hybrid RRF top 20), and
`0.933` (unbounded dense@20 ∪ sparse@20, 22–34 unique candidates). The union misses only
`dense_014` and `dense_017`; a reranker cannot recover those from this candidate pool.

Phase 5 must benchmark all three candidate strategies, not assume hybrid is the default:
`sparse_top20`, `hybrid_top20`, and `dense20_union_sparse20`. Sparse currently has stronger top-rank
development metrics than RRF hybrid. Four cases (`dense_005`, `dense_019`, `dense_021`, and
`dense_029`) have sparse top-5 direct evidence that RRF demotes outside top 5; dense adds evidence
absent from sparse for `dense_008` and `dense_020`. See
`docs/walkthrough-phase-4-closure.md` for the full diagnosis and reproducible checks.

## Multilingual cross-encoder reranking

Phase 5 uses FastEmbed 0.8.0 `TextCrossEncoder` with
`jinaai/jina-reranker-v2-base-multilingual`. Candidate text is the heading breadcrumb, two newlines,
then the unchanged raw chunk text. Scores are ranking signals, not probabilities. The adapter is
lazy: importing the module does not initialize or download the model, and errors never silently
fall back to a retrieval ranking.

The model license was verified from the official model metadata as **CC-BY-NC-4.0**. It is suitable
for this non-commercial benchmark/demo, but must not be assumed suitable for commercial deployment.
Use another licensed model or obtain appropriate rights before a commercial release.

Run an interactive reranked search with an explicit strategy:

```powershell
python -m scripts.search_reranked `
  "What sensor attributes are used by the algorithm?" `
  --strategy union --document-id manual-77d5dae4c2c5 --limit 5
```

Run all three real-model evaluations in the existing Python 3.11 ingestion image, without rebuilding
it. The source tree is bind-mounted and the Compose `fastembed_cache` volume persists the downloaded
model at runtime:

```powershell
docker compose up -d qdrant
docker compose --profile tools run --rm -v "${PWD}:/app" ingestion `
  python -m scripts.evaluate_reranking --strategy all
```

To regenerate only the comparison from existing strategy artifacts, with no model initialization:

```powershell
python -m scripts.evaluate_reranking --comparison-only
```

Measured on 2026-08-06, Python 3.11.15 CPU runtime, frozen 30 qrels and 99 chunks:

| Candidate pool | Hit@5 | MRR@5 | Hit@20 | MRR@20 | Candidate recall | Warm total p95 |
|---|---:|---:|---:|---:|---:|---:|
| Sparse rerank | 0.733 | 0.529 | 0.867 | 0.544 | 0.867 | 9,879.69 ms |
| Hybrid rerank | **0.767** | **0.546** | 0.867 | 0.556 | 0.867 | 8,465.75 ms |
| Union rerank | **0.767** | **0.546** | **0.933** | **0.560** | **0.933** | 11,889.45 ms |

All strategies passed the 3/3 bilingual critical-intent, Hit@5, and MRR@5 gates. None passed the
warm total p95 target of 1.5 seconds. Union is the best observed research strategy, but
`recommended_default_strategy` remains `null`; keep sparse retrieval as the low-latency rollback.
See `docs/walkthrough-phase-5.md` for scenario metrics, critical ranks, failure classes, commands,
and the full validation record.

## Docker

The default Compose file is production-like: source is not bind-mounted and API reload is off.
It contains Qdrant, the API runtime, and an on-demand `ingestion` service in the `tools` profile.
Qdrant is pinned to `qdrant/qdrant:v1.18.3` and its named volume is persistent.

```powershell
docker compose --progress plain build api
docker compose up -d qdrant api
```

The API image installs `.[retrieval]` but not Docling. Build and run ingestion only when needed:

```powershell
docker compose --progress plain --profile tools build ingestion
docker compose --profile tools run --rm ingestion `
  python scripts/index_document.py /data/raw/manual.pdf `
  --page-start 1 --page-end 21 --page-batch-size 4
```

Phase 4.1 baked-image validation passed for the API target: Python 3.11.15, qdrant-client 1.19.0,
FastEmbed 0.8.0, no importable Docling, an empty pre-runtime model-cache directory, `/health`, and
API-to-Qdrant checks for both 99-point collections. `docker image ls` measured the API at 544 MB and
the prior ingestion image at 9.57 GB. The fresh ingestion rebuild was deliberately cancelled while
downloading the 526.6 MB `torch` wheel after a prolonged external package-registry transfer. Do not
claim its new baked-source validation passes until a future build completes. The existing ingestion
image was used only with an explicit source mount for real integration evaluation, which is not
baked-image evidence.

For live reload during development, add the override file:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up api
```

Normal `docker compose up` reuses the fixed `industrial-rag-api:local` image; it does not create a
new image unless one is missing or `build`/`--build` is requested. Do not run `docker system prune`
or `docker volume prune` for this project: the Qdrant and model-cache volumes are intentionally
preserved.

## Known limitations

- OCR for scanned PDFs is not enabled.
- Page-range batch boundaries can change structure-aware chunks and heading context; do not compare
  metrics across different chunk sets.
- Multi-page tables can be split at batch boundaries.
- Docling remains a heavy, on-demand ingestion dependency.
- Dense retrieval remains frozen as the Phase 3A.2 baseline; Phase 4 improves candidates through
  a separate sparse/RRF collection rather than hidden changes to this development set.
- Hybrid retrieval improves the development-set aggregate, but two critical bilingual intents still
  miss top 5 before reranking.
- Phase 5 reranking meets the measured ranking gates but takes 8.47–11.89 seconds at warm p95 on
  the tested CPU, so it is not enabled as a runtime default.
- The selected reranker is licensed CC-BY-NC-4.0 and is not approved here for commercial use.

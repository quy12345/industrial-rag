# Industrial Technical Manual RAG

## Status

**Phase 3A.2 — Dense Baseline Closure and Docker Stabilization**

The repository ingests technical PDF/DOCX documents with Docling, indexes multilingual dense
embeddings in Qdrant, and evaluates retrieval against manually verified direct-evidence qrels.
The FastAPI service currently exposes only `GET /api/v1/health`.

Out of scope for the current phase: sparse/hybrid retrieval, RRF, reranking, LangChain, OpenAI,
answer generation, citations, abstention, and a query endpoint.

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

- `.[retrieval]`: Qdrant client, FastEmbed and dense search.
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

## Dense retrieval development evaluation

`data/eval/dense_smoke.jsonl` is a **30-query retrieval development set**, with 15 Vietnamese and
15 English factual queries. It is not the held-out final evaluation set planned for Phase 6.

Every item has stable `relevant_chunk_ids`. Direct retrieval hits, Hit@k, and MRR use only those
IDs. Expected phrases are validated against the relevant frozen chunks; pages are diagnostics only,
so a same-page unrelated chunk never counts as a hit.

Run the baseline after Qdrant contains exactly the frozen chunk IDs:

```powershell
python scripts/evaluate.py `
  --chunks artifacts/manual-batched.jsonl `
  --limit 20 `
  --output artifacts/metrics/dense-baseline.json
```

The JSON artifact is the source of truth. It records the chunk-set hash, dataset hash, model/index
contract, direct-evidence rank per query, Hit@1/3/5, MRR@5, MRR@20, per-language/category metrics,
critical-query diagnostics, failure cases, and warm-query average/p50/p95 latency. Primary latency
includes query embedding and Qdrant round trip but excludes model initialization, model download,
and one warmup query.

Old page-or-phrase smoke metrics are not directly comparable with this direct-evidence baseline.

Verified 2026-08-05 baseline on the frozen 99 chunks: Hit@1 `0.167`, Hit@3 `0.367`, Hit@5 `0.400`,
MRR@5 `0.269`, MRR@20 `0.298`, p50 latency `15.14 ms`, p95 latency `37.77 ms`, and 18 failure
cases. See `docs/walkthrough-phase-3a2.md` for the command sequence and interpretation.

## Docker

The default Compose file is production-like: source is not bind-mounted and API reload is off.
It contains Qdrant, the API runtime, and an on-demand `ingestion` service in the `tools` profile.
Qdrant is pinned to `qdrant/qdrant:v1.18.3` and its named volume is persistent.

```powershell
docker compose build api
docker compose up -d qdrant api
```

The API image installs `.[retrieval]` but not Docling. Build and run ingestion only when needed:

```powershell
docker compose --profile tools build ingestion
docker compose --profile tools run --rm ingestion `
  python scripts/index_document.py /data/raw/manual.pdf `
  --page-start 1 --page-end 21 --page-batch-size 4
```

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
- Dense retrieval quality is frozen as a baseline; Phase 4 will address candidate quality with
  hybrid retrieval, not with hidden changes to this development set.

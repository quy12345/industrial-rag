# Codebase guide

## Current architecture

```text
PDF/DOCX -> Docling -> structure-aware DocumentChunk -> FastEmbed dense vector
        -> Qdrant named cosine vector `dense` -> ranked RetrievedChunk
```

The current API is deliberately small: `app.main` exposes only `/api/v1/health`. Retrieval remains
explicit Python code so it can be inspected and benchmarked before later orchestration/generation
phases.

## Main modules

- `app/config.py`: Pydantic settings. `embedding_cache_dir` makes FastEmbed cache location explicit.
- `app/ingestion.py`: input validation, Docling conversion, batched PDF processing, stable
  content-based chunk IDs, and atomic JSONL output.
- `app/models.py`: `DocumentChunk`, `RetrievedChunk`, and health models.
- `app/retrieval.py`: embedding input, FastEmbed initialization, Qdrant collection/index/search,
  stable UUIDv5 point IDs, safe re-indexing, and index-manifest validation.
- `app/evaluation.py`: dependency-free typed qrels, frozen-chunk validation, direct-evidence ranks,
  retrieval metrics, group metrics, and latency percentiles.
- `scripts/index_document.py`, `scripts/search_dense.py`, `scripts/evaluate.py`: integration CLIs.

## Dense-index contract

- Collection: `industrial_manual_chunks`
- Vector: named `dense`, cosine distance
- Default model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Dimension: obtained from the model at runtime, not hard-coded
- Index manifest: `artifacts/metrics/dense-index-manifest.json`
- Point ID: UUIDv5 derived from the stable chunk ID

Before search/evaluation, the manifest must agree with collection, vector, model, dimension, and
distance. Evaluation additionally requires Qdrant's indexed chunk IDs to exactly equal the frozen
JSONL chunk set.

## Evaluation boundary

`data/eval/dense_smoke.jsonl` is a retrieval-development set, not a Phase 6 held-out test set.
Ground truth is `relevant_chunk_ids`; expected phrase/page metadata validates and diagnoses qrels but
never changes Hit@k or MRR. Ranks are one-based and reciprocal rank is zero when direct evidence is
outside the candidate limit.

## Dependencies and containers

- Core package: FastAPI runtime/settings only.
- `retrieval` extra: Qdrant/FastEmbed.
- `ingestion` extra: Docling.
- `dev` extra: complete local test/lint dependencies.

`Dockerfile` supplies `api` and `ingestion` targets from a shared retrieval runtime. Compose starts
only API/Qdrant by default; ingestion is profile-gated. The shared `fastembed_cache` volume avoids
downloading the same model separately for API and ingestion.

The ingestion target installs Debian runtime packages `libxcb1`, `libgl1`, and
`libglib2.0-0t64`; Docling's PDF/image stack needs the shared libraries they provide when running
on `python:3.11-slim`.

## Testing

Default pytest uses fake embeddings and in-memory Qdrant. It must not download models, call Docker,
connect to a real Qdrant server, or require an API key. The real manual/index/evaluator flow is an
explicit integration smoke command documented in the README.

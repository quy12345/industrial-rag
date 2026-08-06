# Codebase guide

## Current architecture

```text
PDF/DOCX -> Docling -> structure-aware DocumentChunk
        -> dense FastEmbed vector + BM25 sparse vector -> Qdrant collection v2
Query -> dense top-20 + sparse top-20 -> client-side RRF -> RetrievalCandidate
```

The current API is deliberately small: `app.main` exposes only `/api/v1/health`. Retrieval remains
explicit Python code so it can be inspected and benchmarked before later orchestration/generation
phases.

## Main modules

- `app/config.py`: Pydantic settings for v1 dense and v2 hybrid contracts, BM25 and RRF limits.
- `app/ingestion.py`: input validation, Docling conversion, batched PDF processing, stable
  content-based chunk IDs, and atomic JSONL output.
- `app/models.py`: `DocumentChunk`, backward-compatible `RetrievedChunk`, one-based
  `RetrievalCandidate`, and health models.
- `app/retrieval.py`: embedding input, FastEmbed initialization, Qdrant collection/index/search,
  stable UUIDv5 point IDs, safe re-indexing, and index-manifest validation.
- `app/hybrid_retrieval.py`: FastEmbed BM25 configuration and exact avg-length calculation, v2
  schema/manifest validation, safe dual-vector indexing, sparse search, and deterministic RRF.
- `app/evaluation.py`: dependency-free typed qrels, frozen-chunk validation, direct-evidence ranks,
  retrieval metrics, group metrics, and latency percentiles.
- `scripts/index_document.py`, `scripts/search_dense.py`: v1 dense integration CLIs.
- `scripts/index_hybrid.py`, `scripts/search_hybrid.py`, `scripts/evaluate.py`: v2 indexing/search
  and shared dense/sparse/hybrid evaluation CLIs.

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

## Hybrid-index contract

- Collection v2: `industrial_manual_chunks_v2`; v1 is never recreated, migrated, or deleted.
- Dense vector: named `dense`, dimension 384, cosine.
- Sparse vector: named `sparse`, Qdrant `idf` modifier.
- Sparse model: FastEmbed `Qdrant/bm25`, `disable_stemmer=True`, `k=1.2`, `b=0.75`.
- Frozen-corpus BM25 average length: `72.838384`, persisted in
  `artifacts/metrics/hybrid-index-manifest.json`.
- RRF: one-based component ranks, `sum(1 / (60 + rank))`, then deterministic sort by RRF score,
  best component rank, and chunk ID.

Hybrid indexing first validates the frozen 99-chunk identity, generates every dense and sparse
vector, upserts new deterministic points, and only then removes stale points for that same document.
The hybrid manifest validates all index/fusion settings and the chunk hash before sparse or hybrid
search begins.

## Evaluation boundary

`data/eval/dense_smoke.jsonl` is a retrieval-development set, not a Phase 6 held-out test set.
Ground truth is `relevant_chunk_ids`; expected phrase/page metadata validates and diagnoses qrels but
never changes Hit@k or MRR. Ranks are one-based and reciprocal rank is zero when direct evidence is
outside the candidate limit. The evaluator reports Hit@1/3/5/20, Candidate Recall@20, MRR@5/20,
per-language, and per-retrieval-scenario (`vi -> vi` monolingual; `en -> vi` cross-lingual) metrics.

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
explicit integration smoke command documented in the README. Phase 4 adds offline tests for sparse
schema/IDF, safe re-indexing, document filters, metadata preservation, manifest mismatches, and RRF
duplicate/tie/empty-list behavior.

# Codebase guide

## Current architecture

```text
PDF/DOCX -> Docling -> structure-aware DocumentChunk
        -> dense FastEmbed vector + BM25 sparse vector -> Qdrant collection v2
Query -> dense top-20 + sparse top-20 -> client-side RRF -> RetrievalCandidate
      -> sparse | hybrid | dense/sparse union candidate pool
      -> lazy multilingual cross-encoder -> full reranked pool -> display cutoff
```

The current API is deliberately small: `app.main` exposes only `/api/v1/health`. Retrieval remains
explicit Python code so it can be inspected and benchmarked before later orchestration/generation
phases.

## Main modules

- `app/config.py`: Pydantic settings for v1 dense, v2 hybrid, BM25/RRF, and optional reranker
  contracts. No reranking strategy is configured by default.
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
- `app/candidate_audit.py`: dependency-free candidate-pool normalization, union, coverage, critical
  diagnostics, and RRF-demotion aggregation for the Phase 5 handoff.
- `app/reranking.py`: lazy FastEmbed cross-encoder adapter, exact candidate-text formatting,
  sparse/hybrid/union pool construction, strict output validation, deterministic reranking, stage
  latency, and direct-evidence failure classification.
- `scripts/index_document.py`, `scripts/search_dense.py`: v1 dense integration CLIs.
- `scripts/index_hybrid.py`, `scripts/search_hybrid.py`, `scripts/evaluate.py`: v2 indexing/search
  and shared dense/sparse/hybrid evaluation CLIs.
- `scripts/audit_candidate_pools.py`: explicit real-model/Qdrant audit for dense@20, sparse@20,
  hybrid@20, and dense@20 ∪ sparse@20; not part of default pytest.
- `scripts/generate_phase5_readiness.py`: validates the frozen contract and writes the Phase 5 JSON
  handoff from measured artifacts and live collection metadata.
- `scripts/rerank_runtime.py`: validates both frozen collection manifests and constructs the real
  retrieval/reranking runtime without changing either collection.
- `scripts/search_reranked.py`: explicit-strategy search CLI with component and rerank diagnostics.
- `scripts/evaluate_reranking.py`: evaluates one or all three Phase 5 pools and writes additive
  strategy/comparison JSON artifacts; `--comparison-only` never loads a model.

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

## Reranking contract

- Model: `jinaai/jina-reranker-v2-base-multilingual` through FastEmbed 0.8.0
  `fastembed.rerank.cross_encoder.TextCrossEncoder`.
- License: `CC-BY-NC-4.0`; benchmark/demo use only unless commercial rights are resolved.
- Input format ID: `heading_content_v1`, rendered as `heading > breadcrumb`, two newlines, and the
  unchanged raw chunk text; heading-less chunks use raw text only.
- Sparse pool: v2 sparse top 20. Hybrid pool: v2 dense top 20 plus sparse top 20, RRF `k=60`, then
  top 20. Union pool: v1 dense top 20 plus v2 sparse top 20 with stable-ID de-duplication and no
  pre-rerank truncation.
- The cross-encoder output must contain one indexed, finite score for every input. Final ordering is
  score descending, previous one-based rank ascending, then chunk ID.
- `rerank_score` becomes the final ranking signal while dense, sparse, and RRF scores/ranks remain
  available. The full pool is returned; CLI `--limit` affects display only. No error fallback exists.

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

The canonical post-closure retrieval dependency is `qdrant-client >=1.19.0,<1.20.0`. It was the
client used by the successful Phase 4 Python 3.11 integration; Qdrant server remains pinned to
`v1.18.3`. Historic metrics artifacts are immutable even where older runtime metadata says `1.18.0`.

`Dockerfile` supplies `api` and `ingestion` targets from a shared retrieval runtime. Compose starts
only API/Qdrant by default; ingestion is profile-gated. The shared `fastembed_cache` volume avoids
downloading the same model separately for API and ingestion. Phase 5 downloaded the reranker at
runtime into that volume; the model is not baked into an image.

The ingestion target installs Debian runtime packages `libxcb1`, `libgl1`, and
`libglib2.0-0t64`; Docling's PDF/image stack needs the shared libraries they provide when running
on `python:3.11-slim`.

## Testing

Default pytest uses fake embeddings and in-memory Qdrant. It must not download models, call Docker,
connect to a real Qdrant server, or require an API key. The real manual/index/evaluator flow is an
explicit integration smoke command documented in the README. Phase 4 adds offline tests for sparse
schema/IDF, safe re-indexing, document filters, metadata preservation, manifest mismatches, and RRF
duplicate/tie/empty-list behavior. Phase 4.1 adds offline candidate-pool tests for deterministic
rank/score preservation, union de-duplication, qrel-only candidate recall, scenario aggregation,
critical rows, and RRF-demotion diagnostics.
Phase 5 adds fake indexed cross-encoder tests for malformed outputs, finite scores, ordering and
ties, metadata preservation, no fallback, all candidate pools, document filtering, failure classes,
latency aggregation, CLI contracts, and no eager model initialization. Real model/Qdrant evaluation
remains separate from default pytest.

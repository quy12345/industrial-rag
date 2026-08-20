# Phase 4 walkthrough — Dense + BM25 sparse + client-side RRF

Phase 4 adds a second, independent retrieval collection. It never migrates, recreates, or deletes
the Phase 3A.2 dense collection `industrial_manual_chunks`.

## Architecture and frozen contract

```text
frozen 99 DocumentChunk records
  -> dense MiniLM passages + BM25 passages
  -> industrial_manual_chunks_v2 (named dense + named sparse vectors)

query -> dense top 20
      -> sparse BM25 top 20
      -> client-side RRF (k=60) -> ranked hybrid candidates
```

`dense` is 384-dimensional cosine similarity using
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. `sparse` is FastEmbed 0.8.0
`Qdrant/bm25` and uses Qdrant's `idf` modifier. Both vectors live on the same deterministic UUIDv5
point and retain the same citation-ready payload.

The frozen contract is unchanged:

- document ID: `manual-77d5dae4c2c5`
- chunk count: 99
- chunk ID hash: `bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be`
- development set: 30 factual queries (15 Vietnamese, 15 English-to-Vietnamese cross-lingual)

## BM25 configuration

The manual is Vietnamese, so BM25 deliberately uses `disable_stemmer=True`; that also disables the
English stopword/stemming behavior. The query and passage paths use FastEmbed's own BM25
normalization: `remove_non_alphanumeric`, `SimpleTokenizer`, then the active stem/filter profile.
Heading breadcrumbs plus raw chunk content are embedded; raw `chunk.text` and Qdrant payload are not
rewritten.

The integration index computed `avg_len=72.838384` from those exact 99 passage inputs, with:

```text
model: Qdrant/bm25
k: 1.2
b: 0.75
disable_stemmer: true
```

The real BM25 diagnostic produced non-empty sparse token sets for the technical identifiers
`24 VDC` (2), `IP65` (1), `PLC` (1), `S7-1200` (2), `E-Stop` (2), and `3.5 bar` (3). No custom
identifier normalization was added.

This value and every schema/fusion input are stored in
`artifacts/metrics/hybrid-index-manifest.json`. Sparse/hybrid search rejects a missing or mismatched
manifest rather than silently querying a different index contract.

## Safe indexing and search

From a Python 3.11 environment:

```powershell
python -m scripts.index_hybrid data/raw/manual.pdf --page-batch-size 4
python -m scripts.index_hybrid data/raw/manual.pdf --page-batch-size 4

python -m scripts.search_hybrid "Thuật toán ODA-MD có mục tiêu gì?" `
  --document-id manual-77d5dae4c2c5 --limit 5
```

Indexing validates that the new ingestion output has the frozen chunk metadata before it touches v2.
It creates all dense and sparse vectors first, validates or creates v2, upserts all deterministic
points with `wait=True`, and only then deletes stale IDs of that document. A sparse/dense embedding
or upsert failure therefore cannot trigger stale-point deletion.

The search CLI prints one-based final RRF rank, pages/headings, dense rank/score, sparse rank/score,
RRF score, and a bounded excerpt. These scores are ranking signals, not probabilities.

RRF uses no raw-score addition:

```text
RRF(d) = sum(1 / (60 + rank_component(d)))
```

Duplicate chunk IDs are collapsed. Ties are ordered by decreasing RRF score, increasing best
component rank, then increasing `chunk_id`.

## Evaluate comparable strategies

Each strategy uses the same frozen chunks, qrels, candidate limit and warm-query latency method.
One warmup is excluded; reported latency includes query embedding, the Qdrant round trip, result
mapping, and client RRF for hybrid.

```powershell
python -m scripts.evaluate --strategy dense --limit 20
python -m scripts.evaluate --strategy sparse --limit 20
python -m scripts.evaluate --strategy hybrid --limit 20
```

Artifacts are intentionally separate:

- `dense-baseline.json`: immutable Phase 3A.2 record; never overwritten.
- `dense-baseline-closure.json`: additive Hit@20/scenario metric closure.
- `sparse-baseline.json`: BM25 only.
- `hybrid-baseline.json`: RRF results and machine-readable comparison.

Direct hit means a returned stable `chunk_id` is in `relevant_chunk_ids`. Expected phrase/page data
is diagnostic only. English rows are explicitly **cross-lingual query → Vietnamese evidence**, not
English-document evaluation.

## Actual integration result — 2026-08-06

Python `3.11.15` in the ingestion container, FastEmbed `0.8.0`, Qdrant server `v1.18.3`, and
Qdrant client `1.19.0` were used. The first and second v2 indexes each finished with 99 points.
Collection v1 remained dense-only with 99 points; v2 had 99 points, `dense` 384/cosine, `sparse`
IDF, and a `document_id` keyword payload index.

| Metric | Dense | Sparse | Hybrid | Hybrid − Dense |
|---|---:|---:|---:|---:|
| Hit@1 | 0.167 | 0.333 | 0.267 | +0.100 |
| Hit@3 | 0.367 | 0.500 | 0.400 | +0.033 |
| Hit@5 | 0.400 | 0.633 | 0.533 | +0.133 |
| Hit@20 / Candidate Recall@20 | 0.767 | 0.867 | 0.867 | +0.100 |
| MRR@5 | 0.269 | 0.441 | 0.365 | +0.096 |
| MRR@20 | 0.298 | 0.469 | 0.398 | +0.100 |
| Average latency | 26.25 ms | 2.21 ms | 19.26 ms | -6.99 ms |
| p50 latency | 25.69 ms | 2.16 ms | 19.14 ms | -6.55 ms |
| p95 latency | 35.40 ms | 2.78 ms | 26.16 ms | -9.24 ms |

Hybrid language/scenario split:

| Scenario | Queries | Hit@5 | Hit@20 | MRR@20 | p95 |
|---|---:|---:|---:|---:|---:|
| Vietnamese → Vietnamese (monolingual) | 15 | 0.600 | 0.933 | 0.416 | 26.19 ms |
| English → Vietnamese (cross-lingual) | 15 | 0.467 | 0.800 | 0.380 | 26.16 ms |

## Quality gates and remaining limitation

The aggregate gates pass: Hybrid Hit@5 `0.533 >= 0.50`, Hybrid MRR@20 `0.398 >= 0.298`, Hybrid
Hit@20 `0.867 >=` dense `0.767`, and p95 `26.16 ms < 300 ms`.

The critical top-5 gate is partial. The six labels represent three bilingual critical intents:

| Intent pair | VI direct rank | EN direct rank | Top-5 status |
|---|---:|---:|---|
| `dense_001` / `dense_002` | 4 | 1 | Pass |
| `dense_003` / `dense_004` | 8 | 9 | Fail |
| `dense_005` / `dense_006` | 8 | 16 | Fail |

Hybrid has 14 rows with direct evidence outside top 5. The remaining misses include both
monolingual and cross-lingual questions; BM25 alone performs well on exact Vietnamese technical
terms, while cross-lingual candidate recall remains lower (0.800 vs 0.933). Do not tune qrels,
chunks, or the dense model to change this report.

Phase 5 may add multilingual cross-encoder reranking only over frozen, audited candidate pools. It
must keep this same dataset, collection v2 contract, direct-evidence evaluation, and failure rows.
Phase 4.1 measured candidate recall of 0.867 for both sparse and hybrid top 20, and 0.933 for the
unbounded dense@20 ∪ sparse@20 pool. Sparse is not replaced as the candidate baseline: Phase 5 must
benchmark sparse top 20, hybrid top 20, and the union. See `docs/walkthrough-phase-4-closure.md`.

## Validation record

```text
python -m ruff check .              PASS
python -m pytest -q                 PASS — 70 tests
docker compose config               PASS
two v2 real indexes                 PASS — 99 then 99 points
dense/sparse/hybrid real evaluation PASS
hybrid CLI smoke                    PASS
```

The host `.venv` remains Python 3.13.5. Python 3.11 validation was performed in Docker because the
Windows `py` launcher was unavailable. Phase 4.1 closes the API build-source gap with explicit plain
progress and baked-image checks. Its ingestion rebuild was cancelled during a slow external registry
download; rerun it before claiming full ingestion baked-image closure. No Docker prune or
Qdrant-volume deletion is performed.

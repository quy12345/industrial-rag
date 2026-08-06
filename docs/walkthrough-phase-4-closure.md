# Phase 4.1 walkthrough — closure and Phase 5 readiness

Phase 4.1 freezes the Phase 4 handoff. It does not add a reranker, change qrels, re-chunk the manual,
change embedding models, create a Qdrant v3 schema, or recreate either existing collection.

## Frozen contract and canonical runtime

```text
document_id: manual-77d5dae4c2c5
chunks: 99
sorted chunk-ID SHA-256: bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be
dense model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
sparse model: Qdrant/bm25
candidate limits: dense=20, sparse=20
RRF k: 60
```

The canonical package declaration after closure is `qdrant-client >=1.19.0,<1.20.0`. Real Phase 4
integration ran client 1.19.0, FastEmbed 0.8.0, Python 3.11.15, and Qdrant server `v1.18.3`; that is
why 1.19.x was selected instead of an untested latest client. The server remains pinned at
`qdrant/qdrant:v1.18.3`.

The immutable historic dense artifact may report client 1.18.0. It is intentionally not rewritten:
artifact metadata records the runtime that created a measurement, while `pyproject.toml` records the
canonical compatibility contract for later work.

Collections remain unchanged:

| Collection | Schema | Points |
|---|---|---:|
| `industrial_manual_chunks` | named `dense`, 384 dimensions, cosine | 99 |
| `industrial_manual_chunks_v2` | named `dense` plus named `sparse` with IDF | 99 |

## Candidate-pool audit

Run this only as an explicit real-model/Qdrant integration command; unit tests use fakes.

```powershell
python -m scripts.audit_candidate_pools --limit 20
```

For every one of the 30 development queries the audit independently retrieves dense top 20, sparse
top 20, hybrid RRF top 20, and the unbounded union of dense@20 and sparse@20. The union is
de-duplicated by stable chunk ID, preserves each component's rank/score, is sorted deterministically,
and is never truncated to 20 before candidate recall is calculated. It contains 22–34 candidates.
No raw cosine/BM25 score is fused and no qrel is modified.

Candidate recall means direct evidence is present for a reranker; it is not a top-rank metric.

| Pool | Query candidate recall | Queries with no direct evidence |
|---|---:|---|
| Dense top 20 | 0.767 | `005, 006, 014, 017, 018, 021, 029` |
| Sparse top 20 | 0.867 | `008, 014, 017, 020` |
| Hybrid RRF top 20 | 0.867 | `008, 014, 017, 018` |
| Dense@20 ∪ sparse@20 | 0.933 | `014, 017` |

The union has median 28 candidates (minimum 22, maximum 34). It gives Phase 5 the highest measured
evidence availability, but it is an experiment candidate rather than an automatic production default.
The mandatory Phase 5 comparison is:

```text
sparse_top20
hybrid_top20
dense20_union_sparse20
```

Critical coverage shows why top-rank and candidate metrics must both be retained:

| Critical row | Dense rank | Sparse rank | Hybrid rank | Union contains evidence |
|---|---:|---:|---:|---|
| `dense_001` | 10 | 4 | 4 | yes |
| `dense_002` | 2 | 5 | 1 | yes |
| `dense_003` | 10 | 11 | 8 | yes |
| `dense_004` | 9 | 8 | 9 | yes |
| `dense_005` | absent | 1 | 8 | yes |
| `dense_006` | absent | 8 | 16 | yes |

## RRF diagnosis without tuning

Sparse remains stronger at the top ranks (Hit@5 0.633 versus hybrid 0.533; MRR@20 0.469 versus
0.398). The audit shows four rows where sparse has direct evidence in its top 5 but RRF places it
outside top 5: `dense_005`, `dense_019`, `dense_021`, and `dense_029`. Weak or conflicting dense
ranks can therefore dilute exact sparse evidence when rank-only RRF combines both lists.

Dense is still complementary: it supplies candidate evidence absent from sparse for `dense_008` and
`dense_020`. Candidate coverage is weaker for English-to-Vietnamese retrieval (union 0.933) than
Vietnamese-to-Vietnamese retrieval (union 1.000). These are observations, not a rationale to tune
RRF k=60 on the development set. Weighted RRF and parameter changes remain out of scope.

## Docker and reproducible checks

The API target installs base plus retrieval only. The ingestion target inherits retrieval and adds
Docling. Compose mounts raw documents and artifacts only at runtime for the profile-gated ingestion
service; the API has no source bind mount. Both services share the named `fastembed_cache` at
`/models/fastembed`, and Dockerfile never initializes an embedding model.

```powershell
docker compose config
docker compose --progress plain build api
docker compose --progress plain --profile tools build ingestion
docker compose up -d --force-recreate api
Invoke-RestMethod http://localhost:8000/api/v1/health

docker run --rm industrial-rag-api:local python -c "import app.hybrid_retrieval"
docker run --rm industrial-rag-api:local python -c "import importlib.util; assert importlib.util.find_spec('docling') is None"
docker compose --profile tools run --rm ingestion python -m scripts.search_hybrid --help
docker compose --profile tools run --rm ingestion python -c "import docling"
```

No command above uses `docker system prune`, `docker volume prune`, collection deletion, or a source
bind mount for image validation. The API image's `/models/fastembed` contains zero model files before
runtime; downloading is deferred to the shared cache volume. The API target passed these baked-image
checks with Python 3.11.15, client 1.19.0, FastEmbed 0.8.0, no Docling, health, and a real Qdrant
connection. `docker image ls` measured it at 544 MB (the prior ingestion image is 9.57 GB).

The fresh ingestion target build reached Dockerfile step 12 (`pip install ".[retrieval,ingestion]"`)
and began downloading the 526.6 MB `torch-2.13.0` wheel. It was deliberately cancelled with the
Docker CLI only because the external package-registry transfer was too slow. This is not a source/test
failure, and it did not delete images, cache, containers, volumes, or Qdrant collections. Do not count
ingestion as baked-image PASS; rerun the exact command before claiming Phase 4.1 Docker closure.

## Phase 5 handoff artifact

After a successful candidate audit, generate the complete machine-readable handoff:

```powershell
python -m scripts.generate_phase5_readiness
```

It validates the dense, sparse, hybrid, and audit artifacts against the live frozen chunk contract,
then writes `artifacts/metrics/phase-5-readiness.json`. It includes artifact versions, dataset
identity, collection schemas/counts, runtime versions, benchmark metrics, candidate diagnostics,
critical ranks, Docker validation status, working-tree metadata, recommended pools, limitations, and
blockers. The artifacts are intentionally ignored by Git because they are measured runtime evidence.

## Static validation

```text
python -m ruff check .                              PASS
python -m pytest -q --basetemp C:\tmp\industrial-rag-pytest-phase41-final
                                                     PASS — 74 tests, 2 known warnings
git diff --check                                    PASS
docker compose config                               PASS
```

The explicit `--basetemp` is a local Windows workaround: the host default temporary directory has a
pre-existing `Access denied` ACL on `pytest-of-LENOVO`. Use a writable, fresh directory if an older
basetemp itself has an inherited ACL; this does not affect test behavior or require elevation. Default
unit tests still require no network, real model, Qdrant server, or API key.

## Readiness decision

Phase 5 is `ready_with_documented_deviation`: its input contract is frozen and measurable, but the
Phase 4 bilingual critical direct-evidence top-5 gate remains **PARTIAL (1/3 intent pairs)**. A
reranker must be benchmarked against all three candidate pools on these unchanged qrels; it cannot
recover `dense_014` or `dense_017` from the measured union pool. The 30-query set remains a retrieval
development set, not the held-out Phase 6 final test set.

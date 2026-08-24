# Codebase guide

## Runtime in one view

```text
two ATV320 PDFs
  -> Docling parsing -> stable DocumentChunk records
  -> dense + BM25 sparse vectors -> frozen Qdrant collections

question
  -> dense top-60 + expanded sparse top-40
  -> frozen weighted RRF and coverage reserves -> at most 30 candidates
  -> exact-content deduplication -> lazy multilingual reranker
  -> evidence selection and gate
  -> structured generation -> source-ID validation -> trusted citations
  -> QueryResponse

browser -> Streamlit -> FastAPI -> application service
```

`app.main` exposes `/api/v1/health`, `/api/v1/ready`, and `/api/v1/query`. Streamlit is HTTP-only;
it does not import retrieval, Qdrant, model, ingestion, or provider code.

## Package map

```text
app/
  api/               FastAPI request/auth/error adapters
  application/       query, reranking, indexing, and prompt orchestration
  composition/       concrete runtime object graph
  contracts/         shared HTTP request/response schemas
  domain/            framework-free records, ports, and deterministic policies
  infrastructure/    Docling, Qdrant, FastEmbed, and provider adapters
  bootstrap.py       application composition root
  config.py          canonical environment-backed settings
  main.py            ASGI export

evaluation/          offline datasets, scoring, replay, audits, and metrics
scripts/
  operations/        supported operational command implementations
  evaluation/        supported evaluation command implementations
  archive/           unsupported historical provenance
ui/                  Streamlit HTTP adapter
tests/               deterministic offline behavior and architecture guards
```

There is no root-level `app.*` compatibility layer. Import the module that owns the behavior.

## Production owners

### Inbound and composition

- [`app/api`](../app/api) maps HTTP contracts, authentication, threadpool execution, and sanitized
  errors. It does not build models or contain retrieval logic.
- [`app/bootstrap.py`](../app/bootstrap.py) creates `QueryService` with configuration and concrete
  adapters.
- [`app/composition/retrieval.py`](../app/composition/retrieval.py) validates the frozen runtime and
  binds Qdrant/model state into dense and sparse search adapters. Models remain lazy until needed.
- [`app/config.py`](../app/config.py) is the single environment-backed settings owner. The active
  profile is `phase7`; `phase6` is rejected.

### Application

- [`app/application/query_service.py`](../app/application/query_service.py) coordinates retrieval,
  evidence selection/gating, generation, source validation, citation construction, abstention, and
  sanitized timings.
- [`app/application/reranking_service.py`](../app/application/reranking_service.py) builds candidate
  pools and reranks through `DenseSearchPort`, `SparseSearchPort`, and `CrossEncoder`.
- [`app/application/generation_prompt.py`](../app/application/generation_prompt.py) owns bounded
  evidence formatting, the system/human prompts, and correction text.
- [`app/application/indexing_service.py`](../app/application/indexing_service.py) coordinates guarded
  Phase 7 indexing through explicit adapter protocols.

### Domain

- [`app/domain/documents.py`](../app/domain/documents.py) owns document/chunk records, stable IDs,
  page batching, and input errors.
- [`app/domain/retrieval.py`](../app/domain/retrieval.py) owns retrieval records, search/retriever
  ports, dense-to-candidate mapping, and deterministic union behavior.
- [`app/domain/retrieval_contracts.py`](../app/domain/retrieval_contracts.py) owns the immutable Phase
  7 retrieval contract and collection identity.
- [`app/domain/evidence.py`](../app/domain/evidence.py) owns exact-content evidence selection and the
  evidence gate.
- [`app/domain/generation.py`](../app/domain/generation.py) owns structured generation records and
  the generator port.
- [`app/domain/citations.py`](../app/domain/citations.py) validates generated source IDs and builds
  citations only from trusted retrieval metadata.
- [`app/domain/policies`](../app/domain/policies) owns frozen query analysis, query-role inference,
  RRF/fusion, coverage, and list-completeness policies.

### Infrastructure

- [`app/infrastructure/ingestion`](../app/infrastructure/ingestion) owns Docling conversion,
  structure-aware chunk extraction, and atomic JSONL output.
- [`app/infrastructure/qdrant`](../app/infrastructure/qdrant) owns clients, dense/sparse indexing and
  search, collection validation, filters, manifests, and state-bound search adapters.
- [`app/infrastructure/models/reranker.py`](../app/infrastructure/models/reranker.py) is the lazy
  FastEmbed cross-encoder adapter.
- [`app/infrastructure/generation/langchain_structured.py`](../app/infrastructure/generation/langchain_structured.py)
  is the lazy OpenAI/Gemini structured-output adapter.
- [`app/infrastructure/corpus_artifacts.py`](../app/infrastructure/corpus_artifacts.py) owns active
  corpus paths, hashes, collection protection, and atomic artifact I/O.

## Offline evaluation owners

- [`evaluation/retrieval.py`](../evaluation/retrieval.py) loads/validates retrieval cases and computes
  direct-qrel ranks and aggregate metrics.
- [`evaluation/reranking.py`](../evaluation/reranking.py) computes candidate/reranker failure classes,
  metrics, and latency summaries from completed executions.
- [`evaluation/e2e.py`](../evaluation/e2e.py) scores completed query executions, answer facts,
  citations, abstention, contamination, latency, and quality gates without performing external I/O.
- [`evaluation/phase7_dataset.py`](../evaluation/phase7_dataset.py) owns Phase 7 dataset schemas,
  validation, hashes, and exact-content qrel closure.
- [`evaluation/candidate_audit.py`](../evaluation/candidate_audit.py),
  [`evaluation/replay.py`](../evaluation/replay.py), and
  [`evaluation/retrieval_closure.py`](../evaluation/retrieval_closure.py) own historical audit,
  sanitized replay, and closure aggregation interfaces.

Production modules cannot import `evaluation`; the AST dependency matrix enforces this direction.

## Frozen active contract

- Documents: ATV320 Installation Manual and ATV320 Programming Manual.
- Document IDs:
  - `atv320-installation-manual-en-nve41289-09-c181b4d7f11b`
  - `atv320-programming-manual-en-nve41295-06-f5e9bb48167a`
- Chunks: `2753`.
- Chunk-ID SHA-256:
  `2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d`.
- Collections:
  - `industrial_manual_phase7_dense_v1`
  - `industrial_manual_phase7_hybrid_v1`
- Runtime policy: `PHASE7_RETRIEVAL_CONTRACT` plus
  `PHASE7_CALIBRATION_FUSION_PROFILE`.

The frozen policy keeps dense@60, sparse@40, query glossary expansion, weighted RRF `k=40`, coverage
reserves, a 30-candidate maximum, exact-content deduplication, the Jina multilingual reranker, the
rank-only document-role prior, and the existing relation-list fallback. These values are baseline,
not cleanup targets.

The retired 99-chunk single-manual Phase 6 corpus, both legacy collections, and its tools remain
historical archive material. They are not active runtime/evaluation targets.

## Safety order

```text
retrieve -> rerank -> select evidence -> gate -> generate
         -> validate model source IDs -> build trusted citations
```

Generation never runs before the evidence gate. Model-provided citation metadata is never trusted.
One correction retry may repair invalid source labels; another failure produces the existing safe
abstention. There is no silent provider or retrieval fallback.

## Supported commands

All eight commands use canonical module paths:

```text
scripts.operations.audit_phase7_corpus
scripts.operations.index_phase7_corpus
scripts.operations.ingest_preview
scripts.operations.query_smoke
scripts.operations.validate_query_runtime
scripts.evaluation.validate_phase7_dataset
scripts.evaluation.evaluate_phase7_retrieval_closure
scripts.evaluation.evaluate_phase7_e2e
```

Read [`scripts/README.md`](../scripts/README.md) before any integration command. Archived scripts are
unsupported provenance: do not run them as current operations, and do not depend on their imports.

## Testing and integration

The default suite is offline. It uses fake providers/models and in-memory Qdrant; it must not access
a live provider, download a model, mutate a collection, re-index a corpus, or read raw held-out
payloads.

```powershell
python -m ruff check .
python -m pytest -q
docker compose config --quiet
```

Real Qdrant, model, or provider checks are separate opt-in integration workflows. The exposed
held-out v2 result is regression evidence only and cannot be used for tuning.

## Where to start reading

1. [`app/main.py`](../app/main.py) and [`app/api`](../app/api) for the HTTP edge.
2. [`app/bootstrap.py`](../app/bootstrap.py) for the composition root.
3. [`app/application/query_service.py`](../app/application/query_service.py) for the query use case.
4. [`app/composition/retrieval.py`](../app/composition/retrieval.py) and
   [`app/application/reranking_service.py`](../app/application/reranking_service.py) for retrieval.
5. [`app/domain`](../app/domain) for stable contracts and policies.
6. [`app/infrastructure`](../app/infrastructure) for SDK and storage adapters.
7. [`evaluation`](../evaluation) for offline measurement.

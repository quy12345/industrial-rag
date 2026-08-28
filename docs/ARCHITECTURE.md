# Architecture

This document is the canonical description of the current production structure. Historical design
steps and benchmark lessons belong in [PROJECT_JOURNEY.md](PROJECT_JOURNEY.md); commands and
troubleshooting belong in [OPERATIONS.md](OPERATIONS.md).

## System boundary

Industrial Technical Manual RAG is a modular monolith with four inbound surfaces:

```text
FastAPI -----------+
CLI operations ----+--> application services --> domain contracts and policies
CLI evaluation ----+             ^                         ^
Streamlit --HTTP-->+             |                         |
                              composition           infrastructure adapters
                                                         |
                                               Qdrant / models / providers / Docling
```

Streamlit is deliberately HTTP-only. Production code under `app/` never imports `evaluation/` or
`scripts/`. Evaluation may depend on stable public application and domain interfaces.

## Package ownership

```text
app/
|-- api/                 FastAPI routes, schemas, and HTTP error mapping
|-- application/         Query, reranking, and corpus-indexing use cases
|-- composition/         Runtime assembly of retrieval and generation dependencies
|-- contracts/           Application-facing ports for external capabilities
|-- domain/              Stable entities, frozen contracts, and deterministic policies
`-- infrastructure/      Docling, Qdrant, embeddings, rerankers, providers, and artifacts

evaluation/
|-- dataset.py           Dataset parsing and validation contracts
|-- retrieval.py         Current retrieval metrics and aggregation
`-- e2e.py               Query-record scoring and quality-gate aggregation

scripts/
|-- operations/          Supported operational adapters
|-- evaluation/          Supported evaluation adapters
`-- archive/             Unsupported historical workflows retained as provenance

ui/                      Streamlit HTTP client and presentation
```

Adapters parse input, call an application use case, serialize output, and map errors. Reusable
business behavior does not belong in a route, Streamlit page, or command module.

## Query flow

```text
POST /query
  -> QueryService validates and analyses the question
  -> dense and sparse retrievers produce ranked candidates
  -> rank-only fusion and coverage policy form the bounded candidate pool
  -> reranker scores that pool
  -> evidence policy selects bounded evidence
  -> evidence gate either permits generation or returns an abstention
  -> provider returns a structured answer with source labels
  -> source labels are validated against supplied evidence
  -> trusted citation metadata is constructed by the application
```

The provider never supplies trusted filenames, pages, or chunk metadata. One citation-correction
attempt may repair invalid source labels without rerunning retrieval; a second invalid response is a
safe abstention. Generation cannot run before the evidence gate passes.

## Ingestion and indexing flow

```text
two ATV320 PDFs
  -> Docling page-batch conversion
  -> structure-aware chunking and normalization
  -> deterministic document IDs and stable chunk IDs
  -> preview JSONL plus corpus manifest
  -> explicit guarded embedding and Qdrant indexing
```

`CorpusIndexingService` coordinates the use case. Parsing and Qdrant details remain infrastructure
concerns. Preview mode performs no Qdrant write. Indexing is explicit: the API never re-indexes or
creates a replacement collection as a fallback.

## Composition and configuration

`app/bootstrap.py` is the production composition root. It reads `Settings`, creates external
adapters, and delegates retrieval assembly to `app/composition/retrieval.py`.

Configuration has two distinct owners:

- `Settings` owns environment-specific values such as endpoints, credentials, timeouts, and cache
  directories.
- `ATV320_RETRIEVAL_CONTRACT` and `ATV320_FUSION_PROFILE` own immutable product behavior: collection
  names, vector dimensions, candidate limits, thresholds, and fusion policy.

There is one active corpus contract, identified by `atv320-2025-04-v1`; there is no runtime profile
selector. Partial overrides of the frozen contract are rejected rather than silently producing a
third pipeline.

## Frozen identities

- Documents:
  - `atv320-installation-manual-en-nve41289-09-c181b4d7f11b`
  - `atv320-programming-manual-en-nve41295-06-f5e9bb48167a`
- Chunk count: `2753`
- Ordered chunk-ID hash:
  `2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d`
- Physical Qdrant collections:
  - `industrial_manual_phase7_dense_v1`
  - `industrial_manual_phase7_hybrid_v1`

The collection labels and frozen paths under `data/eval/phase7/` and `artifacts/phase7/` are physical
or historical identities. They are intentionally not renamed by a source-code cleanup.

## Invariants

- API routes, request/response schemas, status mapping, citation validation, and abstention behavior
  are stable public contracts.
- Retrieval ordering, scores, candidate/rerank/evidence boundaries, models, dimensions, and
  thresholds change only in a separately approved algorithm round.
- Stable IDs connect ingestion, Qdrant payloads, qrels, citations, and regression artifacts.
- Production runtime must not reach evaluation or script implementations.
- Evaluation may observe public contracts but cannot influence production decisions.
- Tests remain offline and use fakes or in-memory Qdrant unless an integration command is explicitly
  approved.
- No missing dependency, collection, model, provider, or source identity triggers a silent fallback.

## Decisions

Long-lived architecture decisions are recorded in `docs/adr/`. Current naming and documentation
ownership are defined by
[ADR-006](adr/ADR-006-active-corpus-naming-and-documentation-lifecycle.md). ADR-004 and ADR-005 are
retained as historical decisions and are marked where ADR-006 supersedes their active naming.

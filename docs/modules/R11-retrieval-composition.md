# R11 — Retrieval composition and state-bound search ports

## 1. Goal and scope

R11 gives the active frozen Phase 7 retrieval object graph one canonical owner. API bootstrap and
supported retrieval commands now compose through `app.composition.retrieval`; application
reranking receives state-bound dense and sparse search ports rather than Qdrant clients, vector
names, and embedding models.

The move preserves lazy initialization, frozen runtime validation, sparse rollback, candidate
membership and ordering, reranker behavior, stage timings, metadata, failures, and CLI contracts.
It does not query Qdrant, load a model, run evaluation, or modify any collection.

## 2. Position in the system

```text
FastAPI / supported CLI
          |
          v
app.bootstrap + app.composition.retrieval
          |                  |
          v                  v
application services   infrastructure Qdrant/model adapters
          ^                  |
          +---- domain ports-+
```

`app.bootstrap` remains the backend composition root for the complete `QueryService`.
`app.composition.retrieval` is its focused child composition module for the retrieval and reranking
subgraph. Neither module is a domain or application owner.

## 3. Relevant background concepts

- A **composition root** is the boundary that selects concrete implementations and connects them.
- A **port** describes what an application service needs without naming the SDK or its state.
- A **state-bound adapter** captures infrastructure state once and exposes a small port method.
- **Lazy initialization** delays Qdrant client and model construction until the first actual query.
- A **source-identity anchor** remains byte-for-byte stable because historical artifacts record its
  exact Git blob.

## 4. Input, output, and contracts

`build_query_retriever` accepts resolved `Settings` and an explicit `FrozenRetrievalContract`. It
returns the same `QueryRetriever` variants: union plus reranking by default, or the explicit sparse
rollback when reranking is disabled.

`DenseSearchPort.search` returns ordered `RetrievedChunk` records and retains collection, limit,
document filter, and optional score threshold inputs. `SparseSearchPort.search` returns ordered
`RetrievalCandidate` records with the same collection, limit, and document filter.

The runtime still emits `QueryRetrievalResult` with final candidates, the pre-rerank candidate pool,
retrieval timing, and reranking timing. Error mapping remains fail-closed: retrieval failures become
`RetrievalUnavailableError`, while cross-encoder failures become `RerankerUnavailableError`.

## 5. Step-by-step data flow

1. Bootstrap resolves settings and the immutable Phase 7 contract.
2. `LazyQueryRetriever` stores a factory without opening Qdrant or constructing a model.
3. The first query validates the selected frozen settings combination.
4. Composition creates the Qdrant client and validates the dense and hybrid collection schemas.
5. Frozen point count and stable chunk-ID hash are checked read-only.
6. Composition creates dense, sparse, and cross-encoder model adapters.
7. `QdrantDenseSearcher` and `QdrantSparseSearcher` bind client, vector, and model state once.
8. `RerankPipeline.from_searchers` receives only the two domain ports and ranking configuration.
9. Dense and expanded sparse results follow the unchanged frozen fusion and reranking policies.
10. `UnionRerankRetriever` maps exact stage timings and candidates into `QueryRetrievalResult`.
11. The explicit sparse rollback follows the same bound sparse port without a reranker fallback.

## 6. Responsibilities of changed files

| Path | Responsibility after R11 |
| --- | --- |
| [`app/composition/retrieval.py`](../../app/composition/retrieval.py) | Canonical frozen retrieval construction, lazy adapters, metadata, and read-only collection identity validation. |
| [`app/bootstrap.py`](../../app/bootstrap.py) | Complete query-service and readiness composition using the focused retrieval owner. |
| [`app/domain/retrieval.py`](../../app/domain/retrieval.py) | Retrieval records plus state-bound dense and sparse search ports. |
| [`app/infrastructure/qdrant/search.py`](../../app/infrastructure/qdrant/search.py) | Concrete Qdrant implementations that bind client, vector, and embedding model state. |
| [`app/application/reranking_service.py`](../../app/application/reranking_service.py) | Candidate/reranking orchestration through search ports; legacy callable bridge is isolated for R12. |
| [`scripts/operations/validate_query_runtime.py`](../../scripts/operations/validate_query_runtime.py) | Read-only retrieval smoke using canonical composition. |
| [`scripts/evaluation/evaluate_phase7_retrieval_closure.py`](../../scripts/evaluation/evaluate_phase7_retrieval_closure.py) | Provider-free closure adapter using canonical composition and contracts. |
| [`scripts/operations/index_phase7_corpus.py`](../../scripts/operations/index_phase7_corpus.py) | Explicit indexing adapter importing concrete infrastructure owners directly. |

The supported query smoke command also uses the domain retrieval contract directly. The pinned E2E
entry point and archived workflows remain historical exceptions until R12.

## 7. Important symbols and why they exist

- `build_query_retriever` is the single active selection point for union and sparse rollback.
- `build_union_rerank_runtime` builds the exact frozen union object graph and runtime receipt.
- `LazyQueryRetriever` preserves import-time and application-start laziness.
- `UnionRerankRetriever` converts pipeline execution into the application retrieval contract.
- `SparseRollbackRetriever` keeps the explicit no-reranker operational path.
- `DenseSearchPort` and `SparseSearchPort` remove SDK-shaped state from canonical application wiring.
- `QdrantDenseSearcher` and `QdrantSparseSearcher` implement those ports with existing search
  functions; they do not change search behavior.
- `RerankPipeline.from_searchers` is the canonical constructor for newly composed runtime code.
- `_runtime_metadata` records the exact frozen profile without reading mutable artifacts.

## 8. Before-and-after structure

```text
Before R11
  bootstrap + supported commands -> app.retrieval_runtime
  app.retrieval_runtime           -> root retrieval/reranking facades
  RerankPipeline                  <- client + model + vector state + functions
  temporary architecture debt     = 2 retrieval-composition edges

After R11
  bootstrap + supported commands -> app.composition.retrieval
  composition                    -> canonical domain/application/infrastructure owners
  RerankPipeline.from_searchers  <- DenseSearchPort + SparseSearchPort
  Qdrant search adapters         <- bound client + vector + model state
  temporary architecture debt    = 0
  app.retrieval_runtime          = unchanged source anchor for R12 only
```

R11 temporarily duplicates the composition text instead of editing or deleting the protected
anchor. R12 owns the versioned provenance migration and hard cut; no active runtime imports the old
owner after R11.

## 9. Design decisions and trade-offs

The focused composition module is separate from bootstrap because retrieval construction has its
own collection checks, model adapters, lazy wrapper, and runtime receipt. Keeping it in bootstrap
would replace one large root module with another.

Search ports are state-bound so the application service expresses query intent rather than carrying
Qdrant and model parameters through every call. The existing low-level functions remain canonical
infrastructure operations and are delegated to unchanged.

The legacy callable constructor remains temporarily because the protected root runtime and
compatibility facade still use it. It is explicitly marked for R12; silently changing the historical
anchor would invalidate source provenance. No dependency-injection framework or extra dispatcher is
introduced.

## 10. Tests and the behavior each test protects

| Test area | Protected behavior |
| --- | --- |
| Lazy runtime | One construction on first query and unchanged document filter forwarding. |
| Union adapter | Full candidate order, pre-rerank pool, stage timing, and exception mapping. |
| Sparse rollback | Bound sparse call arguments, zero rerank timing, and fail-closed error mapping. |
| Frozen identity | Exact point count, multi-document chunk-ID hash, and contract constants. |
| Runtime metadata | Exact collections, models, dimensions, limits, profile, batch, and thread values. |
| Search-port constructor | Dense/sparse collections, limits, and document filters reach bound adapters. |
| Composition wiring | No raw client/model parameters cross into canonical pipeline construction. |
| Architecture matrix | Zero temporary layer violations and no active facade imports. |
| Source identity | Both protected E2E blobs and the legacy runtime anchor remain unchanged. |
| Supported CLIs | Parser, output, exit behavior, and canonical imports remain protected offline. |

## 11. Commands and expected results

Focused validation:

```powershell
python -m ruff check app/composition app/application/reranking_service.py `
  app/domain/retrieval.py app/infrastructure/qdrant/search.py `
  tests/test_retrieval_runtime.py tests/test_reranking.py
python -m pytest -q tests/test_retrieval_runtime.py tests/test_reranking.py `
  tests/test_bootstrap.py tests/test_config_contracts.py `
  tests/test_phase7_operational_smoke.py tests/test_phase7_index_cli.py `
  tests/test_architecture_boundaries.py tests/test_source_identity.py
```

Final validation:

```powershell
python -m ruff check .
python -m pytest -q
docker compose config --quiet
git diff --check
```

Expected: all checks remain offline; no model, provider, Qdrant, PDF, or held-out access.

## 12. Small usage example

Application composition remains lazy:

```python
from app.bootstrap import build_query_service
from app.config import Settings

service = build_query_service(Settings())
# Qdrant and retrieval models are still not constructed until service.query(...).
```

A test or alternative composition can inject state-bound ports directly:

```python
pipeline = RerankPipeline.from_searchers(
    dense_searcher=dense_port,
    sparse_searcher=sparse_port,
    cross_encoder=cross_encoder,
    dense_collection="dense",
    hybrid_collection="hybrid",
)
```

## 13. Common failures and debugging

- A model constructed during import means initialization escaped the lazy factory.
- A frozen hash/count failure is a stop condition; never rewrite the expected identity in R11.
- A candidate-order failure means port delegation changed arguments or ordering; inspect the exact
  adapter call before touching ranking policy.
- A missing sparse rollback symbol usually means code bypassed `SparseSearchPort` after a move.
- An architecture failure on an application-to-infrastructure edge means concrete wiring leaked out
  of composition.
- A source-pin failure means a protected anchor changed and must be reverted from this slice.

## 14. Current limitations

- `app.retrieval_runtime` remains byte-for-byte as a historical source-identity anchor until R12.
- Root retrieval, hybrid, and reranking compatibility modules remain for that anchor and archives.
- The legacy callable search protocols and constructor fields remain only to support this window.
- The pinned top-level E2E evaluator intentionally retains historical root imports.
- Real collection/model validation is an explicit integration action and was not run in R11.
- Retrieval algorithms, models, thresholds, collections, and performance are unchanged.

## 15. Self-check questions

1. Why does canonical composition belong outside the application layer?
2. What state do the two Qdrant search adapters bind?
3. Which arguments remain visible through each domain search port?
4. How does `LazyQueryRetriever` protect API startup behavior?
5. Why is the old runtime copied temporarily instead of edited in place?
6. What test proves that no raw client or embedding model is passed to canonical reranking wiring?

## 16. Interview summary

R11 replaces the last active retrieval compatibility imports with one explicit composition module.
The application reranking service now depends on small state-bound search ports while concrete
Qdrant/model state stays in infrastructure and composition. Offline characterization covers exact
frozen metadata, filters, timings, failures, and adapter calls; the dependency matrix has no
temporary violations. A protected historical runtime remains isolated for one versioned R12 cut.

## 17. Validation results and proposed commit

Validation receipts:

| Check | Result |
| --- | --- |
| Pre-change focused pytest, Python 3.11.15 | PASS — 72 tests |
| Post-change focused Ruff | PASS |
| Post-change focused pytest, Python 3.11.15 | PASS — 111 tests |
| Full Ruff | PASS |
| Full pytest, Python 3.11.15 | PASS — 412 tests, 1 dependency warning |
| Docker Compose configuration | PASS |
| Protected data and artifacts | PASS — all 86 files unchanged |
| E2E source pins, legacy runtime anchor, and held-out worktree | PASS — unchanged |
| Local Markdown links and inline source paths | PASS |
| `git diff --check` and exact scope | PASS — 16 files |

Proposed Conventional Commit after review:

```text
refactor: establish canonical retrieval composition
```

## 18. Status

`COMPLETE`

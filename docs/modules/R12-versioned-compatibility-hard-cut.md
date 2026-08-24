# R12 — Versioned compatibility hard cut

## 1. Goal and scope

R12 makes canonical package ownership visible from the file tree. It removes obsolete `app.*`
facades only after moving E2E provenance and reranking diagnostics to their real owners.

- R12A moved E2E scoring to `evaluation.e2e`, moved its command implementation to
  `scripts.evaluation`, and introduced artifact schema v6 with source identity v2.
- R12B removes 16 compatibility modules, removes the callable-based retrieval bridge, and moves
  reranking diagnostics to `evaluation.reranking`.
- R13 later removed the eight top-level supported CLI shims.

No retrieval algorithm, model, threshold, collection, corpus, dataset, provider, API schema, or
answer policy changes in this module.

## 2. Position in the system

```text
FastAPI / supported CLI / Streamlit
                 |
                 v
        application services
                 |
                 v
          domain ports/policies
                 ^
                 |
      infrastructure adapters

evaluation ---> public application/domain interfaces
production --x-> evaluation
```

The hard cut affects internal import paths. It does not change the HTTP boundary or the runtime data
flow.

## 3. Relevant background concepts

- A **facade** re-exports symbols owned elsewhere. It is useful during migration but obscures the
  owner if retained indefinitely.
- A **hard cut** removes that old import path after every supported consumer uses the canonical path.
- **Source identity** hashes the files that materially define an evaluation run.
- An **artifact schema version** identifies the structure and interpretation of a sanitized result.
- A **state-bound port** is an adapter object that already owns its Qdrant client/model/vector state;
  the application service receives only its `search` interface.
- Git history preserves removed implementations and compatibility paths without keeping them in the
  active Python surface.

## 4. Input, output, and contracts

R12 preserved the then-supported E2E invocation and its arguments, approval behavior, scoring rules,
and exit statuses. R13 later changed only the module path to
`python -m scripts.evaluation.evaluate_phase7_e2e`. R12's intentional provenance changes are:

- default output/checkpoint names use `e2e-v6`;
- output `schema_version` is `6`;
- source identity is a version-2 object over canonical owners and the exact system prompt;
- old source-identity/checkpoint formats fail closed and historical v5 artifacts remain untouched.

`RerankPipeline` keeps the same retrieval, candidate, timing, reranking, and error outputs. Its
constructor now accepts `DenseSearchPort` and `SparseSearchPort` objects directly; the legacy raw
client/model/callable constructor path is removed.

The removed `app.*` paths are not public product contracts. Supported code must import the canonical
application, domain, infrastructure, or evaluation module.

## 5. Step-by-step data flow

1. FastAPI resolves the query service through `app.bootstrap`.
2. `app.composition.retrieval` creates Qdrant dense/sparse adapters with bound infrastructure state.
3. It injects those adapters into `app.application.reranking_service.RerankPipeline`.
4. The pipeline retrieves the same dense and sparse candidates, applies the same frozen policies,
   records the same stages, and calls the same lazy cross-encoder adapter.
5. `app.application.query_service` selects/gates evidence before generation, validates source IDs,
   and constructs trusted citations.
6. Offline reranking analysis imports `evaluation.reranking`; production does not.
7. The canonical `scripts.evaluation.evaluate_phase7_e2e` command composes the same public runtime
   and sends completed executions to `evaluation.e2e`.

## 6. Responsibilities of changed files

| Path | Responsibility after R12 |
| --- | --- |
| [`evaluation/e2e.py`](../../evaluation/e2e.py) | Pure E2E scoring, aggregation, facts, and quality gates. |
| [`evaluation/reranking.py`](../../evaluation/reranking.py) | Provider-free reranking diagnostics and aggregation. |
| [`scripts/evaluation/evaluate_phase7_e2e.py`](../../scripts/evaluation/evaluate_phase7_e2e.py) | Approval-gated integration composition, checkpointing, provenance, and sanitized output. |
| Historical `scripts/evaluate_phase7_e2e.py` (removed in R13B) | Temporary R12 command shim. |
| [`app/application/reranking_service.py`](../../app/application/reranking_service.py) | Candidate preparation and reranking through explicit search ports. |
| [`app/composition/retrieval.py`](../../app/composition/retrieval.py) | Concrete model/Qdrant adapter construction and injection. |
| [`app/domain/retrieval.py`](../../app/domain/retrieval.py) | Retrieval records and state-bound application ports. |
| [`tests/test_architecture_boundaries.py`](../../tests/test_architecture_boundaries.py) | Layer matrix, removed-path inventory, and production/evaluation isolation. |
| [`ADR-005`](../adr/ADR-005-compatibility-and-source-identity.md) | Rationale and lifecycle of provenance plus the hard cut. |

The removed facades were `app.citations`, `app.content_identity`, `app.evaluation`,
`app.evidence_selection`, `app.generation`, `app.hybrid_retrieval`, `app.ingestion`, `app.models`,
`app.phase7`, `app.phase7_optimization`, `app.query_expansion`, `app.query_service`, `app.reranking`,
`app.retrieval`, `app.retrieval_runtime`, and `app.domain.policies.ranking`.

## 7. Important symbols and why they exist

- `SOURCE_IDENTITY_VERSION` distinguishes canonical v2 provenance from the historical mapping.
- `SOURCE_IDENTITY_PATHS` is an auditable set of behavior owners rather than facade files.
- `score_phase7_execution` scores one completed query without performing I/O.
- `RerankPipeline` orchestrates candidate preparation and cross-encoder ordering.
- `DenseSearchPort` and `SparseSearchPort` expose only the application-facing search behavior.
- `QdrantDenseSearcher` and `QdrantSparseSearcher` bind infrastructure state at composition time.
- `evaluate_reranked_cases` and `aggregate_rerank_rows` are evaluation-owned diagnostics.

## 8. Before-and-after structure

```text
Before R12
  app/*.py facades                   duplicate import surface
  app/reranking.py                  runtime exports + evaluation diagnostics
  RerankPipeline                    raw SDK state or callable bridge
  app/evaluation_e2e.py             evaluator inside production package
  scripts/evaluate_phase7_e2e.py    full integration implementation

After R12
  app/api|application|domain|infrastructure|composition
                                     one visible production ownership tree
  evaluation/reranking.py            offline diagnostics
  RerankPipeline                     state-bound search ports only
  evaluation/e2e.py                  evaluator outside production
  scripts/evaluation/evaluate_phase7_e2e.py
                                     canonical integration implementation
  scripts/evaluate_phase7_e2e.py     temporary R12 CLI shim; removed in R13B
```

## 9. Design decisions and trade-offs

The hard cut is intentional: keeping another re-export layer would make the shorter tree cosmetic.
Repository-internal callers and tests move together, so a deprecation package would add maintenance
without protecting a user-facing contract.

Reranking diagnostics move to `evaluation` because they consume application executions but never
belong in production reachability. `RerankPipeline` receives bound ports because passing raw clients,
models, vector names, and search callables made the application service aware of adapter assembly.

Archived Phase 6/7 workflows remain byte-preserved provenance. They are unsupported, excluded from
the active import graph, and are not guaranteed to import after canonical owners evolve. Git is the
supported mechanism for reconstructing an old runnable state.

R13 later completed the common top-level CLI cut so R12 did not combine Python package ownership
with command renaming.

## 10. Tests and the behavior each test protects

| Test area | Protected behavior |
| --- | --- |
| Query/API characterization | Request/response schemas, abstention, citations, errors, and timings. |
| Retrieval/reranking | Candidate membership, ordering, filters, metadata, stage timings, and no fallback. |
| Generation/evidence | Prompt bounds, gate order, correction retry, and trusted source validation. |
| E2E/source identity | Scoring, governance, schema v6, checkpoint fail-closed behavior, and exact v2 mapping. |
| Architecture | No removed facade exists; dependency matrix is clean; production cannot reach evaluation. |
| Archive boundary | Historical files remain outside the supported top-level script surface. |

Tests that only asserted facade symbol identity or private helpers of unsupported archived CLIs were
removed. Test count is not a contract; behavior coverage and the complete offline suite are.

## 11. Commands and expected results

```powershell
python -m ruff check .
python -m pytest -q
docker compose config --quiet
git diff --check
```

Expected: all offline checks pass, protected data/artifacts remain byte-identical, and no model,
provider, live Qdrant, re-index, or held-out access occurs.

## 12. Small usage example

Production composes the state-bound pipeline:

```python
pipeline = RerankPipeline(
    dense_searcher=dense_searcher,
    sparse_searcher=sparse_searcher,
    cross_encoder=cross_encoder,
    dense_collection=dense_collection,
    hybrid_collection=hybrid_collection,
)
```

Offline code imports diagnostics directly:

```python
from evaluation.reranking import evaluate_reranked_cases
```

## 13. Common failures and debugging

- `ModuleNotFoundError` for a removed `app.*` facade means the caller must use the owner shown in
  `docs/CODEBASE.md`; do not recreate a shim.
- A source-identity mismatch means a selected behavior owner changed; inspect the diff before any
  approved evaluation.
- A v5 checkpoint rejection is expected under v6; do not rewrite the historical checkpoint.
- If production reaches `evaluation`, inspect the AST dependency test before changing allowlists.
- If a reranking test changes candidate order or timing keys, stop: that is not a structural change.
- Do not run an archived tool merely to diagnose an import failure; restore its historical commit if
  historical reproduction is actually required.

## 14. Current limitations

- R13 removed all eight top-level CLI shims.
- Archived workflows are provenance, not a supported runnable compatibility surface.
- Existing v5 artifacts remain historical receipts and cannot be resumed as v6.
- No real provider/model/Qdrant integration run is part of R12.

## 15. Self-check questions

1. Why is a removed internal import path different from a changed HTTP contract?
2. Why do dense and sparse adapters bind infrastructure state before injection?
3. Why are reranking metrics in `evaluation` rather than `app.application`?
4. Why must a v1 checkpoint fail under source identity v2?
5. Why was top-level CLI removal isolated in R13?
6. How can an archived workflow be reconstructed without keeping active facades?

## 16. Interview summary

R12 removes migration scaffolding after proving every supported consumer uses one canonical owner.
It first versions E2E provenance so historical hashes remain explainable, then deletes the duplicate
`app.*` surface, injects state-bound retrieval ports, and moves diagnostics out of production. The
result is a smaller, honest modular-monolith tree with unchanged runtime decisions and explicit
offline regression protection.

## 17. Validation results and proposed commit

| Check | Result |
| --- | --- |
| R12A full Ruff | PASS |
| R12A full pytest, Python 3.11.15 | PASS — 413 tests, 1 dependency warning |
| R12A Compose/protected-data/links/diff checks | PASS |
| R12B focused Ruff | PASS |
| R12B focused pytest, Python 3.11.15 | PASS — 288 tests, 1 dependency warning |
| R12B full Ruff | PASS |
| R12B full pytest, Python 3.11.15 | PASS — 377 tests, 1 dependency warning |
| Docker Compose configuration | PASS |
| Protected PDFs/data/artifacts | PASS — all 86 files unchanged |
| Removed facade/import inventory | PASS — all 16 paths absent; no supported consumer import |
| Local Markdown links and `git diff --check` | PASS |

Proposed R12B Conventional Commit after review:

```text
refactor: remove legacy application compatibility layer
```

## 18. Status

`COMPLETE`

# R01 — Baseline and characterization coverage

## 1. Goal and scope

R01 creates a deterministic safety net for the Phase 7-only baseline before any package or symbol is
moved. It adds characterization for three gaps identified by the Round 1 audit:

- the public query schema as exposed by FastAPI;
- one golden, cross-stage `QueryExecution` using fake ports;
- the active runtime import path into evaluation code, with an explicit removal owner.

R01 changes no production module. It does not fix dependency violations, alter retrieval or
generation behavior, read benchmark payloads, contact Qdrant, or call a provider.

## 2. Position in the system

R01 sits outside the runtime pipeline as a test boundary:

```text
public schema ─┐
fake ports ────┼─> characterization tests ─> refactor safety signal
source imports ┘

runtime: API -> QueryService -> retrieval -> evidence -> generation -> citations
```

The tests observe the existing pipeline but are not imported by production code.

## 3. Relevant background concepts

A characterization test records intentional, externally observable behavior before restructuring.
It differs from an algorithm-improvement test: it does not define a better score or ranking. It
detects accidental changes while implementations move behind stable contracts.

A golden execution is a small deterministic example whose stage inputs and outputs are asserted
together. The R01 golden uses fakes rather than real models or Qdrant, so it is fast and offline.

An import allowlist is temporary baseline debt. Its entries are not approved architecture. The test
requires an owner for removal and fails if a new forbidden edge appears.

## 4. Input, output, and contracts

### Public HTTP contract

`QueryRequest` requires `question`, forbids unknown fields, accepts optional `document_id`, and
limits `top_k` to 1–10 with a default of 5. The OpenAPI operation remains `POST /api/v1/query`, with
200 and 422 responses, an optional `Authorization` header, and a UTF-8 `QueryResponse` media type.

`QueryResponse` requires `answer` and `abstained`; citations contain trusted chunk, document,
filename, page, heading, and excerpt fields.

### Golden query input

The fake retrieval result contains:

- four pre-rerank candidates;
- three final candidates;
- one exact cross-document duplicate pair;
- one unique installation candidate.

The fake generator first returns an invalid source ID and then a valid corrected answer.

### Golden query output

`QueryExecution` must preserve the full candidate pool, ordered final candidates, selected evidence,
duplicate provenance, two generation attempts, accumulated usage, and trusted citations. Timings are
asserted only for the deterministic retrieval/reranking values; wall-clock durations are not frozen.

## 5. Step-by-step data flow

1. `QueryService.execute` calls `generator.ensure_configured` before retrieval.
2. The fake retriever receives the unchanged question and optional document filter.
3. Evidence selection collapses identical content repeated across documents.
4. Query-derived installation intent selects the installation copy as representative.
5. `top_k=2` bounds the evidence set without truncating diagnostic candidate sets.
6. The evidence gate accepts the valid selected candidates.
7. The first generated answer cites unknown `S9`; citation validation rejects it.
8. The second attempt receives the same evidence object plus sanitized validation errors.
9. Trusted citation metadata is built from the evidence source map.
10. `QueryExecution` returns the response and sanitized stage diagnostics.

Separately, the architecture test parses `app/**/*.py` with the standard-library AST, computes the
dependency closure from `app.main`, and compares production-to-evaluation imports with the exact
owned baseline list.

## 6. Responsibilities of changed files

- [`tests/test_runtime_characterization.py`](../../tests/test_runtime_characterization.py) owns the
  public schema snapshot and golden Phase 7 query execution.
- [`tests/test_architecture_boundaries.py`](../../tests/test_architecture_boundaries.py) owns static
  import discovery and the temporary evaluation-dependency debt inventory.
- [`docs/modules/R01-baseline-characterization.md`](R01-baseline-characterization.md) explains the
  contracts, trade-offs, validation, and handoff to later modules.

No application, test fixture, runtime configuration, script, or benchmark file changes in R01.

## 7. Important symbols and why they exist

- `KNOWN_ACTIVE_RUNTIME_EVALUATION_IMPORTS` is an exact mapping from forbidden import edges to their
  removal module. It currently contains only `app.candidate_audit -> app.evaluation` and
  `app.reranking -> app.evaluation`, both owned by R04.
- `_import_graph` builds internal application dependencies without importing application modules.
- `_reachable_modules` limits the rule to the actual `app.main` dependency closure.
- `RecordingRetriever` exposes stage order and preserves separate pool/final sets.
- `RecordingGenerator` captures correction attempts and evidence object identity.
- `test_golden_phase7_query_execution_preserves_all_stage_boundaries` is the cross-stage safety net
  for later application and adapter cleanup.

## 8. Before-and-after structure

Before R01:

```text
tests/
├── test_query_api.py          # HTTP behavior
├── test_query_service.py      # individual orchestration cases
└── test_retrieval_runtime.py  # profile and runtime cases
```

After R01:

```text
tests/
├── test_architecture_boundaries.py   # import graph guard and owned debt
├── test_runtime_characterization.py  # schema + golden cross-stage execution
├── test_query_api.py
├── test_query_service.py
└── test_retrieval_runtime.py
```

Production structure is unchanged.

## 9. Design decisions and trade-offs

- The schema test asserts stable public fields and media types, not the entire generated OpenAPI
  document. This protects the contract without coupling to unrelated framework metadata.
- The golden test uses real evidence selection, gate, citation validation, and `QueryService`, but
  fake retrieval/generation ports. This exercises orchestration without external side effects.
- Candidate IDs and document IDs are synthetic except for the two frozen Phase 7 document IDs. No
  raw manual or benchmark content is read.
- The import test records only evaluation edges reachable from `app.main`. Offline evaluation modules
  may import evaluation intentionally; they are outside this specific production-runtime guard.
- The allowlist is exact and owned by R04. Removing an edge requires updating the test; adding an edge
  fails immediately. ADR-001 remains for R02, when concrete package boundaries exist.

## 10. Tests and protected behavior

| Test | Protected behavior |
|---|---|
| `test_public_query_schema_matches_the_active_http_contract` | request/response fields, bounds, response codes, auth header, UTF-8 media type |
| `test_golden_phase7_query_execution_preserves_all_stage_boundaries` | stage order, pool/final/evidence boundaries, dedup representative, retry, citations, usage |
| `test_active_runtime_evaluation_imports_match_owned_baseline_debt` | no new production → evaluation import and explicit R04 ownership |
| `test_inbound_adapters_are_not_imported_by_other_application_modules` | application modules do not depend on FastAPI inbound adapters |
| existing `test_query_api.py` | HTTP statuses, sanitized errors, authentication, request ID behavior |
| existing `test_query_service.py` | gate, abstention, retry, privacy, no silent fallback |
| existing `test_retrieval_runtime.py` | Phase 7 profile values, Phase 6 rejection, laziness, frozen identity |

## 11. Commands and expected results

```text
python -m pytest -q tests/test_architecture_boundaries.py tests/test_runtime_characterization.py
python -m pytest -q tests/test_architecture_boundaries.py tests/test_runtime_characterization.py tests/test_query_service.py tests/test_query_api.py tests/test_retrieval_runtime.py tests/test_health.py
python -m ruff check .
python -m pytest -q
git diff --check
```

All commands must use Python 3.11 for completion. Pytest must remain offline and must not use local
credentials, providers, model downloads, external Qdrant, or raw benchmark payloads.

## 12. Small usage example

When a later module moves `QueryService`, the golden test should continue to produce this boundary
shape:

```python
assert [item.chunk_id for item in execution.candidate_pool] == [
    "pre-rerank-only",
    "programming-copy",
    "installation-copy",
    "installation-unique",
]
assert [item.chunk_id for item in execution.evidence_candidates] == [
    "installation-copy",
    "installation-unique",
]
assert execution.generation_attempts == 2
```

Imports may move with the implementation, but the observed ordering and response contract remain.

## 13. Common failures and debugging

- If the schema snapshot fails, inspect whether a public request/response field or media type changed.
  Do not update the assertion during a behavior-preserving refactor without explicit approval.
- If stage order fails, check whether configuration validation moved after retrieval or generation
  moved before the evidence gate.
- If evidence IDs change, inspect query-role inference and exact-content deduplication before assuming
  the golden is stale.
- If the import test reports a new edge, remove the dependency or assign it through an approved module;
  never add an unowned allowlist entry for convenience.
- If local credentials influence collection, verify `tests/conftest.py` still disables `.env` and
  clears matching environment variables.

## 14. Current limitations

- R01 does not remove the two active-runtime evaluation imports; R04 owns that cleanup.
- The import guard describes the current flat `app` package. R02 will establish concrete composition
  and domain seams, then ADR-001 can define the enduring layer rule.
- The golden execution protects one representative successful correction path. Existing focused tests
  remain responsible for all abstention and dependency-error variants.
- Timings derived from `perf_counter` are intentionally not compared exactly.

## 15. Self-check questions

1. Why is an allowlisted dependency still technical debt?
2. Which three candidate sets does `QueryExecution` preserve, and why are they different?
3. Why does the golden test use fake ports but real evidence/citation logic?
4. Which schema details are public contracts and which OpenAPI details are framework noise?
5. What should happen if R04 removes one known evaluation import?

## 16. Interview summary

Before moving modules, R01 converted implicit behavior into executable contracts. A single offline
golden query now proves stage ordering and data boundaries across retrieval, evidence selection,
correction retry, and trusted citation construction. A static import test also prevents production's
known evaluation coupling from growing while assigning its removal to R04.

## 17. Validation results and proposed commit

Current results:

```text
New R01 characterization tests                    PASS — 4 tests
R01 + existing query/API/runtime/health tests     PASS — 53 tests, 1 warning
Full Ruff                                         PASS
Full pytest on Python 3.11.15                     PASS — 327 tests, 1 warning
Full pytest on host Python 3.13.5                 PASS — 327 tests, 1 warning
R01 Markdown local links/trailing whitespace      PASS
git diff --check                                  PASS
```

The warning is the existing third-party Starlette/TestClient deprecation warning. Proposed
Conventional Commit:

```text
test: capture portfolio cleanup baseline
```

No commit is created without a separate explicit user request.

## 18. Status

`COMPLETE` — the public schema, golden query execution, and owned import debt are characterized;
focused and full offline checks pass without production-code changes.

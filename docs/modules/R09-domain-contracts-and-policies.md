# R09 — Domain contracts and ranking policies

## 1. Goal and scope

R09 removes ambiguous model and ranking ownership from the production graph. The module is split into
small vertical slices so each move preserves exact Pydantic identity, retrieval ordering, frozen
policy values, and archived provenance.

The completed R09A slice moves health DTOs and retrieval records to canonical package owners. It
keeps `app.models` as a temporary compatibility export because remaining consumers are migrated in
later R09 slices. Ranking-policy splitting has not started.

## 2. Position in the system

```text
FastAPI health adapter -> app.contracts.health

application / infrastructure / evaluation
                 -> app.domain.retrieval

legacy consumers -> app.models compatibility exports
                             -> canonical owners
```

These records cross several layers but contain no orchestration or infrastructure behavior. Their
canonical ownership must be established before consumers and policies can be simplified safely.

## 3. Relevant background concepts

- A **data transfer object (DTO)** describes data crossing an adapter boundary without implementing
  the use case.
- A **domain record** represents information used by domain and application logic.
- A **compatibility export** makes an old import resolve to the same class object; it is not a second
  definition.
- **Class identity** means `old_path.Symbol is canonical_path.Symbol`, which protects dependency
  injection, Pydantic schemas, and code using exact type comparisons.
- A **vertical slice** migrates one complete path through the layers instead of editing every model
  and policy at once.

## 4. Input, output, and contracts

`HealthResponse` and `ReadinessResponse` accept `status="ok"`, service name, and version and remain
the response models for `/health` and `/ready`.

`RetrievedChunk` represents one dense result. `RetrievalCandidate` carries dense, sparse, RRF, and
reranker signals. Its ranks remain optional one-based integers, metadata retains an independent
default dictionary, and field definition order remains stable for serialization.

`app.models` continues to export all eight historical symbols with exact class identity during the
migration.

## 5. Step-by-step data flow

1. FastAPI constructs health/readiness responses from `app.contracts.health`.
2. The readiness adapter catches the canonical `app.errors.RetrievalError`.
3. Dense adapters produce `RetrievedChunk` records.
4. Retrieval assembly maps them to `RetrievalCandidate` records and applies one-based ranks.
5. Existing consumers importing from `app.models` receive the same canonical class objects.
6. Pydantic validation and serialization execute on the canonical definitions only.

## 6. Responsibilities of changed files

| Path | Responsibility after R09A |
| --- | --- |
| [`app/contracts/health.py`](../../app/contracts/health.py) | Canonical health and readiness HTTP DTOs. |
| [`app/domain/retrieval.py`](../../app/domain/retrieval.py) | Canonical retrieval records, ports, results, and deterministic candidate assembly. |
| [`app/models.py`](../../app/models.py) | Temporary identity-preserving exports; no model definitions. |
| [`app/api/health.py`](../../app/api/health.py) | Thin adapter using canonical contracts and error type. |
| [`tests/test_architecture_boundaries.py`](../../tests/test_architecture_boundaries.py) | Reduced exact debt set after removing three facade edges. |
| [`tests/test_runtime_characterization.py`](../../tests/test_runtime_characterization.py) | Canonical schema, rank, copy, and compatibility identity coverage. |

## 7. Important symbols and why they exist

- `HealthResponse` and `ReadinessResponse` remain distinct so FastAPI keeps two meaningful schema
  names even though their current fields match.
- `RetrievedChunk` is the dense-search boundary record.
- `RetrievalCandidate` is the shared ranking record and owns the one-based rank constraints.
- `QueryRetriever` remains the application-facing retrieval port.
- `QueryRetrievalResult` keeps final candidates, the pre-rerank pool, and separately measured stage
  latency.
- `app.models.__all__` makes the temporary compatibility surface explicit.

## 8. Before-and-after structure

```text
Before R09A
  app.models
    health DTOs
    retrieval records
    query/document aliases

After R09A
  app.contracts.health
    health DTOs
  app.domain.retrieval
    retrieval records and policies
  app.models
    temporary aliases only
```

The dependency matrix now records 15 temporary compatibility edges instead of 18.

## 9. Design decisions and trade-offs

The slice moves definitions before migrating every consumer. This briefly retains the root facade
but prevents a high-risk repository-wide import rewrite. Exact alias identity means no duplicate
Pydantic classes or divergent schemas are created.

The health DTOs are not merged because separate response-model names are useful API documentation and
merging them could alter OpenAPI component identity. Retrieval records remain Pydantic models because
serialization and validation are existing contracts; changing them to dataclasses is out of scope.

The architecture debt allowlist shrinks in the same patch. It never treats a cleaned edge as a
permanent exception.

## 10. Tests and the behavior each test protects

| Test area | Protected behavior |
| --- | --- |
| Compatibility identity | Every `app.models` export is the exact canonical class object. |
| RetrievalCandidate schema | Required field order and four one-based rank constraints. |
| Model copying | Copy updates do not mutate the source candidate or shared metadata. |
| Health endpoint | Exact HTTP 200 JSON response. |
| Readiness endpoint | Injected check and sanitized retrieval 503 behavior. |
| OpenAPI characterization | Existing query and response schema components remain stable. |
| Architecture matrix | Three removed compatibility edges stay removed. |

## 11. Commands and expected results

Focused checks for R09A:

```powershell
python -m ruff check app/contracts/health.py app/domain/retrieval.py `
  app/models.py app/api/health.py tests/test_architecture_boundaries.py `
  tests/test_runtime_characterization.py
python -m pytest -q tests/test_architecture_boundaries.py `
  tests/test_runtime_characterization.py tests/test_health.py
```

Module checkpoint checks:

```powershell
python -m ruff check .
python -m pytest -q
docker compose config --quiet
git diff --check
```

Expected: no provider, model, Qdrant, dataset, or indexing access and no change to protected hashes.

## 12. Small usage example

New code imports the canonical record:

```python
from app.domain.retrieval import RetrievalCandidate

candidate = RetrievalCandidate(
    chunk_id="chunk-1",
    document_id="manual-1",
    filename="manual.pdf",
    text="Disconnect all power.",
    page_numbers=[42],
    headings=["Safety"],
    content_type="text",
    score=0.8,
)
```

An unmigrated consumer may still import `RetrievalCandidate` from `app.models` during R09 and receives
the same class object.

## 13. Common failures and debugging

- Different class objects indicate a copied definition instead of a re-export.
- An OpenAPI component change usually means a DTO was renamed, merged, or wrapped.
- A rank validation failure after a move means `Field(ge=1)` was not preserved.
- An architecture mismatch with three restored edges means a canonical module imported the facade
  again.
- A circular import means a domain module still relies on `app.models` while that facade imports the
  same domain module.

## 14. Current limitations

- Fifteen compatibility edges still use `app.models`, `app.content_identity`, or retrieval facades.
- `app.models` remains importable until all repository consumers migrate.
- `DenseSearcher` and `SparseSearcher` still expose infrastructure-shaped arguments; R11 will replace
  these ports.
- `ranking.py` and `reranking_service.py` remain unsplit.
- No Phase 6 archive ownership or CLI naming changes are included in R09A.

## 15. Self-check questions

1. Why are retrieval records domain objects while health responses are adapter contracts?
2. How does a re-export preserve class identity?
3. Which behavior would break if a rank lost its `ge=1` constraint?
4. Why are health and readiness responses still separate classes?
5. Why does R09A leave `app.models` in place?
6. What must happen before the compatibility facade can be removed?

## 16. Interview summary

R09A replaces an umbrella model module with explicit owners without forcing a risky big-bang import
migration. HTTP DTOs now belong to contracts, retrieval records belong to the domain, and the old
module contains aliases only. Tests prove schema and object identity, while the dependency matrix
shows measurable progress by removing three forbidden edges.

## 17. Validation results and proposed commit

Current R09A validation:

| Check | Result |
| --- | --- |
| Pre-change focused pytest, Python 3.11.15 | PASS — 45 tests, 1 warning |
| Post-change focused Ruff | PASS |
| Post-change focused pytest, Python 3.11.15 | PASS — 46 tests, 1 warning |
| Protected baseline manifest | CREATED — 86 files outside repository |
| Full Ruff | PASS |
| Full pytest, Python 3.11.15 | PASS — 404 tests, 1 warning |
| Docker Compose configuration | PASS |
| Protected data and artifact comparison | PASS — all 86 files unchanged |
| E2E source pins and held-out v2 worktree check | PASS — unchanged |
| Local Markdown links and inline source paths | PASS |
| `git diff --check` and seven-file scope | PASS |

Proposed Conventional Commit after review:

```text
refactor: establish canonical runtime record ownership
```

## 18. Status

`IN_PROGRESS`

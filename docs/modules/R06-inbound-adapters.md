# R06 — Thin inbound adapters

## 1. Goal and scope

R06 makes HTTP, command-line, and Streamlit entry points thin adapters over stable application and
domain contracts. It preserves every public path, schema, status, header, CLI contract, and UI
interaction.

R06A is implemented in this slice. It gives FastAPI one explicit dependency seam and moves
health/readiness route mapping out of the application factory. CLI and Streamlit cleanup remain later
R06 slices and are not described as completed work.

R06B1 is also implemented. It classifies the two active runtime smoke commands under
`scripts.operations` while preserving both historical `python -m scripts.<name>` entry points as
thin compatibility shims. Archived Phase 6 tools remain unsupported and untouched.

R06B2 applies the same classification to the supported ingestion preview and guarded Phase 7 corpus
index command. It moves no indexing policy and executes no ingestion or Qdrant operation.

## 2. Position in the system

```text
HTTP request
    |
    v
FastAPI validation / auth / error mapping
    |
    v
app.application.query_service.QueryService
    |
    v
HTTP response
```

FastAPI is an inbound adapter. It owns transport concerns but not retrieval, evidence, generation, or
ranking decisions. `app.bootstrap` remains the production composition root that creates concrete
dependencies.

## 3. Relevant background concepts

A dependency seam is one module through which an adapter obtains application services and runtime
configuration. It makes composition visible and lets tests replace dependencies without constructing
Qdrant clients, models, or providers.

An application factory constructs a fresh FastAPI object from explicit dependencies. This avoids
hard-coding test globals while preserving the compatibility ASGI export `app.main:app`.

## 4. Input, output, and contracts

`create_app(...)` accepts optional `Settings`, a `QueryServiceProvider`, and a `ReadinessChecker`. It
returns a FastAPI application with the established API prefix and three routes:

- `GET /health` returns a healthy service/version response;
- `GET /ready` returns readiness or a sanitized `retrieval_not_ready` response;
- `POST /query` validates input, authenticates when configured, calls `QueryService`, and maps errors.

The request-ID middleware accepts only 1–64 ASCII letters, digits, dots, underscores, or hyphens.
Invalid or absent values produce a generated ID. Every response keeps the `X-Request-ID` header.

## 5. Step-by-step data flow

1. [`app/main.py`](../../app/main.py) exports the app produced by `create_app()`.
2. The factory resolves settings and a lazy readiness checker.
3. Query and health routers are mounted under the configured prefix.
4. FastAPI dependency injection resolves auth settings and the query-service provider through one
   dependency module.
5. Query request DTO validation trims text and rejects invalid fields before service execution.
6. The query route runs synchronous orchestration in FastAPI's thread pool.
7. Known dependency failures become sanitized HTTP errors; unexpected failures become sanitized 500.
8. Health is process liveness only; readiness invokes the injected frozen-corpus checker.
9. Middleware adds the established safe correlation header without logging request bodies or tokens.
10. Operational smoke shims delegate to canonical Phase 7 command modules without duplicating logic.
11. Ingestion/indexing shims delegate to canonical operational modules; the index command validates
    collection targets before reading chunks, creating models, or opening Qdrant.

## 6. Responsibilities of changed files

- [`app/api/dependencies.py`](../../app/api/dependencies.py) is the explicit FastAPI-facing seam for
  settings, query-service construction, readiness construction, and their public types.
- [`app/api/health.py`](../../app/api/health.py) maps health/readiness outcomes to HTTP DTOs and status.
- [`app/api/app.py`](../../app/api/app.py) constructs FastAPI, mounts routers, installs overrides, and
  owns request-ID middleware.
- [`app/api/query.py`](../../app/api/query.py) validates transport input through DTOs, calls the
  application service, and maps application/infrastructure failures to sanitized HTTP errors.
- [`app/api/auth.py`](../../app/api/auth.py) applies the established optional bearer-token policy.
- [`tests/test_health.py`](../../tests/test_health.py) protects health/readiness behavior, factory
  metadata, request IDs, and dependency alias identity.
- [`tests/test_architecture_boundaries.py`](../../tests/test_architecture_boundaries.py) ensures each
  FastAPI module uses the explicit dependency seam rather than compatibility or composition imports.
- [`scripts/operations/query_smoke.py`](../../scripts/operations/query_smoke.py) owns the supported
  bounded query smoke, sanitized artifact, scenarios, and explicit Phase 7 guard.
- [`scripts/operations/validate_query_runtime.py`](../../scripts/operations/validate_query_runtime.py)
  owns the supported read-only retrieval smoke and its stable parser/output contract.
- [`scripts/query_smoke.py`](../../scripts/query_smoke.py) and
  [`scripts/validate_query_runtime.py`](../../scripts/validate_query_runtime.py) are thin historical
  entry-point shims.
- [`tests/test_phase7_operational_smoke.py`](../../tests/test_phase7_operational_smoke.py) protects
  parser defaults, guards, artifact sanitization, exit behavior, and shim identity.
- [`scripts/operations/ingest_preview.py`](../../scripts/operations/ingest_preview.py) owns the
  supported parse/preview/optional-JSONL command adapter.
- [`scripts/operations/index_phase7_corpus.py`](../../scripts/operations/index_phase7_corpus.py) owns
  guarded Phase 7 indexing composition, manifest output, and the explicit mutation entry point.
- [`scripts/ingest_preview.py`](../../scripts/ingest_preview.py) and
  [`scripts/index_phase7_corpus.py`](../../scripts/index_phase7_corpus.py) preserve historical command
  paths as thin shims.
- [`tests/test_ingestion.py`](../../tests/test_ingestion.py) and
  [`tests/test_phase7_index_cli.py`](../../tests/test_phase7_index_cli.py) protect shim identity,
  parser defaults, failure ordering, atomic output, mutation order, and verification behavior.

## 7. Important symbols and why they exist

- `QueryServiceProvider`: callable contract used by FastAPI dependency overrides.
- `ReadinessChecker`: read-only callable for validating frozen retrieval availability.
- `create_health_router`: constructs health routes around explicit settings and readiness dependencies.
- `create_app`: reusable FastAPI factory for production and offline contract tests.
- `UTF8JSONResponse`: preserves the explicit UTF-8 content type required by legacy clients.
- `require_query_auth`: fail-closed optional bearer authentication dependency.
- `query`: thin async HTTP mapping around synchronous `QueryService.execute`.
- `scripts.operations.query_smoke.main`: supported bounded query smoke orchestration.
- `scripts.operations.validate_query_runtime.main`: supported read-only retrieval smoke orchestration.
- `scripts.operations.ingest_preview.main`: supported document preview adapter with optional JSONL.
- `scripts.operations.index_phase7_corpus.main`: explicit guarded corpus mutation adapter.

## 8. Before-and-after structure

Before R06A:

```text
app.api.app
├── imports bootstrap/config/query compatibility paths
├── creates FastAPI and middleware
└── defines health and readiness routes inline

app.api.query ──> bootstrap + query compatibility facade
app.api.auth  ──> config
```

After R06A:

```text
app.api.app ──> app.api.dependencies <── app.api.query/auth/health
     |
     ├── request-ID middleware
     ├── query router
     └── health router factory

app.api.dependencies ──> application service + bootstrap + canonical config
```

After R06B1:

```text
python -m scripts.query_smoke
        |
        v
thin compatibility shim ──> scripts.operations.query_smoke

python -m scripts.validate_query_runtime
        |
        v
thin compatibility shim ──> scripts.operations.validate_query_runtime

scripts.archive.phase6.* remains unsupported
```

After R06B2, the same pattern also applies to:

```text
scripts.ingest_preview       ──> scripts.operations.ingest_preview
scripts.index_phase7_corpus ──> scripts.operations.index_phase7_corpus
```

## 9. Design decisions and trade-offs

- Dependency symbols are direct aliases, so FastAPI override keys retain object identity.
- Health routes use a router factory because version and readiness behavior are app-instance inputs;
  module-level mutable state would make tests and multiple app instances unsafe.
- Query, auth, and readiness error payloads are not consolidated in R06A because changing their
  construction while moving boundaries would mix structural work with transport-policy redesign.
- HTTP DTOs remain in `app.models` during R06A. R06C owns separating stable shared query contracts and
  updating Streamlit imports in one reviewed slice.
- `app.main:app` remains unchanged because it is the documented Uvicorn and Compose entry point.
- Canonical commands live under an explicit `operations` package so support status is visible without
  renaming established invocations. Shims use direct aliases and contain no runtime policy.
- The bounded query smoke is still an explicit integration command, not a unit-test operation. R06B1
  validates only its fake/no-key paths and never authorizes a real provider call.
- The indexing command remains intentionally explicit and mutation-capable. Packaging does not make
  it safe to run automatically; protected-target validation and frozen archive verification remain.
- The existing indexing command imports frozen-chunk/manifest helpers from evaluation-era modules.
  R07 owns relocating those helpers; R06B2 does not mix that dependency cleanup into CLI packaging.

## 10. Tests and protected behavior

| Test area | Protected behavior |
|---|---|
| health response | exact status, service, and version fields |
| readiness injection | one checker call and sanitized unavailable response |
| app factory | custom title, version, prefix, route set, and middleware |
| request ID | accepted safe caller ID or generated replacement |
| query schema | exact Pydantic/OpenAPI request and response contracts |
| query errors | sanitized 401/422/500/503/504 mappings |
| dependency identity | overrides still target canonical cached dependencies |
| architecture graph | FastAPI modules use one dependency seam |
| CLI facade identity | old command imports resolve to canonical `main` and parser objects |
| retrieval smoke guard | unsupported profile exits before building a retriever |
| query smoke no-key path | sanitized artifact is written without constructing the service |
| ingestion shim/parser | old entry point identity and preview argument contract |
| indexing defaults | exact two manuals, batch/chunker, collection names, and opt-in flags |
| indexing guard | protected targets fail before file/model/Qdrant access |
| indexing order | dense then hybrid per document, followed by exact verification |

## 11. Commands and expected results

R06A pre-move characterization:

```text
python -m pytest -q tests/test_health.py tests/test_query_api.py tests/test_runtime_characterization.py tests/test_bootstrap.py tests/test_architecture_boundaries.py
```

R06A focused validation:

```text
python -m ruff check app/api tests/test_health.py tests/test_query_api.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_health.py tests/test_query_api.py tests/test_runtime_characterization.py tests/test_bootstrap.py tests/test_architecture_boundaries.py
```

R06B1 characterization and focused validation:

```text
python -m pytest -q tests/test_phase7_operational_smoke.py tests/test_retrieval_runtime.py tests/test_config_contracts.py
python -m ruff check scripts/operations scripts/query_smoke.py scripts/validate_query_runtime.py tests/test_phase7_operational_smoke.py
python -m pytest -q tests/test_phase7_operational_smoke.py tests/test_retrieval_runtime.py tests/test_config_contracts.py
python -m scripts.query_smoke --help
python -m scripts.operations.query_smoke --help
python -m scripts.validate_query_runtime --help
python -m scripts.operations.validate_query_runtime --help
```

R06B2 characterization and focused validation:

```text
python -m pytest -q tests/test_ingestion.py tests/test_phase7_index_cli.py tests/test_architecture_boundaries.py
python -m ruff check scripts/operations scripts/ingest_preview.py scripts/index_phase7_corpus.py tests/test_ingestion.py tests/test_phase7_index_cli.py
python -m pytest -q tests/test_ingestion.py tests/test_phase7_index_cli.py tests/test_architecture_boundaries.py
python -m scripts.ingest_preview --help
python -m scripts.operations.ingest_preview --help
python -m scripts.index_phase7_corpus --help
python -m scripts.operations.index_phase7_corpus --help
```

Slice completion:

```text
python -m ruff check .
python -m pytest -q
docker compose config --quiet
git diff --check
```

These tests are offline. They inject fakes and do not call a provider, download a model, query Qdrant,
or run evaluation.

## 12. Small usage example

```python
from app.api.app import create_app

test_app = create_app(
    query_service_provider=lambda: fake_service,
    readiness_checker=lambda: None,
)
```

The same factory serves production defaults when those arguments are omitted.

Supported runtime smoke invocations remain:

```text
python -m scripts.validate_query_runtime --help
python -m scripts.query_smoke --help
python -m scripts.ingest_preview --help
python -m scripts.index_phase7_corpus --help
```

## 13. Common failures and debugging

- A missing dependency override usually means an alias was wrapped or recreated; verify object identity
  in `app.api.dependencies`.
- A changed OpenAPI response content type usually means `UTF8JSONResponse` is no longer the default.
- Readiness constructing a model during app creation means laziness was lost in bootstrap composition.
- A raw exception message in an HTTP response violates sanitization; inspect the route error mapper.
- A changed request-ID header means middleware validation or response attachment order drifted.
- A shim identity failure means operational logic was copied or wrapped instead of directly exported.
- A smoke test constructing a retriever/provider on a rejected or no-key path violates fail-closed
  ordering; inspect the profile/key guard before changing an expected result.
- An index command touching ingestion, frozen chunks, models, or Qdrant before target validation is a
  safety regression; never update the guard-order test to accept it.

## 14. Current limitations

- Evaluation-era frozen-chunk and manifest helpers remain dependencies of the indexing command. R07
  must relocate them without changing manifest or chunk-set semantics.
- Public HTTP DTOs still share `app.models` with internal retrieval records; R06C owns contract split.
- Streamlit still imports `QueryResponse` from broad application models; R06C will preserve its
  HTTP-only client while narrowing imports.
- Docker/Compose entry points remain untouched because R06A changes no executable module path.

## 15. Self-check questions

1. Why must the dependency aliases preserve object identity?
2. Why is health different from readiness?
3. Which module is allowed to know concrete runtime composition?
4. Why does the query route use a thread pool?
5. Which data may an unexpected-error log contain?
6. Why is the health router constructed per app instance?
7. Which R06 slice owns the shared query DTO move?
8. Why are the old smoke module paths retained after classification?
9. Which paths are allowed to perform a real provider or model smoke?
10. Why is the Phase 7 indexing command never run as ordinary unit validation?
11. Which validation must happen before any index input or external dependency is opened?

## 16. Interview summary

R06A turns FastAPI into a clearer inbound boundary without changing its public contract. One explicit
dependency module connects routes to canonical settings, application services, and bootstrap. The app
factory owns only construction, router mounting, overrides, and request correlation; health/readiness
mapping lives in its own route module. Direct aliases preserve dependency override identity, while
offline API characterization proves paths, schemas, statuses, headers, errors, and laziness stay stable.
R06B1 separately makes support status explicit for the two active runtime smoke commands. Canonical
implementations live under `scripts.operations`; historical module invocations remain direct shims,
and characterization proves guards, defaults, sanitized artifacts, and exit behavior stay unchanged.
R06B2 uses the same compatibility pattern for ingestion and indexing commands. The package structure
now distinguishes supported operations from research/evaluation scripts, while exact parser defaults,
guard ordering, atomic preview output, indexing order, and verification remain protected offline.

## 17. Validation results and proposed commit

```text
R06A pre-move characterization    PASS — 44 tests, 1 warning
R06A focused Ruff                 PASS
R06A focused API suite            PASS — 46 tests, 1 warning
R06A full Ruff Python 3.11.15     PASS
R06A full pytest Python 3.11.15   PASS — 392 tests, 1 warning
R06A Compose config               PASS
R06A Markdown links (8) / diff check PASS
R06B1 pre-move characterization  PASS — 25 tests
R06B1 focused Ruff               PASS
R06B1 focused operational suite  PASS — 26 tests
R06B1 old/new help contracts     PASS — 4 commands
R06B1 full Ruff Python 3.11.15   PASS
R06B1 full pytest Python 3.11.15 PASS — 393 tests, 1 warning
R06B1 Compose config             PASS
R06B1 Markdown links (13) / diff check PASS
R06B2 pre-move characterization  PASS — 52 tests
R06B2 focused Ruff               PASS
R06B2 focused ingestion suite    PASS — 54 tests
R06B2 old/new help contracts     PASS — 4 commands
R06B2 full Ruff Python 3.11.15   PASS
R06B2 full pytest Python 3.11.15 PASS — 395 tests, 1 warning
R06B2 Compose config             PASS
R06B2 Markdown links (19) / diff check PASS
```

Proposed commit after user review:

```text
refactor: classify supported ingestion commands
```

## 18. Status

`IN_PROGRESS` — R06A, R06B1, and R06B2 are implemented and validated. Phase 6 CLIs remain archived
and unsupported. Shared HTTP/Streamlit contract work remains open.

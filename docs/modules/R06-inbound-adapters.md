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

R06C1 establishes `app.contracts.query` as the canonical owner of the three public query DTOs. The
old `app.models` symbols remain direct aliases, while UI imports move only in R06C2.

R06C2 moves the Streamlit API client, state helpers, and renderer to that canonical contract. Static
architecture coverage now proves the UI can reach backend code only through `app.contracts.query`.

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
12. The application constructs a canonical `QueryResponse`; FastAPI serializes that same contract and
    compatibility consumers receive the identical class through `app.models`.
13. Streamlit sends JSON only through `RAGAPIClient`, validates the response as `QueryResponse`, then
    passes that DTO to bounded session history and rendering helpers.

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
- [`app/contracts/query.py`](../../app/contracts/query.py) owns `QueryRequest`, `Citation`, and
  `QueryResponse`, including their exact Pydantic validation and schema behavior.
- [`app/models.py`](../../app/models.py) retains internal retrieval/health records and directly
  compatibility-exports the canonical query contracts.
- [`app/application/query_service.py`](../../app/application/query_service.py),
  [`app/domain/citations.py`](../../app/domain/citations.py), and
  [`app/api/query.py`](../../app/api/query.py) consume canonical query contracts while continuing to
  use internal candidate records where required.
- [`tests/test_query_api.py`](../../tests/test_query_api.py) protects facade class identity in addition
  to the established schema, validation, and HTTP mappings.
- [`ui/api_client.py`](../../ui/api_client.py), [`ui/state.py`](../../ui/state.py), and
  [`ui/streamlit_app.py`](../../ui/streamlit_app.py) consume only the shared query contract from the
  backend package; all runtime communication remains HTTP-only.
- [`tests/test_streamlit_api_client.py`](../../tests/test_streamlit_api_client.py) protects canonical
  response typing, exact request bodies, auth, sanitized failures, timeout, and no POST retry.
- [`tests/test_streamlit_app.py`](../../tests/test_streamlit_app.py) protects document options,
  response/history behavior, and page labels using canonical contract instances.

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
- `QueryRequest`: strict, normalized public query input with bounded `top_k`.
- `Citation`: public trusted citation metadata schema.
- `QueryResponse`: public grounded answer or abstention schema shared by API and UI.
- `RAGAPIClient`: the UI's only runtime gateway to health, readiness, and query HTTP endpoints.

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

After R06C1:

```text
app.contracts.query
├── QueryRequest
├── Citation
└── QueryResponse
       ↑             ↑                ↑
application       FastAPI        app.models facade
```

After R06C2:

```text
Streamlit renderer/state
        |
        v
RAGAPIClient ──HTTP──> FastAPI
        |
        v
app.contracts.query.QueryResponse

Streamlit  ✕→  application/domain/infrastructure/config/models facade
```

## 9. Design decisions and trade-offs

- Dependency symbols are direct aliases, so FastAPI override keys retain object identity.
- Health routes use a router factory because version and readiness behavior are app-instance inputs;
  module-level mutable state would make tests and multiple app instances unsafe.
- Query, auth, and readiness error payloads are not consolidated in R06A because changing their
  construction while moving boundaries would mix structural work with transport-policy redesign.
- HTTP DTOs remain in `app.models` during R06A. R06C separates stable shared query contracts first,
  then updates Streamlit imports in a second reviewable slice.
- `app.main:app` remains unchanged because it is the documented Uvicorn and Compose entry point.
- Canonical commands live under an explicit `operations` package so support status is visible without
  renaming established invocations. Shims use direct aliases and contain no runtime policy.
- The bounded query smoke is still an explicit integration command, not a unit-test operation. R06B1
  validates only its fake/no-key paths and never authorizes a real provider call.
- The indexing command remains intentionally explicit and mutation-capable. Packaging does not make
  it safe to run automatically; protected-target validation and frozen archive verification remain.
- The existing indexing command imports frozen-chunk/manifest helpers from evaluation-era modules.
  R07 owns relocating those helpers; R06B2 does not mix that dependency cleanup into CLI packaging.
- Query DTOs stay Pydantic models because validation and JSON schema are part of the established public
  contract. The new owner is framework-neutral: it imports Pydantic but no FastAPI or Streamlit.
- `app.models` uses direct aliases rather than duplicate subclasses so `isinstance`, schema names,
  dependency annotations, and historical imports remain stable.
- R06C1 does not change UI imports in the same slice; this keeps the contract move reviewable before
  enforcing Streamlit's narrowed dependency graph in R06C2.
- R06C2 changes only import ownership. It deliberately keeps the same `httpx.Client`, one request per
  operation, timeout, bearer header, response validation, and sanitized UI error messages.
- No R06D Docker edit is necessary: images already install `app*` packages and copy whole source
  directories, while all supported old script paths remain shims. Compose validation covers paths.

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
| query schema snapshot | exact required/default/bounds/extra-field and OpenAPI `$ref` behavior |
| contract facade identity | old `app.models` query symbols are canonical classes, not wrappers |
| canonical consumers | application, citation policy, and FastAPI use the shared contract owner |
| UI response type | HTTP payload validates to canonical `QueryResponse` |
| UI API-only graph | UI cannot import backend models, services, domain, infrastructure, or config |
| UI transport | exact path/body/auth/timeout/error mapping and no POST retry |

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

R06C1 characterization and focused validation:

```text
python -m pytest -q tests/test_query_api.py tests/test_runtime_characterization.py tests/test_query_service.py tests/test_citations.py tests/test_streamlit_api_client.py tests/test_streamlit_app.py tests/test_architecture_boundaries.py
python -m ruff check app/contracts app/models.py app/api/query.py app/application/query_service.py app/domain/citations.py tests/test_query_api.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_query_api.py tests/test_runtime_characterization.py tests/test_query_service.py tests/test_citations.py tests/test_streamlit_api_client.py tests/test_streamlit_app.py tests/test_architecture_boundaries.py
```

R06C2 characterization and focused validation:

```text
python -m pytest -q tests/test_streamlit_api_client.py tests/test_streamlit_app.py tests/test_query_api.py tests/test_runtime_characterization.py tests/test_architecture_boundaries.py
python -m ruff check ui tests/test_streamlit_api_client.py tests/test_streamlit_app.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_streamlit_api_client.py tests/test_streamlit_app.py tests/test_query_api.py tests/test_runtime_characterization.py tests/test_architecture_boundaries.py
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
- A changed OpenAPI `$ref` or Pydantic required/default set means the DTO move changed public behavior;
  compare canonical and compatibility class identity before editing expected schemas.
- A UI architecture failure means Streamlit reached backend implementation code instead of the shared
  contract; do not whitelist service/config/model imports to make the test pass.

## 14. Current limitations

- Evaluation-era frozen-chunk and manifest helpers remain dependencies of the indexing command. R07
  must relocate them without changing manifest or chunk-set semantics.
- `app.models` keeps query-contract aliases for compatibility. R07 owns evidence-based shim removal,
  not R06.
- UI styling, streaming, uploads, and conversation memory beyond bounded display history are outside
  Round 1.

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
12. Why are query DTOs direct aliases from `app.models` instead of compatibility subclasses?
13. Why does moving a Pydantic class require rechecking OpenAPI `$ref` values?
14. Which single backend package may Streamlit import after R06C2?
15. Why does API-only communication matter even inside one repository?

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
R06C1 then gives the public query schema a single shared owner. Application, citation policy, and
FastAPI use canonical contracts, while direct aliases keep every historical import and schema class
stable before the UI boundary is narrowed in the next slice.
R06C2 completes the inbound boundary: Streamlit imports only shared DTOs and communicates with runtime
capabilities exclusively over FastAPI. MockTransport and static import-graph tests demonstrate that
the refactor changes ownership without changing requests, failures, rendering, or history behavior.

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
R06C1 pre-move characterization  PASS — 83 tests, 1 warning
R06C1 focused Ruff               PASS
R06C1 focused query/schema suite PASS — 85 tests, 1 warning
R06C1 full Ruff Python 3.11.15   PASS
R06C1 full pytest Python 3.11.15 PASS — 397 tests, 1 warning
R06C1 Compose config             PASS
R06C1 Markdown links (25) / diff check PASS
R06C2 pre-move characterization  PASS — 53 tests, 1 warning
R06C2 focused Ruff               PASS
R06C2 focused UI/API suite       PASS — 54 tests, 1 warning
R06C2 full Ruff Python 3.11.15   PASS
R06C2 full pytest Python 3.11.15 PASS — 398 tests, 1 warning
R06C2 Compose config             PASS
R06C2 Markdown links (30) / diff check PASS
```

Proposed commit after user review:

```text
refactor: share query contracts with Streamlit
```

## 18. Status

`COMPLETE` — R06A–R06C2 are implemented and validated. FastAPI and Streamlit are thin inbound
adapters, supported CLIs have explicit operational ownership with stable shims, Phase 6 CLIs remain
archived and unsupported, and no R06D Docker path edit is required.

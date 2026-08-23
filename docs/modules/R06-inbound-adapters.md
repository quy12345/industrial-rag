# R06 — Thin inbound adapters

## 1. Goal and scope

R06 makes HTTP, command-line, and Streamlit entry points thin adapters over stable application and
domain contracts. It preserves every public path, schema, status, header, CLI contract, and UI
interaction.

R06A is implemented in this slice. It gives FastAPI one explicit dependency seam and moves
health/readiness route mapping out of the application factory. CLI and Streamlit cleanup remain later
R06 slices and are not described as completed work.

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

## 7. Important symbols and why they exist

- `QueryServiceProvider`: callable contract used by FastAPI dependency overrides.
- `ReadinessChecker`: read-only callable for validating frozen retrieval availability.
- `create_health_router`: constructs health routes around explicit settings and readiness dependencies.
- `create_app`: reusable FastAPI factory for production and offline contract tests.
- `UTF8JSONResponse`: preserves the explicit UTF-8 content type required by legacy clients.
- `require_query_auth`: fail-closed optional bearer authentication dependency.
- `query`: thin async HTTP mapping around synchronous `QueryService.execute`.

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

## 9. Design decisions and trade-offs

- Dependency symbols are direct aliases, so FastAPI override keys retain object identity.
- Health routes use a router factory because version and readiness behavior are app-instance inputs;
  module-level mutable state would make tests and multiple app instances unsafe.
- Query, auth, and readiness error payloads are not consolidated in R06A because changing their
  construction while moving boundaries would mix structural work with transport-policy redesign.
- HTTP DTOs remain in `app.models` during R06A. R06C owns separating stable shared query contracts and
  updating Streamlit imports in one reviewed slice.
- `app.main:app` remains unchanged because it is the documented Uvicorn and Compose entry point.

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

## 13. Common failures and debugging

- A missing dependency override usually means an alias was wrapped or recreated; verify object identity
  in `app.api.dependencies`.
- A changed OpenAPI response content type usually means `UTF8JSONResponse` is no longer the default.
- Readiness constructing a model during app creation means laziness was lost in bootstrap composition.
- A raw exception message in an HTTP response violates sanitization; inspect the route error mapper.
- A changed request-ID header means middleware validation or response attachment order drifted.

## 14. Current limitations

- Operational CLIs still mix entry-point parsing and reusable operations; R06B will classify and thin
  supported commands without changing old module invocations.
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

## 16. Interview summary

R06A turns FastAPI into a clearer inbound boundary without changing its public contract. One explicit
dependency module connects routes to canonical settings, application services, and bootstrap. The app
factory owns only construction, router mounting, overrides, and request correlation; health/readiness
mapping lives in its own route module. Direct aliases preserve dependency override identity, while
offline API characterization proves paths, schemas, statuses, headers, errors, and laziness stay stable.

## 17. Validation results and proposed commit

```text
R06A pre-move characterization    PASS — 44 tests, 1 warning
R06A focused Ruff                 PASS
R06A focused API suite            PASS — 46 tests, 1 warning
R06A full Ruff Python 3.11.15     PASS
R06A full pytest Python 3.11.15   PASS — 392 tests, 1 warning
R06A Compose config               PASS
R06A Markdown links (8) / diff check PASS
```

Proposed commit after user review:

```text
refactor: thin FastAPI adapters
```

## 18. Status

`IN_PROGRESS` — R06A is implemented and validated. R06 remains open for operational CLI and shared
HTTP/Streamlit contract slices.

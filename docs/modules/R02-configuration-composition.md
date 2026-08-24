# R02 — Canonical configuration and composition root

## 1. Goal and scope

R02 establishes one backend configuration owner, one immutable retrieval-contract owner, and one
composition root while preserving the public runtime. The module is implemented in three bounded
slices:

- **R02A — configuration and frozen contracts:** implemented in the current worktree;
- **R02B — query-service bootstrap:** implemented in the current worktree;
- **R02C — FastAPI app factory and readiness seam:** implemented in the current worktree.

R02A moves immutable Phase 7 corpus identity out of concrete retrieval construction, makes
`app.config` the canonical profile-resolution facade, adds an exact configuration matrix, and keeps
existing runtime/script imports working through explicit compatibility exports.
R02B moves query-service construction and caching from the application module to `app.bootstrap`.
FastAPI and the active query smoke CLI now consume that composition edge.
R02C adds `create_app`, injects query/readiness dependencies, and reduces `app.main` to the stable
ASGI export required by Uvicorn.

## 2. Position in the system

After R02, runtime composition flows in this direction:

```text
environment / .env
        ↓
app.config.Settings
        ↓
app.config.resolve_retrieval_runtime
        ↓
app.domain.retrieval_contracts
        ↓
app.bootstrap
   ├── lazy app.retrieval_runtime builder
   ├── LangChainOpenAIGenerator adapter
   ├── EvidenceGate + QueryService
   └── lazy readiness checker
        ↓
app.api.app.create_app / query smoke CLI
        ↓
app.main:app
```

FastAPI owns only HTTP concerns. Qdrant client construction and frozen-collection validation remain
behind the readiness callable composed by `app.bootstrap`.

## 3. Relevant background concepts

Configuration is mutable process input: endpoints, credentials, selected strategy, auth, and limits
arrive from defaults, environment variables, or `.env`. A frozen retrieval contract is different: it
is immutable package-owned identity for a verified corpus and index. Mixing these categories allows
an environment override to produce a combination that was never validated.

Atomic profile resolution means selecting the entire contract at once. Environment values may not
partially replace collection names, models, BM25 parameters, candidate budgets, or corpus identity.

A composition root is the single edge that knows both abstract application needs and concrete
adapters. `app.bootstrap` is now that edge for query execution. `app.query_service` no longer imports
the concrete generator, lazy retriever implementation, runtime builder, settings cache, or resolver.

## 4. Input, output, and contracts

### `Settings`

Inputs remain Pydantic settings loaded with the existing precedence rules. Explicit constructor
arguments override environment values; empty environment values are ignored; `get_settings()`
continues to return one cached instance until its cache is cleared.

### `resolve_retrieval_runtime`

Input:

```python
Settings
```

Output:

```python
tuple[Settings, FrozenRetrievalContract]
```

The returned settings are a copy. Non-profile settings such as Qdrant URL and API auth remain intact,
while all frozen retrieval fields are replaced atomically by Phase 7 values. Supported execution
combinations remain `union + rerank` and `sparse + no rerank`.

`Settings` rejects `phase6` during construction. The resolver deliberately uses the canonical
`phase7` identity rather than treating later mutation of a validated object as a new profile-selection
path. This preserves R00 readiness behavior and does not create a fallback for constructible settings.

### `build_query_service` and `get_query_service`

`build_query_service(settings)` returns a `QueryService` composed from one resolved settings/contract
pair. It constructs the lightweight generator adapter and evidence gate immediately, while wrapping
the heavy retrieval builder in `LazyQueryRetriever`. `get_query_service()` preserves the previous
zero-argument `lru_cache` singleton behavior over canonical `get_settings()`.

### `build_readiness_checker` and `create_app`

`build_readiness_checker(settings)` returns a zero-argument callable. Building the callable performs
no resolution, client construction, model loading, or network I/O; those operations begin only when
the readiness endpoint invokes it.

`create_app(settings=None, query_service_provider=None, readiness_checker=None)` builds the FastAPI
adapter. Optional providers allow deterministic tests without monkeypatching Qdrant internals. With
no arguments it uses the same canonical settings and query-service cache as before. `app.main:app`
remains the supported ASGI entry point.

## 5. Step-by-step data flow

1. Pydantic constructs `Settings` from explicit inputs, process environment, and the existing `.env`
   policy.
2. `Settings.retrieval_profile` can only be `phase7`.
3. `resolve_retrieval_runtime` asks `retrieval_contract_for("phase7")` for the complete immutable
   contract.
4. The resolver copies settings and replaces every frozen retrieval value together.
5. `validate_retrieval_settings` checks fusion-profile consistency, candidate budgets, model/vector
   identity, BM25 values, and strategy/reranker combinations.
6. `build_query_service` passes the same resolved settings to the evidence gate, generator adapter,
   and lazy retrieval closure.
7. The retriever closure keeps Qdrant/model/reranker construction deferred until the first retrieval.
8. `get_query_service` caches the lightweight composed service.
9. `build_readiness_checker` captures settings but defers resolution and Qdrant construction until
   `/ready` invokes the checker.
10. `create_app` configures UTF-8 responses, request-ID middleware, routes, auth settings, and optional
    test providers.
11. FastAPI dependency injection and `scripts/query_smoke.py` obtain services from `app.bootstrap`.
12. `app.main` exports `app = create_app()` without owning routes or infrastructure.
13. Old contract imports from `app.retrieval_runtime` resolve to canonical objects by identity.

Application construction and query-service composition do not create a Qdrant client, model,
reranker, or provider. The readiness checker creates only a Qdrant client when `/ready` invokes
it; model and provider construction remain absent from that path.

## 6. Responsibilities of changed files

- [`app/domain/retrieval_contracts.py`](../../app/domain/retrieval_contracts.py) owns immutable Phase 7
  document, corpus, index, model, and ranking-profile identity.
- [`app/domain/__init__.py`](../../app/domain/__init__.py) introduces the domain package without
  eagerly importing policies or adapters.
- [`app/config.py`](../../app/config.py) remains the public backend settings facade and now owns atomic
  profile application plus settings-to-contract validation.
- Historical `app/retrieval_runtime.py` (removed in R12B) owned concrete retrieval construction and
  explicitly re-exports moved symbols for compatibility until R07.
- [`app/bootstrap.py`](../../app/bootstrap.py) owns query-service construction, the cached service
  accessor, and lazy read-only readiness composition.
- Historical `app/query_service.py` (removed in R12B) owned orchestration only and depended on
  injected retrieval/generation contracts.
- [`app/api/query.py`](../../app/api/query.py) obtains its service dependency from the composition root.
- [`app/api/app.py`](../../app/api/app.py) owns FastAPI construction, middleware, health/readiness
  routes, and dependency overrides.
- [`app/main.py`](../../app/main.py) is the compatibility ASGI export only.
- Historical `scripts/query_smoke.py` (removed in R13A) used the same composition root as FastAPI.
- [`tests/test_config_contracts.py`](../../tests/test_config_contracts.py) owns configuration
  precedence, cache, exact contract, profile matrix, and compatibility characterization.
- [`tests/test_bootstrap.py`](../../tests/test_bootstrap.py) owns resolved-graph, lazy-construction,
  cache, and readiness characterization.
- [`tests/test_health.py`](../../tests/test_health.py) owns app-factory metadata, routes, request ID,
  readiness success, and sanitized readiness failure.
- [`tests/test_query_api.py`](../../tests/test_query_api.py) creates isolated apps through injected
  service providers instead of mutating global dependency overrides.
- [`docs/modules/R02-configuration-composition.md`](R02-configuration-composition.md) is the single
  learning document updated by all R02 slices.

## 7. Important symbols and why they exist

- `RetrievalProfile` names the only supported profile without reopening Phase 6 compatibility.
- `FrozenDocumentContext` stores trusted title/role metadata used by reranking and evidence rendering.
- `FrozenRetrievalContract` groups values that must be selected as one immutable unit.
- `PHASE7_RETRIEVAL_CONTRACT` is the active verified corpus/index identity.
- `retrieval_contract_for` is a pure lookup over supported profile identity.
- `resolve_retrieval_runtime` projects an immutable contract onto a settings copy.
- `validate_retrieval_settings` rejects partial profile mismatch and invalid execution combinations.
- `build_query_service` is the non-cached factory used to compose one explicit graph.
- `get_query_service` is the cached adapter-facing accessor.
- `ReadinessChecker` is the minimal callable boundary required by the HTTP adapter.
- `build_readiness_checker` composes read-only Qdrant identity validation lazily.
- `create_app` constructs an independently testable FastAPI adapter.
- Compatibility assignments in `app.retrieval_runtime` preserve existing import identity; R07 owns
  their removal after scripts and docs use canonical imports.

## 8. Before-and-after structure

Before R02:

```text
app/config.py             Settings + get_settings
app/retrieval_runtime.py  frozen contract + resolver + validation + builders
app/query_service.py      orchestration + concrete service factory + cache
```

After R02:

```text
app/config.py                       Settings + resolver + validation + cache
app/domain/retrieval_contracts.py   immutable Phase 7 identity
app/retrieval_runtime.py            builders + explicit compatibility exports
app/bootstrap.py                    query + readiness composition
app/query_service.py                injected application orchestration
app/api/app.py                      FastAPI factory + HTTP-only behavior
app/main.py                         compatibility ASGI export
```

## 9. Design decisions and trade-offs

- `Settings` stays in `app.config`; moving it would create unnecessary public-import churn.
- Contract values move, but algorithms do not. The contract temporarily imports the pure Phase 7
  fusion/query profile definitions from their flat modules; R04 owns their policy relocation.
- Resolution stays in the config facade rather than the domain module because applying environment
  settings is a configuration responsibility. The domain module does not import `Settings`.
- Existing callers are not mass-edited. Explicit identity-preserving exports avoid a repository-wide
  change and have an R07 removal owner.
- Query-service callers are deliberately updated because only two supported consumers exist. A shim
  from `app.query_service` back to bootstrap would create the wrong dependency direction or a cycle.
- The heavy retriever remains lazy exactly as before. Service caching and retrieval dependency
  caching remain separate responsibilities.
- Readiness uses a callable rather than a broad manager abstraction. Resolution and Qdrant access
  remain deferred until `/ready` is called, matching startup laziness.
- `create_app` overrides settings only when explicit settings are provided. The default production
  app keeps the cached `get_settings` dependency behavior used by auth.
- `app.main` intentionally exports only `app`; route functions are adapter internals, not public API.
- The exact contract snapshot is intentionally strict because these values identify a frozen corpus,
  not tuneable defaults.
- ADR-002 content is captured here for now. A separate `docs/adr/` record should be created only after
  the user approves that repository convention and the full R02 implementation confirms the design.

## 10. Tests and protected behavior

| Test | Protected behavior |
|---|---|
| `test_settings_precedence_and_cached_identity` | explicit-over-environment precedence and cached singleton identity |
| `test_empty_environment_value_keeps_the_canonical_default` | `env_ignore_empty` behavior |
| `test_phase7_contract_snapshot_is_exact_and_immutable` | every frozen document/index/model/ranking field and dataclass immutability |
| `test_profile_resolution_is_atomic_and_preserves_non_profile_settings` | no field mixing and no loss of unrelated settings |
| `test_supported_runtime_matrix_is_preserved` | union/rerank and sparse rollback combinations |
| `test_unsupported_runtime_matrix_fails_without_fallback` | explicit failure for unsupported combinations |
| `test_retired_phase6_profile_is_rejected_by_settings` | Phase 7-only construction |
| `test_retrieval_runtime_compatibility_exports_are_identity_preserving` | old imports point to canonical symbols, not copies |
| `test_build_query_service_uses_one_resolved_graph_and_keeps_retrieval_lazy` | one resolved graph feeds all components; heavy retriever stays deferred |
| `test_get_query_service_caches_one_service_for_cached_settings` | zero-argument singleton cache behavior |
| `test_readiness_checker_is_lazy_and_uses_the_frozen_phase7_contract` | no eager client, canonical collections/contract, malformed-state defense |
| `test_create_app_preserves_custom_metadata_routes_and_request_id` | factory metadata, route prefix, query route, middleware |
| `test_readiness_calls_the_injected_checker` | readiness adapter invokes its injected port |
| `test_readiness_returns_sanitized_503_when_qdrant_is_unavailable` | unchanged safe 503 mapping |
| existing runtime/health/smoke tests | frozen identity, malformed-state defense, laziness, no provider/model construction |

The focused suite caught and prevented an initial behavior drift where a post-validation monkeypatch
to `retrieval_profile` raised a new `ValueError` during readiness. The implementation was corrected;
production behavior was not changed to make the test pass.

## 11. Commands and expected results

R02 focused validation:

```text
python -m ruff check app/config.py app/domain app/retrieval_runtime.py tests/test_config_contracts.py
python -m pytest -q tests/test_config_contracts.py tests/test_retrieval_runtime.py
python -m pytest -q tests/test_config_contracts.py tests/test_retrieval_runtime.py tests/test_health.py tests/test_query_service.py tests/test_query_api.py tests/test_architecture_boundaries.py tests/test_runtime_characterization.py tests/test_phase7_operational_smoke.py
python -m pytest -q tests/test_bootstrap.py tests/test_query_service.py tests/test_query_api.py tests/test_phase7_operational_smoke.py tests/test_runtime_characterization.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_bootstrap.py tests/test_health.py tests/test_query_api.py tests/test_runtime_characterization.py tests/test_architecture_boundaries.py
```

Module-completion validation:

```text
python -m ruff check .
python -m pytest -q
docker compose config --quiet
git diff --check
```

All pytest commands remain offline. No real Qdrant, model, reranker, provider, calibration, or held-out
evaluation belongs in these checks.

## 12. Small usage example

Canonical configuration imports:

```python
from app.config import Settings, resolve_retrieval_runtime
from app.domain.retrieval_contracts import PHASE7_RETRIEVAL_CONTRACT

resolved, contract = resolve_retrieval_runtime(Settings())

assert contract is PHASE7_RETRIEVAL_CONTRACT
assert resolved.qdrant_collection == contract.dense_collection
assert resolved.dense_candidate_limit == contract.dense_candidate_limit
```

Existing callers may temporarily import these symbols from `app.retrieval_runtime`; they receive the
same objects.

Query adapters use the bootstrap accessor:

```python
from app.bootstrap import get_query_service

service = get_query_service()
assert service is get_query_service()
```

An isolated HTTP adapter can inject fakes without touching global state:

```python
from app.api.app import create_app

test_app = create_app(
    query_service_provider=lambda: fake_service,
    readiness_checker=lambda: None,
)
```

## 13. Common failures and debugging

- A contract snapshot failure means a frozen value changed. Do not update the snapshot unless a
  separately approved baseline change exists.
- A strategy-matrix failure may indicate a silent fallback or a resolver that rewrote
  `retrieval_strategy`/`rerank_enabled`.
- An import-cycle failure usually means domain contracts imported config or a concrete runtime
  builder. Keep settings application in `app.config`.
- A compatibility identity failure means a copied/redeclared contract exists. Re-export the canonical
  object instead.
- If a model or Qdrant client is constructed during `build_query_service`, the lazy retrieval closure
  has been bypassed.
- If FastAPI dependency overrides stop working, confirm the route and tests use the exact
  `app.bootstrap.get_query_service` function object.
- If app import starts contacting Qdrant, confirm `build_readiness_checker` returns a closure and does
  not resolve or create its client until invocation.
- If a custom app loses `/query`, inspect `test_app.openapi()["paths"]`; current FastAPI represents an
  included router with an internal route object that does not expose `.path` directly.
- A local `.env` affecting tests means `tests/conftest.py` no longer disabled env-file loading or
  cleared matching environment variables.

## 14. Current limitations

- Frozen contract types temporarily reference pure policies in `app.phase7_optimization` and
  `app.query_expansion`; R04 will move those policy responsibilities.
- Compatibility exports remain in `app.retrieval_runtime` until R07 verifies all consumers.
- No standalone ADR exists because the `docs/adr/` convention has not been approved yet.

## 15. Self-check questions

1. Why are Qdrant URL and collection identity owned by different concepts?
2. Which settings may remain environment-controlled after frozen profile resolution?
3. Why does the resolver copy `Settings` instead of mutating the cached object?
4. Why is a compatibility export preferable to duplicating a contract constant?
5. Why does readiness resolution occur when the checker is called rather than when the app is built?

## 16. Interview summary

R02A separated mutable process configuration from immutable verified-corpus identity. R02B then
moved concrete query-service wiring into a composition root, leaving `QueryService` focused on its
use case. R02C made FastAPI independently constructible and moved readiness infrastructure behind a
lazy callable, while preserving `app.main:app`. Exact offline tests prove precedence, cache, profile
matrix, one resolved object graph, routes, middleware, and lazy external construction.

## 17. Validation results and proposed commit

Final R02 validation results:

```text
Pre-change query/runtime/health/R01 baseline       PASS — 53 tests, 1 warning
Config + retrieval runtime                         PASS — 19 tests
R02A focused contract suite                        PASS — 68 tests, 1 warning
R02B focused bootstrap/query/API/smoke suite       PASS — 46 tests, 1 warning
R02C focused app/bootstrap/HTTP suite              PASS — 28 tests, 1 warning
Focused Ruff                                       PASS
Full Ruff                                          PASS
Full pytest Python 3.11.15                         PASS — 340 tests, 1 warning
docker compose config --quiet                      PASS
Markdown links (15 local targets) / git diff check PASS
```

Proposed commits for the eventual reviewed module history:

```text
test: characterize runtime configuration selection
refactor: establish canonical runtime composition
refactor: introduce the FastAPI application factory
```

No commit is created without a separate explicit user request.

## 18. Status

`COMPLETE` — R02A–R02C are implemented. Focused checks, full Ruff, the full offline Python 3.11
suite, Compose validation, local Markdown links, and diff checks pass.

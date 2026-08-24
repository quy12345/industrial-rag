# Script lifecycle and supported entry points

The script tree separates current adapters from historical provenance. A file being retained does
not imply that it is supported or safe to execute.

## `operations`

`scripts/operations/` owns current operational implementations:

- corpus audit;
- guarded Phase 7 indexing;
- ingestion preview;
- bounded query smoke;
- read-only query-runtime validation.

Stable top-level modules re-export only `main`, preserving the documented
`python -m scripts.<command>` paths. Reusable domain or infrastructure behavior belongs under `app`,
not inside these adapters.

## `evaluation`

`scripts/evaluation/` owns supported offline/integration evaluation adapters:

- Phase 7 dataset validation;
- provider-free retrieval-closure execution;
- the approval-gated Phase 7 E2E evaluator.

Their reusable schemas and metrics live in `evaluation/`. Supported script implementations must not
import private helpers from another script implementation.

## Phase 7 E2E command

`scripts/evaluation/evaluate_phase7_e2e.py` owns the current implementation. The supported
`scripts/evaluate_phase7_e2e.py` path is a thin `main` shim like the other current commands and is
scheduled for the common R13 CLI hard cut.

R12A introduced artifact schema v6 and source-identity v2 so current artifacts hash canonical
application, domain, infrastructure, and evaluation owners. Historical v5 artifacts remain
immutable, and their checkpoints fail closed under the new identity.

R12B removed the old root `app.*` facades. Supported commands import canonical owners only;
archived workflows remain provenance and are not guaranteed to import under the current tree.

This evaluator can call a real provider. Do not execute it without separate provider/data-egress
approval. Held-out execution remains governance-sensitive, and exposed held-out v2 must not be used
for tuning.

## `archive`

`scripts/archive/phase6/` preserves the retired single-manual development workflow.
`scripts/archive/phase7/` preserves completed construction, migration, calibration, diagnostics,
benchmark, and readiness workflows.

Archive policy:

- source is retained for provenance rather than deleted;
- old top-level module paths are intentionally unavailable;
- archived tools are unsupported and excluded from current runbooks;
- archive code may read real data, initialize models, call providers, mutate Qdrant, or overwrite
  artifacts when invoked, so it must not be run casually;
- no silent compatibility shim redirects an archived command into current behavior.

Read the phase-specific archive README before inspecting or executing historical code.

## Supported top-level inventory

```text
audit_phase7_corpus.py
evaluate_phase7_e2e.py
evaluate_phase7_retrieval_closure.py
index_phase7_corpus.py
ingest_preview.py
query_smoke.py
validate_phase7_dataset.py
validate_query_runtime.py
```

The inventory and import boundaries are enforced by offline architecture tests.

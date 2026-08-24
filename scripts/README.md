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

Invoke these commands through `python -m scripts.operations.<command>`. R13A removed the old
top-level operational shims so this package is both the implementation owner and the supported
entry point. Reusable domain or infrastructure behavior belongs under `app`, not inside these
adapters.

## `evaluation`

`scripts/evaluation/` owns supported offline/integration evaluation adapters:

- Phase 7 dataset validation;
- provider-free retrieval-closure execution;
- the approval-gated Phase 7 E2E evaluator.

Their reusable schemas and metrics live in `evaluation/`. Supported script implementations must not
import private helpers from another script implementation.

## Phase 7 E2E command

`scripts/evaluation/evaluate_phase7_e2e.py` owns the current implementation. The temporary
`scripts/evaluate_phase7_e2e.py` path remains a thin `main` shim until R13B removes all three
evaluation shims together.

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

## Supported command inventory

```text
scripts.operations.audit_phase7_corpus
scripts.operations.index_phase7_corpus
scripts.operations.ingest_preview
scripts.operations.query_smoke
scripts.operations.validate_query_runtime
scripts.validate_phase7_dataset              # temporary shim until R13B
scripts.evaluate_phase7_retrieval_closure    # temporary shim until R13B
scripts.evaluate_phase7_e2e                  # temporary shim until R13B
```

Canonical evaluation implementations already live under `scripts.evaluation`. The inventory,
removed operational paths, and import boundaries are enforced by offline architecture tests.

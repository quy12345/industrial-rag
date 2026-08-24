# ADR-005: Compatibility boundaries and source identity v2

- Status: Accepted
- Date: 2026-08-24
- Scope: R08 baseline for the R09-R14 surface-simplification sequence

## Context

Round 1 established canonical application, domain, infrastructure, and evaluation packages, but the
repository root still contains compatibility modules and two exact source-identity anchors. These
paths make the active code appear larger than it is and allow tests to confuse import identity with
user-visible behavior. The supported script surface has the same problem: eight commands are
implemented through fifteen active files.

Removing the paths without recording their contracts would make later cleanup difficult to review.
Keeping every path indefinitely would preserve accidental structure and prevent the active surface
from becoming concise.

## Decision

Use three explicit categories:

1. A **canonical public interface** owns current behavior and is the only import path used by new
   production or evaluation code.
2. A **compatibility path** temporarily re-exports behavior for a known repository consumer. It is
   structural debt, not an additional owner, and must have an explicit removal module.
3. A **source-identity anchor** is protected byte-for-byte because a historical artifact records its
   Git blob or hashes paths selected by it.

R08 records the current compatibility inventory and forbidden dependency edges as exact sets. R09
through R11 introduce canonical owners while the two source anchors remain unchanged. R12 performs
one explicit provenance migration and removes the compatibility paths. R13 then hard-cuts the old
repository-local CLI module names instead of adding another deprecation layer.

The R12 E2E format will use artifact schema version 6 and source-identity version 2. Historical v5
artifacts remain immutable, old checkpoints fail closed, and Git history retains the original blobs.
The migration changes provenance metadata and import paths only; it does not authorize an evaluation
run, a provider call, retrieval tuning, or a frozen-data change.

## Current source anchors

```text
scripts/evaluate_phase7_e2e.py  7ce4dc9c180fd675eb89e5c06844766b664c6b69
app/evaluation_e2e.py           b8be722d43bc34c8bec00dfc2574d0a6341ab738
```

R08 verifies these values by reconstructing Git blob IDs directly from file bytes. The test does not
need Git metadata and therefore also works in a source-only validation container.

## Consequences

- New dependency violations fail a complete layer-matrix test instead of relying only on a list of
  runtime roots.
- Removing an acknowledged debt edge also requires shrinking the temporary allowlist, so the test
  cannot silently preserve obsolete exceptions.
- R12 and R13 contain intentional internal breaking changes. FastAPI, Streamlit, frozen retrieval,
  and the corresponding CLI argument/output behavior remain unchanged.
- Archived Phase 6 and Phase 7 workflows keep their historical paths. The hard cut applies only to
  active compatibility imports and supported command entry points.
- No production file changes during R08; the module only strengthens observation and governance.

## Rejected alternatives

- Keeping all compatibility modules permanently was rejected because it would keep ownership
  ambiguous.
- Deleting anchors before a versioned migration was rejected because it would obscure artifact
  provenance.
- Copying historical evaluator files into the active tree was rejected because it would duplicate
  implementation rather than preserve history.
- Adding a command dispatcher or dependency-injection framework was rejected because the current
  repository-local commands do not need either abstraction.

## Validation

R08 protects the decision with offline source-pin tests, CLI parser characterization, Pydantic schema
checks, a golden query execution, and an AST dependency matrix. Each later module must also compare
the protected PDF/artifact hash manifest and run the full offline suite before review.

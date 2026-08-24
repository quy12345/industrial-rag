# ADR-004: Evaluation and script lifecycle

- Status: Accepted
- Date: 2026-08-24
- Scope: Round 1 portfolio cleanup

## Context

Evaluation schemas, research helpers, operational adapters, and one-off Phase 7 workflows had grown
across `app/` and top-level `scripts/`. Production modules could appear to own evaluation behavior,
and historical commands looked equally supported as current runtime tools. Some sanitized artifacts
also hash exact evaluator source files, so blindly moving every module would break provenance.

## Decision

Use this dependency direction:

```text
inbound API / CLI / Streamlit
             -> application
             -> domain contracts and ports
             <- infrastructure adapters

evaluation -> public application/domain interfaces
production -X-> evaluation or scripts
```

Evaluation schemas, replay, candidate audits, and metric aggregation have canonical ownership in
`evaluation/`. Current CLI implementations live in either `scripts/operations/` or
`scripts/evaluation/`; stable top-level commands are thin `main` shims.

Completed research, migration, calibration, diagnostics, benchmark, and readiness workflows move to
`scripts/archive/`. They are archived, not dropped, because their source explains historical
artifacts and engineering decisions. Former top-level imports fail instead of silently redirecting
users to unsupported behavior.

At Round 1 closure, two source-identity anchors remained in place:

- `scripts/evaluate_phase7_e2e.py`, blob
  `7ce4dc9c180fd675eb89e5c06844766b664c6b69`;
- `app/evaluation_e2e.py`, blob
  `b8be722d43bc34c8bec00dfc2574d0a6341ab738`.

R12 later versioned E2E provenance, moved both implementations to canonical owners, and removed the
remaining compatibility layer. Git history retains the exact anchor blobs; ADR-005 records the
versioned migration and hard cut.

## Consequences

- Production runtime has a mechanically enforced one-way boundary from evaluation and scripts.
- Supported commands have explicit ownership and stable user-facing invocation paths.
- Historical code remains inspectable without cluttering the active command surface.
- Source pins limit otherwise desirable moves; changing them requires a separate versioned artifact
  migration, outside Round 1.
- Archive source is preserved for provenance, but current-tree import or `--help` compatibility is
  not guaranteed; archived workflows are not current product contracts.

## Rejected alternatives

- Deleting all one-off scripts would make the repository shorter but erase provenance.
- Keeping every old top-level shim would falsely advertise unsupported workflows.
- Moving pinned E2E files and rewriting artifact identity would mix structural refactoring with a
  benchmark migration.
- Introducing a plugin framework, microservices, or a dependency-injection framework has no current
  use case in this modular monolith.

## Validation

Architecture tests enforce production/evaluation direction, supported script inventory, thin shim
identity, removed facades, and the absence of private cross-script imports. Full offline pytest,
Ruff, Compose configuration validation, local Markdown links, source pins, and frozen-data scope are
checked at Round 1 closure.

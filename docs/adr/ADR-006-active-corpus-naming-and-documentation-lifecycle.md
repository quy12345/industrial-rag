# ADR-006: Active corpus naming and documentation lifecycle

- Status: Accepted
- Date: 2026-08-27
- Scope: R14-R17 active-surface simplification

## Context

The repository had one active corpus but still exposed development-stage terminology through
production symbols, a runtime profile selector, and five supported command paths. Documentation also
described the same architecture across module logs, walkthroughs, a plan, an audit, and current
runbooks. The repetition made a completed project look provisional and allowed current facts to
drift.

Some `phase7` values cannot be renamed safely: Qdrant collection names, frozen dataset/artifact paths,
record IDs, and archive directories are physical or historical identities. Renaming those would be a
data migration or provenance rewrite rather than a source cleanup.

## Decision

Use semantic names for reusable behavior and concrete product names for frozen product identity:

- `FusionProfile`, `CorpusIndexingService`, and `EvaluationItem` describe reusable responsibilities;
- `ATV320_RETRIEVAL_CONTRACT` and `ATV320_FUSION_PROFILE` identify the active product contract;
- the active contract ID is `atv320-2025-04-v1`;
- there is no runtime profile selector because only one product pipeline is supported.

Supported command paths use task names (`index_corpus`, `validate_dataset`, `evaluate_e2e`) rather
than milestone names. This is a hard cut: old phase-named modules have no compatibility shims. Frozen
physical names and historical archives remain unchanged.

The active evaluation package has three owners only:

- `evaluation/dataset.py` for dataset contracts;
- `evaluation/retrieval.py` for current retrieval metrics;
- `evaluation/e2e.py` for query-record scoring and quality gates.

The canonical path migration versions diagnostic provenance instead of silently overwriting it. E2E
artifacts use schema 7 and source identity 3; old checkpoints fail closed. Retrieval evaluation uses
schema 2. Existing versioned artifacts remain immutable.

Documentation has one owner per concern:

- `README.md`: project entry point and quickstart;
- `docs/ARCHITECTURE.md`: current structure, flows, configuration, and invariants;
- `docs/OPERATIONS.md`: commands, Docker, validation, and troubleshooting;
- `docs/PROJECT_JOURNEY.md`: chronology and compact bug-to-fix evidence;
- `scripts/README.md`: command inventory and archive lifecycle;
- ADRs: long-lived architecture decisions.

After unique facts are migrated, superseded module logs, walkthroughs, plans, audits, and codebase
overviews are removed from the current tree. Git history preserves their original form; no duplicate
documentation-history tree is created.

## Consequences

- Active modules explain product responsibility without implying an unfinished development phase.
- One frozen contract cannot be partially switched through environment configuration.
- Old Python symbols and command module paths are intentionally breaking internal changes.
- API, Streamlit, retrieval behavior, models, thresholds, corpus identity, Qdrant storage, and frozen
  datasets are unchanged.
- Historical artifacts remain interpretable because physical paths and IDs are not rewritten.
- Current documentation is shorter and has an explicit update location for each kind of fact.

## Rejected alternatives

- Renaming Qdrant collections or frozen directories would require migration and would invalidate
  provenance.
- Keeping a one-value profile selector would preserve accidental complexity without a second product
  contract.
- Compatibility shims would make removed milestone names appear supported indefinitely.
- Copying all old documents into a history directory would preserve the duplication in a new place.
- Mass-rewriting archive imports would turn provenance snapshots into an unsupported second product.

## Validation

Architecture tests enforce the supported command inventory, removed phase-named module paths,
production/evaluation direction, source-path existence, provenance versions, and absence of phase
terminology from active filenames and public identifiers. Full Ruff, offline pytest, Compose config,
Markdown-link validation, and `git diff --check` close the migration.

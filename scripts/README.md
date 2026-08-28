# Script lifecycle

The script tree contains thin supported command adapters and explicit historical archives. A file
retained for provenance is not necessarily importable or safe to execute.

## Supported inventory

```text
scripts.operations.audit_corpus
scripts.operations.index_corpus
scripts.operations.ingest_preview
scripts.operations.query_smoke
scripts.operations.validate_query_runtime
scripts.evaluation.validate_dataset
scripts.evaluation.evaluate_retrieval
scripts.evaluation.evaluate_e2e
```

There are no supported command modules directly under `scripts/`, and the former phase-named module
paths intentionally raise `ModuleNotFoundError`. Use `python -m <module> --help` to inspect the
current parser contract.

## Operations

`scripts/operations/` owns five current adapters:

- audit frozen corpus identity;
- preview or explicitly index the two-manual corpus;
- preview ingestion output;
- run a bounded query smoke check;
- validate the configured query runtime.

Reusable indexing, retrieval, and artifact behavior belongs in `app/`. These commands may initialize
Docling or retrieval models, access Qdrant, and write files. `index_corpus` can write Qdrant unless
`--preview-only` is supplied. See [Operations](../docs/OPERATIONS.md) before using them.

## Evaluation

`scripts/evaluation/` owns three current adapters:

- validate frozen dataset splits;
- evaluate retrieval and write the versioned retrieval artifact;
- run approval-gated end-to-end generation evaluation.

Reusable current contracts live in `evaluation/dataset.py`, `evaluation/retrieval.py`,
`evaluation/e2e.py`, and `evaluation/semantic.py`. Supported command implementations do not import
private helpers from another command. E2E evaluation can call a real provider and must not run
without explicit provider and data-egress approval. Its optional `--semantic-judge ragas` mode uses
the separate evaluator image and requires an independent OpenRouter approval token; the default
`none` mode does not import Ragas. The OpenRouter gateway is pinned to OpenAI upstream endpoints with
provider fallback disabled. The `test` choice is the public regression split; `heldout-v2` is a
separate sealed private split with path and manifest containment checks. Exposed held-out v2 is
regression evidence, not tuning data. See [Operations](../docs/OPERATIONS.md) for the two-run
calibration preflight and sanitized v8 artifact contract.

## Archive

`scripts/archive/phase6/` preserves the retired single-manual development workflow.
`scripts/archive/phase7/` preserves completed dataset construction, migration, calibration,
diagnostics, benchmark, replay, and readiness workflows.

Archive policy:

- source is retained for provenance, not support;
- archived imports may no longer resolve against the current tree;
- no compatibility shim redirects an old command to current behavior;
- archived tools may read sensitive data, initialize models, call providers, mutate Qdrant, or
  overwrite artifacts;
- reconstruct the recorded Git revision if an explicitly approved historical reproduction is ever
  required.

The `phase6` and `phase7` directory labels remain historical identities. Active commands and reusable
runtime symbols use semantic names or the concrete ATV320 product identity instead.

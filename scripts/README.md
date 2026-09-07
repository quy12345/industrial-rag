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
scripts.evaluation.validate_scorecard
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

`scripts/evaluation/` owns four current adapters:

- validate either the legacy frozen pair or the development/confirmation optimization pair;
- evaluate the frozen legacy retrieval baseline or one of six preregistered development-only
  retrieval variants, with trace-only dense/sparse top-100 ceiling probes limited to development,
  apply the canonical evaluation-only evidence top five, and write a sanitized schema-v5
  checkpoint/artifact;
- run approval-gated end-to-end generation and Ragas evaluation.
- validate a sanitized scorecard against private human labels.

Reusable current contracts live in `evaluation/dataset.py`, `evaluation/retrieval.py`,
`evaluation/e2e.py`, `evaluation/semantic.py`, `evaluation/review_inputs.py`, and
`evaluation/human_review.py`. Live E2E evaluation uses the baseline
production generation contract and the three core Ragas metrics. Generator egress and Ragas egress
require independent dataset-specific approval tokens. Optional Answer Relevancy and Context
Precision diagnostics are enabled with `--ragas-diagnostics`.

The compact schema-v19 scorecard exposes Ragas results under their standard names: Factual
Correctness, Faithfulness, Context Recall, Answer Relevancy, and Context Precision without
reference. It reports deterministic answerable-response, unsafe-answer, and citation-reference
validity rates separately; none is relabeled as a Ragas metric or merged into a composite score. It
fixes the final evidence boundary at five and disables retrieval trace capture. Answerable items
require independent natural reference answers with a separate approved review status; expected
fact aliases remain audit annotations and are not rendered as Ragas references. Use
`evaluate_retrieval` for provider-free deep stage diagnostics. Schema v19 is `BASELINE_ONLY`: it
has no thresholds and a valid run exits zero regardless of score.
Development, sealed confirmation, and exposed held-out v2 are the only E2E dataset choices. The
latest executed held-out artifact remains v8 until a newer run is explicitly approved and
completed.

`evaluate_e2e --replay-artifact ... --replay-review ...` re-judges fixed development answers
without calling generation, Qdrant, or retrieval models. It requires only judge approval and explicit
output/checkpoint paths. New private review packs include an identity-bound `inputs.jsonl` captured
before judging, so interrupted review runs can reuse generation. See [Operations](../docs/OPERATIONS.md)
for replay provenance, cost accounting and legacy import limitations. `validate_scorecard` emits
assessment v2 (`REVIEW_REQUIRED`, exit zero for valid reports), with continuous F1 grouped by human
labels, not `F1 == 1` answer-correctness verdicts.

`evaluate_e2e --scope-replay-artifact ... --scope-replay-inputs ...` is development-only and
provider-free. It validates a live artifact and private captured inputs, then emits sanitized
deterministic evidence-scope decisions with separate origin and target policy identities. It does
not accept a judge or generator approval token.

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

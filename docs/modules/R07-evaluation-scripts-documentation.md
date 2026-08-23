# R07 — Evaluation isolation, script archive, and documentation closure

## 1. Goal and scope

R07 completes Round 1 by giving offline evaluation a canonical owner, classifying every command,
archiving completed one-off workflows, removing expired facades, and aligning current documentation
with the active two-manual runtime.

The module preserves retrieval, reranking, evidence, generation, citation, API, dataset, and artifact
behavior. It does not run evaluation, tune algorithms, read raw held-out payloads, call a provider,
download a model, access Qdrant, re-index, or mutate frozen files.

## 2. Position in the system

```text
supported CLI shim
  -> scripts.operations / scripts.evaluation implementation
  -> app application/domain/infrastructure or evaluation public API

offline evaluator
  -> evaluation schemas and metrics
  -> public application/domain records

production runtime  -X-> evaluation / scripts / archived workflows
```

R07 is the final module because it can remove compatibility paths only after R01–R06 stabilize the
runtime interfaces and inbound adapters.

## 3. Relevant background concepts

- A **canonical owner** is the single module where behavior is implemented; compatibility modules
  may re-export it but do not duplicate it.
- A **thin CLI shim** exposes `main` at a stable module path while implementation lives in a clearly
  classified package.
- **Archive, not drop** keeps completed research/migration source as provenance while removing it
  from the supported command surface.
- A **source-identity anchor** cannot move byte-for-byte when an artifact hashes that exact file.
- An **architecture guard** is an offline AST/import test that makes dependency direction executable.

## 4. Input, output, and contracts

Evaluation inputs are typed dataset rows, frozen chunk metadata, sanitized query executions, or
sanitized reranker snapshots. Outputs are deterministic validation results, ranks, aggregate metrics,
or sanitized artifacts. Evaluation must never influence production retrieval decisions.

Supported CLI contracts preserve parser options, exit codes, sanitized error behavior, and top-level
`python -m scripts.<name>` invocation. Archived top-level paths intentionally raise
`ModuleNotFoundError`; this prevents a silent fallback to unsupported behavior.

The exact source pins remain:

```text
scripts/evaluate_phase7_e2e.py  7ce4dc9c180fd675eb89e5c06844766b664c6b69
app/evaluation_e2e.py           b8be722d43bc34c8bec00dfc2574d0a6341ab738
```

## 5. Step-by-step data flow

1. A supported top-level CLI shim imports one canonical `main`.
2. `scripts.operations` coordinates current operational use cases; `scripts.evaluation` coordinates
   current evaluation use cases.
3. Reusable evaluation behavior comes from `evaluation`, never another script implementation.
4. Evaluation consumes public application/domain records and may use infrastructure readers where
   an offline workflow explicitly requires them.
5. Production import guards reject any path back to evaluation, scripts, datasets, or artifacts.
6. Completed workflows live under `scripts.archive`; current docs do not advertise them as commands.

## 6. Responsibilities of changed files

| Path | Responsibility after R07 |
| --- | --- |
| [`evaluation/retrieval.py`](../../evaluation/retrieval.py) | Retrieval dataset schemas, validation, ranks, and metrics. |
| [`evaluation/phase7_dataset.py`](../../evaluation/phase7_dataset.py) | Phase 7 dataset schemas, hashes, qrel closure, and review state. |
| [`evaluation/replay.py`](../../evaluation/replay.py) | Sanitized snapshot validation and deterministic policy replay. |
| [`evaluation/candidate_audit.py`](../../evaluation/candidate_audit.py) | Historical candidate-pool diagnostics. |
| [`evaluation/retrieval_closure.py`](../../evaluation/retrieval_closure.py) | Shared closure aggregation and per-language metrics. |
| [`app/infrastructure/corpus_artifacts.py`](../../app/infrastructure/corpus_artifacts.py) | Active corpus identity, JSONL parsing, hashes, and atomic file operations. |
| [`scripts/operations/`](../../scripts/operations/) | Canonical supported operational adapters. |
| [`scripts/evaluation/`](../../scripts/evaluation/) | Canonical supported evaluation adapters. |
| [`scripts/archive/phase6/`](../../scripts/archive/phase6/) | Unsupported retired single-manual workflows. |
| [`scripts/archive/phase7/`](../../scripts/archive/phase7/) | Unsupported completed Phase 7 workflows. |
| [`scripts/README.md`](../../scripts/README.md) | Current command taxonomy and safety boundary. |
| [`tests/test_architecture_boundaries.py`](../../tests/test_architecture_boundaries.py) | Import direction, shim inventory, archive, and source-pin guards. |

`app.evaluation`, `app.phase7`, and `app.phase7_optimization` remain documented compatibility or
source-identity anchors. `app.candidate_audit` and `app.phase7_replay` were removed after consumers
moved to canonical evaluation modules.

## 7. Important symbols and why they exist

- `EvaluationCase`: stable retrieval-evaluation input contract.
- `evaluate_cases`: provider-free retrieval callback evaluator.
- `aggregate_metrics`: deterministic retrieval metric aggregation.
- `validate_phase7_dataset`: validates schema, IDs, qrels, and review state without runtime access.
- `load_sanitized_snapshot` and `replay_rank_prior`: reproduce rank-only policy decisions without a
  model, raw question, or chunk content.
- `aggregate_candidate_audit`: summarizes historical pool coverage.
- `aggregate_closure_rows`: canonical provider-free closure report.
- `aggregate_closure_rows_by_language`: public grouped closure aggregation; replaces a private CLI
  helper that archived consumers previously imported.

## 8. Before-and-after structure

```text
Before
  app/evaluation*.py + app/phase7*.py own offline behavior
  scripts/*.py mix operations, evaluation, research, migrations, and receipts
  scripts import private helpers from other scripts

After
  evaluation/* owns reusable offline behavior
  app/infrastructure/corpus_artifacts.py owns active corpus file identity
  scripts/operations/* owns supported operations
  scripts/evaluation/* owns supported evaluation adapters
  scripts/archive/* preserves unsupported provenance
  eight documented top-level entry points remain
```

The active top-level inventory is corpus audit, indexing, ingestion preview, query smoke, runtime
validation, dataset validation, retrieval closure, and the pinned E2E evaluator.

## 9. Design decisions and trade-offs

R07 prefers explicit ownership over a broad “utilities” package. Metrics belong to evaluation,
corpus I/O belongs to infrastructure, and CLI modules only coordinate calls.

Archiving keeps about 2,253 lines of completed calibration/research code available for provenance
without presenting it as production surface. Removing old top-level modules is intentional and safer
than silent shims because archived commands can mutate data or access external systems.

The pinned E2E CLI/evaluator remain in historically awkward locations. Preserving artifact identity
is more important than cosmetic uniformity in a behavior-preserving round. A future move requires a
versioned artifact migration, not an unrecorded structural edit. The rationale is recorded in
[ADR-004](../adr/ADR-004-evaluation-and-script-lifecycle.md).

## 10. Tests and protected behavior

| Test area | Behavior protected |
| --- | --- |
| Retrieval/dataset evaluation tests | schemas, hashes, qrels, ranks, aggregate metrics, error contracts |
| E2E evaluation tests | answer facts, citation outcomes, abstention, sanitization, source identity |
| Replay tests | malformed snapshots, deterministic order, policy preservation |
| Candidate/closure tests | pool coverage and exact aggregate formulas |
| CLI tests | parser, `main` shim identity, outputs, and exits for supported commands |
| Architecture tests | production cannot import evaluation/scripts; canonical scripts do not cross-import |
| Archive table guard | archived paths exist and former top-level paths are absent |

Private tests whose only contract was an unsupported one-off CLI helper were removed. Domain/runtime
policy tests remain. Suite size may change; preserved behavior coverage is the acceptance criterion.

## 11. Commands and expected results

Safe default validation:

```powershell
python -m ruff check .
python -m pytest -q
docker compose config --quiet
git diff --check
```

Expected: Ruff and Compose exit 0, offline pytest passes with only the known third-party
Starlette/TestClient warning, and the diff check is empty. These commands must not call a provider,
download a model, or connect to real Qdrant.

Parser-only checks such as the following must exit 0 without executing integration work:

```powershell
python -m scripts.validate_phase7_dataset --help
python -m scripts.evaluate_phase7_retrieval_closure --help
```

## 12. Small usage example

Use the supported offline validator through its stable top-level path:

```powershell
python -m scripts.validate_phase7_dataset --help
```

For reusable code, import from the canonical owner:

```python
from evaluation.retrieval_closure import aggregate_closure_rows

metrics = aggregate_closure_rows(rows)
```

Do not import helper functions from `scripts.*` into application or evaluation code.

## 13. Common failures and debugging

- `ModuleNotFoundError` for an old calibration path is expected; locate provenance under
  `scripts/archive/phase7/` and do not add a shim.
- A supported shim exposing anything beyond canonical `main` indicates adapter drift.
- A source-pin mismatch means a protected E2E file changed; stop instead of updating an artifact.
- An architecture-test failure usually identifies the exact forbidden import edge. Move reusable
  behavior to its owner instead of suppressing the guard.
- Qdrant/model/provider errors during default pytest indicate an invalid test boundary; unit tests
  must use fakes or in-memory adapters.
- A broken historical link should target the archived path, not restore an obsolete command.

## 14. Current limitations

- Source identity prevents relocation of `app.evaluation_e2e` and the top-level E2E CLI.
- `app.evaluation`, `app.phase7`, and `app.phase7_optimization` remain compatibility anchors.
- Archived source is unsupported and may depend on historical artifacts or integrations.
- Held-out v2 is exposed regression evidence and cannot support further tuning.
- Evaluation algorithm improvements, benchmark v3, model changes, latency work, re-indexing, and
  collection migration are deferred to Round 2.

## 15. Self-check questions

1. Why may evaluation import application/domain interfaces while production cannot import evaluation?
2. What distinguishes a thin supported shim from an archived command?
3. Why were completed one-off workflows archived instead of deleted?
4. Which two files are source-identity anchors, and what would authorize moving them?
5. Where should a reusable metric live if two CLIs need it?
6. Which validations prove that structural cleanup did not alter frozen runtime behavior?

## 16. Interview summary

R07 turned an ambiguous mix of runtime, evaluation, research, and migration scripts into an explicit
lifecycle. Offline evaluation now has canonical package ownership and consumes public runtime
contracts in one direction. Supported commands are thin, historical workflows remain inspectable but
unavailable at old top-level paths, and architecture tests enforce the boundary. Two intentionally
unmoved files demonstrate a pragmatic trade-off: provenance and artifact identity outrank cosmetic
package purity.

## 17. Validation results and proposed commit

Key implementation history:

| Outcome | Commit |
| --- | --- |
| Isolate retrieval evaluation and replay | `9f8ff89`, `8ab8d5a` |
| Separate Phase 7 dataset and corpus artifacts | `87a0ce1`, `b9458bc` |
| Classify supported corpus audit | `09a3352` |
| Archive completed dataset/fact/evaluation workflows | `808907b` through `6240aec` |
| Archive weighted rerank and calibration workflows | `44821d0`, `dcf0261` |
| Finalize evaluation and script boundaries | `99ff9f7` |

Last code-checkpoint validation:

| Check | Result |
| --- | --- |
| Focused evaluation/boundary pytest | PASS — `55 passed` |
| Supported closure shim/canonical `--help` | PASS — identical four-option contract |
| Full Ruff | PASS — `All checks passed!` |
| Full offline Python 3.11 pytest | PASS — `384 passed, 1 warning` |
| Docker Compose config | PASS |
| Source pins and frozen scope | PASS — exact blobs retained; no frozen-data/artifact diff |
| Local links and `git diff --check` | PASS |

Documentation-closure validation:

| Check | Result |
| --- | --- |
| Full Ruff | PASS — `All checks passed!` |
| Full offline Python 3.11 pytest | PASS — `384 passed, 1 warning` |
| Docker Compose config | PASS |
| All local Markdown links | PASS — 136 targets checked |
| Inline-code source paths in changed docs | PASS — 266 paths checked |
| R07 structure | PASS — exactly 18 required sections |
| Source pins | PASS — both exact Git blobs unchanged |
| Frozen/runtime scope and `git diff --check` | PASS — documentation-only diff |

Proposed closure commit:

```text
docs: close round 1 portfolio cleanup
```

## 18. Status

`COMPLETE` — R07 and Round 1 have canonical evaluation ownership, an eight-command supported script
surface, archived historical workflows, current runtime documentation, source-identity exceptions,
and automated boundary guards. Round 2 has not started.

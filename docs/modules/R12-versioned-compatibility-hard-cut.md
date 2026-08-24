# R12 — Versioned compatibility hard cut

## 1. Goal and scope

R12 removes the source-identity constraint that kept obsolete root modules alive, then retires the
remaining compatibility layer without changing the application runtime. It is implemented in two
broad vertical slices:

- R12A moves E2E scoring and command implementation to canonical evaluation owners and introduces
  artifact schema v6 with source-identity v2.
- R12B removes the remaining compatibility facades and the legacy callable search bridge after all
  active tests and consumers use canonical imports.

R12A is complete in the current worktree. R12B has not started.

## 2. Position in the system

```text
supported E2E entry point
          |
          v
scripts.evaluation.evaluate_phase7_e2e
          |
          +--> application/domain/infrastructure public owners
          +--> evaluation.e2e + evaluation.phase7_dataset

production API runtime ----x----> evaluation
```

The top-level command remains a thin inbound shim until the common CLI cut in R13. Evaluation can
consume public runtime interfaces; production cannot import evaluation.

## 3. Relevant background concepts

- A **hard cut** removes an internal import path instead of preserving another indefinite shim.
- **Source identity** records hashes of selected source files that materially define an evaluation
  run.
- **Artifact schema version** identifies the structure and interpretation of a sanitized result.
- A **checkpoint identity** prevents partial results from being resumed under different code,
  configuration, corpus, dataset, or provider settings.
- Git history preserves old byte-for-byte source without copying stale implementations into the
  current tree.

## 4. Input, output, and contracts

The supported invocation remains `python -m scripts.evaluate_phase7_e2e` with the same option names,
argparse behavior, approval token, calibration-only item selection, and governance block for the
exposed held-out split.

Scoring records, metric definitions, quality-gate thresholds, sanitization fields, and exit statuses
are unchanged. The intentional provenance changes are:

- default artifact and checkpoint filenames use `e2e-v6`;
- output `schema_version` is `6`;
- run identity includes `artifact_schema_version: 6`;
- `source_identity` is a v2 object containing `version`, canonical file hashes, and the exact system
  prompt hash.

No v5 artifact is read, rewritten, or deleted. Its checkpoint identity cannot equal v2.

## 5. Step-by-step data flow

1. The top-level supported shim delegates to the canonical E2E command.
2. Argparse validates the same dataset, approval, selection, output, and checkpoint options.
3. Governance rejects held-out execution before data or provider access.
4. The command reads only the selected approved split and validates its sealed manifest.
5. Frozen collection identity is validated read-only before query execution.
6. Canonical composition builds the same lazy retrieval and structured-generation object graph.
7. `evaluation.e2e` scores the completed application execution without external I/O.
8. Source identity v2 hashes selected canonical behavior owners rather than facade files.
9. Checkpoint loading compares the entire run identity and rejects v1/v5 or otherwise changed runs.
10. A sanitized schema-v6 result is written atomically only by an explicitly approved real run.

## 6. Responsibilities of changed files

| Path | Responsibility after R12A |
| --- | --- |
| [`evaluation/e2e.py`](../../evaluation/e2e.py) | Pure offline scoring, aggregation, typed-fact matching, and quality gates. |
| [`scripts/evaluation/evaluate_phase7_e2e.py`](../../scripts/evaluation/evaluate_phase7_e2e.py) | Approval-gated integration composition, checkpointing, provenance, and sanitized output. |
| [`scripts/evaluate_phase7_e2e.py`](../../scripts/evaluate_phase7_e2e.py) | Thin supported `main` shim until R13. |
| [`tests/test_evaluation_e2e.py`](../../tests/test_evaluation_e2e.py) | Scoring, governance, parser, manifest, and checkpoint behavior. |
| [`tests/test_source_identity.py`](../../tests/test_source_identity.py) | Exact v2 source mapping and fail-closed v1 checkpoint characterization. |
| [`tests/test_architecture_boundaries.py`](../../tests/test_architecture_boundaries.py) | Canonical evaluator/CLI ownership and shrinking compatibility inventory. |
| [`ADR-005`](../adr/ADR-005-compatibility-and-source-identity.md) | Historical anchors and accepted v2 provenance structure. |

`app/evaluation_e2e.py` is removed; its original blob remains available in Git history.

## 7. Important symbols and why they exist

- `ARTIFACT_SCHEMA_VERSION` makes the output format bump explicit and testable.
- `SOURCE_IDENTITY_VERSION` distinguishes canonical v2 provenance from the historical flat mapping.
- `SOURCE_IDENTITY_PATHS` is an auditable selection of implementation owners, not facade paths.
- `_source_identity` returns a structured receipt with file and prompt hashes.
- `_load_checkpoint` compares the complete run identity and never merges incompatible executions.
- `score_phase7_execution` creates one sanitized record from a completed `QueryExecution`.
- `evaluate_phase7_quality_gates` retains the frozen documented release thresholds.

## 8. Before-and-after structure

```text
Before R12A
  app/evaluation_e2e.py                  evaluator implementation in production package
  scripts/evaluate_phase7_e2e.py         400-line integration implementation
  source identity v1                     hashes 9 legacy facade paths
  artifact/checkpoint                    e2e-v5

After R12A
  evaluation/e2e.py                      canonical offline evaluator
  scripts/evaluation/evaluate_phase7_e2e.py canonical integration implementation
  scripts/evaluate_phase7_e2e.py         thin supported shim
  source identity v2                     hashes canonical behavior owners
  artifact/checkpoint                    e2e-v6; v5 remains immutable
```

R12B will remove the other compatibility modules in the same learning document.

## 9. Design decisions and trade-offs

The evaluator moves outside `app` because it consumes application results but is never a production
runtime dependency. The command stays in `scripts/evaluation` because it owns integration concerns:
approval, filesystem artifacts, live collection validation, and provider execution.

The current top-level command remains temporarily because R13 owns one coherent change to all eight
supported CLI names. This avoids mixing provenance migration with public command removal.

Source identity v2 enumerates canonical owners explicitly. A dependency-graph hash would be harder
to audit and could include unrelated framework code; hashing only the old facades failed to capture
the implementations after R09–R11. The exact approval token is retained to preserve current egress
authorization behavior even though new output filenames are v6.

## 10. Tests and the behavior each test protects

| Test area | Protected behavior |
| --- | --- |
| Evaluator scoring | qrel-only ranks, evidence boundary, facts, citations, abstention, latency, and gates. |
| Parser contract | Same options and errors; only default artifact/checkpoint version changes. |
| Approval/governance | Invalid calibration token and every held-out request fail before execution. |
| Dataset loading | Calibration never opens the held-out path. |
| Checkpoints | Provider changes and v1 source identity both fail closed. |
| Source identity | Exact canonical path inventory, SHA-256 values, version, and prompt hash. |
| Architecture | Evaluator is outside `app`; canonical command imports no root compatibility anchor. |
| Supported shim | Top-level command delegates only to the canonical evaluation command. |

## 11. Commands and expected results

R12A focused validation:

```powershell
python -m ruff check evaluation/e2e.py scripts/evaluation/evaluate_phase7_e2e.py `
  tests/test_evaluation_e2e.py tests/test_source_identity.py `
  tests/test_architecture_boundaries.py
python -m pytest -q tests/test_evaluation_e2e.py tests/test_source_identity.py `
  tests/test_supported_cli_contracts.py tests/test_architecture_boundaries.py `
  tests/test_phase7_operational_smoke.py
```

Final validation for each slice:

```powershell
python -m ruff check .
python -m pytest -q
docker compose config --quiet
git diff --check
```

Expected: no evaluation execution, Qdrant access, model loading, provider call, or held-out read.

## 12. Small usage example

The supported user-facing command remains:

```powershell
python -m scripts.evaluate_phase7_e2e --help
```

Offline code imports the canonical scorer directly:

```python
from evaluation.e2e import score_phase7_execution

record = score_phase7_execution(dataset_item, completed_execution)
```

## 13. Common failures and debugging

- A v5 checkpoint rejection is expected; start a new v6 path rather than modifying the old header.
- A source-identity mismatch means a selected behavior owner changed; review the source diff before
  any real evaluation.
- If `--help` reaches settings or a provider, the shim or canonical module has an import-time side
  effect.
- If production can reach `evaluation.e2e`, the dependency direction has regressed.
- If the held-out path is opened during calibration, stop immediately; do not weaken the guard.
- A scoring difference is not a provenance migration and must be split into a separate bug fix.

## 14. Current limitations

- R12B has not yet removed the remaining root compatibility facades.
- The shared top-level CLI shim layout remains until R13.
- Archived workflows retain historical imports as provenance and are unsupported.
- Existing v5 artifacts remain valid historical receipts but are not resumable as v6.
- No real integration validation was run; runtime algorithms, models, thresholds, and collections are
  unchanged.

## 15. Self-check questions

1. Why must E2E scoring live outside the production package?
2. What makes a source-identity v2 checkpoint incompatible with v1?
3. Why are v6 default filenames necessary even when record metrics are unchanged?
4. Which command behavior remains public during R12A?
5. Why does R13, rather than R12A, remove the top-level shim?
6. Where does Git preserve the two historical source-anchor blobs?

## 16. Interview summary

R12A resolves a provenance trap created by successful earlier refactors: historical artifacts
hashed facade files that no longer owned behavior. The evaluator and command now live at their real
boundaries, schema v6 records a structured v2 identity over canonical implementations, and old
checkpoints fail closed. Public CLI behavior and every scoring decision stay covered offline while
production remains unable to reach evaluation.

## 17. Validation results and proposed commit

R12A receipts so far:

| Check | Result |
| --- | --- |
| Pre-change focused pytest, Python 3.11.15 | PASS — 85 tests |
| Post-change focused Ruff | PASS |
| Post-change focused pytest, Python 3.11.15 | PASS — 86 tests |
| Full Ruff | PASS |
| Full pytest, Python 3.11.15 | PASS — 413 tests, 1 dependency warning |
| Docker Compose configuration | PASS |
| Protected data and artifacts | PASS — all 86 files unchanged |
| Held-out worktree | PASS — unchanged |
| Historical source blobs in Git | PASS — both original IDs retained at the R11 commit |
| Local Markdown links | PASS |
| `git diff --check` and exact scope | PASS — 12 files |

Proposed R12A Conventional Commit after review:

```text
refactor: version e2e provenance and canonicalize evaluator
```

## 18. Status

`IN_PROGRESS`

# R13 — Supported CLI hard cut

## 1. Goal and scope

R13 makes each supported command import path match its implementation owner. It removes the eight
top-level `scripts.<command>` shims in two coherent slices:

- R13A removes five operational shims; canonical commands live in `scripts.operations`.
- R13B removes three evaluation shims; canonical commands live in `scripts.evaluation`.

R13A and R13B are implemented in the current worktree.

## 2. Position in the system

```text
python -m scripts.operations.<command>
                     |
                     v
          application/composition/domain
                     ^
                     |
            infrastructure adapters

python -m scripts.evaluation.<command>
                     |
                     v
        evaluation + public runtime interfaces
```

These modules are inbound adapters. Reusable behavior remains outside `scripts`.

## 3. Relevant background concepts

- A **CLI shim** is a module that only imports and re-exports another command's `main` function.
- A **hard cut** removes the old module instead of maintaining two indefinitely supported names.
- An **entry point** is the module passed to `python -m`; it should reveal the command's owner.
- A **parser contract** includes options, defaults, validation errors, help output, and exit status.

## 4. Input, output, and contracts

R13 keeps parser options, defaults, stdout/stderr, exit codes, artifact formats, validation order,
and external effects for:

- corpus audit;
- guarded Phase 7 indexing;
- ingestion preview;
- bounded query smoke;
- read-only query-runtime validation;
- dataset validation;
- provider-free retrieval closure;
- approval-gated E2E evaluation.

The intentional breaking change is module naming: `scripts.<command>` no longer imports, while
`scripts.operations.<operation>` and `scripts.evaluation.<evaluation-command>` are supported.

## 5. Step-by-step data flow

1. Python resolves the canonical module in `scripts.operations`.
2. Argparse validates arguments before any integration dependency is accessed.
3. The adapter resolves application/configuration/infrastructure interfaces as before.
4. Read-only commands inspect local state; mutating/provider commands still require their existing
   explicit invocation and guards.
5. The command returns the same exit code and sanitized output as the pre-cut implementation.
6. An old top-level module path fails with `ModuleNotFoundError`; no redirect or fallback occurs.

## 6. Responsibilities of changed files

| Path | Responsibility after R13 |
| --- | --- |
| [`scripts/operations`](../../scripts/operations) | Implementation and supported module path for five operational commands. |
| [`scripts/evaluation`](../../scripts/evaluation) | Implementation and supported module path for three evaluation commands. |
| [`tests/test_supported_cli_contracts.py`](../../tests/test_supported_cli_contracts.py) | Table-driven help/error contracts plus rejection of removed paths. |
| [`tests/test_architecture_boundaries.py`](../../tests/test_architecture_boundaries.py) | Exact remaining top-level inventory and canonical dependency guards. |
| [`README.md`](../../README.md) | Current supported invocations and external-effect warnings. |
| [`scripts/README.md`](../../scripts/README.md) | Script ownership, lifecycle, and canonical command inventory. |
| [`docs/CODEBASE.md`](../CODEBASE.md) | Current command map in the codebase guide. |

Per-command behavior tests import their canonical ownership packages directly. The deleted top-level
files owned no logic.

## 7. Important symbols and why they exist

- `SUPPORTED_CLI_OPTIONS` records every current module and its public option surface.
- `REMOVED_CLI_SHIMS` is the exact negative inventory for all eight removed paths.
- `_run_cli` executes help/parser checks in a credential-sanitized subprocess.
- Each canonical command's `main` remains the only execution boundary.

## 8. Before-and-after structure

```text
Before R13A
  scripts/audit_phase7_corpus.py       -> scripts.operations.audit_phase7_corpus
  scripts/index_phase7_corpus.py       -> scripts.operations.index_phase7_corpus
  scripts/ingest_preview.py            -> scripts.operations.ingest_preview
  scripts/query_smoke.py               -> scripts.operations.query_smoke
  scripts/validate_query_runtime.py    -> scripts.operations.validate_query_runtime

After R13
  scripts/operations/*.py              operational implementation + entry point
  scripts/evaluation/*.py              evaluation implementation + entry point
  scripts/*.py                         none
```

## 9. Design decisions and trade-offs

The old names are removed rather than deprecated because these are repository-local commands, all
known consumers are updated together, and another compatibility layer would defeat the simplified
surface. The longer canonical name is useful: it makes operational versus evaluation ownership
visible without opening the file.

Operations and evaluation were cut separately so each review had one responsibility and fewer
meaningful tests/docs. Command behavior remains characterized at the implementation owner.

## 10. Tests and the behavior each test protects

| Test area | Protected behavior |
| --- | --- |
| Table-driven `--help` | Public options and zero exit status at canonical paths. |
| Invalid arguments | Argparse fails before integration access. |
| Removed path guard | Old names fail instead of silently redirecting. |
| Corpus/index tests | Defaults, protected collections, deterministic validation, and sanitized audit output. |
| Ingestion tests | Stable IDs, page/chunk behavior, atomic output, and lazy heavy imports. |
| Smoke tests | Frozen Phase 7 selection, safe no-key output, and Phase 6 rejection. |
| Architecture | No command module remains top-level; no private cross-script imports. |

## 11. Commands and expected results

```powershell
python -m ruff check .
python -m pytest -q
docker compose config --quiet
git diff --check
```

Expected: offline checks pass without Qdrant/model/provider access, and protected files stay
unchanged.

## 12. Small usage example

```powershell
python -m scripts.operations.audit_phase7_corpus --help
python -m scripts.operations.ingest_preview data/raw/ATV320_Installation_manual_EN_NVE41289_09.pdf --limit 3
python -m scripts.operations.validate_query_runtime --help
```

The second command is only an invocation example; R13 validation does not read a manual.

## 13. Common failures and debugging

- `No module named scripts.<command>` is expected after R13; use the owning operations/evaluation
  package.
- An argparse exit code change means the canonical parser contract drifted; do not weaken the test.
- If `--help` initializes Qdrant or a model, inspect import-time construction in the canonical module.
- Never add a redirect shim to fix old documentation; update the command path at its owner.
- An archived command is not a substitute for a removed supported path.

## 14. Current limitations

- All eight old top-level command paths are intentionally unavailable.
- No real operational or evaluation command is executed during offline validation.
- Historical walkthroughs retain historical commands unless they are current runbooks.

## 15. Self-check questions

1. Why is `scripts.operations` both an ownership boundary and a useful command name?
2. Which parser behaviors must survive a module-path hard cut?
3. Why should an old path fail instead of redirecting silently?
4. Which operational commands can mutate Qdrant or call a provider when explicitly run?
5. Why were evaluation shims isolated in R13B?

## 16. Interview summary

R13 removes command aliases after canonical implementations and parser contracts are already stable.
Operational and evaluation module paths now directly name their owners, old paths fail explicitly,
and offline tests prove the command surface, governance, and guard ordering are unchanged.

## 17. Validation results and proposed commit

| Check | Result |
| --- | --- |
| Pre-change focused pytest, Python 3.11.15 | PASS — 89 tests |
| Post-change focused Ruff | PASS |
| Post-change focused pytest, Python 3.11.15 | PASS — 92 tests |
| Full Ruff | PASS |
| Full pytest, Python 3.11.15 | PASS — 380 tests, 1 dependency warning |
| Docker Compose configuration | PASS |
| Protected files, Markdown links, and diff scope | PASS — all 86 protected files unchanged |
| R13B pre-change focused pytest, Python 3.11.15 | PASS — 90 tests |
| R13B focused Ruff | PASS |
| R13B focused pytest, Python 3.11.15 | PASS — 92 tests |
| R13B full Ruff | PASS |
| R13B full pytest, Python 3.11.15 | PASS — 382 tests, 1 dependency warning |
| R13B Docker Compose configuration | PASS |
| R13B protected files, links, and diff scope | PASS — all 86 protected files unchanged |

R13A commit:

```text
f69f8e4 refactor: hard-cut operational CLI shims
```

Proposed R13B Conventional Commit after review:

```text
refactor: hard-cut evaluation CLI shims
```

## 18. Status

`COMPLETE`

# R08 — Simplification baseline

## 1. Goal and scope

R08 creates the safety net for a second, narrowly scoped cleanup pass. It makes current dependency
debt, supported CLI behavior, retrieval-record schemas, and E2E source identity executable as tests
before production modules are moved.

This module changes only tests and documentation. It does not move production code, rename a command,
modify configuration, call an integration, or change retrieval behavior.

## 2. Position in the system

```text
source tree + public contracts
            ↓
R08 offline characterization
            ↓
review gates for R09-R14

runtime request pipeline ── unchanged
```

R08 precedes domain, ingestion, retrieval, evaluation, and CLI structural changes. A later slice can
only remove a compatibility edge after its replacement preserves the contracts captured here.

## 3. Relevant background concepts

- A **characterization test** records correct existing behavior so a refactor cannot change it by
  accident.
- A **dependency matrix** maps each canonical package to the layers it may import.
- A **temporary debt edge** is a forbidden edge that exists today and has an explicit removal module.
- A **Git blob ID** hashes the exact bytes and length of one file; it is stronger than checking only a
  path or commit status.
- A **public CLI contract** consists of invocation path, options, argparse failures, output, and exit
  status. Private helper identity is not part of that contract.

The terminology and future provenance decision are recorded in
[ADR-005](../adr/ADR-005-compatibility-and-source-identity.md).

## 4. Input, output, and contracts

Inputs are Python source paths, import statements, Pydantic models, FastAPI OpenAPI metadata, CLI
arguments, and exact bytes of the two E2E anchors. Tests never load a model, contact Qdrant, call a
provider, or execute held-out evaluation.

Outputs are deterministic pytest assertions. No runtime artifact is produced. A temporary manifest
outside the repository records path, length, and SHA-256 for 86 protected files so validation can
prove that PDFs, frozen inputs, and existing artifacts were not changed.

## 5. Step-by-step data flow

1. The architecture test parses local imports with the standard-library `ast` module.
2. Every `app`, `evaluation`, and `ui` module receives a layer classification.
3. Allowed layer directions are applied to every direct local edge.
4. The actual violations must equal the 18 documented temporary R12 edges exactly.
5. CLI tests start each supported module with `--help` or a parser-invalid argument.
6. Argparse must exit before settings, models, Qdrant, datasets, or providers are accessed.
7. Source-identity tests rebuild both Git blob IDs and the legacy nine-file SHA-256 mapping.
8. Runtime characterization protects the retrieval-record schema and one-based rank validation.

## 6. Responsibilities of changed files

| Path | Responsibility after R08 |
| --- | --- |
| [`tests/test_architecture_boundaries.py`](../../tests/test_architecture_boundaries.py) | Complete layer matrix plus exact temporary R12 anchor inventory. |
| [`tests/test_runtime_characterization.py`](../../tests/test_runtime_characterization.py) | HTTP/query golden behavior and retrieval-candidate schema invariants. |
| [`tests/test_source_identity.py`](../../tests/test_source_identity.py) | Exact Git blobs and legacy E2E source-identity mapping. |
| [`tests/test_supported_cli_contracts.py`](../../tests/test_supported_cli_contracts.py) | Table-driven public option and parser-failure checks for all eight commands. |
| [`ADR-005`](../adr/ADR-005-compatibility-and-source-identity.md) | Compatibility, hard-cut, archive, and provenance-v2 decision. |
| [`R08-simplification-baseline.md`](R08-simplification-baseline.md) | Learning record and validation receipt for this module. |

## 7. Important symbols and why they exist

- `R12_ROOT_MODULE_ANCHORS` is the exact set of root modules that cannot be mistaken for accepted
  target architecture.
- `R12_TEMPORARY_IMPORT_EDGES` records every current canonical-to-compatibility edge.
- `ALLOWED_DEPENDENCY_LAYERS` makes the modular-monolith direction executable.
- `_architecture_layer` rejects an unclassified new local module.
- `PINNED_GIT_BLOBS` preserves the two historical E2E source anchors.
- `LEGACY_SOURCE_IDENTITY_PATHS` captures the source mapping that R12 will replace explicitly.
- `SUPPORTED_CLI_OPTIONS` describes the current eight-command option surface without importing
  private parser helpers.

## 8. Before-and-after structure

```text
Before R08
  six selected production roots define evaluation reachability
  focused tests describe individual import decisions
  source pins exist only in documentation
  CLI tests often rely on shim/helper object identity

After R08
  every canonical app/evaluation/ui module is layer-classified
  all 18 known debt edges are an exact temporary set
  source pins and source mapping are executable contracts
  all eight supported CLI option surfaces have behavior-level coverage
```

The production source tree is intentionally identical before and after R08.

## 9. Design decisions and trade-offs

The matrix supplements existing focused tests instead of replacing them. The matrix catches unknown
edges; focused tests continue to give precise failure messages about important composition seams.

Compatibility modules are classified separately. Their outbound imports are not accepted as target
architecture, while their exact inventory prevents a new root module from hiding in the exemption.
The R12 acceptance condition is to remove this inventory and all temporary edges.

CLI tests use subprocesses because the user-facing contract is `python -m scripts.<command>`. This
costs a few seconds but avoids locking `_build_parser`, constants, or shim object identity. The calls
use only `--help` and parser-invalid inputs, so no integration workflow can start.

## 10. Tests and the behavior each test protects

| Test area | Protected behavior |
| --- | --- |
| Dependency matrix | Domain/application/infrastructure direction and exact current debt edges |
| Existing focused architecture tests | API seam, Streamlit HTTP-only, production/evaluation isolation |
| RetrievalCandidate schema | Required serialization fields and one-based rank validation |
| Golden query execution | candidate, reranked, evidence, retry, citation, usage, and timing boundaries |
| Source identity | two exact blobs, nine legacy file hashes, and prompt hash |
| Supported CLI help | eight importable command paths and their complete option names |
| Invalid CLI arguments | argparse exit 2 before integration access |

## 11. Commands and expected results

Focused validation:

```powershell
python -m ruff check tests/test_architecture_boundaries.py `
  tests/test_runtime_characterization.py tests/test_source_identity.py `
  tests/test_supported_cli_contracts.py
python -m pytest -q tests/test_architecture_boundaries.py `
  tests/test_runtime_characterization.py tests/test_source_identity.py `
  tests/test_supported_cli_contracts.py
```

Final validation:

```powershell
python -m ruff check .
python -m pytest -q
docker compose config --quiet
git diff --check
```

Expected: all checks exit 0, protected-file hashes remain identical, and only the six R08 files are
modified. A test result is reported only for the interpreter on which it actually ran.

## 12. Small usage example

The layer matrix gives a direct review signal. If a domain module imports a Qdrant adapter, the new
edge appears in the `violations` set and fails instead of being hidden behind an unscanned runtime
root.

The public CLI check can be run without executing ingestion or retrieval:

```powershell
python -m scripts.index_phase7_corpus --help
```

## 13. Common failures and debugging

- An unclassified module means it was created outside the known package tree; assign a legitimate
  layer rather than adding a generic exemption.
- A matrix mismatch with an extra edge indicates a new dependency violation.
- A matrix mismatch with a missing edge means cleanup succeeded; remove that edge from the temporary
  set in the same slice.
- A blob mismatch means a protected E2E anchor changed. Stop instead of updating the expected hash.
- CLI help reaching Qdrant or a provider indicates import-time side effects and is a boundary bug.
- A protected-file hash mismatch means a PDF, dataset, or artifact changed outside R08 scope.

## 14. Current limitations

- R08 records compatibility debt but does not remove it.
- The current CLI names and fifteen-file implementation/shim layout remain until R13.
- The E2E evaluator and CLI remain in their legacy locations until the versioned R12 migration.
- Existing tests that assert private shim identity are not removed in this module; behavior-level
  coverage now exists so later CLI slices can retire those assertions safely.
- Retrieval tuning, model changes, threshold changes, provider changes, re-indexing, benchmark v3,
  and performance work remain outside this cleanup sequence.

## 15. Self-check questions

1. Why must the actual violation set equal, rather than contain, the temporary edge set?
2. What distinguishes a compatibility path from a canonical public interface?
3. Why can source anchor bytes not be moved before R12?
4. Which CLI invocations are safe in offline unit tests?
5. What proves that a rank is one-based after `RetrievalCandidate` moves?
6. Why is an unchanged Git worktree insufficient to protect ignored PDFs and artifacts?

## 16. Interview summary

R08 converts refactor assumptions into executable constraints. Instead of claiming that the layers
are clean, the repository classifies every canonical module, records each known exception, protects
public request and retrieval records, executes all supported parser surfaces offline, and pins the
historical evaluator bytes. This allows later commits to remove debt monotonically and makes each
structural change reviewable without running external integrations.

## 17. Validation results and proposed commit

Completed validation:

| Check | Result |
| --- | --- |
| Focused Ruff in the validation container | PASS — Python 3.11.15 |
| Focused offline pytest | PASS — 52 tests |
| Full Ruff | PASS — `All checks passed!` |
| Full offline pytest | PASS — 403 passed, 1 known third-party warning |
| Docker Compose configuration | PASS |
| Protected-file comparison | PASS — 86 paths unchanged |
| Exact E2E source pins | PASS |
| Exposed regression dataset scope | PASS — unchanged |
| New-document local links | PASS |
| `git diff --check` | PASS |

The same full suite also passed under the local Python 3.13.5 `.venv`; the Python 3.11 container result
above is the authoritative R08 validation. No Qdrant service, model, provider, evaluation, or
re-indexing workflow was started.

Proposed Conventional Commit after review:

```text
test: lock surface simplification contracts
```

## 18. Status

`COMPLETE`

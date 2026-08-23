# R07 — Evaluation isolation, script archive, and documentation closure

## 1. Goal and scope

R07 establishes a one-way boundary: offline evaluation may consume public production contracts,
while the production runtime cannot import evaluation code. It will also classify Phase 7 scripts,
archive confirmed one-off tools, remove evidence-backed compatibility shims, and align current
documentation with the active runtime.

R07A1 moves the basic retrieval dataset schemas,
validation, scoring, and aggregate metrics from `app.evaluation` to `evaluation.retrieval`.
`app.evaluation` remains a compatibility facade for existing scripts and evaluator-side modules.

R07A2 moves sanitized snapshot validation and deterministic rank-only replay from
`app.phase7_replay` to `evaluation.replay`. Its old module path also remains a compatibility facade.
The E2E evaluator, Phase 7 dataset helpers, CLI classification, archive work, shim removal, and
documentation closure remain later R07 slices.

## 2. Position in the system

```text
offline dataset / frozen chunks
             |
             v
evaluation.retrieval
  schema -> validation -> retrieval callback -> ranks -> aggregate report
             |
             v
public domain record: app.domain.documents.DocumentChunk

production runtime  -X->  evaluation
legacy evaluator code --> app.evaluation facade --> evaluation.retrieval

sanitized snapshot --> evaluation.replay --> public retrieval/ranking contracts
legacy calibration --> app.phase7_replay facade --> evaluation.replay
```

The new top-level package makes ownership visible in the filesystem. It is installed in retrieval,
API, and ingestion images because supported ingestion tooling still reaches the compatibility facade.
Static dependency tests, rather than package location alone, enforce that production composition and
application services cannot reach it.

## 3. Relevant background concepts

- A **qrel** identifies a chunk manually accepted as direct evidence for an evaluation question.
- A **characterization test** records existing behavior so a structural move cannot silently alter it.
- **MRR** (mean reciprocal rank) rewards relevant evidence appearing near the top of a result list.
- **Nearest-rank percentile** selects an observed latency value at the requested percentile.
- A **compatibility facade** preserves an old import path by exporting the exact canonical objects;
  it contains no duplicate implementation.
- A **source identity hash** ties an artifact to evaluator source files. Moving a hashed source file
  can invalidate identity even when behavior is unchanged.

## 4. Input, output, and contracts

`EvaluationCase` accepts strict Pydantic records containing the query identity, query/evidence
languages, direct-evidence chunk IDs, expected phrases/pages, category, and optional metadata.
Unknown fields, blanks, duplicate qrels, invalid pages, unsupported categories/languages, and
inconsistent retrieval scenarios remain errors.

`load_frozen_chunks` returns `list[DocumentChunk]` using the canonical domain record from
[`app/domain/documents.py`](../../app/domain/documents.py). `evaluate_cases` accepts evaluation cases,
a retrieval callback, and a candidate limit. It returns the same report dictionaries as before,
including per-query diagnostics, overall and grouped metrics, critical-query metrics, and failures.

The public compatibility contract is identity-preserving:

```python
from app.evaluation import EvaluationCase as legacy
from evaluation.retrieval import EvaluationCase as canonical

assert legacy is canonical
```

No schemas, default values, validation order, error messages, rank boundaries, percentile rules, or
aggregate keys are intentionally changed in R07A1.

`snapshot_candidates_to_retrieval` accepts sanitized candidate dictionaries and reconstructs
rank-only `RetrievalCandidate` values with empty text and page fields. `replay_role_prior` applies the
existing deterministic domain ranking policy. R07A2 preserves validation order and messages,
one-based rank rules, score/fingerprint checks, feature metadata, candidate sorting, and exception
translation.

## 5. Step-by-step data flow

1. `load_evaluation_cases` reads UTF-8 JSONL in file order.
2. `EvaluationCase` validates and normalizes each record and derives its retrieval scenario.
3. `load_frozen_chunks` reads the frozen chunk payload into canonical domain `DocumentChunk` values.
4. `validate_cases_against_chunks` verifies direct-evidence IDs, phrases, pages, and document identity.
5. `evaluate_cases` calls the supplied retrieval function with the unchanged question, limit, and
   document ID.
6. Direct-evidence, phrase, and page ranks are calculated with their existing boundaries.
7. `aggregate_rows` computes hit rate, candidate recall, MRR, grouped metrics, and failure lists.
8. Existing consumers may enter through `app.evaluation`; the facade immediately delegates by object
   alias to the canonical implementation.

Sanitized replay follows a separate deterministic flow:

1. Validate required candidate identity, ranks, score, structural counts, and optional fingerprint.
2. Reconstruct empty-content `RetrievalCandidate` values without raw question or evidence content.
3. Require unique chunk IDs and contiguous one-based cross-encoder ranks.
4. Sort by cross-encoder rank and call the public domain role-aware ranking policy.
5. Translate a ranking-policy error into the existing `Phase7ReplayError` contract.

## 6. Responsibilities of changed files

| File | R07 responsibility |
| --- | --- |
| [`evaluation/__init__.py`](../../evaluation/__init__.py) | Marks the offline evaluation package. |
| [`evaluation/retrieval.py`](../../evaluation/retrieval.py) | Canonical retrieval evaluation schemas, validation, scoring, and aggregation. |
| [`evaluation/replay.py`](../../evaluation/replay.py) | Canonical sanitized snapshot validation and deterministic rank-only replay. |
| [`app/evaluation.py`](../../app/evaluation.py) | Temporary import-compatible re-export facade. |
| [`app/phase7_replay.py`](../../app/phase7_replay.py) | Temporary import-compatible replay facade. |
| [`pyproject.toml`](../../pyproject.toml) | Includes `evaluation*` in package discovery and Ruff first-party imports. |
| [`Dockerfile`](../../Dockerfile) | Copies the canonical package into the shared retrieval runtime image. |
| [`tests/test_evaluate.py`](../../tests/test_evaluate.py) | Exercises the canonical module and verifies facade identity. |
| [`tests/test_phase7_replay.py`](../../tests/test_phase7_replay.py) | Protects replay validation/ranking and facade identity. |
| [`tests/test_architecture_boundaries.py`](../../tests/test_architecture_boundaries.py) | Includes `evaluation` in the local import graph and forbids production reachability. |
| This document | Records the implemented boundary and evidence for the whole R07 module. |

## 7. Important symbols and why they exist

- `EvaluationCase`: strict dataset schema and language/scenario invariant owner.
- `EvaluationError`: stable error category for invalid evaluation inputs or constraints.
- `RetrievedLike`: minimal structural retrieval-result contract needed by the evaluator.
- `load_evaluation_cases`: deterministic JSONL loader with duplicate-ID detection.
- `load_frozen_chunks`: frozen corpus loader returning domain records.
- `validate_cases_against_chunks`: qrel-to-corpus integrity guard.
- `chunk_set_metadata`: deterministic chunk-set identity metadata.
- `direct_evidence_rank`: authoritative qrel rank; page/phrase matches remain diagnostic only.
- `evaluate_cases`: per-query scoring coordinator over an injected retrieval callback.
- `aggregate_rows`: stable aggregate and grouped report construction.
- `percentile_nearest_rank`: deterministic observed-value percentile policy.
- `Phase7ReplayError`: stable error category for malformed sanitized snapshots and replay failures.
- `snapshot_candidates_to_retrieval`: validates sanitized rows and reconstructs rank-only candidates.
- `replay_role_prior`: replays the registered domain policy without model, provider, Qdrant, or text.

## 8. Before-and-after structure

Before R07A1:

```text
app/
  evaluation.py       # implementation inside production package
  evaluation_e2e.py
  phase7_replay.py    # replay implementation inside production package
```

After R07A1:

```text
app/
  evaluation.py       # temporary compatibility facade
  evaluation_e2e.py   # unchanged; source identity work is deferred
  phase7_replay.py    # temporary compatibility facade
evaluation/
  __init__.py
  retrieval.py        # canonical implementation
  replay.py           # canonical sanitized replay
```

The move changes ownership and dependency direction, not evaluator behavior.

## 9. Design decisions and trade-offs

The canonical implementation imports `DocumentChunk` from `app.domain.documents`, not the broader
`app.models` facade. Evaluation is allowed to depend inward on public domain contracts; the domain
does not depend back on evaluation.

Direct aliases in `app.evaluation` preserve class/function identity and avoid a second implementation.
Keeping the facade temporarily avoids mixing relocation with dozens of script-classification changes.
It also gives each later script slice an explicit migration point.

`app.evaluation_e2e` is deliberately not moved here. Current evaluation commands include that file in
source identity calculations. Relocating it without an identity transition could invalidate replay or
artifact checks, so R07 will characterize that contract separately.

The package is copied into the shared retrieval image. This is required while the supported guarded
indexing command imports frozen-chunk metadata through the old facade. The important boundary is that
production runtime roots cannot import or reach evaluation; R07 may narrow image contents after script
consumers are classified.

`evaluation.replay` imports candidate and ranking interfaces through `app.domain.retrieval` and
`app.domain.policies.ranking`, not `app.models` or `app.phase7_optimization`. This exposes the intended
dependency direction without moving the shared `RetrievalCandidate` model in the same slice. The
calibration CLI keeps its old import until script classification so source changes remain scoped.

## 10. Tests and protected behavior

[`tests/test_evaluate.py`](../../tests/test_evaluate.py) protects:

- schema strictness and retrieval-scenario derivation;
- JSONL errors and duplicate-ID behavior;
- qrel, phrase, page, and frozen-chunk validation;
- direct-evidence versus diagnostic rank semantics;
- hit rate, recall, MRR, grouped metrics, critical cases, and failure ordering;
- nearest-rank percentiles and candidate-limit errors;
- exact identity between old facade exports and canonical exports.

[`tests/test_architecture_boundaries.py`](../../tests/test_architecture_boundaries.py) protects:

- no production runtime root can reach either `app.evaluation` or `evaluation.*`;
- application services cannot import evaluation;
- the facade points to `evaluation.retrieval`;
- canonical evaluation uses the domain document contract rather than `app.models`.

Candidate-audit, reranking, Phase 7, and indexing CLI suites are included in focused regression checks
because those consumers still enter through the compatibility facade.

[`tests/test_phase7_replay.py`](../../tests/test_phase7_replay.py) protects sanitized reconstruction,
rank-only reordering, empty raw-evidence fields, malformed-record rejection, and exact identity of all
old replay facade exports. The architecture suite additionally proves canonical replay uses the domain
interfaces and remains unreachable from production roots.

## 11. Commands and expected results

Pre-move characterization executed in the offline Python 3.11 validation container:

```text
python -m pytest -q tests/test_evaluate.py tests/test_candidate_audit.py \
  tests/test_reranking.py tests/test_phase7.py tests/test_phase7_index_cli.py \
  tests/test_architecture_boundaries.py
90 passed
```

R07A1 post-move checks are recorded in section 17 after execution. They must not contact a provider,
download a model, write Qdrant, run calibration, or read held-out payloads.

R07A2 pre-move characterization:

```text
python -m pytest -q tests/test_phase7_replay.py tests/test_phase7_optimization.py \
  tests/test_architecture_boundaries.py
55 passed
```

## 12. Small usage example

```python
from pathlib import Path

from evaluation.retrieval import load_evaluation_cases, validate_cases_against_chunks

cases = load_evaluation_cases(Path("path/to/development-cases.jsonl"))
validate_cases_against_chunks(cases, frozen_chunks)
```

The caller owns paths and retrieval execution. Importing the module performs no network or model work.

## 13. Common failures and debugging

- `ModuleNotFoundError: evaluation`: ensure package discovery and the Docker build context include the
  top-level `evaluation` directory.
- Facade identity assertion fails: re-export the canonical symbol directly; do not wrap or subclass it.
- Production dependency test fails: trace the reported root/source/dependency tuple and move evaluation
  coordination out of that production path.
- Metric golden changes: stop. Check for altered ordering, normalization, cutoff, or qrel semantics;
  R07A1 does not authorize metric changes.
- Replay order/error changes: stop. Compare the canonical replay body with the former implementation
  and verify that only dependency imports changed.
- Artifact identity mismatch: stop before updating artifacts. Confirm whether a source-hashed evaluator
  was moved; that transition requires its own characterized slice.

## 14. Current limitations

- `app.evaluation` remains until its consumers are migrated or explicitly retained.
- `app.phase7_replay` remains as a facade until the calibration CLI is classified.
- `app.evaluation_e2e`, Phase 7 dataset helpers, and evaluator CLIs are not isolated yet.
- Evaluation and production packages are still present in the same installed image.
- Historical defaults in `EvaluationCase` are preserved as behavior; changing them is not part of this
  structural slice.
- No evaluation metric, threshold, dataset, artifact, or benchmark result is regenerated.

## 15. Self-check questions

1. Why is page matching diagnostic rather than a direct-evidence hit?
2. What property does a direct alias preserve that a wrapper may not?
3. Why may evaluation import `app.domain.documents` while production cannot import evaluation?
4. Why is `app.evaluation_e2e` excluded from this move?
5. Which test detects a future production-to-evaluation dependency?
6. Why does sanitized replay deliberately reconstruct candidates with empty text?

## 16. Interview summary

R07A1–R07A2 turn evaluation ownership into an enforceable architectural boundary without changing the
retrieval benchmark contract. Retrieval metrics and sanitized replay now live in a top-level offline
package, depend inward on public domain contracts, and remain accessible through identity-preserving
legacy facades. Golden tests and static reachability checks demonstrate that evaluator behavior and
production isolation remain intact.

## 17. Validation results and proposed commit

R07A1 validation used the repository's existing Python 3.11 validation container and was committed as
`9f8ff89 refactor: isolate retrieval evaluation utilities`:

| Check | Result |
| --- | --- |
| Focused Ruff | PASS — `All checks passed!` |
| Focused offline pytest | PASS — `92 passed` |
| Legacy/canonical import identity smoke | PASS |
| Full Ruff | PASS — `All checks passed!` |
| Full offline pytest | PASS — `400 passed, 1 warning` |
| `docker compose config --quiet` | PASS |
| Local Markdown links | PASS — all links in this module document resolve |
| `git diff --check` | PASS |

The warning is the existing Starlette `TestClient`/`httpx` deprecation warning; it does not fail the
suite and R07A1 does not change dependencies.

No provider, model download, Qdrant operation, calibration, or held-out evaluation is authorized.

R07A2 validation:

| Check | Result |
| --- | --- |
| Pre-move characterization | PASS — `55 passed` |
| Focused Ruff | PASS — `All checks passed!` |
| Focused offline pytest | PASS — `57 passed` |
| Legacy/canonical import identity smoke | PASS |
| Implementation-body comparison | PASS — no drift below `Phase7ReplayError` |
| Full Ruff | PASS — `All checks passed!` |
| Full offline pytest | PASS — `402 passed, 1 warning` |
| Local Markdown links | PASS — all links in this module document resolve |
| `git diff --check` | PASS |

Proposed R07A2 commit after user review:

```text
refactor: isolate sanitized evaluation replay
```

## 18. Status

`IN_PROGRESS` — R07A1 is complete and committed. R07A2 is implemented and validated; user review
remains. Later R07 slices remain outside this slice.

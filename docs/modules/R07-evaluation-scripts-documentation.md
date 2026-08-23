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

R07A3 separates offline Phase 7 dataset/qrel contracts into `evaluation.phase7_dataset` and active
corpus identity/file operations into `app.infrastructure.corpus_artifacts`. `app.phase7` becomes a
compatibility facade, while the supported indexing command imports corpus artifacts directly.

R07A4 moves frozen-chunk JSONL parsing and stable chunk-set metadata into the same corpus artifact
infrastructure. The supported indexing command no longer imports `app.evaluation`. Evaluation keeps a
thin error-translation wrapper so its public `EvaluationError` contract remains unchanged.

R07B1 classifies the provider-free Phase 7 dataset validation command under `scripts.evaluation` and
keeps `python -m scripts.validate_phase7_dataset` as a thin compatibility entry point.

R07B2 classifies the redistribution-safe local PDF metadata audit under `scripts.operations`. It
remains a supported reproducibility command because the current corpus walkthrough references it and
its output documents source identity/parsing preconditions without storing substantial manual text.

R07C1 starts evidence-based archival with the completed dataset-v2 migration. The source remains
available under `scripts.archive.phase7`, but its old top-level command is intentionally unsupported
because rerunning it would mutate frozen dataset files.
R07C2 archives the completed typed-fact calibration draft generator for the same reason. The approved,
frozen calibration-v3 identity remains untouched; only the obsolete construction entry point leaves
the supported top-level script surface.
R07C3 archives the completed calibration-v3 approval/freeze command. This removes a top-level command
that can overwrite the active dataset and manifest while preserving its implementation as provenance.
R07C4 archives the earlier answer-fact source-review migration. Its mappings remain available for
provenance, but the command that can rewrite both frozen dataset splits is no longer supported.
R07C5 archives the initial Phase 7 annotation generator. It had no approval token or argument parser
and could overwrite both dataset splits plus the review receipt, so it cannot remain a supported tool.
Remaining CLI classification, archive work, shim removal, and documentation closure remain later R07
slices.

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

offline Phase 7 JSONL --> evaluation.phase7_dataset --> domain documents / retrieval metrics
supported indexing CLI --> app.infrastructure.corpus_artifacts --> retrieval contract
                                      |
                                      +--> frozen DocumentChunk JSONL / stable chunk identity

legacy validation CLI --> scripts.evaluation.validate_phase7_dataset
                              --> evaluation.phase7_dataset / evaluation.retrieval

legacy corpus audit CLI --> scripts.operations.audit_phase7_corpus
                              --> app.infrastructure.corpus_artifacts
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
- An **error translation boundary** lets shared infrastructure expose its own error while a legacy
  evaluator facade preserves the exception category callers already handle.

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

`Phase7DatasetItem` and `ExpectedAnswerFact` preserve the strict Phase 7 annotation schema.
Dataset loaders and validators retain record order, canonical hashes, qrel/phrase/page validation,
review-state rules, and exact-content qrel expansion. Corpus artifact functions preserve streamed
SHA-256 and atomic UTF-8 JSON/JSONL output. Collection names continue to equal the frozen retrieval
contract.

`load_frozen_chunks` preserves JSONL record order and returns canonical domain `DocumentChunk`
instances. Infrastructure raises `CorpusArtifactError`; `evaluation.retrieval.load_frozen_chunks`
translates it to `EvaluationError` with the same message. `chunk_set_metadata` still returns sorted
document IDs and the SHA-256 of newline-joined sorted chunk IDs; it never hashes manual text.

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

Phase 7 dataset/corpus flow is now explicit:

1. Indexing reads active collection identity and filesystem helpers from corpus infrastructure.
2. Offline tools read strict JSONL through `evaluation.phase7_dataset`.
3. Dataset models validate answerability, scenario, qrel, citation, review, and answer-fact invariants.
4. Validators compare qrels with frozen domain chunks and compute canonical summaries/hashes.
5. Existing scripts may still enter through `app.phase7`; each export is the exact canonical object.

Shared frozen-corpus flow:

1. Read UTF-8 JSONL from the explicit caller-supplied path.
2. Reject unreadable, empty, blank, malformed, or duplicate-ID records in the existing order.
3. Return records in file order as domain `DocumentChunk` objects.
4. Derive stable metadata from sorted IDs without reading or hashing chunk text.
5. Indexing handles the infrastructure `ValueError`; evaluation translates it to `EvaluationError`.

## 6. Responsibilities of changed files

| File | R07 responsibility |
| --- | --- |
| [`evaluation/__init__.py`](../../evaluation/__init__.py) | Marks the offline evaluation package. |
| [`evaluation/retrieval.py`](../../evaluation/retrieval.py) | Retrieval evaluation schemas/scoring plus a legacy error-translation wrapper for frozen chunks. |
| [`evaluation/replay.py`](../../evaluation/replay.py) | Canonical sanitized snapshot validation and deterministic rank-only replay. |
| [`evaluation/phase7_dataset.py`](../../evaluation/phase7_dataset.py) | Canonical Phase 7 schemas, dataset validation, hashes, and qrel closure. |
| [`app/evaluation.py`](../../app/evaluation.py) | Temporary import-compatible re-export facade. |
| [`app/phase7_replay.py`](../../app/phase7_replay.py) | Temporary import-compatible replay facade. |
| [`app/phase7.py`](../../app/phase7.py) | Temporary facade over dataset and corpus artifact owners. |
| [`app/infrastructure/corpus_artifacts.py`](../../app/infrastructure/corpus_artifacts.py) | Active corpus constants, frozen-chunk parsing/identity, streamed hash, and atomic local file operations. |
| [`scripts/operations/index_phase7_corpus.py`](../../scripts/operations/index_phase7_corpus.py) | Uses corpus artifacts without importing either evaluation compatibility facade. |
| [`scripts/evaluation/validate_phase7_dataset.py`](../../scripts/evaluation/validate_phase7_dataset.py) | Canonical provider-free dataset validation adapter. |
| [`scripts/validate_phase7_dataset.py`](../../scripts/validate_phase7_dataset.py) | Thin compatibility entry point preserving the documented command. |
| [`scripts/operations/audit_phase7_corpus.py`](../../scripts/operations/audit_phase7_corpus.py) | Canonical local corpus metadata/text-layer audit adapter. |
| [`scripts/audit_phase7_corpus.py`](../../scripts/audit_phase7_corpus.py) | Thin compatibility entry point preserving the walkthrough command. |
| [`scripts/archive/phase7/migrate_phase7_dataset_v2.py`](../../scripts/archive/phase7/migrate_phase7_dataset_v2.py) | Unsupported historical dataset-v2 migration retained for provenance. |
| [`scripts/archive/phase7/draft_phase7_calibration_fact_types.py`](../../scripts/archive/phase7/draft_phase7_calibration_fact_types.py) | Unsupported historical typed-fact calibration-v3 draft generator. |
| [`scripts/archive/phase7/freeze_phase7_calibration_v3.py`](../../scripts/archive/phase7/freeze_phase7_calibration_v3.py) | Unsupported historical calibration-v3 approval/freeze command. |
| [`scripts/archive/phase7/apply_phase7_answer_facts.py`](../../scripts/archive/phase7/apply_phase7_answer_facts.py) | Unsupported historical answer-fact source-review migration. |
| [`scripts/archive/phase7/generate_phase7_annotation_draft.py`](../../scripts/archive/phase7/generate_phase7_annotation_draft.py) | Unsupported initial Phase 7 annotation generator. |
| [`scripts/archive/phase7/README.md`](../../scripts/archive/phase7/README.md) | Archive policy, completion evidence, and no-rerun warning. |
| [`pyproject.toml`](../../pyproject.toml) | Includes `evaluation*` in package discovery and Ruff first-party imports. |
| [`Dockerfile`](../../Dockerfile) | Copies the canonical package into the shared retrieval runtime image. |
| [`tests/test_evaluate.py`](../../tests/test_evaluate.py) | Exercises the canonical module and verifies facade identity. |
| [`tests/test_phase7_replay.py`](../../tests/test_phase7_replay.py) | Protects replay validation/ranking and facade identity. |
| [`tests/test_phase7.py`](../../tests/test_phase7.py) | Exercises canonical dataset contracts and both facade ownership branches. |
| [`tests/test_phase7_index_cli.py`](../../tests/test_phase7_index_cli.py) | Protects supported indexing defaults through canonical corpus constants. |
| [`tests/test_phase7_evaluation_cli.py`](../../tests/test_phase7_evaluation_cli.py) | Protects validation CLI defaults, orchestration, output, errors, and shim identity. |
| [`tests/test_phase7_corpus_audit_cli.py`](../../tests/test_phase7_corpus_audit_cli.py) | Uses fake PDFs to protect sanitized audit behavior without reading manuals. |
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
- `Phase7DatasetItem`: strict answerable/unanswerable annotation and review-state contract.
- `ExpectedAnswerFact`: typed deterministic answer-fact contract used by offline scoring.
- `validate_phase7_dataset(s)`: frozen-corpus qrel integrity and split-isolation guards.
- `dataset_sha256`: line-ending-independent canonical record identity.
- `build_exact_content_equivalence` / `expand_exact_equivalent_qrels`: bounded same-document,
  exact-normalized qrel closure.
- `file_sha256`: streamed source identity without loading a manual into memory.
- `write_json_atomic` / `write_jsonl_atomic`: replace-on-success local artifact writes.
- `CorpusArtifactError`: infrastructure error for invalid local frozen-corpus artifacts.
- `load_frozen_chunks`: strict ordered JSONL-to-domain-record conversion.
- `chunk_set_metadata`: stable corpus identity shared by indexing and evaluation.

## 8. Before-and-after structure

Before R07A1:

```text
app/
  evaluation.py       # implementation inside production package
  evaluation_e2e.py
  phase7.py           # corpus constants, file I/O, and offline dataset implementation
  phase7_replay.py    # replay implementation inside production package
```

After R07A1:

```text
app/
  evaluation.py       # temporary compatibility facade
  evaluation_e2e.py   # unchanged; source identity work is deferred
  phase7.py           # temporary compatibility facade
  phase7_replay.py    # temporary compatibility facade
  infrastructure/
    corpus_artifacts.py # active corpus identity and local artifact I/O
evaluation/
  __init__.py
  retrieval.py        # canonical implementation
  replay.py           # canonical sanitized replay
  phase7_dataset.py   # canonical Phase 7 offline dataset contracts
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

R07A3 does not leave active collection values in the evaluation package. Dense/hybrid names are
derived from `PHASE7_RETRIEVAL_CONTRACT`, giving the supported indexing adapter the same canonical
configuration as runtime composition. Historical protected collection names and corpus-version text
remain explicit frozen constants. File operations stay in infrastructure because they cause local
filesystem side effects; the deterministic dataset module remains side-effect free except explicit
dataset reads.

The frozen-chunk loader cannot keep `EvaluationError` as its canonical infrastructure error without
reversing the dependency direction. `CorpusArtifactError` therefore owns infrastructure failures,
while the evaluation wrapper catches it and raises the same legacy type/message. `chunk_set_metadata`
needs no wrapper and is re-exported as the exact canonical function object.

`app.evaluation_e2e` remains at its historical path in Round 1. Current E2E, retrieval-closure,
runtime-readiness, and calibration-readiness commands hash that exact file. Moving it or replacing it
with a facade would change frozen run identity and require artifact regeneration. Production roots
cannot reach it, so keeping this pinned compatibility owner preserves behavior without contaminating
runtime dependencies. A future relocation requires a versioned identity transition, not a file move.

R07B1 starts CLI classification with the smallest current command that is read-only with respect to
corpus/Qdrant/provider state and is not source-hashed. The old module remains executable and exports
the exact canonical `main`; parser extraction only creates a test seam and does not change options.

The corpus audit belongs in `scripts.operations`, not `scripts.evaluation`: it checks local source-file
preconditions and emits metadata, but computes no retrieval or answer quality metric. It is retained
rather than archived because the active Phase 7 corpus runbook still uses it. Tests substitute a fake
PDF adapter and tiny placeholder files; no vendor manual content is opened or copied.

The dataset-v2 migration is archived without a compatibility shim. Its report exists locally and the
current walkthrough records dataset v2 as source-reviewed, approved, and frozen. No current code or
test imports the script, and no identity builder hashes it. Keeping the former command executable would
suggest that mutating frozen calibration/test files is supported; retaining the source under an
explicit archive preserves provenance without that ambiguity.

The typed-fact draft generator is likewise archived without a shim. Its pre-move test established that
retrieval ground truth stayed unchanged and the output required review. The current README and
evaluation manifest establish that calibration-v3 was subsequently approved and frozen; the manifest
hashes the dataset and draft content, not this generator source. Its historical walkthrough no longer
presents regeneration as a current command.

The calibration-v3 freeze command is archived without a shim after the same evidence check. Its
synthetic pre-move tests proved that approval changed only `review_status` and rejected preapproved or
ground-truth-mutated drafts. The active manifest records the frozen dataset and draft hashes but no
source hash for this command. Leaving a writer for the active dataset and manifest at top level would
misrepresent a completed approval ceremony as routine operation.

The answer-fact migration is also archived without a shim. Its pre-move characterization verified the
42-row mapping and narrow calibration 011/012 correction structure. Dataset approval/freeze happened
after this migration, and no current runtime, evaluator, artifact identity, or supported documentation
imports the command. The historical constants stay in the archived source; they are no longer a live
test contract.

The initial annotation generator is archived without a shim. It has no argument parser or approval
token: calling `main()` reads frozen chunks and overwrites calibration, test, and the review document.
The current tracked review file is instead a metadata-only frozen receipt, and the active calibration-v3
is already approved. Import-only characterization avoids both raw dataset access and mutation while
recording the source contract before removal.

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

Phase 7 dataset tests protect schema/default/error behavior, review rules, qrel validation, canonical
hashes, exact-content closure, source-manifest validation, and facade identity. Indexing CLI tests
protect collection defaults and import direction. Architecture tests prove dataset code depends on
canonical domain/retrieval interfaces while corpus infrastructure cannot depend on evaluation.

Frozen-artifact characterization protects record order, exact metadata hash, unreadable/empty/blank/
invalid/duplicate messages, canonical infrastructure errors, and evaluation error translation. The
indexing import guard now explicitly rejects both `app.phase7` and `app.evaluation`.

Dataset-validation CLI tests protect all four default paths, helper call order, report output, success
exit code, argparse error mapping, and old/new entry-point identity. Static tests reject compatibility
facade imports in the canonical CLI and implementation code in the old shim.

Corpus-audit CLI tests protect the two frozen source descriptors, parser defaults, streamed hashes,
page/text-layer metadata, document cleanup, sanitized output, missing-file errors, and shim identity.

The archive boundary test proves the old migration path is absent, the historical source exists under
`scripts.archive.phase7`, and the archived module uses canonical corpus/evaluation owners rather than
keeping compatibility facades alive.

The typed-fact archive boundary adds the same proof for the completed draft generator. Its former
functional test served as pre-move characterization; unsupported archive tools do not retain a test
contract that would imply ongoing support.

The calibration-freeze archive boundary additionally proves that its historical source uses canonical
domain, infrastructure, and evaluation owners rather than compatibility facades. Its synthetic
approval tests are retained as recorded pre-move characterization, not an ongoing supported contract.

The answer-fact archive boundary proves the completed migration has no old top-level path and depends
directly on canonical corpus-artifact and dataset owners. The mapping-integrity test was executed before
the move and then removed because archive provenance is intentionally unsupported.

The annotation-generator archive boundary proves its former path is absent and its only application
dependency is the canonical corpus-artifact loader. No functional test is added for an unsupported,
unguarded writer.

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

R07A3 pre-move characterization:

```text
python -m pytest -q tests/test_phase7.py tests/test_phase7_index_cli.py \
  tests/test_evaluation_e2e.py tests/test_phase7_calibration.py \
  tests/test_phase7_005_diagnostic.py tests/test_phase7_typed_fact_draft.py \
  tests/test_freeze_phase7_calibration_v3.py tests/test_freeze_phase7_heldout_v2.py \
  tests/test_architecture_boundaries.py
88 passed
```

R07A4 pre-move characterization added to the existing evaluator suite and executed against the old
implementation:

```text
python -m pytest -q tests/test_evaluate.py
15 passed
```

R07B1 pre-move CLI characterization:

```text
python -m scripts.validate_phase7_dataset --help
exit 0; four existing options and description preserved
```

R07B2 pre-move CLI characterization:

```text
python -m scripts.audit_phase7_corpus --help
exit 0; existing `--raw-dir` and `--output` options preserved
```

R07C1 pre-archive characterization:

```text
python -m scripts.migrate_phase7_dataset_v2 --help
exit 0; historical four-option contract recorded before removal
```

R07C2 pre-archive characterization:

```text
pytest -q tests/test_phase7_typed_fact_draft.py
1 passed
python -m scripts.draft_phase7_calibration_fact_types --help
exit 0; historical `--input` and `--output` options recorded before removal
```

R07C3 pre-archive characterization:

```text
pytest -q tests/test_freeze_phase7_calibration_v3.py
2 passed
python -m scripts.freeze_phase7_calibration_v3 --help
exit 0; six historical options recorded before removal
```

R07C4 pre-archive characterization:

```text
pytest -q tests/test_phase7.py::test_source_reviewed_answer_fact_mapping_is_complete_and_strict
1 passed
python -m scripts.apply_phase7_answer_facts --help
exit 0; historical `--calibration`, `--test`, and `--chunks` options recorded before removal
```

R07C5 pre-archive characterization:

```text
import scripts.generate_phase7_annotation_draft
PASS; `main()` has zero parameters and was not called
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
- `app.phase7` remains as a facade until evaluator CLI consumers are classified.
- `app.evaluation_e2e` is deliberately pinned to preserve source identity; relocation is deferred
  until a versioned artifact-identity transition is explicitly approved.
- Most evaluator, calibration, readiness, freeze, and migration CLIs are not classified yet.
- The corpus audit remains optional and requires the ingestion image's existing PDF dependency when
  run against local manuals; unit validation uses a fake and downloads nothing.
- `scripts.archive.phase7` is unsupported: archived tools are provenance, not current commands.
- Evaluation and production packages are still present in the same installed image.
- Frozen chunks and manifests remain explicit operational/reproducibility artifacts; R07A4 does not
  make the API read them per request and does not authorize deleting them.
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
7. Why do collection constants and atomic file writes not belong in the dataset module?
8. Why does evaluation translate `CorpusArtifactError` instead of infrastructure importing
   `EvaluationError`?
9. Why is preserving a source-hashed compatibility file safer than replacing it with a facade?

## 16. Interview summary

R07A1–R07A4 turn evaluation ownership into an enforceable architectural boundary without changing the
retrieval benchmark contract. Retrieval metrics, sanitized replay, and Phase 7 dataset contracts now
live in a top-level offline package and depend inward on public domain contracts. Active corpus
identity, frozen-chunk parsing, and file side effects have a distinct infrastructure owner, so
supported indexing no longer imports evaluation. Identity-preserving facades, error translation,
golden tests, and static reachability checks keep existing consumers stable during migration.

R07B1 applies the same boundary to inbound evaluation adapters: the canonical command is visibly
classified, while the documented legacy invocation remains stable and contains no business logic.

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

R07A2 was committed as `8ab8d5a refactor: isolate sanitized evaluation replay`.

R07A3 validation:

| Check | Result |
| --- | --- |
| Pre-move characterization | PASS — `88 passed` |
| Focused Ruff | PASS — `All checks passed!` |
| Focused offline pytest | PASS — `92 passed` |
| Full facade export identity | PASS |
| UTF-8/atomic artifact writer characterization | PASS |
| Collection contract/import identity smoke | PASS |
| Full Ruff | PASS — `All checks passed!` |
| Full offline pytest | PASS — `406 passed, 1 warning` |
| `docker compose config --quiet` | PASS |
| Local Markdown links | PASS — all links in this module document resolve |
| `git diff --check` | PASS |

R07A3 was committed as `87a0ce1 refactor: separate phase7 dataset and corpus artifacts`.

R07A4 validation:

| Check | Result |
| --- | --- |
| Pre-move frozen-artifact characterization | PASS — `15 passed` |
| Focused Ruff | PASS — `All checks passed!` |
| Focused offline pytest | PASS — `60 passed` |
| Infrastructure/evaluation error translation | PASS |
| Exact chunk-set hash and record order | PASS |
| Supported indexing import guard | PASS |
| Full Ruff | PASS — `All checks passed!` |
| Full offline pytest | PASS — `408 passed, 1 warning` |
| `docker compose config --quiet` | PASS |
| Local Markdown links | PASS — all links in this module document resolve |
| `git diff --check` | PASS |

Proposed R07A4 commit after user review:

```text
refactor: share frozen corpus artifact utilities
```

R07B1 validation:

| Check | Result |
| --- | --- |
| Legacy pre-move `--help` | PASS — exit `0` |
| Focused Ruff | PASS — `All checks passed!` |
| Focused offline pytest | PASS — `29 passed` |
| Legacy/canonical `main` and parser identity | PASS |
| Legacy and canonical `--help` | PASS — identical output, exit `0` |
| Full Ruff | PASS — `All checks passed!` |
| Full offline pytest | PASS — `412 passed, 1 warning` |
| Local Markdown links | PASS — all links in this module document resolve |
| `git diff --check` | PASS |

R07A4 and R07B1 were committed together as:

```text
b9458bc refactor: separate corpus artifacts from evaluation commands
```

R07B2 validation:

| Check | Result |
| --- | --- |
| Legacy pre-move `--help` | PASS — exit `0` |
| Focused Ruff | PASS — `All checks passed!` |
| Focused offline pytest with fake PDFs | PASS — `30 passed` |
| Legacy/canonical exports | PASS — exact object identity |
| Legacy and canonical `--help` | PASS — identical output, exit `0` |
| Source-hash reference audit | PASS — command is not hashed by current identity builders |
| Full Ruff | PASS — `All checks passed!` |
| Full offline pytest | PASS — `416 passed, 1 warning` |
| Local Markdown links | PASS — all links in this module document resolve |
| `git diff --check` | PASS |

Proposed R07B2 commit after user review:

```text
refactor: classify phase7 corpus audit command
```

R07B2 was committed as `09a3352 refactor: classify phase7 corpus audit command`.

R07C1 validation:

| Check | Result |
| --- | --- |
| Legacy pre-archive `--help` | PASS — exit `0`; four options recorded |
| Focused Ruff | PASS — `All checks passed!` |
| Focused architecture pytest | PASS — `28 passed` |
| Archived command `--help` | PASS — exit `0`; four-option contract preserved |
| Removed top-level command | PASS — `scripts.migrate_phase7_dataset_v2` is unavailable as intended |
| Full Ruff | PASS — `All checks passed!` |
| Full offline pytest | PASS — `417 passed, 1 warning` |
| Local Markdown links | PASS — all links in the changed documentation resolve |
| Pinned evaluator source | PASS — blob `b8be722d43bc34c8bec00dfc2574d0a6341ab738` unchanged |
| Artifact/raw-data scope | PASS — no changes under `artifacts/` or `data/raw/` |
| `git diff --check` | PASS |

The warning remains the existing Starlette `TestClient`/`httpx` deprecation warning. R07C1 does not
change dependencies.

Proposed R07C1 commit after user review:

```text
chore: archive completed phase7 dataset migration
```

R07C1 was committed as `808907b chore: archive completed phase7 dataset migration`.

R07C2 validation:

| Check | Result |
| --- | --- |
| Existing pre-archive functional test | PASS — `1 passed` |
| Legacy pre-archive `--help` | PASS — exit `0`; two options recorded |
| Focused Ruff | PASS — `All checks passed!` |
| Focused architecture pytest | PASS — `29 passed` |
| Archived command `--help` | PASS — exit `0`; two-option contract preserved |
| Removed top-level command | PASS — `scripts.draft_phase7_calibration_fact_types` is unavailable as intended |
| Full Ruff | PASS — `All checks passed!` |
| Full offline pytest | PASS — `417 passed, 1 warning` |
| Current-command reference search | PASS — no supported docs/code/tests reference the old path |
| Local Markdown links | PASS — all links in the changed documentation resolve |
| Pinned evaluator source | PASS — blob `b8be722d43bc34c8bec00dfc2574d0a6341ab738` unchanged |
| Frozen-data/artifact scope | PASS — no changes under `data/eval/`, `data/raw/`, or `artifacts/` |
| `git diff --check` | PASS |

The warning remains the existing Starlette `TestClient`/`httpx` deprecation warning. R07C2 does not
change dependencies.

Proposed R07C2 commit after user review:

```text
chore: archive completed phase7 typed-fact draft
```

R07C2 was committed as `b582c87 chore: archive completed phase7 typed-fact draft`.

R07C3 validation:

| Check | Result |
| --- | --- |
| Existing pre-archive synthetic tests | PASS — `2 passed` |
| Legacy pre-archive `--help` | PASS — exit `0`; six options recorded |
| Focused Ruff | PASS — `All checks passed!` |
| Focused architecture pytest | PASS — `30 passed` |
| Archived command `--help` | PASS — exit `0`; six-option contract preserved |
| Removed top-level command | PASS — `scripts.freeze_phase7_calibration_v3` is unavailable as intended |
| Full Ruff | PASS — `All checks passed!` |
| Full offline pytest | PASS — `416 passed, 1 warning` |
| Current-command reference search | PASS — no supported docs/code/tests reference the old path |
| Local Markdown links | PASS — all links in the changed documentation resolve |
| Pinned evaluator source | PASS — blob `b8be722d43bc34c8bec00dfc2574d0a6341ab738` unchanged |
| Frozen-data/artifact scope | PASS — no changes under `data/eval/`, `data/raw/`, or `artifacts/` |
| `git diff --check` | PASS |

The suite count decreases by one because two synthetic unit tests for the unsupported command were
replaced by one archive-boundary test. The warning remains the existing Starlette
`TestClient`/`httpx` deprecation warning. R07C3 does not change dependencies.

Proposed R07C3 commit after user review:

```text
chore: archive completed phase7 calibration freeze
```

R07C3 was committed as `c31fb56 chore: archive completed phase7 calibration freeze`.

R07C4 validation:

| Check | Result |
| --- | --- |
| Existing pre-archive mapping test | PASS — `1 passed` |
| Legacy pre-archive `--help` | PASS — exit `0`; three options recorded |
| Focused Ruff | PASS — `All checks passed!` |
| Focused architecture/dataset pytest | PASS — `43 passed` |
| Archived command `--help` | PASS — exit `0`; three-option contract preserved |
| Removed top-level command | PASS — `scripts.apply_phase7_answer_facts` is unavailable as intended |
| Full Ruff | PASS — `All checks passed!` |
| Full offline pytest | PASS — `416 passed, 1 warning` |
| Current-command reference search | PASS — no supported docs/code/tests reference the old path |
| Local Markdown links | PASS — all links in the changed documentation resolve |
| Pinned evaluator source | PASS — blob `b8be722d43bc34c8bec00dfc2574d0a6341ab738` unchanged |
| Frozen-data/artifact scope | PASS — no changes under `data/eval/`, `data/raw/`, or `artifacts/` |
| `git diff --check` | PASS |

The test count is unchanged because one historical mapping test was replaced by one archive-boundary
test. The warning remains the existing Starlette `TestClient`/`httpx` deprecation warning. R07C4 does
not change dependencies.

Proposed R07C4 commit after user review:

```text
chore: archive completed phase7 answer-fact migration
```

R07C4 was committed as `4f38d17 chore: archive completed phase7 answer-fact migration`.

R07C5 validation:

| Check | Result |
| --- | --- |
| Pre-archive import-only characterization | PASS — zero-parameter `main()` recorded but not called |
| Focused Ruff | PASS — `All checks passed!` |
| Focused architecture pytest | PASS — `32 passed` |
| Archived module import | PASS — `main()` was not called |
| Removed top-level module | PASS — old import path is unavailable as intended |
| Full Ruff | PASS — `All checks passed!` |
| Full offline pytest | PASS — `417 passed, 1 warning` |
| Current-command reference search | PASS — no supported docs/code/tests reference the old path |
| Local Markdown links | PASS — all links in the changed documentation resolve |
| Pinned evaluator source | PASS — blob `b8be722d43bc34c8bec00dfc2574d0a6341ab738` unchanged |
| Frozen-data/artifact scope | PASS — datasets, raw PDFs, artifacts, and review receipt are unchanged |
| `git diff --check` | PASS |

The suite gains one archive-boundary test because no functional test existed to remove. The warning
remains the existing Starlette `TestClient`/`httpx` deprecation warning. R07C5 does not change
dependencies.

Proposed R07C5 commit after user review:

```text
chore: archive completed phase7 annotation generator
```

## 18. Status

`IN_PROGRESS` — R07A1–R07C4 are complete and committed through `4f38d17`. R07C5 is implemented,
validated, and awaiting user review. Later R07 slices remain outside this slice.

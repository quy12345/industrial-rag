
# AGENTS.md — Industrial Technical Manual RAG

## 1. Project mode

This repository is an existing, verified Industrial Technical Manual RAG
prototype. It already runs end-to-end and has tests and benchmark artifacts.

Round 1 incremental, behavior-preserving refactoring is complete. Its outcomes are:

- clearer module responsibilities;
- cleaner dependency boundaries;
- easier maintenance;
- reproducible evaluation;
- portfolio and interview readiness.

Do not rewrite the project from scratch.

Do not change retrieval algorithms, models, thresholds, collections or
evaluation data during a behavior-preserving refactor.

The user approved `R00 — Phase 7-only baseline` on 2026-08-21. R00 retired
`data/raw/manual.pdf` and the Phase 6 corpus from the active product surface.
The repository implementation completed this narrowly scoped transition on
2026-08-22. R01–R07 subsequently completed Round 1 on 2026-08-24. This completion does not authorize
retrieval tuning, Phase 7 data changes, re-indexing, deletion of legacy Qdrant collections, or an
automatic start of Round 2.

## 2. Working language

- Explain plans, results and technical concepts to the user in Vietnamese.
- Write source code, identifiers, tests, logs and commit messages in English.
- Explain unfamiliar technical terms briefly when they first appear.

## 3. Current repository

- Target Python is 3.11.
- Use the same interpreter for all commands.
- Run:

  ```bash
  python -m ruff check .
  python -m pytest -q
  ```

* `app/` contains the production modular monolith: API/application/domain/infrastructure layers plus
  documented compatibility/source-identity anchors.
* `evaluation/` owns offline evaluation schemas, replay and metrics.
* `scripts/` contains thin supported adapters and explicit historical archives; read
  `scripts/README.md` before using integration commands.
* `tests/` must remain offline and use fake models or in-memory Qdrant.

The current directory structure is the completed Round 1 architecture. Further structural moves
require a separately approved module or Round 2 plan.

The pre-R00 Round 1 audit, target architecture and completion record are indexed in
`docs/portfolio-cleanup/00-index.md`. The implementation sequence is proposed
and recorded in `docs/portfolio-cleanup/03-round-1-roadmap.md`. Pre-R00 observations remain as
historical audit evidence rather than descriptions of current ownership.

## 4. Frozen baseline

The repository has one active frozen retrieval contract and one historical
archive contract. Do not mix their collections, corpus identity or runtime
settings.

### Phase 6 historical archive contract

* Document ID: `manual-77d5dae4c2c5`

* Frozen chunks: `99`

* Chunk-ID hash:

  ```text
  bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be
  ```

* Direct-evidence qrels: `30`

* Collections:

  ```text
  industrial_manual_chunks
  industrial_manual_chunks_v2
  ```

This corpus is not an active product or evaluation target. Preserve its
collections and historical evidence. Its unsupported tools live under
`scripts/archive/phase6/`; do not use it for new development or portfolio claims.

### Phase 7 active Compose/demo contract

* Document IDs:

  ```text
  atv320-installation-manual-en-nve41289-09-c181b4d7f11b
  atv320-programming-manual-en-nve41295-06-f5e9bb48167a
  ```

* Frozen chunks: `2753`

* Chunk-ID hash:

  ```text
  2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d
  ```

* Collections:

  ```text
  industrial_manual_phase7_dense_v1
  industrial_manual_phase7_hybrid_v1
  ```

* The immutable runtime profile is defined by
  `PHASE7_RETRIEVAL_CONTRACT` and `PHASE7_CALIBRATION_FUSION_PROFILE`.

`Settings`, `.env.example` and Docker Compose select `phase7` for active API
and local runtime use. `Settings` rejects `phase6`; historical identity exists
only in the archive and destructive-write guards. Do not partially override a
frozen profile or combine values from the two contracts.

Never change chunks, expected evidence, qrels or expected answers to improve
a metric.

Held-out v2 is an exposed regression benchmark. It is no longer an unseen
final benchmark and must not be used for further tuning.

Never claim a metric unless the corresponding command was executed and a
valid artifact was produced.

## 5. Round 1 scope

Round 1 portfolio cleanup is complete.

R00 is the completed prerequisite transition and R01–R07 are the completed behavior-preserving
sequence. Do not reopen a completed module or begin Round 2 without explicit user approval.

Allowed work:

* add characterization tests;
* clarify module responsibilities;
* isolate production code from evaluation;
* consolidate configuration;
* improve names and package boundaries;
* make API, CLI and Streamlit adapters thinner;
* archive confirmed one-off scripts;
* remove confirmed dead code;
* improve documentation;
* preserve existing public behavior.

Deferred to Round 2:

* retrieval tuning;
* reranker optimization;
* embedding model changes;
* pooling or dimension changes;
* Qdrant schema migration;
* collection redesign;
* re-indexing;
* evidence-threshold calibration;
* provider changes;
* benchmark v3;
* performance optimization.

Do not mix Round 2 improvements into Round 1 refactoring.

## 6. Architecture direction

The target is a modular monolith with this dependency direction:

```text
API / CLI / Streamlit
        ↓
Application services
        ↓
Domain contracts and ports
        ↑
Infrastructure adapters
```

Rules:

* Domain code must not import FastAPI, Streamlit, Qdrant, model SDKs,
  provider SDKs or evaluation.
* Application services coordinate use cases through explicit contracts.
* Infrastructure adapters implement external concerns.
* API, CLI and Streamlit must remain thin inbound adapters.
* Streamlit communicates with the application through FastAPI.
* Production runtime must not import evaluation.
* Evaluation may use public application/domain interfaces.
* Configuration must have one canonical source.
* Do not introduce microservices, message queues or a dependency-injection
  framework without a demonstrated use case.

## 7. One module at a time

Each turn may implement only one approved refactoring module or one small
vertical slice of that module.

Do not:

* continue automatically to the next module;
* expand scope because another change appears convenient;
* refactor unrelated files;
* combine structural refactoring with algorithm changes;
* combine bug fixes with refactoring unless explicitly approved;
* modify more than approximately eight meaningful files without proposing
  a smaller slice first.

After completing a module, stop at:

```text
WAITING_FOR_USER_REVIEW
```

Continue only after the user explicitly requests:

```text
TIẾP TỤC MODULE Rxx
```

## 8. Before implementation

Before changing code, report:

1. The problem being addressed.
2. The module’s position in the runtime pipeline.
3. Current input and output.
4. Public behavior that must remain unchanged.
5. Files expected to change.
6. Characterization tests required before refactoring.
7. Validation commands.
8. Work intentionally excluded.
9. Proposed Conventional Commit.

If a public contract, dependency, data schema or architecture decision is
unclear, stop and ask the user.

## 9. Implementation rules

* Read relevant source code and tests before editing.
* Prefer small, reviewable patches.
* Add characterization tests before moving or restructuring risky behavior.
* Keep public behavior unchanged during refactoring.
* Use type hints for public functions and classes.
* Avoid `Any` except at isolated untyped SDK boundaries.
* Do not add dependencies when the standard library or existing dependencies
  are sufficient.
* Do not change a test merely to make incorrect behavior pass.
* Do not create network calls during module import.
* Do not use global mutable clients or models.
* Comments should explain why, not repeat what the code already states.
* Temporary compatibility shims must have an explicit removal module.

## 10. Module documentation

Every refactoring module must create or update exactly one learning document:

```text
docs/modules/Rxx-<module-slug>.md
```

If a module is implemented through multiple slices, update the same document
instead of creating multiple unrelated documents.

The document must contain:

1. Goal and scope.
2. Position in the system.
3. Relevant background concepts.
4. Input, output and contracts.
5. Step-by-step data flow.
6. Responsibilities of changed files.
7. Important symbols and why they exist.
8. Before-and-after structure.
9. Design decisions and trade-offs.
10. Tests and the behavior each test protects.
11. Commands and expected results.
12. Small usage example.
13. Common failures and debugging.
14. Current limitations.
15. Self-check questions.
16. Interview summary.
17. Validation results and proposed commit.
18. Status: `IN_PROGRESS` or `COMPLETE`.

Documentation must describe the code that actually exists. Do not document
planned behavior as if it has already been implemented.

If the user asks for clarification, update the module document with the
clarification instead of leaving the explanation only in chat.

## 11. Testing and evaluation

* Domain and algorithm code requires deterministic unit tests.
* Adapter code requires fake/stub contract tests before real integration tests.
* API tests must cover schemas and error mapping.
* Retrieval tests must use deterministic golden examples.
* Provider tests must use fake providers by default.
* Model downloads never belong in unit tests or Docker builds.
* Real Qdrant/model checks require explicit integration commands.
* Pytest must ignore repository `.env` and matching process settings.
* Local credentials must not affect deterministic tests or appear in output.

Run focused tests first, then the appropriate full checks.

Never report PASS for a command that was not executed.

## 12. Safety

Preserve all four frozen Qdrant collections and all named volumes:

```text
industrial_manual_chunks
industrial_manual_chunks_v2
industrial_manual_phase7_dense_v1
industrial_manual_phase7_hybrid_v1
```

Without explicit user approval, never:

* recreate or delete a Qdrant collection;
* write or delete production Qdrant points;
* re-index the corpus;
* prune Docker;
* build a large image;
* download a large model;
* call Gemini, OpenAI or another real provider;
* run calibration or held-out evaluation;
* read raw held-out payloads;
* send manual content to an external provider.

Query generation may only run after the evidence gate.

Never:

* return unvalidated model citations;
* log API keys;
* log full prompts, questions or evidence content;
* commit `.env`, source PDFs, model caches or raw benchmark payloads;
* silently fall back to another pipeline.

## 13. Git workflow

Do not commit, push, merge, rebase, amend or tag unless the user explicitly
requests it.

Each completed module should produce one or more focused Conventional Commits.

Rules:

* one clear responsibility per commit;
* keep refactor separate from bug fixes;
* keep dependency upgrades separate from structural changes;
* commit module documentation with its code and tests;
* do not fabricate, backdate or rewrite history to hide development work;
* do not include unrelated user changes.

Before a requested commit, show:

```text
git status --short
git diff --stat
tests/checks result
proposed commit message
```

Example commit sequence:

```text
test: capture current query pipeline behavior
refactor: establish application service boundaries
refactor: isolate retrieval and reranking components
refactor: separate evaluation from production runtime
docs: document the cleaned runtime architecture
```

## 14. Definition of Done

A module is complete only when:

* approved scope is implemented;
* public behavior is preserved;
* characterization tests exist where needed;
* relevant tests and checks pass;
* no unrelated code was changed;
* no hidden placeholder or silent fallback was introduced;
* module documentation reflects the actual implementation;
* documentation status is `COMPLETE`;
* a Conventional Commit is proposed;
* the agent stops at `WAITING_FOR_USER_REVIEW`.

## 15. Stop conditions

Stop and ask the user when:

* refactoring requires changing public behavior;
* a confirmed bug requires a separate fix;
* a large dependency must be introduced;
* Qdrant, provider, model download or re-indexing is required;
* the working tree contains overlapping user changes;
* the module affects too many responsibilities;
* tests reveal that the planned scope must expand significantly;
* the baseline and documentation contradict each other.

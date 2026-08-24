# R09 — Domain contracts and ranking policies

## 1. Goal and scope

R09 gives runtime records and deterministic ranking policies one clear canonical owner. It removes
the ambiguous `app.models` dependency from canonical production and evaluation modules and splits a
723-line mixed ranking module into focused policy modules.

The module preserves all Pydantic schemas, ranks, scores, ordering rules, metadata fields, frozen
profiles, API behavior, and historical source-identity anchors. Compatibility facades remain only
for the explicit R12 migration.

## 2. Position in the system

```text
FastAPI health adapter -> app.contracts.health

retrieval adapters -> app.domain.retrieval records
                           |
query text -> query_roles  |
candidate text -> list_completeness
component ranks -> fusion  |
                           v
                 reranking application service
```

These modules sit between retrieval adapters and application orchestration. They are deterministic
and have no Qdrant, model, provider, dataset, or evaluation dependency.

## 3. Relevant background concepts

- A **data transfer object (DTO)** describes data crossing an adapter boundary.
- A **domain record** represents data used by application and domain logic.
- A **policy** is deterministic business logic without external I/O.
- A **compatibility facade** re-exports canonical objects for a known legacy consumer; it does not
  own another implementation.
- **Class identity** means old and canonical imports resolve to the same Python object.
- **Reciprocal-rank fusion (RRF)** combines one-based ranks without treating unrelated raw scores as
  comparable probabilities.

## 4. Input, output, and contracts

`HealthResponse` and `ReadinessResponse` retain the exact HTTP response fields. `RetrievedChunk` and
`RetrievalCandidate` retain their field order, defaults, optional signals, and one-based rank
constraints.

Ranking inputs are query text, trusted document metadata, component ranks, and the immutable
`Phase7FusionProfile`. Outputs are copied `RetrievalCandidate` objects with the same deterministic
score, rank, tie-break, and metadata behavior as before R09.

`app.models`, `app.domain.policies.ranking`, and `app.phase7_optimization` preserve exact export
identity until R12. The frozen Phase 7 profile object is unchanged.

## 5. Step-by-step data flow

1. Dense and sparse adapters create canonical retrieval records.
2. `query_roles` derives an auditable role and confidence from query text only.
3. `fusion` applies generic RRF or the frozen weighted-RRF and coverage policy.
4. The application service sends the selected pool to the cross-encoder port.
5. `fusion` optionally applies the configured rank-only role prior.
6. `list_completeness` optionally derives structural counts and reorders only its bounded window.
7. Evidence selection receives the final canonical candidate list.
8. Compatibility imports resolve to these same functions, classes, constants, and records.

## 6. Responsibilities of changed files

| Path | Responsibility after R09 |
| --- | --- |
| [`app/contracts/health.py`](../../app/contracts/health.py) | Canonical health and readiness DTOs. |
| [`app/domain/retrieval.py`](../../app/domain/retrieval.py) | Retrieval records, application port, and candidate assembly. |
| [`app/domain/policies/query_roles.py`](../../app/domain/policies/query_roles.py) | Query normalization, cue matching, role, and confidence inference. |
| [`app/domain/policies/list_completeness.py`](../../app/domain/policies/list_completeness.py) | List intent, structural features, and bounded completeness ordering. |
| [`app/domain/policies/fusion.py`](../../app/domain/policies/fusion.py) | Generic/weighted RRF, frozen profile, coverage selection, and rank priors. |
| Historical `app/domain/policies/ranking.py` (removed in R12B) | Temporary policy re-export for source compatibility. |
| Historical `app/models.py` (removed in R12B) | Temporary record/DTO re-export for source compatibility. |
| [`app/application/reranking_service.py`](../../app/application/reranking_service.py) | Orchestrates the focused policy interfaces. |
| [`evaluation/replay.py`](../../evaluation/replay.py) | Uses canonical rank-policy interfaces for offline replay. |

Other changed runtime files contain only direct canonical import migrations; their responsibilities
are unchanged.

## 7. Important symbols and why they exist

- `RetrievalCandidate` owns the serialized retrieval/reranking signals and rank constraints.
- `QueryRetriever` is the application-facing retrieval port.
- `QueryRoleInference` records query-derived role, confidence, and cue IDs.
- `Phase7FusionProfile` is the validated immutable shape of the frozen ranking profile.
- `PHASE7_CALIBRATION_FUSION_PROFILE` is the active profile object and retains exact identity.
- `fuse_rrf` and `fuse_weighted_rrf` keep generic and configured fusion visibly separate.
- `select_coverage_preserving_candidates` enforces the fixed reranker candidate budget.
- `apply_role_aware_rank_fusion` applies rank-only priors without mutating reranker scores.
- `apply_relation_list_completeness_fallback` owns the active bounded completeness policy.

## 8. Before-and-after structure

```text
Before R09
  app.models                         8 definitions/aliases mixed together
  policies/ranking.py               723 lines, 3 policy responsibilities
  canonical runtime -> app.models   10 temporary dependency edges

After R09
  contracts/health.py               HTTP DTOs
  domain/retrieval.py               retrieval records and ports
  policies/query_roles.py           query-only role inference
  policies/list_completeness.py      completeness features and ordering
  policies/fusion.py                 fusion profiles and rank combination
  models.py + policies/ranking.py    compatibility exports only
  canonical runtime -> owners        direct imports
```

The architecture debt allowlist falls from 18 edges before R09 to 5. The 723-line mixed module is now
a 71-line explicit facade; no algorithm code remains there.

## 9. Design decisions and trade-offs

Health DTOs stay distinct because merging them could change OpenAPI component identity. Retrieval
records remain Pydantic models because validation and serialization are current contracts.

All fusion logic lives together because generic RRF, weighted RRF, coverage selection, the frozen
profile, and post-rerank priors share the same rank semantics. Query analysis and completeness
features are separate because they operate on different inputs and can be understood independently.

R09 migrates all canonical consumers in one coherent slice. One-line import edits span several files,
but only the policy owners, facade, architecture test, and learning document change meaningful logic
or structure. Archived workflows and protected root anchors keep their paths until R12.

## 10. Tests and the behavior each test protects

| Test area | Protected behavior |
| --- | --- |
| Compatibility identity | Old model and policy exports are the exact canonical objects. |
| RetrievalCandidate schema | Field order, defaults, copied metadata, and one-based ranks. |
| Query-role tests | Bilingual cue boundaries, ambiguity, and confidence. |
| Completeness tests | Feature counts, bounded window, metadata, errors, and ordering. |
| Fusion tests | Formulae, reserves, tie-breaks, role/RRF priors, and validation errors. |
| Reranking tests | Candidate pools, cross-encoder order, timings, and no silent fallback. |
| Evidence/replay tests | Final evidence order and sanitized rank-only replay. |
| Architecture matrix | Canonical consumers cannot return to facades. |

## 11. Commands and expected results

Focused validation:

```powershell
python -m ruff check app/domain/policies app/application app/domain `
  app/infrastructure/qdrant evaluation/replay.py tests/test_architecture_boundaries.py `
  tests/test_phase7_optimization.py
python -m pytest -q tests/test_phase7_optimization.py tests/test_reranking.py `
  tests/test_evidence_selection.py tests/test_phase7_replay.py `
  tests/test_architecture_boundaries.py
```

Module validation:

```powershell
python -m ruff check .
python -m pytest -q
docker compose config --quiet
git diff --check
```

Expected: no external services or data mutation, stable source pins, and identical protected hashes.

## 12. Small usage example

New code imports only the policy it needs:

```python
from app.domain.policies.query_roles import infer_query_role
from app.domain.retrieval import RetrievalCandidate

role = infer_query_role("How should the drive be installed?")
candidate = RetrievalCandidate(
    chunk_id="chunk-1",
    document_id="installation-manual",
    filename="installation.pdf",
    text="Disconnect all power before installation.",
    page_numbers=[42],
    headings=["Safety"],
    content_type="text",
    score=0.8,
)
```

Historical code importing the same objects through `app.models` or
`app.phase7_optimization` still receives identical objects during the migration window.

## 13. Common failures and debugging

- Different object identities indicate a copied definition instead of a re-export.
- A profile or ordering regression indicates code changed during the move; compare the focused
  golden tests before changing expectations.
- A missing metadata key usually means a completeness or role-prior function moved without its full
  update block.
- An architecture mismatch with more than five edges means a canonical module imported a facade.
- An import cycle means `fusion` started depending on completeness; the allowed direction is
  completeness to the shared policy error, never the reverse.
- A source-pin mismatch is a stop condition, not a reason to update the pin.

## 14. Current limitations

- `app.models` and `app.domain.policies.ranking` remain for explicit compatibility consumers.
- Protected root modules such as `app.reranking` and `app.phase7_optimization` remain until R12.
- Five temporary dependency edges remain around content identity and retrieval composition.
- Retrieval ports still expose infrastructure-shaped arguments; R11 owns that cleanup.
- Phase 6 archive ownership and supported CLI naming are outside R09.
- Retrieval tuning, thresholds, models, collections, and performance are unchanged and deferred.

## 15. Self-check questions

1. Why is `RetrievalCandidate` a domain record while health responses are adapter contracts?
2. Why do generic and weighted RRF belong in one module?
3. Which inputs may query-role inference inspect?
4. How does a compatibility re-export preserve object identity?
5. Why does R09 not delete the two facades immediately?
6. What proves that splitting policies did not change candidate ordering?

## 16. Interview summary

R09 replaces two umbrella modules with explicit owners while preserving exact runtime behavior.
Health DTOs and retrieval records now live at their real boundaries; query analysis, list
completeness, and rank fusion are independently readable policies. Canonical runtime/evaluation code
imports those owners directly, compatibility paths contain no implementation, and the executable
dependency matrix reduces known debt from 18 edges to 5.

## 17. Validation results and proposed commit

Validation receipts:

| Check | Result |
| --- | --- |
| R09A pre-change focused pytest, Python 3.11.15 | PASS — 45 tests, 1 warning |
| R09A focused pytest | PASS — 46 tests, 1 warning |
| R09A full pytest | PASS — 404 tests, 1 warning |
| R09A full Ruff, Compose, protected hashes, pins, links, diff scope | PASS |
| R09A commit | `a0d6007 refactor: establish canonical runtime record ownership` |
| R09B pre-change focused pytest, Python 3.11.15 | PASS — 104 tests |
| R09B post-change focused Ruff | PASS |
| R09B post-change focused pytest, Python 3.11.15 | PASS — 104 tests |
| R09B full Ruff | PASS |
| R09B full pytest, Python 3.11.15 | PASS — 404 tests, 1 warning |
| Docker Compose configuration | PASS |
| Protected data and artifacts | PASS — all 86 files unchanged |
| E2E source pins and held-out v2 worktree | PASS — unchanged |
| Local Markdown links and inline source paths | PASS |
| `git diff --check` and 18-file scope | PASS |

Proposed R09B Conventional Commit after review:

```text
refactor: separate ranking policies and canonical imports
```

## 18. Status

`COMPLETE`

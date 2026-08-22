# R04 — Retrieval and reranking boundaries

## 1. Goal and scope

R04 separates framework-neutral retrieval policies from evaluation helpers, Qdrant search, and model
adapters without changing ranking behavior. This document is shared by all R04 slices.

R04A is implemented. It moves dense-result conversion and dense/sparse candidate union into the
domain, retains `app.candidate_audit` as a compatibility facade for these two public imports, and
removes the production reranking path's dependency on that evaluation-oriented module. R04B–R04E
remain pending.

## 2. Position in the system

```text
dense RetrievedChunk results ──┐
                               ├── app.domain.retrieval ──> RetrievalCandidate pool
sparse RetrievalCandidate list ┘                                  │
                                                                  ↓
                                                       app.reranking (transitional)
                                                                  │
                                                                  ↓
                                                      ranked/evidence candidates

evaluation cases ──> app.candidate_audit ──> qrel-oriented audit records
```

R04A changes ownership and dependency direction only. It does not query Qdrant, initialize a model,
call a provider, execute a benchmark, or modify the frozen Phase 7 retrieval profile.

## 3. Relevant background concepts

Candidate assembly is deterministic runtime policy: it converts retrieval results into a common
record and combines dense and sparse pools without mixing incomparable raw scores. This policy does
not need qrels, benchmark cases, or evaluator metrics, so it belongs in the domain.

An evaluation audit is a consumer of runtime candidates. It may compute direct-evidence ranks and
diagnostic pool summaries, but production must not depend on it. A compatibility facade keeps old
imports valid while allowing production code to use the canonical domain owner.

## 4. Input, output, and contracts

`dense_results_to_candidates(results)` accepts an ordered sequence of `RetrievedChunk` objects and
returns `RetrievalCandidate` objects with all citation metadata preserved. `score` and `dense_score`
both receive the original dense score. One-based `dense_rank` is assigned after sorting by descending
dense score and then ascending `chunk_id`.

`union_dense_sparse_candidates(dense_candidates, sparse_candidates)` returns one candidate per
`chunk_id`. A dense candidate remains the representative when both pools contain the same chunk; only
its `sparse_score` and `sparse_rank` are copied from the sparse candidate. Sparse-only candidates are
preserved. Output order is the lowest available component rank, then `chunk_id`. The function does
not calculate a fused score or rank.

The two existing imports from `app.candidate_audit` remain valid and are the same function objects as
the canonical domain exports.

## 5. Step-by-step data flow

1. Dense search returns `RetrievedChunk` records with citation metadata and a raw dense score.
2. `dense_results_to_candidates` copies those fields into the common candidate representation.
3. Candidates are sorted by the established score/chunk-ID tie-break rule.
4. One-based dense ranks are assigned without modifying any other provenance field.
5. Sparse search supplies candidates containing sparse score and rank provenance.
6. `union_dense_sparse_candidates` indexes dense candidates by `chunk_id`.
7. Sparse-only candidates are added; duplicates enrich the dense representative with sparse fields.
8. The union is ordered by its best component rank and stable chunk-ID tie-break.
9. Reranking consumes the canonical domain functions directly.
10. Evaluation audit code may consume the same functions through its temporary compatibility exports.

## 6. Responsibilities of changed files

- [`app/domain/retrieval.py`](../../app/domain/retrieval.py) owns deterministic, SDK-free candidate
  conversion and union policy.
- [`app/candidate_audit.py`](../../app/candidate_audit.py) owns qrel/evaluation audit behavior and
  temporarily re-exports the two moved functions for compatibility.
- [`app/reranking.py`](../../app/reranking.py) consumes candidate assembly from the domain instead of
  importing the evaluation-oriented audit module.
- [`tests/test_candidate_audit.py`](../../tests/test_candidate_audit.py) protects the exact candidate
  mapping, stable union behavior, and compatibility-export identity.
- [`tests/test_architecture_boundaries.py`](../../tests/test_architecture_boundaries.py) proves that
  active reranking imports the domain owner and no longer reaches evaluation through candidate audit.

## 7. Important symbols and why they exist

- `dense_results_to_candidates`: canonical mapping from dense search records to ranked candidates.
- `union_dense_sparse_candidates`: deterministic pool union that preserves component provenance and
  deliberately avoids mixing raw dense and sparse scores.
- `candidate_assembly` in `app.candidate_audit`: explicit facade reference that makes temporary
  compatibility ownership visible.
- `KNOWN_ACTIVE_RUNTIME_EVALUATION_IMPORTS`: executable debt inventory; after R04A it records only the
  remaining direct `app.reranking → app.evaluation` dependency owned by R04.

## 8. Before-and-after structure

Before R04A:

```text
app.candidate_audit   runtime candidate assembly + evaluation audit
        ↑
app.reranking        production path imports evaluation-oriented module
```

After R04A:

```text
app.domain.retrieval  canonical candidate assembly
        ↑                         ↑
app.reranking         app.candidate_audit (compatibility + evaluation audit)
```

## 9. Design decisions and trade-offs

- The algorithms were moved without changing sort keys, rank numbering, representative selection, or
  copied fields. Renaming or simplifying them would obscure behavior preservation.
- The domain functions continue to use established Pydantic records from `app.models`. Moving all
  retrieval records is a wider contract change and is not needed for R04A.
- Compatibility uses direct aliases rather than wrappers, preserving object identity and avoiding a
  second implementation.
- The facade remains evaluation-oriented and therefore may import `app.evaluation`; the architecture
  guard now proves it is unreachable from `app.main` through the reranking edge.
- The remaining direct import from `app.reranking` to `app.evaluation` is documented debt for R04D/E,
  not hidden or prematurely waived.

## 10. Tests and protected behavior

| Test area | Protected behavior |
|---|---|
| exact dense snapshot | every mapped field, score duplication, and one-based rank |
| dense ties | descending score followed by stable `chunk_id` order |
| candidate union | deduplication, dense representative, sparse provenance merge, stable order |
| facade identity | old audit imports resolve to canonical domain functions |
| architecture graph | reranking imports domain assembly and not candidate audit |
| debt inventory | only explicitly owned evaluation imports remain reachable from production |
| reranking/query suites | candidate pools, ranking, evidence boundaries, and application consumers stay stable |

The exact dense snapshot was added before the implementation move and passed against the old owner.

## 11. Commands and expected results

R04A characterization:

```text
python -m pytest -q tests/test_candidate_audit.py tests/test_reranking.py
```

R04A focused validation:

```text
python -m ruff check app/domain/retrieval.py app/candidate_audit.py app/reranking.py tests/test_candidate_audit.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_candidate_audit.py tests/test_reranking.py tests/test_retrieval_runtime.py tests/test_query_service.py tests/test_architecture_boundaries.py
```

Slice-completion validation:

```text
python -m ruff check .
python -m pytest -q
git diff --check
```

Unit tests must run offline with fake models and in-memory stores. No command in R04A requires source
PDFs, model downloads, a provider, live Qdrant, or held-out benchmark payloads.

## 12. Small usage example

```python
from app.domain.retrieval import dense_results_to_candidates
from app.models import RetrievedChunk

results = [
    RetrievedChunk(
        chunk_id="chunk-a",
        document_id="manual-a",
        filename="manual-a.pdf",
        text="Disconnect power.",
        score=0.8,
    )
]
candidates = dense_results_to_candidates(results)
assert candidates[0].dense_rank == 1
assert candidates[0].dense_score == 0.8
```

Existing `from app.candidate_audit import dense_results_to_candidates` imports remain compatible.

## 13. Common failures and debugging

- A changed dense snapshot indicates field mapping or rank assignment drift; do not update the golden
  output during this structural refactor.
- A union-order failure usually means the best component rank or `chunk_id` tie-break changed.
- If sparse metadata disappears for a duplicate, verify the dense representative is copied with both
  sparse fields rather than replaced.
- If the architecture debt test includes `app.candidate_audit`, inspect active imports for a new
  production edge to the evaluation facade.
- A domain boundary failure means an evaluator, Qdrant, or model dependency leaked into the pure
  policy module.

## 14. Current limitations

- RRF, query analysis, and role/list policies remain in their transitional owners until R04B.
- Dense and sparse search remain in compatibility modules until R04C.
- Cross-encoder construction and reranking orchestration remain combined until R04D.
- `app.reranking` still directly imports evaluation types/helpers; R04E cannot enable the final
  production-to-evaluation prohibition until those runtime/evaluator responsibilities are separated.
- Compatibility exports in `app.candidate_audit` need an explicit removal owner before Round 1 closes.

## 15. Self-check questions

1. Why is candidate union domain policy rather than evaluation behavior?
2. Which candidate represents a chunk present in both dense and sparse pools?
3. Which sparse fields are merged into that representative?
4. What are the exact dense and union tie-break rules?
5. Why does `app.candidate_audit` still export the moved functions?
6. Which production-to-evaluation import remains after R04A?

## 16. Interview summary

R04A removed an inverted dependency in the retrieval path. Deterministic candidate conversion and
dense/sparse union now live in a framework-neutral domain module, while qrel-oriented audit remains an
evaluation consumer. Direct compatibility aliases preserve old imports. Exact record snapshots,
ranking goldens, and a static import-graph test demonstrate that dependency direction improved without
changing scores, ranks, provenance, or candidate ordering.

## 17. Validation results and proposed commit

Current R04A validation:

```text
Pre-refactor candidate/reranking characterization  PASS — 34 tests
R04A focused candidate/runtime/architecture suite  PASS — 70 tests
Focused Ruff                                      PASS
Full Ruff                                         PASS
Full pytest Python 3.11.15                        PASS — 359 tests, 1 warning
Markdown links (5 local targets) / git diff check PASS
```

Proposed commit after user review:

```text
refactor: separate candidate assembly from evaluation
```

## 18. Status

`IN_PROGRESS` — R04A is implemented and focused checks pass. R04 remains open for the policy, search
adapter, reranking adapter, and final production/evaluation boundary slices.

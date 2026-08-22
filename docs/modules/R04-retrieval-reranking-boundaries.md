# R04 — Retrieval and reranking boundaries

## 1. Goal and scope

R04 separates framework-neutral retrieval policies from evaluation helpers, Qdrant search, and model
adapters without changing ranking behavior. This document is shared by all R04 slices.

R04A, R04B1, and R04B2a are implemented. R04A moves dense-result conversion and dense/sparse candidate union
into the domain, retains `app.candidate_audit` as a compatibility facade for these two public
imports, and removes the production reranking path's dependency on that evaluation-oriented module.
R04B1 moves unweighted reciprocal-rank fusion (RRF) out of the Qdrant-facing hybrid facade. R04B2a
moves deterministic Vietnamese technical query expansion into domain policy. Query-role/list and
weighted-fusion policies remain for R04B2b; R04C–R04E remain pending.

## 2. Position in the system

```text
dense RetrievedChunk results ──┐
                               ├── app.domain.retrieval ──> RetrievalCandidate pool
sparse RetrievalCandidate list ┘                                  │
                                                                  ↓
                                                   app.domain.policies.fusion
                                                                  │
                                                                  ↓
                                                       app.reranking (transitional)
                                                                  │
                                                                  ↓
                                                      ranked/evidence candidates

evaluation cases ──> app.candidate_audit ──> qrel-oriented audit records

question ──> app.domain.policies.query_analysis ──> sparse query augmentation
```

R04A–R04B2a change ownership and dependency direction only. They do not query Qdrant, initialize a
model, call a provider, execute a benchmark, or modify the frozen Phase 7 retrieval profile.

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

`fuse_rrf(dense_candidates, sparse_candidates, rrf_k, final_limit)` preserves one-based reciprocal
rank contributions, dense-candidate representation for duplicates, sparse provenance merge, bounded
output, and the established score/rank fields. Invalid bounds or missing component ranks continue to
raise `RetrievalError` with the same messages. `app.hybrid_retrieval.fuse_rrf` remains the identical
function object for compatibility.

`augment_vietnamese_technical_query(query)` preserves the immutable
`vi_technical_glossary_v1` profile, glossary order, NFKC/casefold matching, English-term deduplication,
matched-rule order, stripped no-op output, and blank-query `ValueError`. The top-level
`app.query_expansion` exports are aliases to the canonical domain objects.

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
11. RRF validates positive `rrf_k` and `final_limit` plus required one-based component ranks.
12. Each component contributes `1 / (rrf_k + component_rank)` without mixing raw scores.
13. Duplicate chunk IDs keep the dense representative and receive sparse score/rank provenance.
14. Candidates sort by descending RRF score, best component rank, then `chunk_id`.
15. The bounded result receives the fused `score` and one-based `rrf_rank`.
16. Query analysis normalizes the question with NFKC, case-folding, and surrounding-whitespace trim.
17. Glossary rules are visited in their frozen order and append only matched, absent English terms.
18. Sparse retrieval receives the expanded string; dense retrieval continues to use the original query.

## 6. Responsibilities of changed files

- [`app/domain/retrieval.py`](../../app/domain/retrieval.py) owns deterministic, SDK-free candidate
  conversion and union policy.
- [`app/domain/policies/fusion.py`](../../app/domain/policies/fusion.py) owns deterministic unweighted
  RRF policy without Qdrant or model dependencies.
- [`app/domain/policies/__init__.py`](../../app/domain/policies/__init__.py) marks the framework-neutral
  retrieval-policy package without eager exports.
- [`app/domain/policies/query_analysis.py`](../../app/domain/policies/query_analysis.py) owns the
  frozen Vietnamese technical glossary and its deterministic lexical augmentation.
- [`app/candidate_audit.py`](../../app/candidate_audit.py) owns qrel/evaluation audit behavior and
  temporarily re-exports the two moved functions for compatibility.
- [`app/hybrid_retrieval.py`](../../app/hybrid_retrieval.py) retains sparse/hybrid Qdrant coordination
  and exposes the canonical RRF policy through a compatibility alias.
- [`app/reranking.py`](../../app/reranking.py) consumes candidate assembly from the domain instead of
  importing the evaluation-oriented audit module, and consumes RRF directly from its domain owner.
- [`app/query_expansion.py`](../../app/query_expansion.py) remains a compatibility facade with an
  explicit public export list.
- [`app/retrieval_runtime.py`](../../app/retrieval_runtime.py) consumes query expansion directly from
  the canonical domain policy.
- [`app/domain/retrieval_contracts.py`](../../app/domain/retrieval_contracts.py) resolves the frozen
  query-expansion profile without depending on the top-level compatibility facade.
- [`tests/test_candidate_audit.py`](../../tests/test_candidate_audit.py) protects the exact candidate
  mapping, stable union behavior, and compatibility-export identity.
- [`tests/test_architecture_boundaries.py`](../../tests/test_architecture_boundaries.py) proves that
  active reranking imports the domain owner and no longer reaches evaluation through candidate audit.
- [`tests/test_query_expansion.py`](../../tests/test_query_expansion.py) protects exact lexical outputs,
  identifiers, errors, and compatibility-export identity.

## 7. Important symbols and why they exist

- `dense_results_to_candidates`: canonical mapping from dense search records to ranked candidates.
- `union_dense_sparse_candidates`: deterministic pool union that preserves component provenance and
  deliberately avoids mixing raw dense and sparse scores.
- `fuse_rrf`: canonical one-based reciprocal-rank fusion with fixed stable tie-breaks and errors.
- `QUERY_EXPANSION_PROFILE`: frozen identity of the active lexical augmentation policy.
- `TECHNICAL_QUERY_GLOSSARY`: ordered Vietnamese-to-English technical term mapping.
- `augment_vietnamese_technical_query`: deterministic query-only transform used before sparse search.
- `candidate_assembly` in `app.candidate_audit`: explicit facade reference that makes temporary
  compatibility ownership visible.
- `KNOWN_ACTIVE_RUNTIME_EVALUATION_IMPORTS`: executable debt inventory; after R04A it records only the
  remaining direct `app.reranking → app.evaluation` dependency owned by R04.

## 8. Before-and-after structure

Before R04A–R04B2a:

```text
app.candidate_audit   runtime candidate assembly + evaluation audit
        ↑
app.reranking        production path imports evaluation-oriented module

app.hybrid_retrieval  Qdrant sparse search + client-side RRF policy

app.query_expansion   query policy implementation and historical source-identity path
```

After R04A–R04B2a:

```text
app.domain.retrieval  canonical candidate assembly
        ↑                         ↑
app.reranking         app.candidate_audit (compatibility + evaluation audit)

app.domain.policies.fusion  canonical RRF policy
        ↑                         ↑
app.reranking         app.hybrid_retrieval (compatibility + Qdrant search)

app.domain.policies.query_analysis  canonical lexical query policy
        ↑
app.retrieval_runtime + app.query_expansion compatibility facade
```

## 9. Design decisions and trade-offs

- The algorithms were moved without changing sort keys, rank numbering, representative selection, or
  copied fields. Renaming or simplifying them would obscure behavior preservation.
- The domain functions continue to use established Pydantic records from `app.models`. Moving all
  retrieval records is a wider contract change and is not needed for R04A.
- Compatibility uses direct aliases rather than wrappers, preserving object identity and avoiding a
  second implementation.
- RRF keeps `RetrievalError` as its established public failure contract. Consolidating exception
  ownership would be a separate contract decision, not part of a behavior-preserving move.
- The policy package has no eager `__init__` exports, keeping canonical owners explicit and avoiding
  hidden import side effects.
- Historical benchmark artifacts remain immutable provenance. Their recorded source hashes continue
  to describe the code that produced them; they are not rewritten or presented as results from the
  refactored source. R07 must make future evaluation identity hash canonical policy files.
- Evaluation is not rerun during this move. Updating source-identity scripts and generating new
  benchmark artifacts are separate R07 responsibilities.
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
| exact RRF snapshot | duplicate representation, all component/fused fields, and one-based output rank |
| RRF facade identity | legacy hybrid import resolves to the canonical domain function |
| RRF architecture graph | reranking and hybrid facade both name the domain policy owner |
| exact query outputs | glossary order, matched rules, identifiers, no-op, and blank-query error |
| query facade identity | legacy exports resolve to canonical policy objects |
| query architecture graph | runtime and frozen contract import the domain owner, not the facade |

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

R04B1 characterization and focused validation:

```text
python -m pytest -q tests/test_hybrid_retrieval.py tests/test_reranking.py
python -m ruff check app/domain/policies app/hybrid_retrieval.py app/reranking.py tests/test_hybrid_retrieval.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_hybrid_retrieval.py tests/test_reranking.py tests/test_retrieval_runtime.py tests/test_query_service.py tests/test_architecture_boundaries.py
```

R04B2a characterization and focused validation:

```text
python -m pytest -q tests/test_query_expansion.py tests/test_config_contracts.py tests/test_retrieval_runtime.py
python -m ruff check app/domain/policies/query_analysis.py app/query_expansion.py app/retrieval_runtime.py app/domain/retrieval_contracts.py tests/test_query_expansion.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_query_expansion.py tests/test_config_contracts.py tests/test_retrieval_runtime.py tests/test_reranking.py tests/test_query_service.py tests/test_architecture_boundaries.py
```

Slice-completion validation:

```text
python -m ruff check .
python -m pytest -q
git diff --check
```

Unit tests must run offline with fake models and in-memory stores. No R04A–R04B2a command requires
source PDFs, model downloads, a provider, live Qdrant, or held-out benchmark payloads.

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
Existing `from app.query_expansion import augment_vietnamese_technical_query` imports also remain
compatible and resolve to the canonical domain function.

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
- A changed Vietnamese expansion means a glossary literal, rule order, normalization, or term-dedup
  condition drifted; do not accept a changed golden during this move.
- A source hash mismatch against an old metrics artifact is expected historical provenance after a
  refactor. Do not overwrite the artifact or claim its metrics were produced by current source.

## 14. Current limitations

- Query-role/list analysis, weighted RRF, reserves, and post-rerank policies remain in their
  transitional owner until R04B2b.
- Dense and sparse search remain in compatibility modules until R04C.
- Cross-encoder construction and reranking orchestration remain combined until R04D.
- `app.reranking` still directly imports evaluation types/helpers; R04E cannot enable the final
  production-to-evaluation prohibition until those runtime/evaluator responsibilities are separated.
- Compatibility exports in `app.candidate_audit` need an explicit removal owner before Round 1 closes.
- Future evaluation source identity still names compatibility paths; R07 must switch it to canonical
  implementation paths before any new artifact is treated as comparable current evidence.

## 15. Self-check questions

1. Why is candidate union domain policy rather than evaluation behavior?
2. Which candidate represents a chunk present in both dense and sparse pools?
3. Which sparse fields are merged into that representative?
4. What are the exact dense and union tie-break rules?
5. Why does `app.candidate_audit` still export the moved functions?
6. Which production-to-evaluation import remains after R04A?
7. Why does RRF combine ranks rather than raw dense and sparse scores?
8. Why must historical artifact hashes remain unchanged after moving query policy code?

## 16. Interview summary

R04A removed an inverted dependency in the retrieval path. Deterministic candidate conversion and
dense/sparse union now live in a framework-neutral domain module, while qrel-oriented audit remains an
evaluation consumer. R04B1 similarly separates unweighted RRF from Qdrant sparse search. Direct
compatibility aliases preserve old imports. R04B2a makes the frozen lexical query transform a domain
policy and removes a reverse dependency from frozen contracts. Exact record snapshots, lexical and
ranking goldens, and static
import-graph tests demonstrate improved dependency direction without changing scores, ranks,
provenance, candidate ordering, limits, or error behavior.

## 17. Validation results and proposed commit

Current R04A–R04B2a validation:

```text
Pre-refactor candidate/reranking characterization  PASS — 34 tests
R04A focused candidate/runtime/architecture suite  PASS — 70 tests
R04A full pytest Python 3.11.15                    PASS — 359 tests, 1 warning
R04A commit                                       b1d2172
Pre-refactor RRF/reranking characterization        PASS — 41 tests
R04B1 focused hybrid/runtime/architecture suite    PASS — 78 tests
Focused Ruff                                       PASS
Full Ruff                                          PASS
Full pytest Python 3.11.15                         PASS — 361 tests, 1 warning
Markdown links (8 local targets) / git diff check  PASS
Pre-refactor query-policy characterization         PASS — 23 tests
R04B2a focused query/runtime/architecture suite    PASS — 82 tests
R04B2a full Ruff                                   PASS
R04B2a full pytest Python 3.11.15                  PASS — 363 tests, 1 warning
R04B2a Markdown links (13 targets) / diff check    PASS
```

Proposed commit after user review:

```text
refactor: move retrieval policies into the domain
```

## 18. Status

`IN_PROGRESS` — R04A, R04B1, and R04B2a are implemented and focused checks pass. R04 remains open for
query-role/list and weighted-fusion policies, search adapters, reranking adapter, and the final
production/evaluation boundary slices.

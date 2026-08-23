# R04 — Retrieval and reranking boundaries

## 1. Goal and scope

R04 separates framework-neutral retrieval policies from evaluation helpers, Qdrant search, and model
adapters without changing ranking behavior. This document is shared by all R04 slices.

R04A, all R04B slices, and all R04C slices are implemented. R04A moves dense-result conversion and dense/sparse
candidate union into the domain, retains `app.candidate_audit` as a compatibility facade for these public
imports, and removes the production reranking path's dependency on that evaluation-oriented module.
R04B1 moves unweighted reciprocal-rank fusion (RRF) out of the Qdrant-facing hybrid facade. R04B2a
moves deterministic Vietnamese technical query expansion into domain policy. R04B2b moves query-role,
list-completeness, weighted-RRF, reserve, and post-rerank policies into the same domain policy layer.
R04C1 moves dense Qdrant search into its infrastructure owner. R04C2 does the same for sparse search,
payload mapping, and component rank assignment. R04D–R04E remain pending.

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

query + ranked candidates ──> app.domain.policies.ranking ──> bounded ranked candidates

question + dense model ──> app.infrastructure.qdrant.dense ──> Qdrant ──> RetrievedChunk

question + sparse model ──> app.infrastructure.qdrant.hybrid ──> Qdrant ──> ranked candidates
```

R04A–R04C change ownership and dependency direction only. Validation does not query live Qdrant,
initialize a model, call a provider, execute a benchmark, or modify the frozen Phase 7 profile.

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

`Phase7FusionProfile` and the ranking-policy functions preserve all frozen names and bounds. Query
role/list inference uses query text only; candidate policies use ranks and sanitized metadata only.
Weighted RRF, component reserves, list windows, post-rerank rank priors, metadata fields, stable
tie-breaks, and `Phase7OptimizationError` remain unchanged. `app.phase7_optimization` explicitly
re-exports every established public symbol from the canonical domain module.

`dense_search(...)` preserves trimmed-question validation, one query embedding, float conversion,
named-vector selection, optional document filter and score threshold, payload/vector flags, result
order, citation metadata mapping, and exact `RetrievalError` normalization. `app.retrieval` and
`app.hybrid_retrieval` retain compatibility aliases to the canonical adapter function.

`sparse_search(...)` preserves trimmed-question validation, one BM25 query embedding, sparse-vector
validation, named-vector/filter request fields, metadata allowlisting, score conversion, and
deterministic descending-score/`chunk_id` ranks. Its exception types and messages remain unchanged.
`app.hybrid_retrieval.sparse_search` remains the identical canonical function object.

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
19. Query-only cues infer role/confidence and list/relation intent without qrels or expected answers.
20. Weighted RRF applies the frozen component weights and optional query-role multiplier to ranks.
21. Coverage selection retains mandatory dense/sparse reserves inside the fixed candidate budget.
22. Post-rerank policies use rank-only priors and bounded ranks 5–10 list windows.
23. Evidence selection and reranking import these policies directly from their canonical domain owner.
24. Dense search validates input and converts exactly one query embedding to plain floats.
25. The adapter sends the established named-vector/filter/threshold request to Qdrant.
26. Returned payloads are validated and mapped in Qdrant response order to `RetrievedChunk`.
27. Sparse search validates input and converts one BM25 result to a Qdrant sparse vector.
28. Qdrant payloads map to candidates with source-path/character-count metadata when present.
29. Candidates sort by descending sparse score, then `chunk_id`, before one-based rank assignment.

## 6. Responsibilities of changed files

- [`app/domain/retrieval.py`](../../app/domain/retrieval.py) owns deterministic, SDK-free candidate
  conversion and union policy.
- [`app/domain/policies/fusion.py`](../../app/domain/policies/fusion.py) owns deterministic unweighted
  RRF policy without Qdrant or model dependencies.
- [`app/domain/policies/__init__.py`](../../app/domain/policies/__init__.py) marks the framework-neutral
  retrieval-policy package without eager exports.
- [`app/domain/policies/query_analysis.py`](../../app/domain/policies/query_analysis.py) owns the
  frozen Vietnamese technical glossary and its deterministic lexical augmentation.
- [`app/domain/policies/ranking.py`](../../app/domain/policies/ranking.py) owns query-role/list
  inference, weighted RRF, coverage reserves, list fallbacks, and post-rerank rank fusion.
- [`app/infrastructure/qdrant/dense.py`](../../app/infrastructure/qdrant/dense.py) now owns dense search
  and payload mapping in addition to established dense embedding/indexing infrastructure.
- [`app/infrastructure/qdrant/hybrid.py`](../../app/infrastructure/qdrant/hybrid.py) now owns sparse
  search, sparse payload mapping, and component rank assignment beside BM25/hybrid infrastructure.
- [`app/candidate_audit.py`](../../app/candidate_audit.py) owns qrel/evaluation audit behavior and
  temporarily re-exports the two moved functions for compatibility.
- [`app/hybrid_retrieval.py`](../../app/hybrid_retrieval.py) retains sparse/hybrid Qdrant coordination
  for legacy `hybrid_search` and exposes canonical dense/sparse search and RRF compatibility aliases.
- [`app/retrieval.py`](../../app/retrieval.py) is now a pure compatibility facade over dense Qdrant
  infrastructure and manifests.
- [`app/reranking.py`](../../app/reranking.py) consumes candidate assembly from the domain instead of
  importing the evaluation-oriented audit module, and consumes RRF directly from its domain owner.
- [`app/query_expansion.py`](../../app/query_expansion.py) remains a compatibility facade with an
  explicit public export list.
- [`app/phase7_optimization.py`](../../app/phase7_optimization.py) remains an explicit compatibility
  facade for historical scripts, replay code, and imports.
- [`app/retrieval_runtime.py`](../../app/retrieval_runtime.py) consumes query expansion directly from
  the canonical domain policy.
- [`app/domain/retrieval_contracts.py`](../../app/domain/retrieval_contracts.py) resolves the frozen
  query-expansion and fusion profiles without depending on top-level compatibility facades.
- [`app/evidence_selection.py`](../../app/evidence_selection.py) consumes query-role inference from
  the canonical domain owner.
- [`tests/test_candidate_audit.py`](../../tests/test_candidate_audit.py) protects the exact candidate
  mapping, stable union behavior, and compatibility-export identity.
- [`tests/test_architecture_boundaries.py`](../../tests/test_architecture_boundaries.py) proves that
  active reranking imports the domain owner and no longer reaches evaluation through candidate audit.
- [`tests/test_query_expansion.py`](../../tests/test_query_expansion.py) protects exact lexical outputs,
  identifiers, errors, and compatibility-export identity.
- [`tests/test_phase7_optimization.py`](../../tests/test_phase7_optimization.py) protects role/list
  inference, feature scoping, ranking bounds, reserves, metadata, errors, and facade identity.

## 7. Important symbols and why they exist

- `dense_results_to_candidates`: canonical mapping from dense search records to ranked candidates.
- `union_dense_sparse_candidates`: deterministic pool union that preserves component provenance and
  deliberately avoids mixing raw dense and sparse scores.
- `fuse_rrf`: canonical one-based reciprocal-rank fusion with fixed stable tie-breaks and errors.
- `QUERY_EXPANSION_PROFILE`: frozen identity of the active lexical augmentation policy.
- `TECHNICAL_QUERY_GLOSSARY`: ordered Vietnamese-to-English technical term mapping.
- `augment_vietnamese_technical_query`: deterministic query-only transform used before sparse search.
- `Phase7FusionProfile`: immutable validated weighted-fusion and post-rerank configuration.
- `infer_query_role`, `infer_list_intent`, `infer_relation_list_intent`: query-only, auditable signals.
- `fuse_weighted_rrf`, `select_coverage_preserving_candidates`: bounded pre-rerank ordering policy.
- `apply_role_aware_rank_fusion`: rank-only post-rerank prior that preserves cross-encoder scores.
- `apply_*_list_completeness_*`: bounded runtime and sanitized-metadata replay of list ordering.
- `dense_search`: canonical dense Qdrant query adapter with fail-closed payload/error mapping.
- `sparse_search`: canonical BM25/Qdrant query adapter with deterministic component ranks.
- `candidate_assembly` in `app.candidate_audit`: explicit facade reference that makes temporary
  compatibility ownership visible.
- `KNOWN_ACTIVE_RUNTIME_EVALUATION_IMPORTS`: executable debt inventory; after R04A it records only the
  remaining direct `app.reranking → app.evaluation` dependency owned by R04.

## 8. Before-and-after structure

Before R04A–R04C:

```text
app.candidate_audit   runtime candidate assembly + evaluation audit
        ↑
app.reranking        production path imports evaluation-oriented module

app.hybrid_retrieval  Qdrant sparse search + client-side RRF policy

app.query_expansion   query policy implementation and historical source-identity path

app.phase7_optimization  role/list/fusion implementation and historical phase-named owner

app.retrieval  dense Qdrant search implementation + indexing compatibility exports

app.hybrid_retrieval  sparse Qdrant search implementation + hybrid coordinator
```

After R04A–R04C:

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

app.domain.policies.ranking  canonical role/list/fusion policy
        ↑
runtime/domain consumers + app.phase7_optimization compatibility facade

app.infrastructure.qdrant.dense  canonical dense index + search adapter
        ↑
runtime consumers + app.retrieval/app.hybrid_retrieval compatibility facades

app.infrastructure.qdrant.hybrid  canonical sparse/hybrid index + search adapter
        ↑
runtime consumers + app.hybrid_retrieval compatibility facade/coordinator
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
- Historical profile/symbol names remain because artifacts and scripts use them. Renaming those
  contracts would add no dependency benefit and would obscure behavior preservation.
- Dense search stays beside dense indexing because both adapt the same Qdrant vector/payload schema
  and embedding boundary. No separate abstraction is introduced for a single concrete backend.
- Sparse search follows the same ownership rule. `hybrid_search` stays in the facade because it is a
  legacy cross-adapter coordinator; production runtime uses explicit dense/sparse injected operations.
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
| role/list goldens | bilingual boundaries, identifiers, scoped feature counts, and ranks 5–10 window |
| weighted ranking goldens | weights, reserves, budget, rank-only priors, metadata, and stable errors |
| ranking facade identity | legacy classes, profile, errors, and functions are canonical objects |
| ranking architecture graph | active runtime/domain consumers bypass the phase-named facade |
| exact dense query snapshot | named vector, converted query, filter, limit, flags, and threshold |
| dense in-memory tests | response order, scores, citation payload, filters, and invalid payload errors |
| dense facade identity | historical imports resolve to the canonical infrastructure adapter |
| dense architecture graph | reranking/hybrid runtime bypass `app.retrieval` for dense search |
| exact sparse query snapshot | sparse vector, named vector, filter, limit, and payload/vector flags |
| sparse in-memory tests | metadata allowlist, document filter, score/rank ordering, and errors |
| sparse facade identity | historical import resolves to the canonical hybrid infrastructure adapter |
| sparse architecture graph | reranking/runtime bypass `app.hybrid_retrieval` for sparse search |

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

R04B2b characterization and focused validation:

```text
python -m pytest -q tests/test_phase7_optimization.py tests/test_evidence_selection.py tests/test_reranking.py tests/test_config_contracts.py
python -m ruff check app/domain/policies/ranking.py app/phase7_optimization.py app/domain/retrieval_contracts.py app/evidence_selection.py app/reranking.py tests/test_phase7_optimization.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_phase7_optimization.py tests/test_evidence_selection.py tests/test_reranking.py tests/test_config_contracts.py tests/test_retrieval_runtime.py tests/test_query_service.py tests/test_architecture_boundaries.py
```

R04C1 characterization and focused validation:

```text
python -m pytest -q tests/test_retrieval.py tests/test_hybrid_retrieval.py tests/test_reranking.py
python -m ruff check app/infrastructure/qdrant/dense.py app/retrieval.py app/hybrid_retrieval.py app/reranking.py tests/test_retrieval.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_retrieval.py tests/test_hybrid_retrieval.py tests/test_reranking.py tests/test_retrieval_runtime.py tests/test_query_service.py tests/test_architecture_boundaries.py
```

R04C2 characterization and focused validation:

```text
python -m pytest -q tests/test_hybrid_retrieval.py tests/test_reranking.py tests/test_retrieval_runtime.py
python -m ruff check app/infrastructure/qdrant/hybrid.py app/hybrid_retrieval.py app/reranking.py app/retrieval_runtime.py tests/test_hybrid_retrieval.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_hybrid_retrieval.py tests/test_reranking.py tests/test_retrieval_runtime.py tests/test_query_service.py tests/test_architecture_boundaries.py
```

Slice-completion validation:

```text
python -m ruff check .
python -m pytest -q
git diff --check
```

Unit tests must run offline with fake models and in-memory stores. No R04A–R04C command requires
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
Historical `app.phase7_optimization` imports remain compatible in the same way.

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
- A role/list or fusion golden failure means a cue boundary, feature scope, rank window, weight,
  reserve, tie-break, or metadata contract drifted; do not tune the expected result during R04.
- A dense query snapshot failure means vector conversion, named-vector selection, filter, threshold,
  or payload/vector flags changed; do not update the expected call during an adapter move.
- A sparse snapshot or rank failure means vector conversion, named-vector/filter fields, metadata,
  score/`chunk_id` tie-break, or one-based ranking drifted; do not update the golden during R04C.

## 14. Current limitations

- Legacy `hybrid_search` remains in the compatibility facade; active production runtime composes the
  canonical dense and sparse adapter functions directly.
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
9. Why does post-rerank role fusion preserve `rerank_score` and write a separate rank-derived score?
10. Why does dense search belong in Qdrant infrastructure rather than the retrieval domain?
11. Why is raw sparse score preserved as provenance while rank is assigned deterministically?

## 16. Interview summary

R04A removed an inverted dependency in the retrieval path. Deterministic candidate conversion and
dense/sparse union now live in a framework-neutral domain module, while qrel-oriented audit remains an
evaluation consumer. R04B1 similarly separates unweighted RRF from Qdrant sparse search. Direct
compatibility aliases preserve old imports. R04B2a makes the frozen lexical query transform a domain
policy and removes a reverse dependency from frozen contracts. R04B2b completes the policy boundary
with query-only role/list analysis, weighted fusion, reserves, and bounded post-rerank ordering. Exact
record snapshots, lexical and ranking goldens, and static
import-graph tests demonstrate improved dependency direction without changing scores, ranks,
provenance, candidate ordering, limits, or error behavior. R04C1 moves dense search behind the Qdrant
adapter while preserving exact request and response mapping contracts. R04C2 completes the Qdrant
search boundary for sparse BM25 queries and removes production imports through the hybrid facade.

## 17. Validation results and proposed commit

Current R04A–R04C validation:

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
Pre-refactor ranking-policy characterization       PASS — 69 tests
R04B2b focused ranking/runtime/architecture suite  PASS — 108 tests
R04B2b full Ruff                                   PASS
R04B2b full pytest Python 3.11.15                  PASS — 365 tests, 1 warning
R04B2b Markdown links (17 targets) / diff check    PASS
Pre-refactor dense-search characterization         PASS — 69 tests
R04C1 focused dense/runtime/architecture suite     PASS — 108 tests
R04C1 full Ruff                                    PASS
R04C1 full pytest Python 3.11.15                   PASS — 367 tests, 1 warning
R04C1 Markdown links (19 targets) / diff check     PASS
Pre-refactor sparse-search characterization        PASS — 52 tests
R04C2 focused sparse/runtime/architecture suite    PASS — 83 tests
R04C2 full Ruff                                    PASS
R04C2 full pytest Python 3.11.15                   PASS — 369 tests, 1 warning
R04C2 Markdown links (20 targets) / diff check     PASS
```

Proposed commit after user review:

```text
refactor: isolate sparse search infrastructure
```

## 18. Status

`IN_PROGRESS` — R04A–R04C are implemented and focused checks pass. R04 remains open for the
reranking adapter and the final
production/evaluation boundary slices.

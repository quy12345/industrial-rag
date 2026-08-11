# Codebase guide

## Current architecture

```text
PDF/DOCX -> Docling -> structure-aware DocumentChunk
        -> dense FastEmbed vector + BM25 sparse vector -> Qdrant collection v2
Query -> dense top-20 + sparse top-20 -> client-side RRF -> RetrievalCandidate
      -> sparse | hybrid | dense/sparse union candidate pool
      -> optional Phase 7 same-document exact-content deduplication
      -> lazy multilingual cross-encoder -> full reranked pool -> display cutoff
      -> evidence gate -> structured grounded generation
      -> source-ID validation -> trusted citation builder -> QueryResponse

Phase 7.4.1--7.5 frozen calibration/runtime override:
Query -> dense top-60 + expanded sparse top-40 after vi_technical_glossary_v1
      -> query-only role inference -> weighted RRF k=40 + dense@5/sparse@24 reserves
      -> exact-content deduplication -> maximum 30 candidates -> unchanged Jina reranker
      -> rank-only `0.50` document-role prior with offset 20 for strong/weak cues
      -> frozen batch size 8 and runtime-default ONNX threads; never an expected-document filter
```

`app.main` exposes `/api/v1/health`, `/api/v1/ready`, and `/api/v1/query`. Retrieval, reranking, evidence gating, and
citations remain explicit Python; LangChain is limited to prompt orchestration, OpenAI Responses or
Gemini OpenAI-compatible Chat Completions invocation, and provider-native structured output.

## Main modules

- `app/config.py`: Pydantic settings for retrieval, reranking, evidence/generation limits, and the
  selected OpenAI or Gemini provider. Union+rerank is the Phase 6 default; sparse/no-rerank is
  rollback.
- `app/ingestion.py`: input validation, Docling conversion, batched PDF processing, stable
  content-based chunk IDs, and atomic JSONL output.
- `app/models.py`: ingestion/retrieval models plus the public query request, response, and trusted
  citation contracts.
- `app/retrieval.py`: embedding input, FastEmbed initialization, Qdrant collection/index/search,
  stable UUIDv5 point IDs, safe re-indexing, and index-manifest validation.
- `app/hybrid_retrieval.py`: FastEmbed BM25 configuration and exact avg-length calculation, v2
  schema/manifest validation, safe dual-vector indexing, sparse search, and deterministic RRF.
- `app/evaluation.py`: dependency-free typed qrels, frozen-chunk validation, direct-evidence ranks,
  retrieval metrics, group metrics, and latency percentiles.
- `app/content_identity.py`: dependency-free NFKC/case/whitespace identity for exact equivalence;
  it is deliberately not semantic similarity.
- `app/phase7.py`: offline schema, validation, hashing, exact-content qrel closure, and review state
  for the separate two-manual corpus; it never calls Qdrant or a provider.
- `app/evaluation_e2e.py`: offline scoring of a completed query execution: qrel-only ranks,
  citation outcomes, abstention confusion matrix, typed deterministic answer facts, strict-phrase
  and token-overlap diagnostics, document contamination, component qrel ranks, and latency
  summaries. It stores no answer/evidence text.
- `app/query_expansion.py`: frozen, deterministic Vietnamese technical glossary used only to append
  translated query terms before Phase 7 sparse retrieval; it contains no qrels, pages, document IDs,
  expected answers, model call, or held-out-specific rule.
- `app/phase7_optimization.py`: query-only bilingual document-role inference with Unicode-safe cue
  boundaries and strong/weak confidence, bounded weighted RRF, coverage-preserving candidate
  membership, and post-rerank rank-only role prior. It has no Qdrant,
  provider, qrel, expected-page, or answer-fact dependency.
- `app/phase7_replay.py`: validates sanitized reranker snapshots and replays rank-only priors without
  a model, Qdrant, provider, raw question, or chunk text.
- `app/candidate_audit.py`: dependency-free candidate-pool normalization, union, coverage, critical
  diagnostics, and RRF-demotion aggregation for the Phase 5 handoff.
- `app/reranking.py`: lazy FastEmbed cross-encoder adapter, exact candidate-text formatting,
  sparse/hybrid/union pool construction, strict output validation, deterministic reranking, stage
  latency, and direct-evidence failure classification.
- `app/retrieval_runtime.py`: artifact-independent frozen contract, live Qdrant hash/schema checks,
  lazy union runtime, and sparse rollback composition shared by API and Phase 5 scripts.
- `app/generation.py`: deterministic bounded evidence blocks, strict `GeneratedAnswer`, prompt
  injection boundary, and a lazy LangChain adapter. OpenAI uses Responses with `store=false`;
  Gemini uses Google's OpenAI-compatible Chat Completions endpoint.
- `app/citations.py`: referential source-ID validation and deterministic citation construction from
  retrieved metadata.
- `app/query_service.py`: retrieve → gate → generate → validate/retry → respond orchestration,
  abstention policy, and internal stage timings.
- `app/api/query.py`: threadpool handoff and sanitized HTTP error mapping only.
- `app/api/auth.py`: optional constant-time bearer-token guard, disabled unless configured.
- `scripts/index_document.py`, `scripts/search_dense.py`: v1 dense integration CLIs.
- `scripts/index_hybrid.py`, `scripts/search_hybrid.py`, `scripts/evaluate.py`: v2 indexing/search
  and shared dense/sparse/hybrid evaluation CLIs.
- `scripts/audit_candidate_pools.py`: explicit real-model/Qdrant audit for dense@20, sparse@20,
  hybrid@20, and dense@20 ∪ sparse@20; not part of default pytest.
- `scripts/generate_phase5_readiness.py`: validates the frozen contract and writes the Phase 5 JSON
  handoff from measured artifacts and live collection metadata.
- `scripts/rerank_runtime.py`: validates both frozen collection manifests and constructs the real
  retrieval/reranking runtime without changing either collection.
- `scripts/search_reranked.py`: explicit-strategy search CLI with component and rerank diagnostics.
- `scripts/evaluate_reranking.py`: evaluates one or all three Phase 5 pools and writes additive
  strategy/comparison JSON artifacts; `--comparison-only` never loads a model.
- `scripts/validate_query_runtime.py`: read-only real union/sparse runtime smoke without OpenAI.
- `scripts/query_smoke.py`: bounded real-provider smoke and sanitized Phase 6 artifact writer.
- `scripts/audit_phase7_corpus.py`, `scripts/index_phase7_corpus.py`: source audit and guarded
  indexing for the isolated ATV320 collections.
- `scripts/freeze_phase7_calibration_v3.py`: copies the review-required typed calibration draft to an
  approved v3 file and manifest only after an exact human token; it cannot modify held-out.
- `scripts/evaluate_phase7_e2e.py`: resumable integration evaluator that validates the approved
  manifest and live frozen hash before it sends anything to a provider. Calibration and held-out use
  distinct provider-approval tokens, so held-out cannot run accidentally after calibration approval.
- `scripts/calibrate_phase7_retrieval.py`: provider-free, calibration-only dense/sparse 20--60
  union/RRF/expanded-RRF ablation, including the fixed top-30 reranker budget; it never loads the
  cross-encoder or executes held-out queries.
- `scripts/audit_phase7_retrieval_failures.py`: sanitized top-200 dense/sparse audit for calibration
  004/005/010, including source/index integrity and query-variant ranks; no provider or reranker.
- `scripts/calibrate_phase7_weighted_fusion.py`: retrieval-only 144-profile weighted-RRF/role/reserve
  ablation and Pareto shortlist; no reranker, provider, or held-out query.
- `scripts/evaluate_phase7_weighted_rerank.py`: local Jina evaluation of at most six shortlisted
  calibration profiles, with deterministic profile selection and no generation provider.
- `scripts/evaluate_phase7_retrieval_closure.py`: runs the frozen Phase 7.4.1 retrieval and local Jina
  reranker on answerable calibration only; no provider and no held-out execution.
- `scripts/create_phase7_reranker_snapshot.py`: makes a sanitized baseline Jina snapshot once; replay
  artifacts include IDs/ranks/scores and roles but exclude questions and raw evidence.
- `scripts/calibrate_phase7_role_prior.py`: runs six-fold, provider-free rank-prior selection from a
  snapshot; the fold mapping is evaluation-only and never imported by the runtime.
- `scripts/benchmark_phase7_reranker_cpu.py`: runs the bounded CPU micro/full ablation without Docker
  builds, re-indexing, provider calls, or held-out questions.
- `scripts/generate_phase7_fact_evaluator_readiness.py` and
  `scripts/generate_phase7_runtime_readiness.py`: fail-closed sanitized readiness artifacts for the
  typed-fact review boundary and provider-approval boundary.
- `scripts/rescore_phase7_calibration_facts.py`: derives deterministic text-fact decisions from the
  prior sanitized v2 diagnostics when raw provider answers are unavailable.
- `scripts/migrate_phase7_dataset_v2.py`: offline migration that revokes approval, adds answer facts,
  and expands qrels only across same-document exact-content duplicates.
- `scripts/apply_phase7_answer_facts.py`: explicit source-review mapping for 42 answerable rows plus
  the documented calibration 011/012 qrel correction; it reads no retrieval/provider output.

## Dense-index contract

- Collection: `industrial_manual_chunks`
- Vector: named `dense`, cosine distance
- Default model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Dimension: obtained from the model at runtime, not hard-coded
- Index manifest: `artifacts/metrics/dense-index-manifest.json`
- Point ID: UUIDv5 derived from the stable chunk ID

Before search/evaluation, the manifest must agree with collection, vector, model, dimension, and
distance. Evaluation additionally requires Qdrant's indexed chunk IDs to exactly equal the frozen
JSONL chunk set.

## Hybrid-index contract

- Collection v2: `industrial_manual_chunks_v2`; v1 is never recreated, migrated, or deleted.
- Dense vector: named `dense`, dimension 384, cosine.
- Sparse vector: named `sparse`, Qdrant `idf` modifier.
- Sparse model: FastEmbed `Qdrant/bm25`, `disable_stemmer=True`, `k=1.2`, `b=0.75`.
- Frozen-corpus BM25 average length: `72.838384`, persisted in
  `artifacts/metrics/hybrid-index-manifest.json`.
- RRF: one-based component ranks, `sum(1 / (60 + rank))`, then deterministic sort by RRF score,
  best component rank, and chunk ID.

Hybrid indexing first validates the frozen 99-chunk identity, generates every dense and sparse
vector, upserts new deterministic points, and only then removes stale points for that same document.
The hybrid manifest validates all index/fusion settings and the chunk hash before sparse or hybrid
search begins.

## Reranking contract

- Model: `jinaai/jina-reranker-v2-base-multilingual` through FastEmbed 0.8.0
  `fastembed.rerank.cross_encoder.TextCrossEncoder`.
- License: `CC-BY-NC-4.0`; benchmark/demo use only unless commercial rights are resolved.
- Input format ID: `heading_content_v1`, rendered as `heading > breadcrumb`, two newlines, and the
  unchanged raw chunk text; heading-less chunks use raw text only.
- Sparse pool: v2 sparse top 20. Hybrid pool: v2 dense top 20 plus sparse top 20, RRF `k=60`, then
  top 20. Union pool: v1 dense top 20 plus v2 sparse top 20 with stable-ID de-duplication and no
  pre-rerank truncation.
- The cross-encoder output must contain one indexed, finite score for every input. Final ordering is
  score descending, previous one-based rank ascending, then chunk ID.
- `rerank_score` becomes the final ranking signal while dense, sparse, and RRF scores/ranks remain
  available. The full pool is returned; CLI `--limit` affects display only. No error fallback exists.

The historical Phase 5 contract above remains unchanged. Phase 7.4 uses input format
`document_context_heading_content_v2`, prepending trusted document title and role. Its frozen pool is
dense@60 plus expanded sparse@40 after `vi_technical_glossary_v1`, weighted RRF `k=40`, sparse weight
`1.25`, dense@5/sparse@24 coverage reserves, and a maximum 30-candidate Jina rerank. The final
rank-only role prior preserves raw Jina scores and never filters by expected document, so it does not
leak evaluation ground truth.

## Evaluation boundary

`data/eval/dense_smoke.jsonl` is a retrieval-development set, not a Phase 7 held-out test set.
Ground truth is `relevant_chunk_ids`; expected phrase/page metadata validates and diagnoses qrels but
never changes Hit@k or MRR. Ranks are one-based and reciprocal rank is zero when direct evidence is
outside the candidate limit. The evaluator reports Hit@1/3/5/20, Candidate Recall@20, MRR@5/20,
per-language, and per-retrieval-scenario (`vi -> vi` monolingual; `en -> vi` cross-lingual) metrics.

## Dependencies and containers

- Core package: FastAPI runtime/settings only.
- `retrieval` extra: Qdrant/FastEmbed, with FastEmbed directly pinned to `0.8.0` to preserve the
  evaluated embedding behavior.
- `ingestion` extra: Docling.
- `llm` extra: `langchain-core` and `langchain-openai`.
- `dev` extra: complete local test/lint, retrieval, ingestion, and LLM dependencies.

The canonical post-closure retrieval dependency is `qdrant-client >=1.19.0,<1.20.0`. It was the
client used by the successful Phase 4 Python 3.11 integration; Qdrant server remains pinned to
`v1.18.3`. Historic metrics artifacts are immutable even where older runtime metadata says `1.18.0`.

`Dockerfile` supplies `api` and `ingestion` targets from a shared retrieval runtime. API adds only
the LLM extra; ingestion does not receive LangChain/OpenAI. Compose starts API/Qdrant by default;
ingestion is profile-gated. The shared `fastembed_cache` volume supplies models at runtime; weights
are not baked into either image.

The ingestion target installs Debian runtime packages `libxcb1`, `libgl1`, and
`libglib2.0-0t64`; Docling's PDF/image stack needs the shared libraries they provide when running
on `python:3.11-slim`.

## Testing

Default pytest uses fake embeddings and in-memory Qdrant. It must not download models, call Docker,
connect to a real Qdrant server, or require an API key. The real manual/index/evaluator flow is an
explicit integration smoke command documented in the README. Phase 4 adds offline tests for sparse
schema/IDF, safe re-indexing, document filters, metadata preservation, manifest mismatches, and RRF
duplicate/tie/empty-list behavior. Phase 4.1 adds offline candidate-pool tests for deterministic
rank/score preservation, union de-duplication, qrel-only candidate recall, scenario aggregation,
critical rows, and RRF-demotion diagnostics.
Phase 5 adds fake indexed cross-encoder tests for malformed outputs, finite scores, ordering and
ties, metadata preservation, no fallback, all candidate pools, document filtering, failure classes,
latency aggregation, CLI contracts, and no eager model initialization. Real model/Qdrant evaluation
remains separate from default pytest.
Phase 6 adds fake generation, evidence, citation, correction-retry, query-service, HTTP mapping, lazy
runtime, and security/logging tests. The original canonical Python 3.11 run passes 160 tests with one
known third-party Starlette/TestClient deprecation warning. The additive Gemini and UTF-8 response
regressions bring the local suite to 162, and the real adapter constructs in the Python 3.11 API image; no default test
calls Qdrant, FastEmbed, a reranker, or a generation provider.

Phase 7.4.1--7.5 adds offline Unicode/boundary role-inference, confidence, rank-prior replay,
malformed-snapshot, CPU-profile selection, fact-readiness, and runtime-readiness tests. Canonical
Python 3.11.15 validation passes 239 tests with the same one third-party warning; default pytest
still performs no network/model/Qdrant call.

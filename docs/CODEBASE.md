# Codebase guide

## Current architecture

```text
PDF/DOCX -> Docling -> structure-aware DocumentChunk
        -> dense FastEmbed vector + BM25 sparse vector -> Qdrant collection v2
Query -> dense top-60 + expanded sparse top-40 -> frozen weighted RRF/reserves
      -> maximum 30 RetrievalCandidate records
      -> same-document exact-content deduplication
      -> lazy multilingual cross-encoder -> full reranked pool -> display cutoff
      -> evidence gate -> structured grounded generation
      -> source-ID validation -> trusted citation builder -> QueryResponse

Browser -> Streamlit UI -> FastAPI health/ready/query HTTP endpoints
        -> no direct Qdrant, embedding, reranker, ingestion, or provider import

Phase 7.4.1--7.5 frozen calibration/runtime override:
Query -> dense top-60 + expanded sparse top-40 after vi_technical_glossary_v1
      -> query-only role inference -> weighted RRF k=40 + dense@5/sparse@24 reserves
      -> exact-content deduplication -> maximum 30 candidates -> unchanged Jina reranker
      -> rank-only `0.50` document-role prior with offset 20 for strong/weak cues
      -> exact cross-document duplicate evidence selection -> actual top-k generation context
      -> frozen batch size 8 and runtime-default ONNX threads; never an expected-document filter
```

`app.main` exposes `/api/v1/health`, `/api/v1/ready`, and `/api/v1/query`. Retrieval, reranking, evidence gating, and
citations remain explicit Python; LangChain is limited to prompt orchestration, OpenAI Responses or
Gemini OpenAI-compatible Chat Completions invocation, and provider-native structured output.

## Main modules

Round 1 established these canonical owners. The current simplification pass is removing the
remaining explicit compatibility paths through versioned hard cuts:

```text
app/api                 inbound HTTP adapters
app/application         use-case orchestration and application contracts
app/domain              framework-free records, ports, and policies
app/infrastructure      Qdrant, model, provider, and corpus adapters
evaluation              offline datasets, replay, audits, and metrics
scripts/operations      supported operational CLI implementations
scripts/evaluation      supported evaluation CLI implementations
scripts/archive         unsupported historical provenance
ui                      HTTP-only Streamlit adapter
```

- `app/config.py`: Pydantic settings for retrieval, reranking, evidence/generation limits, selected
  OpenAI or Gemini provider, and the only supported `phase7` retrieval profile. Python and Compose
  use the same active collections; `phase6` is rejected.
- `app/ingestion.py`: input validation, Docling conversion, batched PDF processing, stable
  content-based chunk IDs, and atomic JSONL output.
- `app/models.py`: ingestion/retrieval models plus the public query request, response, and trusted
  citation contracts.
- `app/retrieval.py`: embedding input, FastEmbed initialization, Qdrant collection/index/search,
  stable UUIDv5 point IDs, safe re-indexing, and index-manifest validation.
- `app/hybrid_retrieval.py`: FastEmbed BM25 configuration and exact avg-length calculation, v2
  schema/manifest validation, safe dual-vector indexing, sparse search, and deterministic RRF.
- `app/evaluation.py`: compatibility/source-identity anchor over canonical offline retrieval
  evaluation. New evaluation code imports `evaluation.retrieval` directly.
- `app/content_identity.py`: dependency-free NFKC/case/whitespace identity for exact equivalence;
  it is deliberately not semantic similarity.
- `app/phase7.py`: compatibility/source-identity anchor. Canonical two-manual dataset schemas,
  validation, hashing, qrel closure, and review state live in `evaluation/phase7_dataset.py`; active
  corpus file identity lives in `app/infrastructure/corpus_artifacts.py`.
- `evaluation/e2e.py`: offline scoring of a completed query execution: qrel-only ranks,
  citation outcomes, abstention confusion matrix, typed deterministic answer facts, strict-phrase
  and token-overlap diagnostics, bounded lexical inflection for text facts, span-aware negation,
  document contamination, full-ranking versus actual-evidence qrel ranks, and latency summaries. It
  stores no answer/evidence text.
- `app/evidence_selection.py`: removes exact normalized content duplicated across source documents
  before top-k. It chooses provenance from query-derived role plus existing rank, retains equivalent
  source IDs in metadata, and does not import qrels/expected documents or collapse near-duplicates.
- `app/query_expansion.py`: frozen, deterministic Vietnamese technical glossary used only to append
  translated query terms before Phase 7 sparse retrieval; it contains no qrels, pages, document IDs,
  expected answers, model call, or held-out-specific rule.
- `app/phase7_optimization.py`: query-only bilingual document-role inference with Unicode-safe cue
  boundaries and strong/weak confidence, bounded weighted RRF, coverage-preserving candidate
  membership, post-rerank rank-only role prior, historical whole-chunk `list_completeness_v1`, and
  active `relation_list_completeness_v1`. The active fallback requires query-derived list,
  key/button, switch/change and technical-ID cues, then counts only targets after that relation in
  one candidate clause. It moved calibration 010 from rank 6 to 5 without runtime qrel access. This
  module has no Qdrant, provider, qrel, expected-page, expected-document, or answer-fact dependency.
- `evaluation/replay.py`: validates sanitized reranker snapshots and replays rank-only priors without
  a model, Qdrant, provider, raw question, or chunk text. The expired `app.phase7_replay` facade has
  been removed.
- `evaluation/candidate_audit.py`: dependency-free candidate-pool normalization, union, coverage,
  critical diagnostics, and RRF-demotion aggregation for the historical Phase 5 handoff.
- `evaluation/retrieval_closure.py`: shared provider-free closure aggregation, including aggregate
  and per-language metrics; CLI modules consume this public evaluator instead of importing private
  helpers from another script.
- `app/reranking.py`: lazy FastEmbed cross-encoder adapter, exact candidate-text formatting,
  sparse/hybrid/union pool construction, strict output validation, deterministic reranking, stage
  latency, and direct-evidence failure classification.
- `app/retrieval_runtime.py`: the artifact-independent Phase 7 frozen contract, atomic resolution,
  live Qdrant hash/schema checks, lazy union runtime, and sparse rollback composition shared by API
  and integration scripts. Builders require an explicit contract.
- `app/generation.py`: deterministic bounded evidence blocks, strict `GeneratedAnswer`, prompt
  injection boundary, and a lazy LangChain adapter. OpenAI uses Responses with `store=false`;
  Gemini uses Google's OpenAI-compatible Chat Completions endpoint.
- `app/citations.py`: referential source-ID validation and deterministic citation construction from
  retrieved metadata.
- `app/query_service.py`: retrieve/rerank → select actual evidence → gate → generate →
  validate/retry → respond orchestration, abstention policy, full-rank/evidence provenance, and
  internal stage timings.
- `app/api/query.py`: threadpool handoff and sanitized HTTP error mapping only.
- `app/api/auth.py`: optional constant-time bearer-token guard, disabled unless configured.
- `ui/config.py`: server-side API URL/timeout/token settings and stable ATV320 document labels.
- `ui/api_client.py`: one-shot HTTPX health/readiness/query client with strict response parsing and
  sanitized error mapping; it never logs request/response content.
- `ui/state.py`: dependency-light bounded display-history and citation page helpers.
- `ui/streamlit_app.py`: chat/session/sidebar rendering only; all retrieval and generation stays
  behind the FastAPI boundary.
- `scripts/archive/phase6/`: unsupported Phase 3–6 indexing, search, evaluation, reranking, readiness,
  and validation tools. Their historical contract is local to the archive; mutating commands require
  separate explicit approval and are never active Phase 7 entrypoints.
- `scripts/validate_query_runtime.py`: read-only real union/sparse runtime smoke without OpenAI.
- `scripts/query_smoke.py`: bounded real-provider smoke and sanitized Phase 7 artifact writer.
- `scripts/audit_phase7_corpus.py`, `scripts/index_phase7_corpus.py`: source audit and guarded
  indexing for the isolated ATV320 collections.
- `scripts/archive/phase7/freeze_phase7_calibration_v3.py`: unsupported completed approval tool that
  copied the reviewed typed calibration draft to the frozen v3 file and manifest.
- `scripts/evaluate_phase7_e2e.py`: supported thin entry point for the resumable implementation in
  `scripts/evaluation/evaluate_phase7_e2e.py`. It validates the approved manifest and live frozen
  hash before provider egress, writes schema-v6 artifacts with canonical source-identity v2, and
  keeps held-out execution blocked by governance.
- `scripts/archive/phase7/calibrate_phase7_retrieval.py`,
  `audit_phase7_retrieval_failures.py`, and `calibrate_phase7_weighted_fusion.py`: unsupported
  completed Phase 7 retrieval experiments retained for provenance.
- `scripts/archive/phase7/evaluate_phase7_weighted_rerank.py`: unsupported completed local-Jina
  evaluation of the Phase 7.4 weighted-fusion shortlist, superseded by Phase 7.4.1–7.5 closure.
- `scripts/evaluate_phase7_retrieval_closure.py`: supported thin compatibility entry point for the
  frozen Phase 7.4.1 retrieval closure. Its implementation lives in
  `scripts/evaluation/evaluate_phase7_retrieval_closure.py`; metrics live in
  `evaluation/retrieval_closure.py`. It uses local Jina on answerable calibration only, with no
  provider and no held-out execution.
- `scripts/archive/phase7/create_phase7_reranker_snapshot.py` and
  `calibrate_phase7_role_prior.py`: unsupported completed snapshot/replay selection workflow.
- `scripts/archive/phase7/diagnose_phase7_calibration_005.py`: unsupported completed three-attempt
  provider diagnostic retained with its private-debug and sanitization guards for provenance.
- `scripts/archive/phase7/aggregate_phase7_calibration_stability.py`,
  `generate_phase7_calibration_closure_readiness.py`, and `benchmark_phase7_reranker_cpu.py`:
  unsupported completed stability, readiness, and CPU-measurement workflows.
- `scripts/archive/phase7/generate_phase7_fact_evaluator_readiness.py` and
  `scripts/archive/phase7/generate_phase7_runtime_readiness.py`: unsupported completed generators for
  the historical typed-fact review and pre-egress provider-approval receipts.
- `scripts/archive/phase7/rescore_phase7_calibration_facts.py`: unsupported completed derivation of
  deterministic text-fact decisions from the prior sanitized v2 diagnostics.
- `scripts/archive/phase7/migrate_phase7_dataset_v2.py`: unsupported completed migration that revoked
  approval, added answer facts, and expanded qrels only across same-document exact-content duplicates.
- `scripts/archive/phase7/draft_phase7_calibration_fact_types.py`: unsupported completed generator for
  the review-required typed-fact calibration-v3 draft.
- `scripts/archive/phase7/apply_phase7_answer_facts.py`: unsupported completed source-review migration
  for 42 answerable rows and the documented calibration 011/012 qrel correction.
- `scripts/archive/phase7/generate_phase7_annotation_draft.py`: unsupported initial annotation
  generator retained for provenance; rerunning it would overwrite both frozen dataset splits.
- `scripts/archive/phase7/freeze_phase7_dataset.py`: unsupported completed dataset-v2 approval/freeze
  command; the active evaluator uses calibration-v3 and its v3 manifest.
- `scripts/archive/phase7/freeze_phase7_heldout_v2.py`: unsupported completed approval/freeze command
  for the private replacement held-out v2 dataset.
- `scripts/archive/phase7/evaluate_phase7_heldout_v2.py`: unsupported completed one-shot provider
  evaluator for the private replacement held-out v2 dataset.

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

`Dockerfile` supplies `api` and `ingestion` targets from a shared retrieval runtime. API adds the
LLM extra without Docling. Ingestion now adds Docling and the LLM extra, because its on-demand Phase 7
E2E CLI uses the same structured generator; this deliberately makes only the profile-gated tools image
heavier. The independent `ui` target starts again from `python:3.11-slim` and installs only base plus
`.[ui]`; it contains no retrieval, Docling or LangChain stack. Compose can start API/Qdrant/UI, and
keeps ingestion profile-gated. The shared `fastembed_cache` volume supplies models to API/ingestion
at runtime; weights are not baked into images.

The ingestion target installs Debian runtime packages `libxcb1`, `libgl1`, and
`libglib2.0-0t64`; Docling's PDF/image stack needs the shared libraries they provide when running
on `python:3.11-slim`.

## Testing

Default pytest uses fake embeddings and in-memory Qdrant. It must not download models, call Docker,
connect to a real Qdrant server, or require an API key. Real model/index/evaluator flows are explicit
integration commands documented in the README. Historical Phase 4 added offline tests for sparse
schema/IDF, safe re-indexing, document filters, metadata preservation, manifest mismatches, and RRF
duplicate/tie/empty-list behavior. Phase 4.1 adds offline candidate-pool tests for deterministic
rank/score preservation, union de-duplication, qrel-only candidate recall, scenario aggregation,
critical rows, and RRF-demotion diagnostics.
Phase 5 adds fake indexed cross-encoder tests for malformed outputs, finite scores, ordering and
ties, metadata preservation, no fallback, all candidate pools, document filtering, failure classes,
latency aggregation, CLI contracts, and no eager model initialization. Real model/Qdrant evaluation
remains separate from default pytest.
Historical Phase 6 added fake generation, evidence, citation, correction-retry, query-service, HTTP mapping, lazy
runtime, and security/logging tests. The original canonical Python 3.11 run passes 160 tests with one
known third-party Starlette/TestClient deprecation warning. The additive Gemini and UTF-8 response
regressions bring the local suite to 162, and the real adapter constructs in the Python 3.11 API image; no default test
calls Qdrant, FastEmbed, a reranker, or a generation provider.

Phase 7.4.1--7.5 adds offline Unicode/boundary role-inference, confidence, rank-prior replay,
malformed-snapshot, CPU-profile selection, fact-readiness, and runtime-readiness tests. Canonical
Python 3.11.15 validation passes 239 tests with the same one third-party warning; default pytest
still performs no network/model/Qdrant call.

The Streamlit layer adds offline tests for frozen profile selection/readiness, HTTP payload/auth
behavior, response schema validation, sanitized 401/422/503/504/network errors, bounded Unicode
session history, and citation page ordering. Tests import only the client/pure state helpers; the
actual Streamlit module is imported and health-checked in its dedicated image. The 2026-08-19
canonical Python 3.11 run passes `308 tests` with the same one third-party warning.

R00 validation passes `323 tests` with the same warning on both the host Python 3.13.5 environment
and the existing ingestion image's target Python 3.11.15. The Python 3.11 run is offline: source and
pytest packages are mounted read-only, with network and plugin autoload disabled.

Round 1's last code checkpoint passes `384 tests` on Python 3.11 with the same known third-party
warning. The count changed as private tests for unsupported CLIs were replaced by consolidated
archive and architecture guards; test count itself is not a contract. Current acceptance depends on
behavior coverage, full-suite PASS, source pins, and unchanged frozen data/artifacts.

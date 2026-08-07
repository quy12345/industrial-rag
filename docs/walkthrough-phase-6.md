# Phase 6 walkthrough — Query API, grounded generation, citations and abstention

## Outcome and status

Phase 6 was implemented on branch `feat/phase-6-query-api` from Phase 5 commit `07c074b`. The
working tree was clean before branch creation. No commit, push, merge, stash, re-index, collection
deletion, volume deletion, prune, or ingestion-image build was performed.

Status:

```text
Implementation:                         COMPLETE
Original Phase 6 Python 3.11 suite:    PASS — 160 tests
Current suite after provider/UTF-8:   PASS — 162 tests locally
Gemini adapter, Python 3.11 image:     PASS — offline construction
API Docker build/runtime:              PASS
Frozen Qdrant v1/v2:                   PASS — 99 / 99, hash unchanged
Real retrieval/reranking smoke:        PASS
Real OpenAI/Gemini provider smoke:     NOT RUN — provider key unavailable
Production readiness:                  NOT CLAIMED; Phase 7
```

The current roadmap is Phase 5 reranking, deferred Phase 5.1 optimization, Phase 6
query/generation/citations, then Phase 7 held-out evaluation and production hardening.

## Final architecture

```text
POST /api/v1/query
    ↓ Pydantic request validation
Lazy QueryService
    ↓ generator configuration check; no model load when key is missing
Frozen retrieval runtime
    ├── v1 dense top 20 ────┐
    └── v2 sparse top 20 ───┤ stable-ID union
                            ↓
            Jina multilingual cross-encoder
                            ↓
                  reranked final top_k
                            ↓
                       evidence gate
                            ↓
              deterministic S1…Sn blocks
                            ↓
  Provider-native strict GeneratedAnswer
                            ↓
       validate source IDs; at most one correction
                            ↓
     build citation metadata from RetrievalCandidate
                            ↓
                 QueryResponse or abstention
```

FastAPI runs the synchronous CPU/network pipeline through its threadpool. The route performs HTTP
mapping only. Heavy dense/sparse/reranker instances are lazy and cached, never created per request.

## Public API contract

Request:

```json
{
  "question": "Thuật toán nào phát hiện dữ liệu cảm biến bất thường?",
  "document_id": "manual-77d5dae4c2c5",
  "top_k": 5
}
```

- `question`: required, stripped, non-empty.
- `document_id`: optional; when supplied it must be non-empty and is forwarded as a server-side
  filter to dense and sparse Qdrant queries.
- `top_k`: default 5, accepted range 1–10, applied after reranking.
- Unknown request fields are rejected.

Response:

```json
{
  "answer": "...",
  "abstained": false,
  "abstention_reason": null,
  "citations": [
    {
      "chunk_id": "...",
      "document_id": "manual-77d5dae4c2c5",
      "filename": "manual.pdf",
      "page_numbers": [2],
      "headings": ["..."],
      "excerpt": "..."
    }
  ]
}
```

The public response never exposes dense, sparse, RRF, or reranker scores. Citations are always a
list; abstentions always return an empty list, while non-abstained responses require at least one
citation.

PowerShell must send Unicode JSON as UTF-8 bytes:

```powershell
$body = @{
  question = "Thuật toán nào phát hiện dữ liệu cảm biến bất thường?"
  document_id = "manual-77d5dae4c2c5"
  top_k = 5
} | ConvertTo-Json

$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod http://localhost:8000/api/v1/query `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $bytes
```

## Frozen retrieval and rollback

The API image does not contain `scripts/` or `artifacts/`, so production runtime cannot depend on
host manifests or frozen JSONL. `app/retrieval_runtime.py` packages the minimal immutable contract
and validates live Qdrant before constructing models:

```text
document_id:       manual-77d5dae4c2c5
chunk count:       99
chunk-ID hash:     bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be
dense v1:          industrial_manual_chunks / dense / 384 / cosine
hybrid v2:         industrial_manual_chunks_v2 / dense + sparse/IDF
dense model:       sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
sparse model:      Qdrant/bm25
BM25 avg_len:      72.83838383838383
reranker:          jinaai/jina-reranker-v2-base-multilingual
candidates:        dense 20, sparse 20, RRF k 60
```

Default accuracy-first configuration:

```text
RETRIEVAL_STRATEGY=union
RERANK_ENABLED=true
```

Explicit rollback:

```text
RETRIEVAL_STRATEGY=sparse
RERANK_ENABLED=false
```

Other combinations fail clearly. Sparse rollback validates/uses v2 only and does not create the
dense or cross-encoder model. There is no silent fallback when configured reranking fails.

The historic Phase 5 benchmark measured union Hit@5 `0.767`, MRR@5 `0.546`, candidate recall
`0.933`, critical bilingual intents `3/3`, and warm p95 `11,889.45 ms`. Accuracy is the reason for
the default; latency remains an explicit limitation.

## Generation provider contract

Dependency ranges:

```text
langchain-core >=1.5.1,<1.6.0
langchain-openai >=1.4.1,<1.5.0
```

The API build resolved `langchain-core 1.5.3`, `langchain-openai 1.4.1`, and `openai 2.53.0` on
Python 3.11.15. Full `langchain`, agents, retriever abstractions, vector stores, tools, file search,
web search, conversation memory, and Assistants API are not used.

The lazy `ChatOpenAI` adapter has two explicit modes. OpenAI remains the default:

```text
model:                 gpt-5.6-terra
use_responses_api:     true
output_version:        responses/v1
store:                 false
reasoning.effort:      low
max output tokens:     800
timeout:               60 seconds
provider max retries:  1
```

Gemini uses Google's OpenAI-compatible endpoint and Chat Completions:

```text
generation_provider:   gemini
model:                 gemini-3.5-flash-lite
base_url:              https://generativelanguage.googleapis.com/v1beta/openai/
use_responses_api:     false
reasoning_effort:      minimal
max output tokens:     800
timeout:               60 seconds
provider max retries:  1
```

Gemini configuration:

```text
GENERATION_PROVIDER=gemini
GEMINI_API_KEY=<YOUR_GEMINI_API_KEY>
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_REASONING_EFFORT=minimal
```

Google documents `gemini-3.5-flash-lite` as supporting structured outputs. The compatibility API
uses Chat Completions, so merely putting a Gemini key in `OPENAI_API_KEY` is insufficient and risks
sending it to the wrong endpoint. `OPENAI_STORE=false` remains a project safety invariant, but it
does not control Gemini data use; check the selected Google AI pricing/data tier before sending
sensitive documents.

Official references: [OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai),
[Gemini 3.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite), and
[Gemini pricing/data-use table](https://ai.google.dev/gemini-api/docs/pricing).

No temperature is forced. Model, reasoning, timeout, output limit, and retry count are typed env
settings. The API key is a Pydantic `SecretStr`, has no default, and is never written to logs,
artifacts, source, or the image. `OPENAI_STORE=true` is rejected by the query generator.

Structured output uses the Pydantic model as its only schema source:

```text
GeneratedAnswer
├── answer: string
├── source_ids: list[string]
└── insufficient_evidence: boolean
```

LangChain calls provider-native JSON-schema structured output with `strict=true` and
`include_raw=true`; raw provider data is used only to normalize optional usage metadata or detect a
refusal and is never returned publicly.

## Evidence and prompt-injection boundary

After reranking, candidates receive deterministic request-local labels `S1`, `S2`, and so on. The
backend keeps the authoritative map; the model never invents position-to-metadata mapping.

Each block includes source/chunk/document/filename/pages/heading followed by:

```text
<untrusted_document>
raw chunk content
</untrusted_document>
```

The prompt states that document content is reference data, not instructions; it forbids outside
knowledge, invented facts, changed numbers/units, following instructions embedded in documents, or
revealing the system prompt. The answer must use the question language and directly supported
source IDs.

Total formatted context is bounded to 24,000 characters. If needed, raw content is truncated
deterministically with `[…truncated…]` while source headers/mapping are retained. Public excerpts
default to 400 characters and are cut on Python Unicode code-point boundaries.

Questions and full evidence are not logged. Internal logs contain only stage timings and an
abstention/error class.

## Evidence gate and abstention

The deterministic pre-provider gate supports:

| Reason | Behavior |
|---|---|
| `no_candidates` | HTTP 200 abstention; no generation call |
| `invalid_candidate_metadata` | HTTP 200 abstention; no document sent to provider |
| `configured_score_gate_failed` | HTTP 200 abstention; only active if explicitly configured |
| `llm_insufficient_evidence` | HTTP 200 from valid structured model abstention |
| `llm_refusal` | HTTP 200; provider refusal is not treated as an answer |
| `citation_validation_failed` | HTTP 200 after the single correction also fails |

`EVIDENCE_SCORE_THRESHOLD` defaults to `None`. The 30-query retrieval development set does not
contain enough truly unanswerable data to calibrate a confidence threshold. BM25/reranker scores
are ranking signals, not probabilities; threshold calibration is Phase 7.

## Citation validation and correction

For a grounded answer:

- answer must be non-empty;
- `insufficient_evidence` must be false;
- at least one source ID must be present;
- every ID must exist in the request's evidence map;
- duplicate IDs/chunks are deduplicated in first-occurrence order;
- a requested document filter must match every citation.

A model abstention is valid only with `insufficient_evidence=true` and an empty source list. Unknown
IDs, no citation, empty answer, or abstention with citations are invalid.

One correction is allowed. It receives validation errors and allowed IDs, but uses the exact same
question/evidence; retrieval and reranking are not repeated. If correction fails, no unvalidated
answer is returned.

Filename, pages, headings, chunk ID, document ID, and excerpt are copied from the trusted candidate,
not model output. This guarantees referential validity. It does not prove every cited source
semantically supports every sentence; that held-out evaluation belongs to Phase 7.

## HTTP error mapping

| Condition | HTTP | Public code |
|---|---:|---|
| Request validation | 422 | FastAPI validation detail |
| Valid abstention | 200 | `abstention_reason` in response |
| Qdrant/retrieval failure | 503 | `retrieval_unavailable` |
| Reranker failure | 503 | `reranker_unavailable` |
| Missing selected-provider key | 503 | `llm_not_configured` |
| Generation provider unavailable | 503 | `llm_unavailable` |
| Generation provider timeout | 504 | `llm_timeout` |
| Unexpected error | 500 | `internal_error` |

Public errors contain a code and generic message only. They do not include exception messages,
tracebacks, provider responses, credentials, prompts, questions, or evidence.

Successful JSON responses explicitly declare `charset=utf-8`. This avoids mojibake in Windows
PowerShell 5.1, whose `Invoke-RestMethod` can otherwise decode `application/json` using a legacy code
page. For complete console output, pipe the returned object through `ConvertTo-Json -Depth 10` rather
than relying on PowerShell's abbreviated table view.

## Tests and measured validation

Canonical one-shot Python 3.11 validation used the existing ingestion image with a source mount; it
did not build ingestion:

```powershell
docker compose --profile tools run --rm --no-deps `
  -v "${PWD}:/workspace" -w /workspace ingestion `
  sh scripts/validate_phase6.sh
```

Measured output:

```text
python -m ruff check .                                     PASS
python -m pytest -q --basetemp /tmp/industrial-rag-phase6  PASS
160 passed, 1 warning in 8.47 s
```

After adding the Gemini provider route and UTF-8 response regression, the local full suite passes
`162 tests`; the baked Python
3.11 API image also constructs the real LangChain structured-output runnable in Gemini mode. A full
162-test run is not claimed in the API image because that image intentionally excludes Docling and
therefore cannot collect ingestion tests.

The warning is the existing third-party Starlette/TestClient deprecation. Tests do not globally
suppress or mass-upgrade dependencies to hide it. The offline suite covers request validation,
union/sparse behavior, post-rerank top-k, document filtering, gate short-circuit, deterministic
evidence, prompt-injection boundaries, provider kwargs, refusal/timeout/unavailability, source-ID
validation, Unicode citations, correction reuse, no fallback, sanitized HTTP errors, and no eager
model initialization.

## Docker validation

Only the API target was rebuilt:

```powershell
docker compose --progress plain build api
docker compose up -d qdrant api
```

Results:

```text
API build:                       PASS — 1m15s
Image size:                      146,177,894 bytes (146.18 MB / 139.41 MiB)
Python:                          3.11.15
langchain-core/openai/openai:    1.5.3 / 1.4.1 / 2.53.0
Docling in API:                  absent
Baked model files:              0
GET /api/v1/health:              200
POST /api/v1/query, no key:      503 llm_not_configured
Shared runtime model cache:      1.3 GB named volume
```

The image itself contains no model weights. The shared volume supplies the already downloaded Jina
model only at runtime. No API key, raw manual, metrics artifact, or `.env` is baked into the image.

No ingestion image build was run. The earlier heavy ingestion rebuild remains deferred after slow
external package downloads; this does not block Phase 6 API validation.

## Real runtime smoke

Read-only retrieval smoke, with source bind-mounted into the API image:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm --no-deps api `
  python scripts/validate_query_runtime.py
```

Measured results:

| Strategy | Candidates | Retrieval | Rerank |
|---|---:|---:|---:|
| Union + rerank | 23 | 32.20 ms | 13,897.50 ms |
| Sparse rollback | 14 | 5.63 ms | 0 ms |

The union result confirms API runtime can validate the frozen collections and load the shared Jina
cache. The timing is one smoke query, not a replacement for Phase 5 p95. FastEmbed emitted its known
mean-pooling compatibility warning; no model/version/index was changed in response.

Before and after Phase 6:

```text
industrial_manual_chunks:     99 points, frozen hash PASS
industrial_manual_chunks_v2:  99 points, frozen hash PASS
```

## Real OpenAI status and artifact

The environment and `.env` did not configure `OPENAI_API_KEY`, so no provider request was made. The
sanitized artifact was generated in the API image:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm --no-deps `
  -e PHASE6_BASE_COMMIT=07c074b603d2925f681f27c6a5c0d0a33314428f api `
  python scripts/query_smoke.py
```

`artifacts/metrics/phase-6-query-smoke.json` is ignored runtime evidence. It records model,
strategy, `store=false`, runtime versions, status `not_run`, and reason `api_key_unavailable`. It
does not contain an API key, question, answer, prompt, evidence, citation text, or raw response.

When a key is configured safely, the script runs at most three scenarios: Vietnamese answerable,
English query over Vietnamese evidence, and clearly unanswerable. It records only sanitized status,
citation counts/validity, stage latency, and optional token usage.

## Limitations and Phase 7 handoff

- Jina reranker CPU p95 remains approximately 11.89 seconds and its license is CC-BY-NC-4.0; it is
  acceptable for this benchmark/demo, not approved for commercial deployment.
- The current corpus is a 21-page Vietnamese technical research PDF, not a representative set of
  industrial operation/service manuals.
- Real Responses API behavior remains unverified until a key is configured; no result is inferred.
- Referential citation validity is enforced; semantic citation correctness is not yet measured.
- No calibrated answerability threshold exists; unanswerable precision/recall are unknown.
- OCR, cross-batch heading/table continuity, multi-page table handling, auth, rate limiting,
  streaming, multi-turn memory, production tracing, load tests, and deployment remain out of scope.

Phase 7 must introduce a separate held-out answerable/unanswerable set, a real industrial corpus,
answer/citation/abstention metrics, provider cost/token/latency measurements, reranker
latency/license resolution, readiness/restore procedures, and production hardening. It must not
tune and report final quality on the Phase 3A.2 retrieval development set.

# Industrial Technical Manual RAG

## Status

**Phase 7 calibration closure PARTIAL — held-out BLOCKED_GOVERNANCE**

Phase 6 implementation and offline/Docker correctness validation are complete. FastAPI exposes
`GET /api/v1/health` and `POST /api/v1/query`. The accuracy-first runtime uses dense@20 ∪ sparse@20,
multilingual cross-encoder reranking, an evidence gate, OpenAI/Gemini structured generation,
referential citation validation, and deterministic citations built from Qdrant payloads. Sparse
top-20 without reranking remains the explicit operational rollback.

An interactive Gemini request has returned an answer through the full API path. Phase 7 added a
separate two-manual ATV320 corpus and multiple calibration-only diagnostics. The latest closure fixes
confirmed evaluator false negatives, seals calibration loading from held-out access, and removes
exact cross-document duplicate evidence before the generation cutoff. It does not pass the final
calibration gate: item 010 still has direct evidence at full rank 6 and therefore outside top 5, while
the historical item-005 answer cannot be reconstructed from sanitized artifacts. No new provider run
was authorized in this closure.

The current held-out set is also `BLOCKED_GOVERNANCE`: historical tracked documentation mirrored its
content and the old calibration CLI parsed both splits. The current CLI no longer opens the held-out
JSONL in calibration mode and now refuses held-out execution, but Git history cannot be made unseen.
Use a new access-controlled final set or explicitly downgrade the current one to diagnostic status.

The canonical runtime is Python 3.11 with `qdrant-client >=1.19.0,<1.20.0`, direct-pinned FastEmbed
`0.8.0`,
and Qdrant server `v1.18.3`. The historic dense artifact that records client `1.18.0` is retained
unchanged; it is a historic measurement, not the dependency declaration used after closure.

For a module-by-module architecture explanation, Qdrant mental model, code-redundancy audit, query
debugging guide, and interview preparation, see `docs/project-deep-dive.md`.

## Requirements and installation

The supported development/runtime version is **Python 3.11**. Use one interpreter consistently:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
```

Extras are separated by responsibility:

- `.[retrieval]`: Qdrant client, FastEmbed, dense retrieval, sparse BM25, and RRF.
- `.[ingestion]`: Docling document parsing.
- `.[llm]`: `langchain-core` and `langchain-openai` for Responses structured generation.
- `.[dev]`: test/lint tooling plus retrieval, ingestion, and LLM dependencies.

`EMBEDDING_CACHE_DIR` is optional locally. In Docker it is set to `/models/fastembed` and backed by
a shared named volume; model weights are never baked into either image.

## Phase 7 isolated industrial-corpus evaluation

Phase 7 keeps the legacy 99-chunk development collections untouched. The new ATV320 corpus uses
`industrial_manual_phase7_dense_v1` and `industrial_manual_phase7_hybrid_v1`, both frozen at 2,753
points with stable-ID fingerprint
`2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d`.

The active calibration-v3 set has 12 answerable and 8 unanswerable cases. The current held-out set
has 30 answerable and 15 unanswerable cases. The files remain frozen, but the held-out split must not
be described as statistically unseen because of historical repository exposure. Verify the approved
dataset contract locally without models or Qdrant:

```powershell
python -m scripts.validate_phase7_dataset
```

The reproducible freeze command is shown below. It has already completed for the current files; the
historical v1 approval token is intentionally rejected:

```powershell
python -m scripts.freeze_phase7_dataset `
  --approval-token "APPROVE PHASE 7 DATASET V2"
```

The real end-to-end evaluator refuses draft datasets. Calibration mode validates only
`calibration-v3.jsonl`, frozen chunks, and manifest metadata; it obtains the held-out hash from the
manifest without opening `test.jsonl`. It then validates live Qdrant hashes and writes a sanitized
resumable artifact containing only IDs/ranks/metrics:

```powershell
python -m scripts.evaluate_phase7_e2e --dataset calibration `
  --provider-approval-token "APPROVE PHASE 7 CALIBRATION V5 STABILITY EGRESS"
```

This command sends calibration questions and retrieved manual evidence to the configured generation
provider. Do not run it until the code is reviewed and the exact new approval is granted. Held-out
execution is currently refused with `BLOCKED_GOVERNANCE`, even if the historical token is supplied.

The historical dataset-v1 Gemini calibration on 2026-08-09 completed all 20 records but failed its
quality gate: Hit@5 `0.583`, MRR@5 `0.444`, direct-evidence citation rate `0.500`, and total p95
`11.585 s`. Its reported phrase accuracy `0.417` is retained only as historical evidence: that metric
incorrectly compared Vietnamese answers with English evidence wording. It must not be compared with
the dataset-v2 answer-fact metric. The held-out benchmark has not been run; see
`docs/walkthrough-phase-7.md`.

The sealed evaluator now writes `phase-7-calibration-e2e-v5.json` and a separate v5 checkpoint, so
rerunning it cannot overwrite v1--v4 evidence. Its identity includes prompt/evaluator/runtime hashes,
provider settings, library versions, top-k, and correction count.

The active calibration-v3 SHA-256 is
`8e3e7798d667cccef0b3ac16aa292fb1f18381dc6e754a7b9dace1250755fa46`. The manifest records held-out
SHA-256 `68c9c52e745a7616a869a2f55024964501dfb0cb537bbe6ff91dac5fbcae3c54`; calibration code reads this
hash from the manifest and does not open the held-out dataset.

The historical dataset-v2 diagnostics calibration completed all 20 rows with candidate recall/Hit@5
`0.667`, strict contiguous fact accuracy `0.500`, direct-evidence citation rate `0.667`, and
abstention precision/recall `1.000/1.000`. Phase 7.4 now keeps strict phrase matching as a diagnostic
and uses deterministic typed facts as the headline metric. A sanitized rescore of the same answers
raises deterministic fact accuracy to `8/12 = 0.667`; this is a derived evaluator result, not a new
provider run.

Phase 7.4.1 preserves that retrieval profile but separates the weak fusion role signal from a strong
post-rerank rank-only prior. The selected profile is weighted RRF `k=40`, sparse weight `1.25`,
dense@5/sparse@24 coverage reserves, fusion multiplier `0.10`, then a `0.50` document-role prior with
rank offset `20` for strong and weak inferences. It is selected from a sanitized Jina snapshot using
six bilingual intent folds; raw Jina scores remain intact for diagnostics.

The final provider-free closure measures candidate recall `12/12 = 1.000`, Hit@5 `11/12 = 0.917`,
MRR@5 `0.875`, wrong-document top-1 `0/12`, and wrong-document top-5 `8/60 = 0.133`. English is
`6/6`; Vietnamese is `5/6`; calibration 010 is rank 6; reranker input is capped at 30. This passes
the contamination gate without an expected-document filter, qrel change, re-index, or model change.

Phase 7.5 freezes the final reranker budget at 30 and batch size 8 (runtime-default ONNX threads).
Three repetitions of all 12 calibration questions measure rerank p95 `6.996 s` and total p95
`7.027 s`, a 47.8% improvement from the 13.399 s baseline while retaining the closure metrics. A
separate one-pass closure run measures rerank p95 `9.706 s`; it is not statistically comparable to
the repeated CPU benchmark. The corpus, datasets, collections, MiniLM, Jina, and FastEmbed `0.8.0` remain frozen. Artifacts are
sanitized and contain zero provider calls and zero held-out questions.

Typed calibration-v3 is approved and active; qrels/pages/phrases remain unchanged. Historical
calibration v4 completed all 20 rows but failed release: deterministic fact accuracy was `8/12`,
wrong-document citations were `2`, and item 010 direct evidence was excluded by the actual top-5
generation cutoff. The new generic evaluator makes `block`/`blocked` and `contact`/`contacts` match,
while identifier, numeric/unit, ordering, sign, and negation controls remain strict. Historical item
005 remains unknown because v4 intentionally stored no raw answer or match span.

The 2026-08-11 provider-free snapshot-v2/replay tested the registered CE-rank/RRF-rank grid and the
single `list_completeness_v1` fallback. Neither moves item 010 from full rank 6 into actual evidence
top 5, so the fallback is not activated and no failed profile becomes a runtime default. Exact
cross-document duplicate selection does reduce active-profile wrong-document evidence from `8/60`
to `7/60`, while preserving candidate recall `12/12`, Hit@5 `11/12`, and MRR@5 `0.875`.

Canonical ingestion-container Python 3.11.15 validation: Ruff PASS and pytest
`279 passed, 1 warning`. The repository `.venv` is Python 3.13.5 and independently produced the same
result, but it is not the canonical interpreter and was not overwritten. The warning is the known
third-party Starlette/TestClient deprecation.

Run the provider-free Phase 7.4.1--7.5 diagnostics only after Qdrant and the shared FastEmbed cache are
available. These commands never call Gemini/OpenAI and never execute held-out questions:

```powershell
python -m scripts.calibrate_phase7_weighted_fusion
python -m scripts.evaluate_phase7_retrieval_closure `
  --output artifacts/metrics/phase-7-contamination-closure-v4.json
python -m scripts.create_phase7_reranker_snapshot
python -m scripts.calibrate_phase7_role_prior
python -m scripts.generate_phase7_calibration_closure_readiness
```

`diagnose_phase7_calibration_005` and full v5 calibration require new, separate provider-egress
approvals and are intentionally not included in provider-free reproduction. Three full runs must be
aggregated with `scripts.aggregate_phase7_calibration_stability`; the worst run, not the best run, is
the release headline. See `docs/walkthrough-phase-7-calibration-closure.md` for the sealed data flow,
matcher policy, replay evidence, stop decision, artifacts, and governance boundary.

## Quickstart: khởi động, test, chạy demo và dừng

Các lệnh Docker được chạy tại repository root. Không cần activate `.venv` để build hoặc chạy
container:

```powershell
cd D:\Git\industrial-rag
```

### 1. Cấu hình provider

Tạo hoặc cập nhật `.env`. Ví dụ dùng Gemini OpenAI-compatible:

```text
GENERATION_PROVIDER=gemini
GEMINI_API_KEY=YOUR_NEW_GEMINI_API_KEY
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_REASONING_EFFORT=minimal
GEMINI_TEMPERATURE=0

OPENAI_MAX_OUTPUT_TOKENS=800
OPENAI_TIMEOUT_SECONDS=60
OPENAI_MAX_RETRIES=1
OPENAI_STORE=false

RETRIEVAL_STRATEGY=union
RERANK_ENABLED=true
```

Không commit `.env`, không ghi key vào Dockerfile/source và không dán key vào terminal output hoặc
chat. Nếu key từng xuất hiện trong log, phải revoke và tạo key mới trước khi tiếp tục.

### 2. Build và khởi động

Build riêng API sau lần checkout đầu tiên hoặc sau khi source/Dockerfile/dependencies thay đổi:

```powershell
docker compose --progress plain build api
```

Không cần build ingestion để chạy query API. Khởi động Qdrant và API:

```powershell
docker compose up -d qdrant api
docker compose ps
```

Kiểm tra health và UTF-8 response header:

```powershell
$health = Invoke-WebRequest -UseBasicParsing http://localhost:8000/api/v1/health
$health.StatusCode
$health.Headers["Content-Type"]
```

Kết quả mong đợi là HTTP `200` và `application/json; charset=utf-8`.

### 3. Chạy unit tests

Local tests cần cùng một interpreter Python 3.11; activate `.venv` chỉ cần cho workflow local này,
không ảnh hưởng Docker:

```powershell
python --version
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q --basetemp ".pytest-tmp-$PID"
```

`python --version` phải là 3.11.x. Test suite tự bỏ qua repository `.env` và các biến settings của
shell, nên không gọi provider thật hoặc đưa API key vào assertion output. Nếu Windows báo
`PermissionError: [WinError 5]`, dùng một tên `--basetemp` mới nằm trong workspace; không cần chạy
toàn bộ terminal bằng Administrator chỉ để che lỗi ACL của thư mục temp cũ.

Kiểm tra Compose mà không khởi động service:

```powershell
docker compose config --quiet
```

### 4. Chạy query demo

Windows PowerShell 5.1 nên gửi request bằng UTF-8 bytes và lưu kết quả vào biến để tránh table
formatter cắt ngắn answer:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$body = @{
    question = "Thuật toán nào được sử dụng để phát hiện dữ liệu cảm biến bất thường?"
    document_id = "manual-77d5dae4c2c5"
    top_k = 5
} | ConvertTo-Json

$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)

$result = Invoke-RestMethod http://localhost:8000/api/v1/query `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body $bytes

$result.answer
$result.citations | Format-List *
$result | ConvertTo-Json -Depth 10
```

Request đầu tiên có thể chậm vì runtime phải khởi tạo dense model và multilingual reranker. Xem log
service mà không in secret/prompt/evidence:

```powershell
docker compose logs --tail 100 api
```

### 5. Sau khi chỉnh sửa

| Thay đổi | Lệnh cần chạy |
|---|---|
| Chỉ sửa `.env` hoặc API key | `docker compose up -d --force-recreate api` |
| Sửa `app/`, Dockerfile hoặc dependencies | `docker compose build api` rồi `docker compose up -d --force-recreate api` |
| Sửa Python source khi đang dùng dev override | Compose tự reload; dependency change vẫn phải rebuild API |
| Sửa ingestion code/dependencies | Chỉ build ingestion thủ công khi thực sự cần ingestion |

Dev mode bind-mount source và bật Uvicorn reload:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d qdrant api
```

### 6. Dừng an toàn

Dừng hằng ngày nhưng giữ container, collection, Qdrant storage và model cache:

```powershell
docker compose stop api qdrant
```

Khởi động lại các container đã dừng:

```powershell
docker compose start qdrant api
```

Nếu muốn xóa container/network của Compose nhưng vẫn giữ named volumes:

```powershell
docker compose down
```

Không dùng `docker compose down -v`, `docker system prune` hoặc `docker volume prune`; các lệnh đó có
thể xóa Qdrant collection hoặc FastEmbed model cache cần giữ lại.

### Rebuild the tools ingestion image when needed

The `ingestion` Docker target now installs `.[retrieval,ingestion,llm]`. It can therefore run the
on-demand Phase 7 E2E CLI, which imports `langchain-core` and `langchain-openai`, as well as Docling
ingestion. This makes only the profile-gated tools image heavier; the API target remains Docling-free.

Run this manually from the repository root after pulling this change. It does not download FastEmbed
or Jina model weights during the Docker build:

```powershell
docker compose --progress plain --profile tools build ingestion
```

The 2026-08-11 rebuilt tools image passed Python 3.11.15 plus Docling/`langchain-core`/
`langchain-openai` import checks. Docker reports `9.66 GB` total and `9.515 GB` unique disk usage via
`docker system df -v`; image metadata reports `3,255,679,310` bytes via `docker image inspect`.
These values use different Docker storage accounting, so both are retained rather than presenting a
single misleading number. Size optimization is deferred; correctness came first, and no prune was
run.

## Ingestion and dense indexing

Place the document under `data/raw/`. The checked retrieval-development baseline is frozen against
`artifacts/manual-batched.jsonl`: the 21-page manual, batch size 4, document ID
`manual-77d5dae4c2c5`, and 99 chunks.

```powershell
python scripts/index_document.py data/raw/manual.pdf `
  --page-start 1 `
  --page-end 21 `
  --page-batch-size 4
```

The index manifest records collection/model compatibility. Indexing parses and embeds before it
updates Qdrant, upserts new points, then removes stale points for only the indexed document.
Chunk and point IDs are deterministic. Re-indexing the same frozen input must leave 99 points.

Search a specific indexed document:

```powershell
python scripts/search_dense.py `
  "What sensor attributes are used by the algorithm?" `
  --document-id manual-77d5dae4c2c5 `
  --limit 5
```

Dense cosine scores are ranking signals, not probabilities.

## Retrieval development evaluation

`data/eval/dense_smoke.jsonl` is a **30-query retrieval development set**, with 15 Vietnamese and
15 English factual queries. It is not the held-out final evaluation set planned for Phase 7.

Every item has stable `relevant_chunk_ids`. Direct retrieval hits, Hit@k, and MRR use only those
IDs. Expected phrases are validated against the relevant frozen chunks; pages are diagnostics only,
so a same-page unrelated chunk never counts as a hit.

The evaluator supports the same qrels and chunk set for all strategies. English queries are reported
as **cross-lingual** (`en` query → `vi` evidence); Vietnamese queries are monolingual.

Run the immutable dense metric closure after Qdrant contains exactly the frozen chunk IDs:

```powershell
python scripts/evaluate.py --strategy dense `
  --chunks artifacts/manual-batched.jsonl `
  --limit 20 `
  --output artifacts/metrics/dense-baseline-closure.json
```

The JSON artifact is the source of truth. It records the chunk-set hash, dataset hash, model/index
contract, direct-evidence rank per query, Hit@1/3/5/20, Candidate Recall@20, MRR@5, MRR@20,
per-language/scenario/category metrics,
critical-query diagnostics, failure cases, and warm-query average/p50/p95 latency. Primary latency
includes query embedding and Qdrant round trip but excludes model initialization, model download,
and one warmup query.

Old page-or-phrase smoke metrics are not directly comparable with this direct-evidence baseline.

Verified 2026-08-05 baseline on the frozen 99 chunks: Hit@1 `0.167`, Hit@3 `0.367`, Hit@5 `0.400`,
MRR@5 `0.269`, MRR@20 `0.298`, p50 latency `15.14 ms`, p95 latency `37.77 ms`, and 18 failure
cases. The immutable `dense-baseline.json` is not overwritten; this additive closure recorded
Hit@20 `0.767`, p50 `25.69 ms`, and p95 `35.40 ms` on 2026-08-06.

## Hybrid BM25 + RRF retrieval

Dense collection v1, `industrial_manual_chunks`, remains intact for the dense baseline. Hybrid data
uses the independent `industrial_manual_chunks_v2` collection: named `dense` (384-d cosine) and
named `sparse` (Qdrant IDF modifier). Each point keeps the same deterministic UUIDv5 and
citation-ready payload as v1. `document_id` has a keyword payload index.

BM25 uses FastEmbed 0.8.0 `Qdrant/bm25` with `disable_stemmer=True`, `k=1.2`, `b=0.75`, and the
exact FastEmbed BM25 preprocessing/tokenizer result computed from the frozen 99 chunks:
`avg_len=72.838384`. It does not use English stopword removal/stemming for the Vietnamese corpus.
The independent manifest is `artifacts/metrics/hybrid-index-manifest.json`; hybrid search refuses
to run if its schema, models, sparse IDF configuration, chunk hash, or RRF configuration mismatch.

Index v2 only after ingestion reproduces the frozen set:

```powershell
python -m scripts.index_hybrid data/raw/manual.pdf --page-batch-size 4
python -m scripts.index_hybrid data/raw/manual.pdf --page-batch-size 4
```

Search with 20 dense candidates, 20 sparse candidates, RRF `k=60`, and bounded output. Ranks are
one-based. RRF is `sum(1 / (k + rank))`; raw cosine and BM25 scores are never added together and
all scores are ranking signals, not probabilities.

```powershell
python -m scripts.search_hybrid "Thuật toán ODA-MD có mục tiêu gì?" `
  --document-id manual-77d5dae4c2c5 --limit 5
```

```powershell
python -m scripts.evaluate --strategy sparse --limit 20
python -m scripts.evaluate --strategy hybrid --limit 20
```

The 2026-08-06 Python 3.11 container run measured the following development-set results:

| Metric | Dense | Sparse | Hybrid | Hybrid − Dense |
|---|---:|---:|---:|---:|
| Hit@1 | 0.167 | 0.333 | 0.267 | +0.100 |
| Hit@3 | 0.367 | 0.500 | 0.400 | +0.033 |
| Hit@5 | 0.400 | 0.633 | 0.533 | +0.133 |
| Hit@20 | 0.767 | 0.867 | 0.867 | +0.100 |
| MRR@5 | 0.269 | 0.441 | 0.365 | +0.096 |
| MRR@20 | 0.298 | 0.469 | 0.398 | +0.100 |
| p50 latency | 25.69 ms | 2.16 ms | 19.14 ms | -6.55 ms |
| p95 latency | 35.40 ms | 2.78 ms | 26.16 ms | -9.24 ms |

Hybrid clears the Hit@5, MRR@20, Hit@20, and p95 targets, but only one of the three bilingual
critical intents has direct evidence in top 5. See `docs/walkthrough-phase-4.md` for per-language,
scenario, critical, and failure diagnostics. This remains a retrieval-development set, not the
held-out Phase 7 end-to-end evaluation.

## Phase 5 candidate-pool handoff

Candidate recall answers a different question from Hit@k: whether a reranker could see direct
evidence at all. It is measured before any reranking over the same 30 qrels and frozen 99 chunks.

```powershell
python -m scripts.audit_candidate_pools --limit 20
python -m scripts.generate_phase5_readiness
```

The generated JSON artifacts are ignored runtime evidence: `artifacts/metrics/candidate-pool-audit.json`
and `artifacts/metrics/phase-5-readiness.json`. The 2026-08-06 audit found query-level candidate
coverage of `0.767` (dense top 20), `0.867` (sparse top 20), `0.867` (hybrid RRF top 20), and
`0.933` (unbounded dense@20 ∪ sparse@20, 22–34 unique candidates). The union misses only
`dense_014` and `dense_017`; a reranker cannot recover those from this candidate pool.

Phase 5 must benchmark all three candidate strategies, not assume hybrid is the default:
`sparse_top20`, `hybrid_top20`, and `dense20_union_sparse20`. Sparse currently has stronger top-rank
development metrics than RRF hybrid. Four cases (`dense_005`, `dense_019`, `dense_021`, and
`dense_029`) have sparse top-5 direct evidence that RRF demotes outside top 5; dense adds evidence
absent from sparse for `dense_008` and `dense_020`. See
`docs/walkthrough-phase-4-closure.md` for the full diagnosis and reproducible checks.

## Multilingual cross-encoder reranking

Phase 5 uses FastEmbed 0.8.0 `TextCrossEncoder` with
`jinaai/jina-reranker-v2-base-multilingual`. Candidate text is the heading breadcrumb, two newlines,
then the unchanged raw chunk text. Scores are ranking signals, not probabilities. The adapter is
lazy: importing the module does not initialize or download the model, and errors never silently
fall back to a retrieval ranking.

The model license was verified from the official model metadata as **CC-BY-NC-4.0**. It is suitable
for this non-commercial benchmark/demo, but must not be assumed suitable for commercial deployment.
Use another licensed model or obtain appropriate rights before a commercial release.

Run an interactive reranked search with an explicit strategy:

```powershell
python -m scripts.search_reranked `
  "What sensor attributes are used by the algorithm?" `
  --strategy union --document-id manual-77d5dae4c2c5 --limit 5
```

Run all three real-model evaluations in the existing Python 3.11 ingestion image, without rebuilding
it. The source tree is bind-mounted and the Compose `fastembed_cache` volume persists the downloaded
model at runtime:

```powershell
docker compose up -d qdrant
docker compose --profile tools run --rm -v "${PWD}:/app" ingestion `
  python -m scripts.evaluate_reranking --strategy all
```

To regenerate only the comparison from existing strategy artifacts, with no model initialization:

```powershell
python -m scripts.evaluate_reranking --comparison-only
```

Measured on 2026-08-06, Python 3.11.15 CPU runtime, frozen 30 qrels and 99 chunks:

| Candidate pool | Hit@5 | MRR@5 | Hit@20 | MRR@20 | Candidate recall | Warm total p95 |
|---|---:|---:|---:|---:|---:|---:|
| Sparse rerank | 0.733 | 0.529 | 0.867 | 0.544 | 0.867 | 9,879.69 ms |
| Hybrid rerank | **0.767** | **0.546** | 0.867 | 0.556 | 0.867 | 8,465.75 ms |
| Union rerank | **0.767** | **0.546** | **0.933** | **0.560** | **0.933** | 11,889.45 ms |

All strategies passed the 3/3 bilingual critical-intent, Hit@5, and MRR@5 gates. None passed the
warm total p95 target of 1.5 seconds. Union is the best observed research strategy, but
the historical Phase 5 artifact keeps `recommended_default_strategy=null`. Phase 6 explicitly
selects union as its accuracy-first API path and keeps sparse as the low-latency rollback.
See `docs/walkthrough-phase-5.md` for scenario metrics, critical ranks, failure classes, commands,
and the full validation record.

## Grounded query API

Phase 6 deliberately keeps retrieval, reranking, evidence gating, and citation validation in
project code. LangChain is used only to orchestrate the prompt, invoke the selected generation
provider, and parse the provider-native `GeneratedAnswer` schema:

```text
question
→ dense v1 top 20 ∪ sparse v2 top 20
→ Jina multilingual rerank
→ final top_k evidence
→ evidence gate
→ strict structured output (OpenAI Responses or Gemini Chat Completions)
→ source-ID validation and at most one correction
→ trusted Qdrant citation metadata
```

Default configuration:

```text
RETRIEVAL_STRATEGY=union
RERANK_ENABLED=true
GENERATION_PROVIDER=openai
OPENAI_MODEL=gpt-5.6-terra
OPENAI_REASONING_EFFORT=low
OPENAI_MAX_OUTPUT_TOKENS=800
OPENAI_TIMEOUT_SECONDS=60
OPENAI_MAX_RETRIES=1
OPENAI_STORE=false
```

Set `OPENAI_API_KEY` in the local `.env`; never put it in source, Dockerfile, commands, or chat.
Health remains available without the key. A query then returns sanitized HTTP 503
`llm_not_configured` before loading retrieval models.

Gemini is supported through Google's OpenAI-compatible Chat Completions endpoint; no Google SDK is
required. Use a Gemini key from Google AI Studio and keep the OpenAI key unset:

```text
GENERATION_PROVIDER=gemini
GEMINI_API_KEY=<YOUR_GEMINI_API_KEY>
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_REASONING_EFFORT=minimal
GEMINI_TEMPERATURE=0
OPENAI_MAX_OUTPUT_TOKENS=800
OPENAI_TIMEOUT_SECONDS=60
OPENAI_MAX_RETRIES=1
OPENAI_STORE=false

RETRIEVAL_STRATEGY=union
RERANK_ENABLED=true
```

The base URL is essential: without it, an OpenAI-compatible client would send the Gemini key to the
OpenAI endpoint. Gemini compatibility currently uses Chat Completions, not the OpenAI Responses
path used by the OpenAI provider. Structured output and the project's citation validator remain
active in both modes. Google's data-use terms depend on the Gemini billing tier; `OPENAI_STORE=false`
does not control Google's retention or model-improvement policy.

References: [Google OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai) and
[Gemini 3.5 Flash-Lite model card](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite).

PowerShell request:

```powershell
$body = @{ question = "Thuật toán nào phát hiện dữ liệu cảm biến bất thường?"; top_k = 5 } |
  ConvertTo-Json
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod http://localhost:8000/api/v1/query `
  -Method Post -ContentType "application/json; charset=utf-8" -Body $bytes
```

On Windows PowerShell 5.1, assign the result and use `ConvertTo-Json` instead of its default table
formatter so long answers and nested citations are not abbreviated:

```powershell
$result = Invoke-RestMethod http://localhost:8000/api/v1/query `
  -Method Post -ContentType "application/json; charset=utf-8" -Body $bytes
$result | ConvertTo-Json -Depth 10
```

The API explicitly returns `application/json; charset=utf-8` for compatibility with that legacy
client. PowerShell 7 also handles UTF-8 JSON correctly.

Request fields are `question`, optional `document_id`, and `top_k` from 1–10. A successful response
contains `answer`, `abstained`, optional `abstention_reason`, and a citation list. Citation filename,
pages, headings, and excerpt come from trusted retrieved payloads, never from model-generated
metadata. Valid no-evidence/model/refusal/citation-validation abstentions return HTTP 200; Qdrant,
reranker, and provider dependency failures return 503, provider timeouts return 504, and malformed
requests return 422.

Explicit low-latency rollback:

```text
RETRIEVAL_STRATEGY=sparse
RERANK_ENABLED=false
```

The optional evidence score threshold defaults to `None`: reranker/BM25 scores are ranking signals,
not probabilities, and no threshold is claimed as calibrated. See `docs/walkthrough-phase-6.md` for
the prompt-injection boundary, citation contract, retry behavior, smoke status, and reproduction.

## Docker

The default Compose file is production-like: source is not bind-mounted and API reload is off.
It contains Qdrant, the API runtime, and an on-demand `ingestion` service in the `tools` profile.
Qdrant is pinned to `qdrant/qdrant:v1.18.3` and its named volume is persistent.

```powershell
docker compose --progress plain build api
docker compose up -d qdrant api
```

The API image installs base + `.[retrieval]` + `.[llm]`, but not Docling. Build and run ingestion
only when needed:

```powershell
docker compose --progress plain --profile tools build ingestion
docker compose --profile tools run --rm ingestion `
  python scripts/index_document.py /data/raw/manual.pdf `
  --page-start 1 --page-end 21 --page-batch-size 4
```

The Phase 6 API build completed in 1m15s. Docker inspection measured `146,177,894` bytes
(146.18 MB decimal); Python is 3.11.15, `langchain-core` is 1.5.3, `langchain-openai` is 1.4.1,
and `openai` is 2.53.0. Docling is not importable and the unmounted baked cache contains zero model
files. Health returned 200 and a configured-without-key query returned sanitized 503. The existing
shared runtime cache remains 1.3 GB; it is a named volume, not part of the image.

No Phase 6 ingestion build was run. The prior ingestion image remains available for on-demand tools;
its fresh rebuild remains a manual/deferred task after the earlier external package-download delay.

For live reload during development, add the override file:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up api
```

Normal `docker compose up` reuses the fixed `industrial-rag-api:local` image; it does not create a
new image unless one is missing or `build`/`--build` is requested. Do not run `docker system prune`
or `docker volume prune` for this project: the Qdrant and model-cache volumes are intentionally
preserved.

## Known limitations

- OCR for scanned PDFs is not enabled.
- Page-range batch boundaries can change structure-aware chunks and heading context; do not compare
  metrics across different chunk sets.
- Multi-page tables can be split at batch boundaries.
- Docling remains a heavy, on-demand ingestion dependency.
- Dense retrieval remains frozen as the Phase 3A.2 baseline; Phase 4 improves candidates through
  a separate sparse/RRF collection rather than hidden changes to this development set.
- Hybrid retrieval improves the development-set aggregate, but two critical bilingual intents still
  miss top 5 before reranking.
- Accuracy-first union reranking is now the Phase 6 default, but it remains slow on CPU: the Phase 5
  warm p95 was 11.89 seconds and the Phase 6 real smoke measured 13.90 seconds for one rerank. Sparse
  rollback measured 5.63 ms retrieval with no reranker.
- The selected reranker is licensed CC-BY-NC-4.0 and is not approved here for commercial use.
- Gemini has returned an answer in an interactive request, but the sanitized bounded provider smoke
  artifact is still pending; the OpenAI path remains untested. Offline structured-output, provider
  routing, and HTTP behavior are covered by fakes.
- Referential citation IDs are validated in Phase 6, but semantic claim support is a Phase 7 metric.

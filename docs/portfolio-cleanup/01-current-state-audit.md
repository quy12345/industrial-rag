# Audit trạng thái hiện tại

> **R00 delta — 2026-08-22:** phần audit bên dưới mô tả snapshot trước R00. Hiện tại active runtime
> chỉ chấp nhận Phase 7; `manual.pdf` đã bị loại khỏi local corpus; Phase 3–6 CLIs nằm dưới
> `scripts/archive/phase6/`; root `scripts/` chỉ còn Phase 7 utilities và generic `ingest_preview`.
> Production `app.retrieval_runtime` chỉ sở hữu `PHASE7_RETRIEVAL_CONTRACT`, builders yêu cầu explicit
> contract, và full offline suite hiện pass 323 tests trên cả host Python 3.13.5 lẫn ingestion-image
> Python 3.11.15. Các bảng/số dòng phía dưới vẫn được giữ làm audit evidence tại thời điểm
> 2026-08-20, không phải inventory sau R00.

## 1. Phạm vi và bằng chứng audit

Audit được thực hiện trên worktree ngày 2026-08-20. Trạng thái ban đầu:

```text
branch: feat/phase-7-heldout-evaluation
HEAD:   d3b500d add streamlit demo
dirty:  M AGENTS.md
```

`AGENTS.md` là thay đổi có sẵn của người dùng (338 dòng thêm, 18 dòng xóa theo
`git diff --stat`) và không bị sửa. Không có `AGENTS.md` lồng sâu hơn. Hai mươi commit gần nhất cho
thấy repository được xây theo Phase 3–7, sau đó thêm Streamlit; lịch sử hiện tại được giữ nguyên.

Audit đã đọc/đối chiếu:

- toàn bộ module trong `app/` và `ui/`, import graph và các public symbols;
- toàn bộ test files và từng test contract;
- operational/evaluation CLI, shell validation, cross-script imports và vai trò của từng nhóm;
- `pyproject.toml`, `.env.example`, Dockerfiles, Compose, CI, `.gitignore`, source manifest;
- README, `docs/CODEBASE.md`, `docs/PROJECT_JOURNEY.md`, `docs/plan.md` và walkthroughs;
- metadata của tracked JSONL/frozen chunks và danh sách ignored artifacts.

Không đọc local `.env`, raw PDF, model cache, ignored private held-out payload hoặc raw question,
answer/evidence của held-out. Với các script chứa annotation literals, audit chỉ dùng module
docstring, symbols, imports, arguments và write boundaries; đây là giới hạn có chủ đích để tuân thủ
benchmark governance.

## 2. Python, dependency và entry points

`pyproject.toml` yêu cầu Python `>=3.11`; base dependencies là FastAPI/Pydantic và optional groups
gồm `retrieval`, `ingestion`, `llm`, `ui`, `dev`. Các version quan trọng đã được pin/range trong file,
đáng chú ý `qdrant-client[fastembed]>=1.19,<1.20`, `fastembed==0.8.0` và
`docling>=2.117,<2.118`.

Entry points thực tế:

| Loại | Entry point | Contract hiện tại |
|---|---|---|
| FastAPI | `app.main:app` | `/api/v1/health`, `/api/v1/ready`, `/api/v1/query` |
| Streamlit | `ui/streamlit_app.py:render` | chỉ gọi FastAPI qua `ui.api_client.RAGAPIClient` |
| Ingestion/index | `scripts.ingest_preview`, `scripts.index_document`, `scripts.index_hybrid`, `scripts.index_phase7_corpus` | explicit CLI; có filesystem/Qdrant mutation |
| Search/smoke | `scripts.search_dense`, `scripts.search_hybrid`, `scripts.search_reranked`, `scripts.validate_query_runtime`, `scripts.query_smoke` | manual/integration diagnostics |
| Evaluation | `scripts.evaluate`, `scripts.evaluate_reranking`, `scripts.evaluate_phase7_e2e`, `scripts.evaluate_phase7_heldout_v2` cùng nhóm calibration/readiness | reproducible evaluation, một số lệnh cần Qdrant/model/provider approval |

Docker có bốn concerns: `retrieval-runtime` base, `api`, `ui` và tool-only `ingestion`. Compose bảo vệ
Qdrant bằng named volumes `qdrant_data` và `fastembed_cache`; API image không copy `scripts/`, raw
data hay artifacts. CI chạy Ruff và pytest trên Python 3.11.

## 3. Kiến trúc đang tồn tại

Code hiện tại là một modular monolith theo lịch sử phát triển, nhưng chưa theo layer:

```text
Browser
  -> ui/streamlit_app.py
  -> ui/api_client.py --HTTP--> app/main.py + app/api/*
                                  -> app/query_service.py
                                     -> app/retrieval_runtime.py
                                        -> retrieval/hybrid/reranking
                                     -> evidence_selection + gate
                                     -> generation
                                     -> citations

scripts/index_* -> ingestion -> retrieval + hybrid_retrieval -> Qdrant

scripts/evaluate_* -> evaluation/phase7/evaluation_e2e
                   -> public query/retrieval runtime

production query -> reranking -> candidate_audit -> evaluation   [sai dependency direction]
                           \----> evaluation                     [sai dependency direction]
```

Điểm tốt đã có sẵn:

- heavy models và provider SDK được lazy-import/lazy-initialize;
- `QueryService.execute` không phụ thuộc FastAPI;
- Streamlit dùng HTTP, không truy cập Qdrant/model/provider trực tiếp;
- Qdrant collection validation fail closed và không tự recreate collection sai schema;
- test suite dùng fakes/in-memory Qdrant, vô hiệu `.env` trong `tests/conftest.py`;
- provider output phải qua citation validation, không có silent retrieval/reranker fallback.

Vấn đề kiến trúc trung tâm là folder `app/` đồng thời chứa HTTP DTO, application service, domain
policy, Qdrant/FastEmbed/Docling/provider adapters và evaluation. Tên folder không cho biết dependency
direction; một số module còn import ngược từ production vào evaluation.

## 4. Runtime flows

### 4.1 Query API đang chạy

Compose đặt `RETRIEVAL_PROFILE=phase7`; local `Settings` mặc định `phase6`.

```text
POST /api/v1/query
  -> QueryRequest validation + optional bearer auth
  -> QueryService.execute
  -> generator.ensure_configured() trước retrieval
  -> LazyQueryRetriever
  -> dense@60 + query-expanded sparse@40
  -> weighted RRF k=40 + component reserves -> tối đa 30 candidates
  -> exact-content dedup trong cùng document trước rerank
  -> Jina cross-encoder, batch 8
  -> rank-only role prior + relation-list fallback đã đóng băng
  -> full post-rerank candidates
  -> exact cross-document evidence selection, rồi top_k
  -> EvidenceGate
  -> deterministic S1..Sn evidence bundle
  -> selected provider structured generation
  -> source-ID validation; tối đa một correction với cùng evidence
  -> trusted Citation metadata -> QueryResponse
```

`app.query_service.QueryExecution` cố ý giữ ba boundary khác nhau:

- `candidate_pool`: pre-rerank pool, Phase 7 tối đa 30;
- `candidates`: toàn bộ post-rerank ordered pool;
- `evidence_candidates`: post-selection/post-dedup `top_k` thực sự gửi provider.

`app.reranking.deduplicate_candidates_by_content` chỉ gộp exact normalized text trong **cùng
document**. `app.evidence_selection.select_evidence_candidates` xử lý exact duplicates giữa các
document, ưu tiên role suy ra chỉ từ query rồi mới cắt `top_k`. Hai policy này không được nhập làm
một nếu chưa có characterization tests riêng.

Phase 6 rollback vẫn là public operational behavior: `union + rerank` là đường chuẩn, còn
`sparse + rerank_disabled` là rollback explicit. Những tổ hợp khác fail; không silent fallback.

### 4.2 Ingestion và stable identity

`app.ingestion.ingest_document` nhận PDF/DOCX, validate path/range, lazy-load Docling, convert theo
page batches và normalize thành `DocumentChunk`. PDF có `hierarchical|hybrid` chunker; DOCX không
nhận page batching.

- `build_document_id`: NFKD filename stem → ASCII slug + 12 hex đầu của SHA-256 file bytes.
- `build_chunk_id`: canonical document/pages/headings/text + `occurrence_index`, SHA-256 lấy 16 hex;
  format `<document_id>_p<first-page>_h<digest>`.
- `write_chunks_jsonl`: UTF-8 và atomic replace; serialization lỗi không ghi đè output cũ.

`scripts.index_phase7_corpus` preview hoặc đọc frozen JSONL, kiểm tra hai manual, từ chối protected
Phase 3–6 collection names, index dense/hybrid, xác minh exact chunk-ID set và có thể chạy idempotent
second pass. Đây là mutating integration command, không phải runtime import.

### 4.3 Indexing và Qdrant

`app.retrieval.index_chunks` và `app.hybrid_retrieval.index_hybrid_chunks`:

- embed toàn bộ input trước mutation;
- validate hoặc tạo đúng collection schema khi CLI index explicit gọi;
- dùng UUID5 từ `chunk_id` qua `build_point_id`;
- upsert trước, chỉ xóa stale points của cùng document sau khi upsert thành công;
- ghi/validate dense và hybrid manifests ở lớp hiện tại.

`app.retrieval.dense_search` và `app.hybrid_retrieval.sparse_search` nhận question, limit và optional
`document_id`, trả candidates/payload đã validate. `fuse_rrf` dùng one-based ranks, công thức
`1 / (k + rank)`, stable tie-break theo component rank rồi `chunk_id`; không trộn raw cosine/BM25.

### 4.4 Readiness

`app.main.ready` resolve profile, tạo Qdrant client và gọi
`app.retrieval_runtime.validate_frozen_runtime`. Nó chỉ kiểm tra collection count/hash/schema; không
load embedding, reranker hoặc provider. Failure được sanitize thành HTTP 503
`retrieval_not_ready`.

### 4.5 Streamlit

`ui.streamlit_app.render` tạo UI, giữ tối đa 20 lượt trong session, render answer/citations và filter
document. `RAGAPIClient` gọi health/ready/query bằng `httpx`, validate response qua
`app.models.QueryResponse`, sanitize error và không retry POST. UI không import Qdrant, FastEmbed,
Jina, LangChain hoặc provider key.

### 4.6 Evaluation

`app.evaluation` cung cấp qrel schemas và retrieval metrics; `app.phase7` chứa dataset/fact schemas,
hashing và validation; `app.evaluation_e2e` score `QueryExecution`; `app.phase7_replay` replay
sanitized snapshots. Các CLI evaluation gọi public application/runtime interfaces và ghi sanitized
artifacts.

Hướng này đúng ở phía evaluator → application. Tuy nhiên `app.reranking` chứa cả
`evaluate_reranked_cases`, `classify_rerank_failure`, `aggregate_rerank_rows` và import trực tiếp
`app.evaluation`; `app.candidate_audit` cũng import `EvaluationCase`. Vì query runtime import
`app.reranking`, production import graph hiện kéo evaluation vào process dù không chạy evaluator.

## 5. Audit theo component

| Component và symbols | Responsibility, input → output | Dependencies và side effects | Tests hiện có | Debt và vị trí đích |
|---|---|---|---|---|
| `app/models.py`: `DocumentChunk`, `RetrievedChunk`, `RetrievalCandidate`, `QueryRequest`, `Citation`, `QueryResponse` | Gộp ingestion/domain/ranking/HTTP DTO | Pydantic; không I/O | ingestion, retrieval, reranking, query API, UI | `SPLIT`: domain records và API contracts không nên cùng module |
| `app/ingestion.py`: `ingest_document`, `build_document_id`, `build_chunk_id`, `write_chunks_jsonl` | File → stable chunks/JSONL | filesystem; lazy Docling; atomic write | `tests/test_ingestion.py` | Tách pure identity/policies khỏi Docling và file adapter; giữ thuật toán nguyên vẹn |
| `app/retrieval.py`: `index_chunks`, `dense_search`, manifests | Dense embedding, Qdrant schema/index/search và manifest | FastEmbed, Qdrant, filesystem; index mutation | `tests/test_retrieval.py` | `SPLIT` infrastructure client/index/search khỏi stable mapping/contract |
| `app/hybrid_retrieval.py`: sparse/index/search, `fuse_rrf`, manifests | BM25 sparse + hybrid collection + pure RRF | FastEmbed/Qdrant/filesystem; import private helpers từ `retrieval.py` | `tests/test_hybrid_retrieval.py` | `SPLIT`; RRF là policy, Qdrant/FastEmbed là adapters; private cross-module imports cần bỏ |
| `app/candidate_audit.py` | Candidate conversion/union và evaluation audit | import `app.evaluation` | `tests/test_candidate_audit.py` | `SPLIT/MOVE`: runtime candidate assembly ra domain/application; audit metrics ra evaluation |
| `app/phase7_optimization.py`, `app/query_expansion.py` | Frozen deterministic query-only ranking policies | pure Python + `RetrievalCandidate`; không I/O | Phase 7 optimization/expansion tests | `KEEP` behavior; `RENAME/MOVE` để bỏ tên phase lịch sử sau khi có golden tests |
| `app/reranking.py`: `RerankPipeline`, `FastEmbedCrossEncoder`, `execute_rerank`, evaluator helpers | Candidate pool, dedup, Jina adapter, ranking policy và evaluation trong 725 dòng | lazy FastEmbed; imports Qdrant retrieval và evaluation | `tests/test_reranking.py` | `SPLIT` runtime policy/port/adapter/evaluator; đây là coupling ưu tiên cao |
| `app/evidence_selection.py` | Exact cross-document selection và provenance | pure policy; query-role inference | `tests/test_evidence_selection.py` | `KEEP/MOVE` vào domain policy; không đổi semantics |
| `app/query_service.py`: `EvidenceGate`, `QueryService`, `get_query_service` | Orchestrate full query, abstention, retries, timings; đồng thời composition root | application + concrete retriever/generator/config; logs timings | `tests/test_query_service.py` | Giữ service, chuyển contracts/policies thích hợp; move factory sang composition root |
| `app/generation.py`: `AnswerGenerator`, `GeneratedAnswer`, `format_evidence`, `LangChainOpenAIGenerator` | Port, domain DTO, prompt formatting và provider adapter | lazy LangChain/OpenAI-compatible HTTP; provider call | `tests/test_generation.py` | `SPLIT`; class name gây hiểu nhầm vì cũng phục vụ Gemini |
| `app/citations.py`: `validate_generated_answer`, `build_citations` | Referential validation và trusted citation build | pure nhưng import DTO từ generation/models | `tests/test_citations.py` | `MOVE` vào domain/application policy; dependency vào concrete generation module cần đảo |
| `app/retrieval_runtime.py`: contracts, profiles, adapters, validation, builders | Frozen contracts, profile resolution, lazy runtime, concrete composition, Qdrant validation | config + retrieval/hybrid/reranking adapters | `tests/test_retrieval_runtime.py` | `SPLIT`: immutable contracts/policy, application port, infra construction/validation |
| `app/main.py`, `app/api/query.py`, `app/api/auth.py` | FastAPI app, routes, middleware, auth/error mapping, readiness | FastAPI; main tạo global settings/app và trực tiếp Qdrant composition | API/health tests | Query route khá mỏng; app factory/readiness còn concrete và khó composition-test |
| `ui/*` | HTTP client, config, presentation/session state | Streamlit/httpx; network từ server-side UI | Streamlit client/app tests | Đúng HTTP boundary; `UISettings` là nguồn env riêng và UI phụ thuộc broad `app.models` |
| `app/evaluation.py`, `app/evaluation_e2e.py`, `app/phase7.py`, `app/phase7_replay.py` | Dataset/qrels/metrics/scoring/replay | application public records; filesystem dataset reads | nhiều deterministic evaluator tests | `MOVE` ra top-level `evaluation/`; production không được import lại |
| `scripts/ingest_preview.py`, index/search/query CLIs | Inbound operational/manual adapters | filesystem, Qdrant, model/provider tùy lệnh | partial CLI parser/error tests | Giữ CLI mỏng; search CLIs hiện import evaluation chỉ để validate frozen chunks |
| Phase 7 calibration/readiness/migration scripts | Research/evaluation orchestration và sanitized artifacts | nhiều script import private functions từ script khác | nhiều script-level tests | Phân nhóm `evaluation`/`archive`; tránh biến `scripts` thành reusable library ngầm |
| `app/config.py`, `.env.example`, Compose, `ui/config.py`, frozen constants | Runtime/environment/profile/UI configuration | env/.env; cached settings | settings cases nằm rải trong retrieval/generation/UI tests | Ownership phân mảnh; defaults `phase6`/Compose `phase7` dễ hiểu sai |
| Docker/Compose/CI | Reproducible images, services, checks | build/network/volumes khi chạy | không có unit test; lịch sử có manual validation | `KEEP` trong Vòng 1, chỉ cập nhật import/copy paths tối thiểu |
| README/walkthroughs | Hướng dẫn và historical evidence | không runtime | không có link/doc test tổng quát | Stale claims, broken link, lịch sử lẫn current state; cần final polish |

## 6. Baseline và invariants phải bảo vệ

### 6.1 HTTP contract

- `POST /api/v1/query` nhận `question` stripped/non-empty, optional stripped/non-empty
  `document_id`, `top_k` mặc định 5 và range 1–10; extra fields bị từ chối.
- Response chỉ có `answer`, `abstained`, nullable `abstention_reason`, `citations`; không lộ scores
  hoặc diagnostics.
- Citation chỉ lấy từ trusted candidate: `chunk_id`, `document_id`, `filename`, positive/sorted pages,
  headings và Unicode-safe excerpt.
- Abstention hợp lệ trả HTTP 200, citations rỗng; lỗi dependency map 503/504; unexpected map sanitized
  500; validation lỗi 422.
- JSON success có `charset=utf-8`; mọi response có safe `X-Request-ID`.
- Optional bearer auth dùng constant-time comparison và giữ nguyên 401/503 contracts.

### 6.2 Identity và storage contract

- `build_document_id`, `build_chunk_id`, duplicate occurrence và UUID5 point-ID không đổi.
- `DocumentChunk` payload fields, Qdrant vector names, dimensions, distance, sparse IDF schema và
  document filter không đổi.
- Index phải fail trước stale deletion nếu embedding/upsert thất bại; chỉ xóa stale points của đúng
  document sau successful upsert.
- Dense/hybrid manifests, corpus identity, point count và sorted chunk-ID hash không đổi.
- Không create/delete/mutate/re-index collection trong refactor.

### 6.3 Frozen retrieval contracts

Phase 6 legacy/development:

```text
document_id:        manual-77d5dae4c2c5
chunks:             99
chunk-ID hash:      bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be
direct qrels:       30
dense/hybrid:       industrial_manual_chunks / industrial_manual_chunks_v2
dense/sparse/RRF:   20 / 20 / k=60
```

Phase 7 active Compose/demo:

```text
documents:          2 ATV320 manuals
chunks:             2753
chunk-ID hash:      2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d
dense collection:   industrial_manual_phase7_dense_v1
hybrid collection:  industrial_manual_phase7_hybrid_v1
dense/sparse/RRF:   60 / 40 / k=40
rerank budget/batch:30 / 8
active profile:     weighted RRF, reserves, query-only role prior offset 40,
                    relation-list fallback; exact-content selection
```

Model IDs, MiniLM dimension 384, BM25 parameters/average lengths, query expansion glossary,
candidate budgets, fusion weights, reranker và provider contracts đều frozen trong Vòng 1.

### 6.4 Ranking/evidence/generation boundaries

- Dense/sparse component ranks one-based; RRF deterministic tie-break không đổi.
- `candidate_pool`, `candidates`, `evidence_candidates` giữ đúng nghĩa và thứ tự.
- Same-document dedup trước rerank và cross-document dedup trước `top_k` không được nhập hoặc đổi thứ
  tự.
- Document filter phải đi vào cả dense và sparse Qdrant query.
- Evidence gate chạy trước format/provider; threshold mặc định `None`, score chỉ là rank signal.
- Provider configuration được kiểm tra trước retrieval.
- Prompt/evidence labels deterministic, max context 24.000 chars, raw chunk nằm trong untrusted block.
- Generation tối đa hai attempts; correction dùng cùng question/evidence, không retrieve/rerank lại.
- Model abstention/refusal/citation failure trả safe abstention; unvalidated citations không bao giờ
  ra public response.
- Không log key, full question, prompt, answer hoặc evidence; không silent fallback.

### 6.5 UI và evaluation boundary

- Streamlit chỉ giao tiếp qua FastAPI HTTP, không gọi application object trực tiếp và không retry
  POST.
- Evaluation được phép dùng public application/domain interfaces; production không được import
  evaluation.
- Held-out v2 chỉ là exposed regression benchmark; không tune runtime từ kết quả đó.

## 7. Benchmark metadata đã kiểm tra

Chỉ metadata/path/count được kiểm tra, không đọc raw held-out rows:

| Tracked asset | Metadata quan sát | Vai trò |
|---|---:|---|
| `artifacts/phase7/frozen-chunks.jsonl` | 2.753 JSONL rows | frozen Phase 7 corpus; runtime không đọc file này |
| `data/eval/dense_smoke.jsonl` | 30 rows | Phase 3–6 direct-evidence development qrels |
| `data/eval/phase7/calibration.jsonl` | 20 rows | historical calibration |
| `data/eval/phase7/calibration-v3.jsonl` | 20 rows | active approved calibration contract |
| `data/eval/phase7/calibration-v3-draft.jsonl` | 20 rows | historical/review draft |
| `data/eval/phase7/test.jsonl` | 45 rows | exposed historical split; raw rows không đọc |
| `data/sources.yaml` | two ATV320 source records | provenance/roles, local PDFs ignored |

Metrics manifests/checkpoints phần lớn nằm trong ignored `artifacts/metrics/`; private held-out v2
nằm dưới ignored `data/eval/phase7/private-heldout-v2/`. API image không dùng chúng. Canonical
dataset hashes trong Pydantic serialization không được nhầm với raw file SHA-256.

## 8. Test coverage hiện tại

Test suite được thiết kế offline: `tests/conftest.py` đặt `Settings.model_config["env_file"] = None`,
xóa các env keys tương ứng và dùng fakes/in-memory Qdrant. Coverage quan trọng đã có:

| Behavior | Tests tiêu biểu |
|---|---|
| Stable document/chunk IDs, batching, Docling failure, atomic JSONL | `tests/test_ingestion.py` |
| Dense point IDs/schema/index safety/search/filter/manifests | `tests/test_retrieval.py` |
| BM25 average length, hybrid schema, sparse search, RRF/manifests | `tests/test_hybrid_retrieval.py` |
| Candidate pool/dedup/Jina ordering/ties/fail closed/Phase 7 budget | `tests/test_reranking.py`, `tests/test_phase7_optimization.py` |
| Frozen contracts, profile resolution, lazy models, no fallback | `tests/test_retrieval_runtime.py` |
| Evidence duplicate policy | `tests/test_evidence_selection.py` |
| Prompt boundary, context limit, provider config/laziness/errors | `tests/test_generation.py` |
| Referential citation validation/build | `tests/test_citations.py` |
| Full orchestration, boundaries, gate, retry, safe logs | `tests/test_query_service.py` |
| API schema/auth/error/UTF-8/request ID/readiness | `tests/test_query_api.py`, `tests/test_health.py` |
| Streamlit HTTP/no retry/config/session state | `tests/test_streamlit_api_client.py`, `tests/test_streamlit_app.py` |
| Dataset/evaluator/fact/replay/governance | Phase 7/evaluation test group |

Không chạy test suite trong lượt audit. `python` trên PATH là Windows Store alias không chạy được;
`.venv/Scripts/python.exe` là Python 3.13.5, không phải target Python 3.11. Vì đây chỉ là docs audit,
không dùng interpreter khác để tạo một claim sai. README ghi historical validation `308 passed`,
nhưng đó không phải kết quả của lượt này.

### Characterization gaps phải đóng trước refactor

1. Một golden `QueryExecution` contract cho active Phase 7 qua fake ports, khóa stage order và cả ba
   candidate boundaries trong một test duy nhất.
2. Import-architecture test chứng minh production entry points không import `app.evaluation*` hoặc
   package `evaluation` đích.
3. Composition tests cho `get_query_service`/composition root và app factory: profile resolution,
   singleton/laziness, readiness không load model/provider.
4. Config precedence/default tests: Python default Phase 6, Compose-selected Phase 7, `.env` ignored
   trong pytest, UI-only env ownership.
5. Characterization của public schemas/status/error headers ở mức snapshot/schema, trước khi split
   `app.models`.
6. CLI contract tests cho các lệnh sẽ giữ: parser/default/exit code/sanitized output và “không
   mutation trước validation”. Không khóa textual output của script định archive nếu không là public
   contract.
7. Import/lazy tests cho Docling/FastEmbed/LangChain adapters sau khi di chuyển.
8. Docker/Compose static contract: API/UI copy paths, profile env và named volumes không đổi.

Không biến bug đã xác nhận thành contract. Nếu characterization phát hiện mâu thuẫn giữa code và
public docs/schema, dừng module và tách bug fix riêng.

## 9. Technical debt có dẫn chứng

### TD-01 — Production import evaluation

`app/reranking.py` import metrics/schemas từ `app.evaluation`; `app/candidate_audit.py` import
`EvaluationCase`; `app.retrieval_runtime.py` import `RerankPipeline`; query runtime vì thế kéo
evaluation transitively. Đây là vi phạm trực tiếp mục tiêu production ✕→ evaluation.

### TD-02 — Module trộn nhiều loại responsibility

`app/reranking.py` trộn port, FastEmbed adapter, candidate assembly, runtime policy và evaluator;
`app/retrieval.py`/`app/hybrid_retrieval.py` trộn pure mapping, SDK client, indexing, search và
manifest; `app/generation.py` trộn protocol/DTO/prompt/provider adapter; `app/models.py` trộn domain
với HTTP DTO. Tên file không biểu diễn layer và làm thay đổi nhỏ có blast radius lớn.

### TD-03 — Configuration và composition ownership phân mảnh

`app.config.Settings` là env source chính nhưng `ui.config.UISettings` parse env độc lập;
`app.retrieval_runtime` chứa immutable profiles; `get_query_service` vừa là application module vừa
composition root; `app.main.ready` tự compose Qdrant. `.env.example` và Python default là Phase 6,
Compose là Phase 7. Behavior này hợp lệ nhưng khó nhìn và chưa có một tài liệu canonical.

### TD-04 — Evaluation/script sprawl tạo package ngầm

39 Python files và một shell file trong `scripts/` trộn operational CLI, evaluator, calibrator,
freezer, migration,
diagnostic và readiness generators. Nhiều script import private helpers từ script khác, ví dụ
`evaluate_phase7_heldout_v2` import `_phase7_settings`/`_source_identity` từ
`evaluate_phase7_e2e`, còn calibration/readiness import `_per_language`/aggregate helpers từ script
khác. Reusable logic vì thế không có stable package contract.

### TD-05 — Inbound adapters chưa hoàn toàn mỏng

`app.api.query` chủ yếu chỉ map HTTP và là điểm tốt. Ngược lại `app.main` giữ global settings/app,
readiness biết Qdrant/frozen runtime; `get_query_service` biết concrete `LazyQueryRetriever` và
LangChain adapter. Search/index CLIs cũng assemble nhiều concrete dependencies. Điều này làm test
composition và thay adapter khó hơn cần thiết.

### TD-06 — Phase names che khuất product responsibilities

`phase7_optimization`, `PHASE7_CALIBRATION_FUSION_PROFILE`, `phase7.py` và nhiều script names kể lịch
sử hơn là mô tả responsibility. Không được rename vội vì artifacts/tests import các symbols này;
rename phải có compatibility shim và removal slice.

### TD-07 — Documentation không còn một current source of truth

- README liên kết `docs/project-deep-dive.md`, nhưng file không tồn tại.
- `docs/CODEBASE.md` và `docs/walkthrough-phase-7-5.md` ghi post-rerank offset 20; source frozen
  `PHASE7_CALIBRATION_FUSION_PROFILE` là offset 40.
- `docs/walkthrough-phase-7-corpus.md` còn ghi held-out chưa chạy/unseen, trong khi README và closure
  docs ghi held-out v2 đã đo và hiện chỉ là regression benchmark.
- `docs/plan.md` và `docs/PROJECT_JOURNEY.md` trộn roadmap lịch sử với status hiện tại.

Các khác biệt trên là documentation debt; không được dùng để thay source contract trong refactor.

### TD-08 — `AGENTS.md` đã đúng hướng nhưng cần một follow-up riêng

Working-tree `AGENTS.md` đã chuyển từ chỉ dẫn ngắn sang incremental refactor, nên không còn yêu cầu
rebuild repository. Những điểm nên sửa sau khi người dùng duyệt, không sửa trong lượt này:

- phân biệt rõ frozen Phase 6 legacy contract với active Phase 7 Compose/demo contract;
- nói rõ held-out v2 private đã executed/exposed chỉ là regression observation;
- sửa closing fence bốn backticks sau validation commands;
- bổ sung link tới bộ planning này khi nó được duyệt.

## 10. Bảng quyết định Vòng 1

`DROP` chỉ được dùng khi có bằng chứng không còn import, CLI usage, docs/CI invocation hoặc giá trị
reproducibility. Audit hiện chưa chứng minh an toàn cho file nào, nên không đề xuất drop.

| Quyết định | File/symbol hoặc nhóm | Lý do | Rủi ro | Test cần trước thay đổi |
|---|---|---|---|---|
| KEEP | Frozen chunks/qrels/contracts, `PHASE6_RETRIEVAL_CONTRACT`, `PHASE7_RETRIEVAL_CONTRACT` | Baseline identity và runtime behavior | Mất collection compatibility | hash/count/profile contract tests |
| KEEP | `build_document_id`, `build_chunk_id`, `build_point_id` algorithms | Stable persisted identity | tạo orphan/duplicate points | golden identity vectors |
| KEEP | RRF/Phase 7 policy outputs, candidate budgets/models/thresholds | Vòng 1 không tune | ranking drift | deterministic ranking goldens |
| KEEP | `EvidenceGate`, citation validation và no-fallback semantics | Safety/public behavior | provider call sớm hoặc answer không grounded | service/gate/citation tests |
| KEEP | Streamlit HTTP-only architecture và named Docker volumes | Đúng target boundary/safety | UI bypass API hoặc mất data | client/no-direct-import/static Compose tests |
| REFACTOR | `app/config.py`, profile resolution, UI config documentation | Một ownership rõ hơn | đổi env precedence/default | config matrix tests |
| REFACTOR | `QueryService` internals | Là application service đúng hướng nhưng imports concrete modules | stage-order drift | golden `QueryExecution` test |
| REFACTOR | API app construction/readiness | Tạo composition seams, route mỏng | status/header/lazy drift | app factory/API/readiness tests |
| MOVE | Pure ID/content/ranking/citation policies | Domain không phụ thuộc SDK/framework | circular imports/shim kéo dài | unit goldens + import test |
| MOVE | Docling/FastEmbed/Qdrant/LangChain implementations | Infrastructure adapters | eager model/network import | lazy construction and fake adapter contracts |
| MOVE | `app.evaluation*`, evaluator part của `app.phase7`, `app.phase7_replay` | Evaluation chỉ phụ thuộc public interfaces | script/test import break | evaluator suite + production import guard |
| MOVE | Stable operational/evaluation CLI vào `scripts/operations`/`scripts/evaluation` | Làm rõ execution class | module invocation path break | CLI parser/exit contract tests + temporary shims nếu cần |
| SPLIT | `app/models.py` | HTTP DTO, chunk và candidate khác ownership | public import break | schema/model serialization snapshots |
| SPLIT | `app/retrieval.py`, `app/hybrid_retrieval.py` | Pure policy, SDK, search/index, manifests đang trộn | Qdrant payload/index drift | existing retrieval/hybrid suite + adapter contracts |
| SPLIT | `app/reranking.py`, `app/candidate_audit.py` | Runtime/evaluation coupling trực tiếp | ranking/evaluation drift | rerank goldens + import guard |
| SPLIT | `app/generation.py` | Port/DTO/formatter/provider adapter | prompt/provider kwargs drift | generation suite/hash characterization |
| SPLIT | `app/retrieval_runtime.py` | Contract, port, adapter, validation, factory trộn nhau | profile/collection validation drift | frozen runtime and composition tests |
| MERGE | Runtime candidate conversion/union từ `candidate_audit` với candidate assembly package | Một implementation dùng chung, không phụ thuộc evaluator | đổi rank/provenance | candidate audit + rerank pool goldens |
| MERGE | Reusable Phase 7 hash/artifact/aggregation helpers đang nằm private trong scripts | Script không còn import private script functions | artifact identity drift | script unit tests + sanitized artifact snapshots |
| RENAME | `LangChainOpenAIGenerator` → provider-neutral adapter name | Class phục vụ OpenAI và Gemini | public import break | provider routing tests + compatibility alias |
| RENAME | Runtime `phase7_*` policy modules/symbols sau khi tách evaluator | Tên responsibility rõ hơn lịch sử | artifact source hash/import break | profile mapping/golden/hash-impact stop check |
| ARCHIVE | `generate_phase7_annotation_draft.py`, `apply_phase7_answer_facts.py`, `migrate_phase7_dataset_v2.py`, `draft_phase7_calibration_fact_types.py`, old freeze/rescore/targeted diagnostic scripts sau xác nhận | One-off dataset construction/migration đã hoàn tất | mất reproducibility hoặc docs command cũ | reference search, import graph, CLI tests; giữ archive README |
| ARCHIVE | Historical readiness/calibration selection scripts đã bị supersede | Không thuộc production package | nhầm current vs historical evidence | xác nhận walkthrough/artifact lineage trước move |
| DROP | Không có candidate đã chứng minh | Không xóa chỉ vì tên cũ hoặc ít import | mất audit trail | cần bằng chứng độc lập và user approval |
| DEFER | Embedding/pooling/dimension/model/provider/Qdrant schema/index changes | Round 2 hoặc data migration | behavior/data change | không áp dụng Vòng 1 |
| DEFER | Retrieval/reranker tuning, threshold/latency/Docker optimization | Algorithm/performance change | benchmark overfit | không áp dụng Vòng 1 |
| DEFER | New benchmark/held-out tuning/deployment architecture | Ngoài portfolio cleanup | governance/scope expansion | không áp dụng Vòng 1 |

## 11. Kết luận audit

Hệ thống đã có các safety properties và deterministic tests đủ tốt để refactor theo lát nhỏ. Thứ tự
an toàn là: khóa baseline/import boundary → canonicalize composition/config → tách ingestion/indexing
→ tách retrieval/reranking và loại production-evaluation imports → tách generation/application → làm
mỏng inbound adapters → cuối cùng mới di chuyển evaluation/scripts và sửa current documentation.

Không nên bắt đầu bằng rename/move hàng loạt. Các module lớn phải dùng compatibility imports có thời
hạn, và mỗi slice phải chạy focused tests trước full Python 3.11 checks.

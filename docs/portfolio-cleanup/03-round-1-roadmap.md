# Roadmap Vòng 1 — Portfolio Cleanup

> **R00 prerequisite completed — 2026-08-22:** roadmap R01–R07 áp dụng cho active Phase 7-only
> baseline. Phase 6 collections được bảo toàn nhưng tools đã archive và profile bị production
> configuration từ chối. Không module nào được khôi phục compatibility branch này.

## 1. Nguyên tắc thực thi

Roadmap có bảy module theo dependency order. Mỗi lượt sau planning chỉ làm một module đã duyệt hoặc
một vertical slice của module đó, rồi dừng ở `WAITING_FOR_USER_REVIEW`. Không tự động chuyển module.

Một **vertical slice** là thay đổi end-to-end nhỏ, có test và compatibility boundary riêng. Nó giúp
module lớn không vượt khoảng tám file có nội dung đáng kể trong một lượt, nhưng mọi slice của cùng
module cập nhật chung một file `docs/modules/Rxx-*.md`.

Quy tắc áp dụng cho mọi module:

- characterization tests trước structural move có rủi ro;
- giữ public imports/CLI bằng shim có removal owner khi cần;
- focused tests trước, sau đó `python -m ruff check .` và `python -m pytest -q` bằng cùng Python 3.11;
- không chạy real Qdrant/model/provider/evaluation nếu không có approval riêng;
- không đổi algorithm, config value, data, schema, collection hoặc public behavior;
- module docs mô tả code đã tồn tại và có status `IN_PROGRESS`/`COMPLETE`;
- nếu phát hiện bug hoặc baseline/docs contradiction ảnh hưởng contract, dừng và tách quyết định.

## 2. Thứ tự implementation

```text
R01 Baseline & characterization
  -> R02 Configuration & composition root
    -> R03 Ingestion & indexing boundaries
      -> R04 Retrieval & reranking boundaries
        -> R05 Grounded query application
          -> R06 Inbound adapters
            -> R07 Evaluation, scripts & documentation closure
```

R01 khóa behavior để các module sau có safety net. R02 tạo composition seams. R03/R04 làm sạch các
outbound adapters từ data path lên query path. R05 hoàn thiện application layer. R06 chỉ làm mỏng
inbound adapters sau khi application interface ổn định. R07 chuyển evaluation/scripts/docs cuối cùng
để tránh liên tục sửa imports và walkthrough trong các module trước.

## 3. Module plans

### R01 — Baseline and characterization coverage

1. **Mã và tên:** `R01 — Baseline and characterization coverage`.
2. **Mục tiêu:** khóa bằng test các public contracts và stage boundaries quan trọng trước khi di
   chuyển code; thêm automated import rule làm acceptance test cho target architecture.
3. **Vấn đề hiện tại:** test chi tiết đã nhiều nhưng chưa có một golden Phase 7 `QueryExecution`,
   config/profile matrix tổng hợp hoặc import-boundary test. Historical `308 passed` không phải
   validation hiện tại.
4. **Files/symbols hiện tại:** `tests/conftest.py`, `tests/test_query_service.py`,
   `tests/test_query_api.py`, `tests/test_retrieval_runtime.py`, `QueryService.execute`,
   `QueryExecution`, `resolve_retrieval_runtime`, public Pydantic models.
5. **Đích sau module:** thêm tối đa hai test modules chuyên trách, ví dụ
   `tests/test_architecture_boundaries.py` và `tests/test_runtime_characterization.py`; chưa đổi package
   production.
6. **Thứ tự:** (a) snapshot schema/error/profile contracts; (b) golden fake-port query execution khóa
   pool/final/evidence sets và stage order; (c) import graph test ban đầu ghi rõ violation được phép có
   removal target R04/R07; (d) document baseline và gaps.
7. **Behavior giữ nguyên:** toàn bộ API, IDs, profile values, ranking/evidence/generation/abstention,
   lazy imports và log privacy.
8. **Characterization tests:** API JSON schema/status/header; Phase 7 atomic profile resolution và
   explicit rejection của retired Phase 6 profile;
   same-doc vs cross-doc dedup order; generator config-before-retrieval; one retry/same evidence;
   production import inventory.
9. **Checks:** focused new tests; existing query/API/runtime tests; full Ruff/pytest Python 3.11;
   `git diff --check`. Không gọi Qdrant/provider.
10. **Cố ý chưa làm:** không sửa production import violations, không move/rename, không sửa bug hoặc
    docs lịch sử.
11. **Rủi ro/stop:** test vô tình khóa bug; source/docs mâu thuẫn làm expected behavior không xác định;
    target Python 3.11 không sẵn có. Dừng thay vì dùng Python khác để claim completion.
12. **Acceptance criteria:** deterministic/offline; các invariants có tên rõ; test không dựa `.env`;
    baseline violation list có owner; full checks pass.
13. **Tài liệu:** `docs/modules/R01-baseline-characterization.md`.
14. **Commit đề xuất:** `test: capture portfolio cleanup baseline`.
15. **Dependency:** không phụ thuộc module cleanup; là prerequisite của R02–R07.
16. **Ước lượng:** **SMALL**; một slice, khoảng 4–6 meaningful files kể cả module doc.

### R02 — Canonical configuration and composition root

1. **Mã và tên:** `R02 — Canonical configuration and composition root`.
2. **Mục tiêu:** có một backend `Settings` ownership, một immutable profile resolver và một
   composition root; giữ `app.main:app` cùng lazy singleton behavior.
3. **Vấn đề hiện tại:** `get_query_service` nằm trong application module, readiness tự tạo Qdrant,
   và `retrieval_runtime` vẫn trộn Phase 7 contract/validation/builders. R00 đã thống nhất local,
   `.env.example` và Compose ở Phase 7; R02 làm rõ ownership chứ không đổi frozen values.
4. **Files/symbols hiện tại:** `app/config.py:Settings/get_settings`,
   `app/retrieval_runtime.py:FrozenRetrievalContract/resolve_retrieval_runtime`,
   `app/query_service.py:get_query_service`, `app/main.py:app/ready`, `app/api/query.py` dependency.
5. **Đích:** `app/config.py` canonical facade; `app/domain/retrieval_contracts.py` cho frozen profiles;
   `app/bootstrap.py` cho `build_query_service`/readiness composition; `app/api/app.py:create_app`;
   `app/main.py` chỉ export compatibility ASGI object.
6. **Thứ tự/slices:** R02A khóa config matrix rồi move immutable contracts/resolver; R02B move query
   factory/caching sang bootstrap; R02C tạo app factory/readiness dependency seam. Mỗi slice không quá
   khoảng tám meaningful files và cập nhật cùng module doc.
7. **Behavior giữ nguyên:** env names/validation/cache, Phase 7-only selection,
   collection/model/profile values, startup laziness, health/readiness/query routes.
8. **Characterization tests:** config precedence; exact resolved contract fields; cached service builds
   heavy dependencies lazily; readiness count/hash không load model/provider; app factory preserves
   middleware/router/status.
9. **Checks:** config/runtime/health/query tests; new composition tests; full Ruff/pytest; static
   `docker compose config --quiet` nếu command không lộ secret; `git diff --check`.
10. **Cố ý chưa làm:** không đổi provider/Qdrant endpoints/defaults; không split retrieval algorithms;
    UI process config vẫn ở `ui/config.py`; không thêm DI framework.
11. **Rủi ro/stop:** global cache identity đổi, circular imports, route path/header drift, profile fields
    bị env-mix. Dừng nếu app factory cần public API change.
12. **Acceptance criteria:** query/readiness được compose từ một edge; application service không tạo
    concrete adapters; profile resolution atomic; `uvicorn app.main:app` không đổi; no eager SDK work.
13. **Tài liệu:** `docs/modules/R02-configuration-composition.md`.
14. **Commits đề xuất:** `test: characterize runtime configuration selection`; sau đó
    `refactor: establish canonical runtime composition`. Module doc đi cùng từng slice update.
15. **Dependency:** cần R01; mở seam cho R03–R06; shim cleanup chốt ở R07.
16. **Ước lượng:** **MEDIUM**; 3 slices.

### R03 — Ingestion and indexing boundaries

1. **Mã và tên:** `R03 — Ingestion and indexing boundaries`.
2. **Mục tiêu:** tách stable document/chunk identity và indexing use cases khỏi Docling, FastEmbed,
   Qdrant và manifest I/O mà không thay output/index semantics.
3. **Vấn đề hiện tại:** `app/ingestion.py` trộn identity, batching, SDK conversion và JSONL;
   `app/retrieval.py`/`app/hybrid_retrieval.py` trộn pure mapping, schema, model, index/search và
   manifests; hybrid import private dense helpers.
4. **Files/symbols hiện tại:** `app/models.py:DocumentChunk`, `app/ingestion.py`,
   `app/retrieval.py:index_chunks/build_point_id`, `app/hybrid_retrieval.py:index_hybrid_chunks`,
   `scripts/ingest_preview.py`, `scripts/index_phase7_corpus.py`; legacy indexing CLIs hiện ở
   `scripts/archive/phase6/` và không thuộc supported target.
5. **Đích:** `app/domain/documents.py`; `app/infrastructure/ingestion/docling.py`;
   `app/infrastructure/qdrant/{client,dense,hybrid,manifests}.py`; optional thin
   `app/application/ingestion_service.py`/`indexing_service.py`; old modules là facade tạm.
6. **Thứ tự/slices:** R03A identity/chunk records + Docling adapter; R03B dense embedding/Qdrant
   indexing/manifests; R03C sparse/hybrid indexing/manifests và bỏ private cross-module imports;
   R03D chuyển supported ingestion/index CLI composition nếu cần. Không chạy index.
7. **Behavior giữ nguyên:** supported file/range rules, page batching, chunk order/metadata/IDs,
   atomic JSONL, embedding text, UUID5, payload schema, vector schema, embed-before-mutate,
   upsert-before-stale-delete, manifests và protected collection guard.
8. **Characterization tests:** golden ID vectors including duplicate occurrence/Unicode; Docling fake
   status normalization; exact payload/filter snapshots; dense/hybrid failure ordering; manifest
   roundtrip/mismatch; Phase 7 CLI validation occurs before mutation.
9. **Checks:** ingestion/retrieval/hybrid tests, relevant CLI tests, full Ruff/pytest, import laziness,
   `git diff --check`. Không build ingestion image, download model, connect/write Qdrant hoặc re-index.
10. **Cố ý chưa làm:** không đổi chunker/profile/batch size/model/pooling/dimension/BM25/Qdrant schema;
    không sửa long-chunk/table limitations.
11. **Rủi ro/stop:** stable ID/payload drift, order thay đổi, hidden SDK type leak vào domain, cần
    Qdrant migration hoặc hơn tám files trong một slice. Bất kỳ vấn đề nào đều dừng.
12. **Acceptance criteria:** domain imports không chứa Docling/FastEmbed/Qdrant; old public behavior
    qua facade còn nguyên; all indexing safety tests pass; no collection operation executed.
13. **Tài liệu:** `docs/modules/R03-ingestion-indexing-boundaries.md`.
14. **Commits đề xuất:** `refactor: separate document ingestion from Docling`; `refactor: isolate dense
    indexing infrastructure`; `refactor: isolate hybrid indexing infrastructure`; optional
    `refactor: thin ingestion and indexing commands`.
15. **Dependency:** dùng config/composition conventions R02; tạo Qdrant/model adapters cho R04.
16. **Ước lượng:** **LARGE**; 3–4 slices.

### R04 — Retrieval and reranking boundaries

1. **Mã và tên:** `R04 — Retrieval and reranking boundaries`.
2. **Mục tiêu:** tách pure retrieval/ranking policy, runtime candidate assembly, Qdrant search và Jina
   adapter; chấm dứt production import evaluation.
3. **Vấn đề hiện tại:** `app/reranking.py` 725 dòng chứa runtime lẫn evaluator; `candidate_audit`
   chứa helpers runtime lẫn qrel audit; `retrieval_runtime` biết concrete adapters; phase-named policy
   khó giải thích; production query transitively import `app.evaluation`.
4. **Files/symbols hiện tại:** `app/candidate_audit.py`, `app/reranking.py:RerankPipeline/execute_rerank`,
   `app/retrieval_runtime.py`, `app/phase7_optimization.py`, `app/query_expansion.py`, dense/sparse
   search functions và evaluator functions cuối reranking module.
5. **Đích:** `app/domain/retrieval.py`, `app/domain/policies/{fusion,query_analysis}.py`, domain ports;
   `app/infrastructure/qdrant` search adapters; `app/infrastructure/models/reranker.py`; application
   retrieval adapter/orchestrator; evaluator helpers tạm sang `evaluation/` hoặc compatibility module.
6. **Thứ tự/slices:** R04A tách candidate conversion/union khỏi audit; R04B move pure RRF/query/role
   policies bằng re-export; R04C isolate dense/sparse search adapters; R04D isolate cross-encoder và
   RerankPipeline; R04E bật import guard cấm production → evaluation. Mỗi slice cập nhật module doc.
7. **Behavior giữ nguyên:** dense/sparse limits, query expansion, weighted RRF, reserves, max 30,
   same-document dedup, candidate text format, Jina model/batch/threads, score/rank/tie/provenance,
   role/list fallbacks, document filter và explicit sparse rollback/no fallback.
8. **Characterization tests:** fixed candidate goldens cho Phase 6 RRF và Phase 7 profile; exact
   candidate pool order/provenance; dedup groups; reranker ties/failures; lazy one-time model; profile
   validation/count/hash; import graph không evaluation.
9. **Checks:** candidate/retrieval/hybrid/reranking/runtime/optimization/expansion tests; query service
   golden; full Ruff/pytest; `git diff --check`. Không real model/Qdrant/benchmark.
10. **Cố ý chưa làm:** không tune ranking, đổi model/license, latency, profile names trong artifact,
    thresholds hoặc candidate budgets; không “simplify” two-stage dedup.
11. **Rủi ro/stop:** một stable tie hoặc rank field đổi; evaluator import cần public runtime symbol chưa
    có; source hash là benchmark identity và rename làm invalid artifact; slice lan sang tuning. Dừng
    để quyết định compatibility/ADR.
12. **Acceptance criteria:** production import guard pass; domain policy SDK-free; runtime/evaluator
    outputs deterministic bằng existing goldens; no silent fallback; old imports có owner/removal date.
13. **Tài liệu:** `docs/modules/R04-retrieval-reranking-boundaries.md`.
14. **Commits đề xuất:** `refactor: separate candidate assembly from evaluation`; `refactor: isolate
    retrieval policies and adapters`; `refactor: isolate the reranking adapter`; `test: enforce the
    production evaluation boundary`.
15. **Dependency:** cần R01–R03; gỡ coupling trước R05; evaluation file relocation hoàn tất ở R07.
16. **Ước lượng:** **LARGE**; 4–5 slices.

### R05 — Grounded generation, citations and query application

1. **Mã và tên:** `R05 — Grounded generation, citations and query application`.
2. **Mục tiêu:** làm rõ application use case và domain policies cho evidence/citations; đặt provider
   SDK sau port mà không đổi prompt, retry hoặc answer behavior.
3. **Vấn đề hiện tại:** `app/generation.py` trộn protocol, Pydantic output, prompt formatter và
   LangChain adapter; tên `LangChainOpenAIGenerator` sai nghĩa với Gemini; citations phụ thuộc concrete
   generation module; `QueryService` vẫn import nhiều modules cụ thể.
4. **Files/symbols hiện tại:** `app/generation.py`, `app/citations.py`,
   `app/evidence_selection.py`, `app/query_service.py:EvidenceGate/QueryService/QueryExecution`,
   `app/errors.py`, query/citation/candidate models.
5. **Đích:** `app/domain/ports.py:AnswerGenerator`; domain policies evidence/citations; provider-neutral
   generation DTO; `app/infrastructure/generation/langchain_structured.py`; clean
   `app/application/query_service.py`; compatibility facades ở old paths nếu cần.
6. **Thứ tự/slices:** R05A move generation DTO/port/formatter và preserve prompt bytes; R05B move
   concrete LangChain adapter, thêm provider-neutral name + alias; R05C move citation/evidence gate
   policies; R05D simplify QueryService imports bằng ports/contracts.
7. **Behavior giữ nguyên:** provider selection/kwargs/store=false/temperature/reasoning/timeouts,
   prompt and untrusted block, context/excerpt limits, ensure-configured-before-retrieval, gate,
   abstention messages/reasons, two attempts/same evidence, usage aggregation, citations và safe logs.
8. **Characterization tests:** prompt/system hash or exact structural snapshot; evidence labels and
   truncation; OpenAI/Gemini kwargs; failure mapping; correction evidence identity; source-ID
   validation/dedup/filter; all abstention paths; no provider construction at import.
9. **Checks:** generation/citations/evidence/query-service tests; API tests; query golden; full
   Ruff/pytest; `git diff --check`. Không provider call.
10. **Cố ý chưa làm:** không sửa prompt wording, model/provider, threshold, answer quality, streaming,
    memory, semantic citation scoring hoặc retry policy.
11. **Rủi ro/stop:** prompt byte drift ảnh hưởng benchmark identity, exception mapping khác, usage bị
    double count, citation metadata lấy từ model, “cleanup” đổi abstention. Dừng và tách bug/behavior
    decision nếu xảy ra.
12. **Acceptance criteria:** application chỉ phụ thuộc contracts/ports/policies; provider SDK chỉ ở
    infrastructure; API responses và QueryExecution giống baseline; no unvalidated citation.
13. **Tài liệu:** `docs/modules/R05-grounded-query-application.md`.
14. **Commits đề xuất:** `refactor: isolate structured generation infrastructure`; `refactor: clarify
    grounded query policies and service`.
15. **Dependency:** cần stable retrieval port từ R04 và composition R02; cung cấp interface ổn định cho
    R06 và evaluator R07.
16. **Ước lượng:** **MEDIUM** đến **LARGE**; 3–4 slices.

### R06 — Thin FastAPI, CLI and Streamlit adapters

1. **Mã và tên:** `R06 — Thin inbound adapters`.
2. **Mục tiêu:** inbound code chỉ parse/validate/map/call application; Streamlit tiếp tục HTTP-only;
   stable operational CLIs không chứa reusable domain/infrastructure logic.
3. **Vấn đề hiện tại:** query route mỏng nhưng main/readiness và CLI còn concrete composition; UI
   import broad `app.models`; CLI default/error contracts chưa được bảo vệ đồng đều.
4. **Files/symbols hiện tại:** `app/main.py`, `app/api/query.py`, `app/api/auth.py`, operational scripts,
   `ui/api_client.py`, `ui/config.py`, `ui/state.py`, `ui/streamlit_app.py`, `app.models` HTTP DTOs.
5. **Đích:** `app/api/app.py`, `dependencies.py`, route modules; `app/contracts/query.py`; supported
   `scripts/operations/*`; UI chỉ import shared query contracts và giữ riêng URL/timeout/token config.
6. **Thứ tự/slices:** R06A hoàn thiện API routes/dependencies quanh R02 app factory; R06B chuyển từng
   nhóm operational CLI với old `python -m scripts.<name>` shims; R06C split public HTTP schemas và
   chuyển UI imports; R06D static Docker copy/entry-point updates nếu path đổi.
7. **Behavior giữ nguyên:** paths/schemas/status/UTF-8/request-ID/auth; CLI args/defaults/exit codes và
   mutation guards; UI document IDs/options, history limit, error sanitization, timeout, no POST retry,
   API-only communication.
8. **Characterization tests:** OpenAPI/Pydantic schema; all API error mappings/headers; app dependency
   injection with fakes; selected CLI parser/output/error tests; UI MockTransport/no-retry; forbidden
   UI imports; Docker entry points/copy paths.
9. **Checks:** API/health/Streamlit/CLI tests; full Ruff/pytest; `docker compose config --quiet`;
   optional image smoke chỉ với approval, không build mặc định; `git diff --check`.
10. **Cố ý chưa làm:** không đổi UI design/features, auth architecture, rate limiting, TLS, deployment,
    upload/re-index workflow hoặc API version.
11. **Rủi ro/stop:** CLI module paths là external contract chưa rõ; moving schemas kéo FastAPI into UI
    image; app factory làm middleware order đổi; Docker cần large rebuild. Dừng và xin quyết định nếu
    không thể giữ shim/static validation.
12. **Acceptance criteria:** route/CLI files nhỏ và không chứa algorithm; UI import graph chỉ
    contracts/httpx/Streamlit; existing entry points vẫn chạy về mặt import/parser; API tests pass.
13. **Tài liệu:** `docs/modules/R06-inbound-adapters.md`.
14. **Commits đề xuất:** `refactor: thin FastAPI adapters`; `refactor: thin operational commands`;
    `refactor: share stable query contracts with Streamlit`.
15. **Dependency:** cần R02 composition và R05 public application contracts; script archive chốt R07.
16. **Ước lượng:** **MEDIUM** đến **LARGE**; 3–4 slices.

### R07 — Evaluation isolation, script archive and documentation closure

1. **Mã và tên:** `R07 — Evaluation isolation, script archive and documentation closure`.
2. **Mục tiêu:** hoàn tất one-way evaluation boundary, phân loại scripts, loại compatibility shims đã
   hết hạn và làm README/current docs phản ánh runtime thực tế.
3. **Vấn đề hiện tại:** evaluator/dataset modules nằm trong `app`; scripts import private script
   helpers; historical one-offs lẫn operational commands; README có broken/stale links và
   walkthroughs mâu thuẫn active profile/held-out status.
4. **Files/symbols hiện tại:** `app/evaluation.py`, `app/evaluation_e2e.py`, evaluator part
   `app/phase7.py`, `app/phase7_replay.py`, Phase 7 evaluation/calibration/readiness/migration scripts,
   README, `docs/CODEBASE.md`, `docs/PROJECT_JOURNEY.md`, `docs/plan.md`, walkthroughs và user-modified
   `AGENTS.md`.
5. **Đích:** top-level `evaluation/*`; `scripts/evaluation/*`; `scripts/archive/phase7/*` kèm archive
   index; current README/architecture docs; historical walkthroughs giữ nhưng gắn nhãn/supersession;
   old import/CLI shims chỉ giữ khi public contract yêu cầu.
6. **Thứ tự/slices:** R07A move schemas/metrics/scoring/replay và shared artifact helpers; R07B chuyển
   current evaluator CLIs; R07C audit references rồi archive confirmed one-offs; R07D README/docs link
   and status polish; R07E remove shims đã có bằng chứng. `AGENTS.md` chỉ sửa sau explicit approval vì
   đang có overlapping user changes.
7. **Behavior giữ nguyên:** dataset validation/hashes, qrel/fact metrics, quality gates, approval
   tokens/governance blocks, checkpoint/run identity, sanitized artifacts, current evaluation CLI
   behavior và production runtime behavior.
8. **Characterization tests:** dataset canonical hashes/schemas; metric goldens; calibration never
   opens held-out; approval/path guards; checkpoint identity; sanitization forbidden-content tests;
   production import guard; CLI compatibility/reference inventory; Markdown links.
9. **Checks:** toàn bộ evaluation/Phase 7/script tests; production import guard; full Ruff/pytest;
   Markdown link checker; `git diff --check`; scope/reference checks. Không chạy calibration/held-out,
   provider, model hoặc Qdrant.
10. **Cố ý chưa làm:** không thay evaluator semantics/thresholds/data, không tune theo held-out, không
    tạo benchmark v3, không xóa historical evidence, không rewrite Git history.
11. **Rủi ro/stop:** archive candidate còn docs/CI/import consumer; source file hash nằm trong run
    identity; raw held-out cần mở để move; user changes trong `AGENTS.md` overlap; docs và source
    contract chưa thống nhất. Dừng và xin approval/decision.
12. **Acceptance criteria:** production graph không evaluation; evaluator dùng public interfaces;
    scripts có taxonomy/index và không private cross-script imports; không `DROP` nếu chưa chứng minh;
    README current flow đúng source; links pass; module doc `COMPLETE`.
13. **Tài liệu:** `docs/modules/R07-evaluation-scripts-documentation.md`.
14. **Commits đề xuất:** `refactor: move evaluation behind application interfaces`; `chore: archive
    superseded phase 7 utilities`; `docs: align runtime architecture and runbooks`. Không gộp bug fix
    hoặc AGENTS overlap vào các commit này.
15. **Dependency:** module cuối, cần interfaces/shims từ R01–R06; có thể tách R07D docs sau code slices
    nhưng không declare module complete trước đó.
16. **Ước lượng:** **LARGE**; 4–5 slices.

## 4. Characterization test matrix theo thời điểm

| Trước module | Test phải có |
|---|---|
| R02 | config/profile matrix, public API/app construction, lazy singleton/readiness |
| R03 | stable IDs, payload/manifests, mutation ordering, lazy Docling/FastEmbed |
| R04 | deterministic candidate/rank/provenance goldens, dedup stages, no fallback, import graph |
| R05 | prompt/evidence structure, provider kwargs, gate/retry/citations/abstentions/log privacy |
| R06 | HTTP schemas/status/headers, CLI args/exit, UI no-retry/HTTP-only, Docker entry points |
| R07 | evaluator hashes/metrics/governance/sanitization, script references, Markdown links |

Mỗi characterization test phải mô tả behavior có chủ đích. Nếu test chỉ ghi lại một kết quả đáng ngờ
không có public contract hoặc safety rationale, cần xác nhận trước khi coi nó là invariant.

## 5. Acceptance criteria toàn Vòng 1

- dependency direction và automated import rules pass;
- production runtime không import evaluation/scripts/data artifacts;
- backend config/profile/composition ownership rõ và defaults không đổi;
- domain/application không import FastAPI, Streamlit, Qdrant, Docling, FastEmbed, LangChain/provider
  SDK hoặc evaluation;
- infrastructure implement ports, không quyết định use-case policy;
- API/CLI/UI mỏng; Streamlit vẫn HTTP-only;
- frozen Phase 7 identity, algorithms, collections, manifests và public query behavior không đổi;
- legacy Phase 6 collections/archive evidence vẫn được bảo toàn nhưng không trở lại production;
- characterization + existing tests pass bằng cùng Python 3.11 interpreter;
- không có real provider/model download/Qdrant mutation/re-index/held-out run;
- scripts được phân loại, historical utilities archive thay vì xóa thiếu bằng chứng;
- bảy module docs phản ánh code thật và status `COMPLETE`;
- README/current architecture đúng, historical metrics có provenance và không bị claim lại;
- Git commits tập trung, không chứa unrelated user changes.

## 6. Git history plan

Git history đề xuất tuyến tính theo module/slice, không squash giả tạo và không rewrite lịch sử cũ:

```text
test: capture portfolio cleanup baseline
test: characterize runtime configuration selection
refactor: establish canonical runtime composition
refactor: separate document ingestion from Docling
refactor: isolate dense indexing infrastructure
refactor: isolate hybrid indexing infrastructure
refactor: separate candidate assembly from evaluation
refactor: isolate retrieval policies and adapters
refactor: isolate the reranking adapter
test: enforce the production evaluation boundary
refactor: isolate structured generation infrastructure
refactor: clarify grounded query policies and service
refactor: thin FastAPI adapters
refactor: thin operational commands
refactor: share stable query contracts with Streamlit
refactor: move evaluation behind application interfaces
chore: archive superseded phase 7 utilities
docs: align runtime architecture and runbooks
```

Đây là proposal, không phải số commit bắt buộc. Một slice chỉ đáng thành commit khi:

- thay đổi đúng một responsibility và diff review được trong vài phút;
- focused/full checks tương ứng đã thực sự chạy;
- module doc được cập nhật trong cùng commit;
- compatibility shim/removal target được ghi rõ;
- `git status --short` và `git diff --stat` không lẫn thay đổi người dùng;
- commit message mô tả outcome, không mô tả Phase mơ hồ.

Bug fix phát hiện trong quá trình refactor phải dừng, được người dùng duyệt và dùng commit `fix:` riêng.
Algorithm improvement/dependency upgrade không nằm trong Git history Vòng 1. Không commit `.env`, raw
PDFs, caches, private/raw benchmark payload hoặc generated metrics.

## 7. Risks và stop conditions

| Risk | Cách kiểm soát | Stop condition |
|---|---|---|
| Stable ID/ranking drift | golden fixtures và exact ordering assertions | bất kỳ ID/rank/profile output đổi |
| Public import/CLI break | facade/shim + `rg` consumers + CLI tests | không xác định được external contract |
| Circular dependency khi move | move ports/records trước adapters | cần import ngược layer để tiếp tục |
| Eager model/network construction | import/lazy tests | import module tạo model/client/network |
| Data/Qdrant mutation | chỉ offline tests/fakes | module cần re-index/schema/collection write |
| Prompt/provider behavior drift | prompt/provider kwargs characterization | source/prompt identity hoặc answer behavior đổi |
| Evaluation contamination | import guard/governance tests | cần raw held-out hoặc runtime rule từ qrel/fact |
| Scope quá tám files | chia vertical slice | slice vẫn chạm quá nhiều responsibilities |
| Dirty worktree overlap | status/diff trước mỗi module | file người dùng thay đổi trùng target, đặc biệt `AGENTS.md` |
| Docs-source contradiction | source + tests là evidence, hỏi user khi public contract không rõ | không thể xác định behavior được phê duyệt |
| Python/tool mismatch | dùng cùng Python 3.11 | chỉ có interpreter khác hoặc dependency download cần thiết |

## 8. Deferred to Round 2

Danh sách này là backlog, không phải implementation của Vòng 1:

- thay embedding model, pooling, dimension hoặc FastEmbed compatibility behavior;
- thay sparse model/BM25 processing, dense/sparse limits, RRF k/weights/reserves;
- tune query expansion, query role, list/relation fallback hoặc reranker order;
- thay Jina model, batch/thread strategy, quantization, CPU/GPU hoặc license resolution;
- evidence threshold calibration, score normalization hoặc semantic support scoring;
- đổi prompt/provider/model/retry/temperature có chủ đích;
- latency/performance optimization, caching redesign và load testing;
- Qdrant schema/collection redesign, migration, re-index hoặc volume changes;
- chunker/OCR/table/multi-page continuity improvement;
- Docker image size/build optimization hoặc dependency upgrade;
- tune từ exposed held-out v2, tạo benchmark v3 hoặc dùng UI queries làm tuning set;
- new deployment, auth/rate-limit/TLS/streaming/memory/multi-user architecture;
- microservices, queues hoặc DI framework.

Nếu một mục trên được phát hiện trong audit/refactor, chỉ ghi limitation/backlog và tiếp tục module nếu
không ảnh hưởng behavior. Nếu nó là điều kiện bắt buộc để hoàn tất module, dừng và xin mở scope.

## 9. Quyết định cần duyệt trước implementation

1. Giữ top-level `app` và áp dụng package tree trong target architecture.
2. Chấp nhận thứ tự bảy module R01 → R07 và cách chia slices.
3. Giữ Phase 7 là active local/Compose contract; bảo toàn Phase 6 collections/archive evidence.
4. Dùng compatibility facade có removal target thay cho move hàng loạt.
5. Archive, không drop, nhóm Phase 7 one-off scripts sau reference audit ở R07.
6. `AGENTS.md` đã được R00 cập nhật cho incremental refactoring và Phase 7-only baseline.
7. Bắt đầu bằng R01 chỉ khi người dùng gửi yêu cầu implementation riêng.

# Target architecture cho Vòng 1

> **R00 constraint update — 2026-08-22:** target architecture chỉ có một active Phase 7 runtime
> contract. Phase 6 identity thuộc historical archive và collection-protection guards, không phải
> profile option. R02 có thể di chuyển Phase 7 contract ownership nhưng không được khôi phục selector
> hoặc production dependency vào archive.

## 1. Nguyên tắc

Target là **modular monolith**: một codebase và một API process, nhưng boundaries được thể hiện bằng
package/import direction. Không thêm microservice, queue, service mesh hay dependency-injection
framework.

```text
FastAPI / operational CLI / Streamlit HTTP client
                       |
                       v
              Application services
                       |
                       v
          Domain contracts, ports, policies
                       ^
                       |
            Infrastructure adapters

Evaluation ----------> public application/domain interfaces
Production runtime -X-> evaluation
```

Composition root là ngoại lệ có chủ đích: nó được phép biết cả application ports và concrete
infrastructure adapters để nối object graph. Không module business nào được import ngược composition
root.

## 2. Package tree mục tiêu

Giữ top-level package `app` trong Vòng 1 để tránh một repository-wide rename không mang giá trị
behavioral. Tree dưới đây là đích cuối Vòng 1; mỗi module chỉ tạo phần cần thiết cho slice của nó.

```text
app/
├── __init__.py
├── config.py                       # canonical backend Settings + load/cache policy
├── bootstrap.py                    # composition root factories
├── main.py                         # compatibility ASGI export: app = create_app()
├── contracts/
│   └── query.py                    # QueryRequest/QueryResponse/Citation/health DTOs
├── domain/
│   ├── documents.py                # DocumentChunk and stable document/chunk identity
│   ├── retrieval.py                # RetrievedChunk, RetrievalCandidate, rank records
│   ├── errors.py                   # framework/SDK-neutral application/domain failures
│   ├── ports.py                    # retrieval, generation, ingestion/indexing protocols
│   ├── retrieval_contracts.py      # immutable active Phase 7 frozen contract
│   └── policies/
│       ├── content_identity.py
│       ├── fusion.py               # unchanged RRF/weighted RRF behavior
│       ├── query_analysis.py        # unchanged expansion/role/list policies
│       ├── evidence.py             # selection and evidence gate policy
│       └── citations.py            # referential validation/build rules
├── application/
│   ├── query_service.py            # query use case and QueryExecution diagnostics
│   ├── ingestion_service.py        # file-to-chunks orchestration if a service seam is useful
│   └── indexing_service.py         # explicit index orchestration, no hidden startup mutation
├── infrastructure/
│   ├── ingestion/
│   │   └── docling.py              # Docling conversion/chunking adapter
│   ├── qdrant/
│   │   ├── client.py
│   │   ├── dense.py
│   │   ├── hybrid.py
│   │   ├── manifests.py
│   │   └── readiness.py
│   ├── models/
│   │   ├── embeddings.py           # FastEmbed dense/sparse adapters
│   │   └── reranker.py             # Jina/FastEmbed cross-encoder adapter
│   └── generation/
│       └── langchain_structured.py # OpenAI/Gemini-compatible structured adapter
└── api/
    ├── app.py                      # create_app
    ├── auth.py
    ├── dependencies.py             # obtains composed services/checkers
    └── routes/
        ├── health.py
        └── query.py

evaluation/
├── datasets.py                     # former app.evaluation/app.phase7 schema/validation pieces
├── retrieval_metrics.py
├── answer_metrics.py
├── query_scoring.py                # consumes QueryExecution/public contracts
├── replay.py
└── artifacts.py                    # reusable sanitized I/O/identity helpers

scripts/
├── operations/                     # supported ingest/index/search/query/readiness CLIs
├── evaluation/                     # current reproducible benchmark/evaluator CLIs
└── archive/
    └── phase7/                     # superseded drafts, migrations, calibrations, diagnostics

ui/
├── api_client.py                   # imports app.contracts.query only
├── config.py                       # UI-process-only URL/timeout/token
├── state.py
└── streamlit_app.py
```

Tree là direction, không phải lệnh tạo toàn bộ folder ngay lập tức. Nếu một abstraction không có hai
consumers hoặc không làm dependency direction rõ hơn, không tạo nó chỉ để khớp sơ đồ.

## 3. Responsibility và dependency rules

| Package | Owns | Được import | Không được import/own |
|---|---|---|---|
| `app.contracts` | Stable public HTTP DTOs và serialization contract | Pydantic, stdlib | FastAPI routes, SDK clients, application orchestration |
| `app.domain` | Stable records, ports, pure deterministic policies, frozen invariants | stdlib, Pydantic nếu cần validation | FastAPI, Streamlit, Qdrant, Docling, FastEmbed, LangChain, evaluation |
| `app.application` | Use-case order, candidate boundaries, abstention/retry workflow, sanitized diagnostics | domain/contracts/ports | FastAPI, Streamlit, Qdrant/Docling/FastEmbed/provider SDK, evaluation |
| `app.infrastructure` | External SDK construction, serialization to external systems, filesystem/network/model side effects | domain ports/records, canonical config at construction edge | API/UI, evaluation policies, use-case decisions |
| `app.api` | HTTP parsing, auth, threadpool handoff, status/error/header mapping | contracts, application service interface, dependency providers | retrieval algorithm, model creation, Qdrant payload mapping |
| `app.bootstrap` | Construct settings-specific implementations and caches | application, domain, infrastructure | business rules |
| `evaluation` | Dataset governance, metrics, scoring, replay, sanitized artifacts | public domain/application/contracts | imported by `app`, provider/retrieval decisions |
| `scripts` | CLI argument parsing, explicit command invocation, exit/summary | supported application/evaluation interfaces | reusable business logic hidden in private script helpers |
| `ui` | HTTP client and demo presentation | `app.contracts.query`, httpx, Streamlit | application service, Qdrant, model/provider SDK |

### Domain ports

Ports là Python `Protocol` hoặc nhỏ hơn nếu một callable đủ dùng. Chúng diễn tả điều application cần,
không diễn tả SDK đang dùng. Ports tối thiểu dự kiến:

- `QueryRetriever.retrieve(question, document_id) -> QueryRetrievalResult`;
- `AnswerGenerator.ensure_configured()` và `generate(...) -> GenerationResult`;
- `DocumentParser.parse(...) -> Sequence[DocumentChunk]` nếu ingestion service thực sự cần seam;
- dense/sparse index/search ports chỉ khi operational use case được đưa vào application service.

Không tạo repository/manager/factory abstraction chung chung. Existing `QueryRetriever` và
`AnswerGenerator` là điểm xuất phát, được di chuyển chứ không thiết kế lại semantics.

## 4. Configuration ownership

### 4.1 Canonical source

`app.config.Settings` tiếp tục là nguồn canonical cho **backend runtime environment** trong Vòng 1.
Giữ Pydantic validation, `.env` behavior, `SecretStr`, defaults và `get_settings()` cache để tránh
behavior drift. Có thể tách file implementation nội bộ sau, nhưng `app.config` phải là public facade
ổn định cho đến removal module được duyệt.

Phân loại rõ ba loại giá trị:

1. **Environment settings**: Qdrant endpoint/timeout, selected profile, provider credentials/model,
   API auth và presentation limits. Ownership: `Settings`.
2. **Immutable retrieval contracts**: collection names, document IDs, chunk counts/hashes, model and
   ranking profile values của Phase 7. Ownership: `app.domain.retrieval_contracts`; không override
   từng field tùy ý từ env khi profile frozen được chọn.
3. **UI process settings**: FastAPI URL, client timeout và bearer token. Ownership: `ui.config` vì UI
   là process riêng; chúng không được lặp backend retrieval/provider settings.

`.env.example`, Docker Compose và test fixtures là projections/consumers của canonical schema, không
phải nguồn truth độc lập. Documentation phải nói rõ:

- local Python và Compose API đều chọn `phase7`;
- `phase6` bị configuration validation từ chối;
- pytest bỏ qua `.env` và process settings;
- không đổi default/profile trong Vòng 1.

### 4.2 Profile resolution

Một function thuần nhận `Settings.retrieval_profile` và trả về toàn bộ immutable Phase 7 contract.
Nó phải giữ atomic behavior hiện tại: mutable overrides không được mix vào frozen values.
Readiness, query composition và evaluation cùng dùng public resolver/contract, nhưng evaluation không
được sở hữu hoặc sửa contract.

## 5. Composition root

`app.bootstrap` là nơi duy nhất nối concrete runtime graph:

```text
Settings
  -> resolve frozen retrieval contract
  -> construct lazy Qdrant/FastEmbed/Jina retriever adapter
  -> construct structured provider adapter
  -> construct EvidenceGate from configured threshold
  -> construct QueryService through domain ports
```

Các factory dự kiến:

- `build_query_service(settings: Settings) -> QueryService`;
- `build_readiness_checker(settings: Settings) -> ReadinessChecker`;
- cached accessors ở composition edge nếu cần giữ singleton/lazy behavior.

`app.api.app.create_app(settings=None)` cấu hình middleware, routes và dependency providers. `app.main`
chỉ export ASGI object để `uvicorn app.main:app` không đổi. Tests có thể truyền fake services/checkers
mà không monkeypatch Qdrant internals.

Operational CLIs cũng là composition roots nhỏ: parse arguments, lấy canonical settings, gọi một use
case/factory, map exception sang exit code. Chúng không gọi private helper của CLI khác.

## 6. Adapter boundaries

### 6.1 Inbound adapters

- FastAPI sở hữu HTTP validation/status/headers, không sở hữu retrieval/generation decisions.
- CLI sở hữu arguments/printing/exit codes, không sở hữu index/retrieval algorithms.
- Streamlit sở hữu UI state/rendering và chỉ gọi HTTP; không được “tối ưu” bằng direct service import.

### 6.2 Outbound adapters

- Docling adapter sở hữu document conversion và SDK status mapping.
- Qdrant adapters sở hữu schema/payload/filter/upsert/scroll/query details.
- FastEmbed adapters sở hữu dense/sparse/cross-encoder construction và lazy model errors.
- LangChain structured adapter sở hữu provider SDK kwargs, structured output parsing, refusal/error
  mapping; tên adapter phải provider-neutral vì implementation phục vụ cả OpenAI và Gemini.
- Filesystem manifest/artifact adapters sở hữu atomic serialization, nhưng data contract thuộc domain
  hoặc evaluation package tương ứng.

Mọi adapter có fake/stub contract test; unit tests không download model, gọi provider hoặc Qdrant
thật.

## 7. Production/evaluation boundary

Target import rules:

```text
allowed:
  evaluation -> app.application
  evaluation -> app.domain
  scripts.evaluation -> evaluation + app public interfaces

forbidden:
  app.* -> evaluation.*
  app.* -> scripts.*
  ui.*  -> evaluation.*
  API image runtime -> data/eval or artifacts
```

Các thay đổi cụ thể:

- chuyển candidate conversion/union dùng runtime ra domain/application trước;
- chuyển evaluator functions cuối `app/reranking.py` ra `evaluation/retrieval_metrics.py`;
- chuyển audit-only functions của `app/candidate_audit.py` ra evaluation;
- chuyển `app.evaluation`, `app.evaluation_e2e`, dataset/evaluator phần của `app.phase7` và
  `app.phase7_replay` ra `evaluation/` theo slices;
- giữ compatibility imports chỉ khi CLI/tests chưa chuyển trong cùng slice, gắn removal target R07;
- thêm import-architecture test chạy không cần optional SDKs.

Evaluation tiếp tục được phép gọi `QueryService` và đọc `QueryExecution`; application không biết qrel,
expected fact, query ID, fold, metric threshold hoặc artifact path.

## 8. Public contracts và invariants

### 8.1 Public Python/HTTP contracts cần giữ

- `uvicorn app.main:app` và ba endpoint paths;
- `QueryRequest`, `QueryResponse`, `Citation` JSON schema và validation;
- HTTP status/error code/body, UTF-8 content type, request ID và auth behavior;
- supported `python -m scripts.<existing-name>` paths cho đến khi compatibility/removal được duyệt;
- existing stable model serialization cho chunks/candidates/artifacts;
- `Settings` env names/defaults/validation và selected-provider properties.

### 8.2 Runtime invariants

- Phase 7 identity/count/hash/profile values không đổi; legacy collections vẫn được bảo vệ;
- document/chunk/point IDs và Qdrant payload/filter/index safety không đổi;
- dense/sparse/RRF/rerank ordering, stable ties và exact dedup semantics không đổi;
- `candidate_pool` → `candidates` → `evidence_candidates` boundaries không đổi;
- top-k chỉ sau final evidence selection;
- evidence gate trước provider; config failure trước retrieval;
- correction tối đa một lần với cùng evidence; no silent fallback;
- citations chỉ từ trusted source map; abstentions và sanitized errors không đổi;
- no import-time network/model construction;
- logs/artifacts không chứa secret, raw question/prompt/answer/evidence;
- Streamlit gọi FastAPI bằng HTTP và không retry POST.

### 8.3 Data/evaluation invariants

- không sửa frozen chunks, qrels, expected phrases/facts/answers để tăng metric;
- không đọc/tune theo raw held-out v2;
- historical metrics không được re-label thành validation mới;
- evaluation artifacts không trở thành production runtime input.

## 9. Chỉ di chuyển, cần thiết kế lại và không đụng

| Mức thay đổi | Thành phần | Cách xử lý |
|---|---|---|
| Chỉ di chuyển/split | pure ID/content/RRF/evidence/citation policies; evaluation metrics/replay; concrete SDK adapters | giữ function bodies/output, sửa imports và thêm facade/shim tạm thời |
| Thiết kế boundary | ports placement, composition root, app factory, canonical config/profile ownership, script package taxonomy | dùng characterization tests trước; review từng vertical slice |
| Rename có kiểm soát | provider adapter và phase-named runtime policies | alias cũ + removal module; không đổi artifact identity im lặng |
| Không đụng Vòng 1 | models, dimensions, weights, budgets, thresholds, collections, qrels/chunks, provider behavior, latency optimization, Docker topology | ghi Deferred to Round 2 |

## 10. Compatibility strategy

1. Thêm characterization/import tests trước.
2. Tạo module đích và chuyển implementation nhỏ nhất.
3. Giữ old import path bằng re-export mỏng khi public scripts/tests còn phụ thuộc.
4. Chuyển consumers theo một vertical slice, không broad search/replace toàn repo.
5. Gắn shim với module removal cụ thể; không để shim vô thời hạn.
6. Sau focused tests, chạy full Ruff/pytest bằng cùng Python 3.11 interpreter.
7. Chỉ remove shim khi `rg`, CLI contracts và docs chứng minh không còn consumer.

Không dùng compatibility shim để silent fallback hoặc giữ hai implementations.

## 11. ADR cần tạo trong quá trình refactor

ADR (Architecture Decision Record) là bản ghi ngắn về quyết định kiến trúc và trade-off. Chỉ tạo ADR
khi module implementation xác nhận quyết định; tài liệu planning này không giả vờ ADR đã accepted.

| ADR dự kiến | Module | Quyết định phải ghi |
|---|---|---|
| `ADR-001-layer-boundaries-and-import-rules.md` | R01/R02 | Giữ `app`, dependency direction, composition-root exception và automated import guard |
| `ADR-002-configuration-and-frozen-profiles.md` | R02 | `Settings` canonical, UI-only config, Phase 7-only selection và immutable contract |
| `ADR-003-candidate-and-evidence-boundaries.md` | R04/R05 | Nghĩa của pool/final/evidence sets, hai exact-dedup stages và top-k location |
| `ADR-004-evaluation-and-script-lifecycle.md` | R07 | Evaluation one-way dependency, operational/evaluation/archive taxonomy, archive-not-drop policy |
| `ADR-005-compatibility-imports.md` | khi shim đầu tiên cần thiết | public import/CLI guarantees, owner và removal module cho từng shim |

ADR nằm dưới `docs/adr/` nếu người dùng duyệt convention đó. Module docs vẫn là tài liệu học bắt buộc
và phải link ADR liên quan; ADR không thay thế `docs/modules/Rxx-*.md`.

## 12. Target state acceptance

Target architecture được coi là đạt sau R07 khi:

- automated import guard chứng minh production packages không import evaluation/scripts;
- API/CLI/UI chỉ gọi application/public contracts;
- domain không import framework/SDK/evaluation;
- Qdrant/Docling/FastEmbed/LangChain chỉ xuất hiện trong infrastructure/inbound adapter phù hợp;
- một backend `Settings` schema và một immutable profile resolver được dùng bởi composition;
- Streamlit vẫn HTTP-only;
- all frozen/API/ranking/evidence/citation behavior được tests bảo vệ;
- scripts được phân loại có README/index, không có private cross-script library imports;
- README có current architecture/runbook rõ, walkthrough lịch sử được gắn nhãn;
- không có data/model/collection/provider/metric thay đổi trong Git diff của Vòng 1.

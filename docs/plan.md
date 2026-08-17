# Industrial Technical Manual RAG — Project Plan

> Phase 6 implementation and correctness snapshot — 2026-08-06
>
> Phase 1 through Phase 6 are implemented. The frozen retrieval-development set has 30 manually
> checked queries (15 VI / 15 EN) against the 99-chunk batch-4 artifact with fingerprint
> `bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be`.
>
> Phase 6 static validation is Ruff PASS and pytest PASS — 160 tests on Python 3.11.15. The API
> image contains retrieval + LLM dependencies without Docling/model weights and measures
> 146,177,894 bytes. Union+rerank is now the accuracy-first query runtime and sparse/no-rerank is
> rollback. Real OpenAI smoke is `NOT RUN — API key unavailable`; correctness is proven offline and
> through Docker/Qdrant, not claimed for the real provider. The host `.venv` remains Python 3.13.5.
> Additive Gemini routing and the UTF-8 response regression later raised the local suite to 162 tests and passed
> Python 3.11 adapter construction; real Gemini generation remains `NOT RUN` without a provider key.

## 1. Mục tiêu dự án

Dự án xây dựng hệ thống RAG hỏi đáp trên tài liệu kỹ thuật công nghiệp dạng PDF/DOCX,
hỗ trợ tiếng Việt và tiếng Anh. Kiến trúc mục tiêu:

```text
PDF / DOCX
    ↓
Docling parsing
    ↓
Structure-aware chunks
    ↓
Dense + sparse representations
    ↓
Qdrant
    ↓
Hybrid retrieval + RRF
    ↓
Cross-encoder reranking
    ↓
LangChain + OpenAI LLM
    ↓
Answer + citations + abstention
    ↓
FastAPI
```

LangChain chỉ được sử dụng cho orchestration và generation. Ingestion, chunk metadata,
retrieval, fusion, reranking, citation validation và evaluation được giữ explicit để có thể
benchmark, debug và giải thích rõ từng bước.

## 2. Trạng thái tổng quan

| Phase | Trạng thái | Kết quả chính |
|---|---|---|
| Phase 1 | Hoàn thành | FastAPI scaffold, health endpoint, settings, test/lint, Docker Compose và CI |
| Phase 2 | Hoàn thành | Docling ingestion, structure-aware chunking, batching và atomic JSONL |
| Phase 3A | Hoàn thành | Dense embedding, Qdrant indexing và dense search |
| Phase 3A.1 | Hoàn thành | Stable chunk IDs, safe re-index, manifest và smoke evaluation |
| Phase 3A.2 | Hoàn thành | Direct-evidence evaluation, 30-query development set, dependency split, Docker stabilization và real Docker/Qdrant validation |
| Phase 4 | Closed; critical gate partial | Sparse BM25 vectors, collection v2, client-side RRF, strategy evaluator, real benchmark; sparse is currently stronger than hybrid, and 2/3 critical intents miss hybrid top 5 |
| Phase 4.1 | Implementation complete; ingestion Docker closure cancelled | Canonical Qdrant client 1.19.x, frozen candidate-pool audit, Phase 5 readiness artifact, API baked-image validation và documented external Docker deviation |
| Phase 5 | Implementation complete; quality PARTIAL | Ba multilingual reranking strategies đã benchmark; ranking gates và critical 3/3 PASS, CPU latency FAIL, không đặt runtime default |
| Phase 5.1 | Deferred | Reranker optimization/quantization/license replacement không nằm trong Phase 6 |
| Phase 6 | Implementation/offline/Docker correctness complete; real provider NOT RUN | Query API, evidence gate, Responses structured generation, citations, abstention, sparse rollback |
| Phase 7 | Đang thực hiện | Provider-free calibration closure PASS; typed-fact draft và provider calibration đang chờ review/approval trước held-out |

Thứ tự triển khai đã chốt:

```text
Phase 3A.2 → Phase 4 → Phase 4.1 closure → Phase 5 → Phase 6 → Phase 7
```

Phase 6 trước đây được gọi là Phase 3B. Tên lịch sử chỉ được giữ tại đây; roadmap hiện hành dùng
Phase 6 cho query/generation và Phase 7 cho final evaluation/hardening.

---

## 3. Kết quả thực tế đã đạt qua Phase 6

| Phase | Kết quả đã xác nhận |
|---|---|
| Phase 1 | FastAPI scaffold, `GET /api/v1/health`, settings, CI, Ruff, pytest và Docker Compose được thiết lập. |
| Phase 2 | Docling ingestion PDF/DOCX, structure-aware chunks, atomic JSONL và page batching hoạt động; OCR scanned PDF vẫn ngoài phạm vi. |
| Phase 3A | Dense multilingual MiniLM, Qdrant named vector `dense` 384/cosine, dense indexing và document filter hoạt động. |
| Phase 3A.1 | Stable chunk IDs, deterministic UUIDv5 point IDs, safe re-index, dense manifest và frozen 99-chunk contract được chốt. |
| Phase 3A.2 | Direct-evidence qrels 30 câu, evaluator direct hit, dependency/Docker split, Qdrant v1 99 points và dense baseline immutable được xác nhận. |
| Phase 4 | V2 dense+sparse/IDF, BM25 `Qdrant/bm25`, client-side RRF, manifest v2, sparse/hybrid CLIs và evaluator strategy đã hoàn tất; re-index v2 vẫn 99 points. |
| Phase 4.1 | Candidate-pool audit và readiness artifact chốt dense/sparse/hybrid/union coverage mà không thay frozen contract. |
| Phase 5 | Lazy multilingual cross-encoder, ba candidate pools, strict indexed-score validation, reranked search/evaluation CLI và real 30-query benchmark đã hoàn tất. |
| Phase 6 | `POST /api/v1/query`, frozen runtime validation, union reranker default, sparse rollback, evidence gate, typed Responses generation, deterministic citations và abstention đã hoàn tất. |

Kết quả benchmark trên cùng 30 qrels/frozen chunks:

| Metric | Dense | Sparse | Hybrid |
|---|---:|---:|---:|
| Hit@5 | 0.400 | **0.633** | 0.533 |
| Hit@20 | 0.767 | **0.867** | 0.867 |
| MRR@20 | 0.298 | **0.469** | 0.398 |
| p95 latency | 35.40 ms | **2.78 ms** | 26.16 ms |

Sparse BM25 hiện là retrieval baseline mạnh nhất cho manual tiếng Việt và development set này.
Hybrid RRF vẫn tốt hơn dense, nhưng không được mô tả là tốt hơn sparse. Hai trong ba bilingual
critical intents chưa có direct evidence trong hybrid top 5 trước reranking; Phase 5 đã xử lý được
critical gate 3/3 trên cả ba pool mà không thay qrels, chunks hay dense model, nhưng chưa đạt latency.

## 4. Tóm tắt các phase tiếp theo

- **Phase 5.1:** tiếp tục deferred; chỉ mở lại khi chọn hướng tối ưu CPU, quantization hoặc model có
  license thương mại phù hợp.
- **Phase 7:** held-out final evaluation, hardening và production readiness; không dùng development
  set hiện tại làm số liệu final.

Chi tiết/acceptance criteria của các phase sau bên dưới được giữ nguyên.

---

## 5. Các phase đã hoàn thành

## Phase 1 — Application scaffold

Commit lịch sử: `b5549b8 phase 1`

### Mục tiêu

Tạo nền tảng ứng dụng, cấu hình, kiểm thử và môi trường chạy trước khi thêm xử lý tài liệu.

### Đã triển khai

- FastAPI application tại `app/main.py`.
- Endpoint liveness `GET /api/v1/health`.
- Pydantic `Settings` đọc biến môi trường và file `.env`.
- Response model `HealthResponse`.
- Pytest và FastAPI `TestClient`.
- Ruff với Python target 3.11.
- Dockerfile cho API.
- Docker Compose gồm API và Qdrant.
- GitHub Actions CI cài `.[dev]`, chạy Ruff và pytest.

### Kết quả xác nhận

```text
GET /api/v1/health
status: 200
body: {"status":"ok","service":"industrial-rag","version":"0.1.0"}
```

Health endpoint chỉ là liveness check, không phụ thuộc trạng thái Qdrant.

---

## Phase 2 — Document ingestion

Commit lịch sử: `71da1aa phase 2`

### Mục tiêu

Chuyển PDF/DOCX thành các chunk có cấu trúc và metadata đủ tin cậy để index và citation.

### Đã triển khai

- Chỉ chấp nhận `.pdf` và `.docx`, không phân biệt hoa thường.
- Từ chối file không tồn tại, directory và extension không hỗ trợ.
- Document ID deterministic từ slug filename và SHA-256 toàn bộ nội dung file.
- Docling `DocumentConverter` và `HierarchicalChunker`.
- Page provenance, heading breadcrumb và content type từ metadata Docling.
- `DocumentChunk` JSON-serializable.
- CLI `scripts/ingest_preview.py`.
- Xuất UTF-8 JSONL.
- PDF page-range batching để giảm peak memory.
- OCR tắt cho PDF digital có text layer.
- Layout, table và OCR batch size đặt bằng 1.
- Chỉ chấp nhận `ConversionStatus.SUCCESS`.
- `PARTIAL_SUCCESS`, `FAILURE` hoặc status không xác định đều raise `IngestionError`.
- Ghi JSONL qua temporary file rồi atomic replace.
- Chunk index liên tục qua tất cả page batches.
- DOCX không bị áp dụng PDF batching.

### Kết quả manual thực tế

Manual: `data/raw/manual.pdf`

```text
Document ID: manual-77d5dae4c2c5
Page coverage: 1–21
Batch size 4: 99 chunks
Batch size 8: 98 chunks
```

Sự khác nhau 98/99 chunks đến từ Docling boundary tại page batches. Batch size không được xem
là universal optimum và các batch size khác nhau không được giả định sẽ tạo cùng chunk set.

### Hạn chế còn lại

- OCR chưa hỗ trợ scanned PDF trong profile hiện tại.
- Heading context không được tái dựng qua page-batch boundary.
- Multi-page table có thể bị chia tại batch boundary.
- Docling vẫn là dependency nặng và tạo Docker image lớn.

---

## Phase 3A — Dense indexing and retrieval

Commit lịch sử: `7677913 Phase 3A`

### Mục tiêu

Tạo dense vector cho từng chunk, index vào Qdrant và cung cấp ranked dense retrieval.

### Đã triển khai

- `qdrant-client[fastembed]`.
- FastEmbed multilingual dense model.
- Passage dùng `passage_embed()`; query dùng `query_embed()`.
- Embedding input gồm heading context và raw content.
- Không sửa `chunk.text` và không lưu embedding text trong payload.
- Dimension được lấy từ model probe, không hard-code.
- Embedding theo mini-batch.
- Shared Qdrant collection, không tạo collection theo từng document.
- Named vector `dense`, cosine distance.
- Deterministic UUIDv5 point ID từ `chunk_id` và fixed namespace.
- Citation-ready payload.
- CLI `scripts/index_document.py` và `scripts/search_dense.py`.
- Optional server-side `document_id` filter.
- Empty question và invalid limit bị từ chối.
- In-memory Qdrant tests với fake deterministic embedding model.

### Runtime versions đã xác nhận

```text
docling: 2.117.0
docling-core: 2.90.0
qdrant-client: 1.18.0
fastembed: 0.8.0
Qdrant server: 1.18.3
embedding model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
embedding dimension: 384
collection: industrial_manual_chunks
vector name: dense
distance: cosine
```

---

## Phase 3A.1 — Audit and stabilization

Commit hiện tại: `01df8f6 phase 3A.1`

### Mục tiêu

Audit Phase 1–3A, sửa correctness và data-loss risk trước khi thêm generation hoặc API query.

### Stable chunk IDs

Chunk ID cũ phụ thuộc `document_id + page + global chunk_index`, nên một chunk mới ở đầu tài
liệu có thể làm đổi ID của các chunk phía sau.

Chunk ID mới dùng:

```text
document_id
+ sorted page numbers
+ heading breadcrumb
+ normalized raw text
+ deterministic duplicate occurrence index
→ short SHA-256
```

Đặc tính đạt được:

- Same canonical chunk tạo cùng ID.
- Changed text tạo ID mới.
- Unrelated earlier chunk không làm đổi ID phía sau.
- Duplicate identical chunks vẫn phân biệt được.
- `chunk_index` tiếp tục nằm trong metadata để ordering.

Đổi batch size 4 sang 8 giữ nguyên 85/99 canonical IDs; các ID còn lại thay đổi do chính chunk
content hoặc boundary đã thay đổi.

### Safe re-index

Luồng cũ xóa points của document trước khi upsert. Luồng mới:

```text
1. Parse toàn bộ document thành công.
2. Embed toàn bộ chunks thành công.
3. Lấy existing point IDs của document.
4. Upsert toàn bộ new points với wait=True.
5. Chỉ sau khi upsert thành công mới xóa stale old point IDs.
```

Regression tests xác nhận embedding failure hoặc upsert failure không xóa old points; document
khác không bị ảnh hưởng.

### Index manifest

Artifact runtime `artifacts/metrics/dense-index-manifest.json` ghi collection/vector name,
embedding model/dimension, distance và ingestion profile. Search từ chối chạy nếu manifest thiếu
hoặc không khớp model/config, kể cả khi model mới có cùng dimension với model cũ.

### Dense smoke evaluation

- Dataset 13 câu factual tiếng Việt và tiếng Anh.
- Hit Rate@1, @3, @5, MRR và average query latency.
- Page/phrase relevance matching.

Kết quả audit:

```text
pip install -e ".[dev]": PASS
Ruff: PASS
pytest: PASS — 47 tests
Docker Compose config: PASS
Qdrant connection: PASS
Manual indexing: PASS — 99 points
Second indexing: PASS — vẫn 99 points
Point IDs: deterministic và khớp payload chunk IDs
```

Dense smoke baseline:

```text
Hit Rate@1: 0.385
Hit Rate@3: 0.615
Hit Rate@5: 0.615
MRR: 0.500
Average latency: khoảng 24 ms, không tính model load
```

Multilingual MPNet candidate đã được benchmark trên cùng chunks nhưng không cải thiện Hit@5 và
làm MRR thấp hơn, nên default MiniLM được giữ nguyên.

### Blocker hiện tại

Direct answer-bearing chunks cho ba câu critical chưa xuất hiện ổn định trong top 5. Metric hiện
tại cũng có thể tạo false positive vì bất kỳ chunk nào cùng expected page đều được xem là relevant.
Retrieval phải được hoàn thiện trước khi gửi context vào LLM.

### Docker hiện tại

```text
industrial-rag-api image: khoảng 9.26 GB
Qdrant image: khoảng 265 MB
Qdrant volume tại thời điểm audit: khoảng 483 MB
Build cache tại thời điểm audit: khoảng 12.73 GB
```

Dockerfile không được tạo lại mỗi lần chạy; phần tăng dung lượng chủ yếu là image layers và build
cache. Repository hiện chưa có `.dockerignore`, và API image đang mang cả dependency ingestion nặng.

---

## 6. Chi tiết phase và roadmap còn lại

## Phase 3A.2 — Dense baseline closure and Docker stabilization

> Hoàn thành. Dense baseline immutable đã được xác nhận trên frozen 99 chunks; API/ingestion Docker
> đã được tách dependency và real Qdrant validation pass. Nội dung dưới đây được giữ như design/acceptance
> record lịch sử; kết quả thực tế được tóm tắt tại mục 3.

### Mục tiêu

Làm evaluation phản ánh đúng direct evidence, tái lập baseline trên Python 3.11 và tách API image
khỏi ingestion dependencies.

### Evaluation hardening

- Mở rộng smoke set lên khoảng 30 câu factual song ngữ.
- Đọc manual/chunk JSONL để xác nhận thủ công evidence cho từng câu.
- Thêm `relevant_chunk_ids` dựa trên stable chunk IDs.
- `expected_phrases` phải xuất hiện trong chính chunk trả về.
- `expected_pages` chỉ dùng chẩn đoán, không đủ để kết luận direct-evidence hit.
- Giữ cùng chunk set cho mọi model/retrieval comparison.
- Báo cáo Hit Rate@1/3/5, MRR, direct-evidence rank, p50/p95 latency và failure cases.
- Đóng băng dense MiniLM baseline; không đổi model nếu không có metric tốt hơn.

### Python environment

- Tạo lại `.venv` bằng Python 3.11.
- Cài `python -m pip install -e ".[dev]"`.
- Chỉ dùng `python -m ruff` và `python -m pytest` từ cùng `.venv`.
- Xác nhận local runtime, CI và Docker cùng Python 3.11.

### Docker split

- Thêm `.dockerignore` loại `.git`, `.venv`, test caches, raw documents và artifacts khỏi context.
- Chia dependency groups thành base API, retrieval, ingestion, LLM và development.
- API image cài base + retrieval, không cài Docling.
- Ingestion image cài Docling + retrieval và chỉ chạy on demand.
- Thêm Compose profile `tools` cho ingestion service.
- Pin `qdrant/qdrant:v1.18.3` thay cho `latest`.
- Dùng shared FastEmbed model cache volume; không bake model weights vào image.
- Không tự động prune hoặc xóa Qdrant volume.

Luồng dự kiến:

```powershell
docker compose up -d qdrant api
docker compose --profile tools run --rm ingestion `
  python scripts/index_document.py data/raw/manual.pdf --page-batch-size 4
```

### Acceptance gate

- Dense baseline tái lập được trên cùng 99 chunks.
- Evaluator không tính same-page unrelated chunk là direct hit.
- Ba critical questions có direct-evidence rank chính xác.
- Ruff và pytest pass trên Python 3.11.
- Docker Compose config và cả hai image build pass.
- API image mục tiêu dưới 3 GB, không chứa Docling hoặc model weights.
- Unchanged rebuild dùng cache và không tải lại model.

---

## Phase 4 — Hybrid retrieval and RRF (design record and implementation result)

> Implemented and benchmarked on 2026-08-06. The collection v1 and immutable dense baseline remain
> separate from v2. See `docs/walkthrough-phase-4.md` for commands, artifacts, and diagnostics.

### Mục tiêu

Kết hợp semantic dense retrieval với exact-keyword sparse retrieval để cải thiện direct evidence.

### Qdrant schema v2

- Tạo physical collection mới `industrial_manual_chunks_v2`.
- Không recreate hoặc xóa dense collection hiện tại.
- Named vectors:
  - `dense`: 384 dimensions, cosine.
  - `sparse`: Qdrant sparse vector với IDF modifier.
- Sparse model: `Qdrant/bm25`, có trong FastEmbed 0.8 hiện tại.
- Một point chứa cả dense và sparse representations.
- Giữ deterministic UUIDv5 point ID và payload hiện tại.
- Manifest schema v2 bổ sung schema version, sparse model/config, fusion parameters và profile.
- Chỉ chuyển `QDRANT_COLLECTION` sang v2 sau khi index và validation hoàn tất.
- Giữ collection cũ để rollback; không tự động xóa.

### Sparse indexing and search

- Passage sparse input dùng heading context + raw content.
- Query dùng `SparseTextEmbedding.query_embed()`.
- Sparse embedding theo mini-batch.
- Optional `document_id` filter được thực hiện trong Qdrant.
- Không lưu sparse/dense vectors trong payload.

### Explicit RRF

- Dense và sparse search chạy riêng để giữ khả năng benchmark.
- Candidate configuration:

```text
dense candidates: 20
sparse candidates: 20
RRF k: 60
hybrid-only final results: 5
reranker candidate pool: 20
```

- RRF score tính client-side theo rank; không trộn raw dense và sparse scores.
- Tie-break deterministic bằng best component rank rồi `chunk_id`.
- Giữ `dense_search()` cho baseline và backward compatibility.
- Thêm typed `RetrievalCandidate` chứa chunk metadata, component ranks/scores và RRF score.
- Mọi score chỉ là ranking signal, không phải probability.

### Tests

- Sparse collection schema và IDF modifier.
- Sparse passage/query embedding calls.
- Sparse payload không chứa vector.
- Deterministic RRF ordering và duplicate collapse.
- Document filter áp dụng cho dense và sparse.
- Hybrid re-index không duplicate và xóa đúng stale points.
- Collection v1 không bị thay đổi.
- Unit tests dùng fake sparse model và in-memory Qdrant.

### Acceptance gate

- Ba critical questions có direct evidence trong hybrid top 5.
- Direct Hit@5 tăng ít nhất 0.10 tuyệt đối so với dense baseline.
- MRR không thấp hơn dense baseline.
- Index/re-index, document isolation và manifest validation pass.
- p95 hybrid query dưới 300 ms, không tính initial model load.

### Kết quả thực tế 2026-08-06

- FastEmbed `0.8.0` API được xác minh: `SparseTextEmbedding`, `passage_embed`, `query_embed`, và
  `Qdrant/bm25` đều có mặt. BM25 chạy `disable_stemmer=true`, `k=1.2`, `b=0.75`, và
  `avg_len=72.838384` tính bằng đúng FastEmbed tokenizer/preprocessing trên frozen 99 chunks.
- Collection v2 `industrial_manual_chunks_v2` có 99 points sau index và vẫn 99 sau re-index; mỗi
  point có named dense vector 384/cosine, named sparse vector/IDF, stable UUIDv5 ID, và keyword
  index `document_id`. Collection v1 `industrial_manual_chunks` vẫn giữ schema dense-only và 99
  points.
- Baseline artifacts tách riêng: `dense-baseline.json` không bị overwrite; additive dense closure,
  sparse và hybrid nằm lần lượt ở `dense-baseline-closure.json`, `sparse-baseline.json`, và
  `hybrid-baseline.json`.

| Metric | Dense | Sparse | Hybrid | Hybrid − Dense |
|---|---:|---:|---:|---:|
| Hit@1 | 0.167 | 0.333 | 0.267 | +0.100 |
| Hit@3 | 0.367 | 0.500 | 0.400 | +0.033 |
| Hit@5 | 0.400 | 0.633 | 0.533 | +0.133 |
| Hit@20 | 0.767 | 0.867 | 0.867 | +0.100 |
| MRR@5 | 0.269 | 0.441 | 0.365 | +0.096 |
| MRR@20 | 0.298 | 0.469 | 0.398 | +0.100 |
| p95 | 35.40 ms | 2.78 ms | 26.16 ms | -9.24 ms |

- Hybrid Hit@5, MRR@20, Hit@20/candidate recall, and p95 targets pass. The critical-intent top-5
  gate is partial: only the first bilingual intent is top 5; the other two pairs have ranks
  `8/9` and `8/16` (VI/EN). Do not alter qrels, chunks, or the dense model to hide this result.
- Unit suite: 70 passed. The integration run used Python 3.11.15 in the ingestion container;
  host `.venv` remains Python 3.13.5. The next allowed improvement is Phase 5 reranking.

---

## Phase 4.1 — Phase 4 closure and Phase 5 readiness

> Closure audit on 2026-08-06. The frozen 99 chunks and historic baselines remain immutable. This
> phase added no reranker, model, qrel, chunking, or collection-schema change. API baked-image
> validation passed; the fresh ingestion target build was cancelled during a slow package-registry
> download and must be rerun before full Docker closure.

### Closure decisions

- Canonical package contract: `qdrant-client >=1.19.0,<1.20.0`; the successful real Phase 4
  integration used 1.19.0 with Python 3.11.15 and FastEmbed 0.8.0.
- Qdrant server stays `qdrant/qdrant:v1.18.3`. Collection v1 remains dense 384/cosine with 99
  points; v2 remains dense 384/cosine plus sparse/IDF with 99 points. Neither collection is
  recreated or deleted.
- The old dense artifact that reports client 1.18.0 remains a historical artifact. Documentation
  distinguishes it from the canonical dependency; it is not rewritten to make historical runs look
  different.

### Candidate-pool audit

The real audit uses exactly the frozen qrels and retrieves each independent 20-candidate pool before
any reranking. Candidate recall is not Hit@k: it means at least one direct-evidence qrel is available
to a reranker.

| Pool | Candidate recall | Missing qrels | Candidate count |
|---|---:|---|---|
| dense top 20 | 0.767 | `005,006,014,017,018,021,029` | 20 maximum |
| sparse top 20 | 0.867 | `008,014,017,020` | 7–20 |
| hybrid RRF top 20 | 0.867 | `008,014,017,018` | 20 |
| dense@20 ∪ sparse@20 | 0.933 | `014,017` | 22–34; median 28 |

The recommended Phase 5 experiments are therefore `sparse_top20`, `hybrid_top20`, and
`dense20_union_sparse20`; hybrid is not declared the default. RRF demotes sparse top-5 evidence
outside top 5 for `dense_005`, `dense_019`, `dense_021`, and `dense_029`, while dense contributes
evidence absent from sparse for `dense_008` and `dense_020`. Cross-lingual union coverage is 0.933
versus 1.000 monolingual; missing `dense_014` is cross-lingual and `dense_017` is monolingual.

`scripts/audit_candidate_pools.py` writes `artifacts/metrics/candidate-pool-audit.json` and
`scripts/generate_phase5_readiness.py` writes `artifacts/metrics/phase-5-readiness.json`. Both are
ignored runtime artifacts. The readiness status remains `ready_with_documented_deviation`: aggregate
Phase 4 gates pass, but the bilingual critical top-5 gate remains 1/3 and Phase 5 must prove any
reranker improvement on the unchanged development set.

See `docs/walkthrough-phase-4-closure.md` for the complete command and Docker validation record.

## Phase 5 — Multilingual cross-encoder reranking

> Implemented and benchmarked on 2026-08-06. Correctness/static validation PASS; overall quality
> `PARTIAL` because the ranking gates pass but no strategy meets the CPU p95 latency gate. No
> reranking runtime default is configured.

### Implemented contract

- FastEmbed 0.8.0 `TextCrossEncoder` with
  `jinaai/jina-reranker-v2-base-multilingual`; initialization is lazy and never occurs on import.
- Official model metadata and FastEmbed registry both report `CC-BY-NC-4.0`, ONNX about 1.11 GB,
  and 1K/sliding-window context. This is a non-commercial benchmark/demo model, not an approved
  commercial deployment dependency.
- Candidate text format `heading_content_v1`: breadcrumb joined with ` > `, two newlines, then raw
  chunk text; the raw payload is never rewritten.
- Strategies are explicit: sparse v2 top 20; hybrid v2 dense@20 + sparse@20 → RRF k=60 → top 20;
  union v1 dense@20 ∪ v2 sparse@20 with stable-ID de-duplication and no pre-rerank truncation.
- Full pool is returned. Display `--limit` is applied only after reranking. Final ordering is score
  descending, previous rank ascending, then chunk ID.
- Missing/duplicate/out-of-range model indices, wrong output count, non-finite score, invalid input,
  or inference exception raises `RerankingError`; there is no silent fallback.
- Dense, sparse, RRF/union, rerank ranks and scores plus metadata remain available for diagnostics.

### Real benchmark

Frozen input remained 30 qrels, 15 VI/15 EN, 99 chunks, hash
`bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be`.

| Strategy | Hit@5 | MRR@5 | Hit@20 | MRR@20 | Candidate recall | Warm total p95 |
|---|---:|---:|---:|---:|---:|---:|
| Sparse rerank | 0.733 | 0.529 | 0.867 | 0.544 | 0.867 | 9,879.69 ms |
| Hybrid rerank | 0.767 | 0.546 | 0.867 | 0.556 | 0.867 | 8,465.75 ms |
| Union rerank | 0.767 | 0.546 | 0.933 | 0.560 | 0.933 | 11,889.45 ms |

All three strategies recover 3/3 bilingual critical intent pairs in top 5 and pass Hit@5 ≥ 0.633
and MRR@5 ≥ 0.485. Union is the best observed research strategy because it preserves the 0.933
candidate recall and highest MRR@20. Every strategy fails warm total p95 < 1.5 seconds, therefore
`recommended_default_strategy=null`; sparse retrieval remains the operational rollback.

Union still has unrecoverable candidate misses `dense_014` and `dense_017`. Its top-5 reranker misses
are `dense_007`, `dense_008`, `dense_018`, `dense_021`, and `dense_022`; `dense_008` and `dense_018`
remain present but rank 18. Results are development-set evidence and are not a held-out Phase 6 test.

### Validation and artifacts

- Python 3.11.15 container: Ruff PASS; pytest PASS — 99 tests, one known third-party
  Starlette/TestClient warning.
- Real VI search, EN→VI search, six critical queries, validation-error smoke, and all 30 queries ×
  three strategies PASS against live Qdrant.
- Runtime artifacts are additive and ignored by Git:
  `phase-5-candidate-audit.json`, `rerank-sparse.json`, `rerank-hybrid.json`,
  `rerank-union.json`, and `phase-5-comparison.json`.
- Model download happened at runtime in the shared FastEmbed cache, not in Docker build. The heavy
  ingestion rebuild remains deferred to the user:
  `docker compose --progress plain --profile tools build ingestion`.

See `docs/walkthrough-phase-5.md` for commands, per-scenario metrics, critical ranks, latency
methodology, warnings, rollback, and remaining gates.

---

## Phase 6 — Query API, grounded generation, citations and abstention

> Implemented 2026-08-06. Offline/Docker correctness PASS; real OpenAI smoke `NOT RUN` because no
> API key was configured. The system is not described as production-ready.

### Implemented contract

- `POST /api/v1/query` accepts a stripped non-empty question, optional non-empty `document_id`, and
  `top_k` 1–10/default 5. Public citations contain trusted chunk/document/filename/page/heading/text
  excerpt metadata; retrieval scores are not exposed.
- Default runtime is v1 dense@20 ∪ v2 sparse@20 → full Jina multilingual rerank → final top-k.
  `RETRIEVAL_STRATEGY=sparse` plus `RERANK_ENABLED=false` is the only rollback combination.
- Runtime is lazy and artifact-independent. It checks the 99-point collection counts, both schemas,
  stable chunk-ID hash, frozen model names, dimensions, BM25 settings, and candidate limits directly
  against Qdrant. Missing generation-provider configuration is reported before models load.
- Evidence is labeled `S1…Sn`, bounded to 24,000 characters, and enclosed as untrusted document
  data. The system prompt prohibits outside knowledge/document instructions and preserves technical
  numbers, units, and identifiers.
- `GeneratedAnswer(answer, source_ids, insufficient_evidence)` is provider-native strict structured
  output through LangChain/OpenAI Responses. Configuration is `gpt-5.6-terra`, reasoning `low`, 800
  output tokens, 60-second timeout, one provider retry, and mandatory `store=false`.
- An additive provider switch supports `gemini-3.5-flash-lite` through Google's OpenAI-compatible
  Chat Completions endpoint. It keeps the same schema/citation gates and does not add the Google SDK;
  real Gemini smoke remains pending until a Gemini key is configured.
- The gate abstains on no candidates, invalid metadata, or an explicitly configured score threshold.
  Threshold defaults to `None`; no score is described as a probability or calibrated confidence.
- Source IDs are referentially validated. Unknown/out-of-context IDs, missing citations, empty
  answers, or contradictory abstention output receive at most one correction using the same evidence
  without re-retrieval/reranking. Final citation metadata comes only from retrieved payloads.
- Valid abstentions are HTTP 200. Retrieval/reranker/provider dependency failures are 503, provider
  timeout is 504, validation is 422, and unexpected failures are sanitized 500 responses.

### Measured validation

- Python 3.11.15: Ruff PASS; pytest PASS — 160 tests, one known Starlette/TestClient warning.
- API image build PASS in 1m15s; size `146,177,894` bytes. Python 3.11.15,
  `langchain-core 1.5.3`, `langchain-openai 1.4.1`, and `openai 2.53.0` are present; Docling and baked
  model files are absent.
- Docker health 200; query without key 503 `llm_not_configured`; no raw question/evidence in logs.
- Real union runtime: 23 candidates, retrieval 32.20 ms, rerank 13,897.50 ms. Sparse rollback:
  14 candidates, retrieval 5.63 ms, rerank 0 ms. These are smoke measurements, not a latency
  benchmark replacing Phase 5 p95.
- Qdrant v1/v2 remain 99 points with hash
  `bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be`.
- `artifacts/metrics/phase-6-query-smoke.json` records `not_run/api_key_unavailable`; no real answer,
  prompt, evidence, key, or provider response is stored.

See `docs/walkthrough-phase-6.md` for API examples, error/abstention tables, Docker commands,
structured-output details, validation evidence, and rollback.

---

## Phase 7 — End-to-end evaluation, real industrial corpus and production hardening

### Historical checkpoint (2026-08-09)

- Separate ATV320 Installation/Programming corpus is frozen with 2,753 chunks and stable-ID hash
  `2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d`.
- Protected Phase 3--6 v1/v2 collections remain 99/99 points. Phase 7 uses only
  `industrial_manual_phase7_dense_v1` and `industrial_manual_phase7_hybrid_v1`.
- Dataset v1 was approved and used only for calibration. Diagnosis found that generated-answer
  scoring reused English evidence phrases and that qrels omitted some exact duplicate chunks.
- Dataset v2 is source-reviewed and frozen: 65/65 rows are approved and all 42 answerable rows have
  language-aware `expected_answer_facts`. Exact-content closure added 2 calibration and 14 held-out
  qrels without broad phrase matching. Manual review also corrected calibration 011/012 from an
  unrelated page-355 qrel to direct reference-mode evidence on page 45.
- Implemented: multi-document frozen-runtime validation, sanitized resumable E2E scorer, Qdrant
  readiness, request correlation IDs, bounded Qdrant timeout, optional bearer auth, and API Docker
  liveness healthcheck.
- Current Python 3.11.15 validation: Ruff `PASS`; pytest `239 passed, 1 warning`; Compose config and
  `git diff --check` PASS. The warning is the known third-party Starlette/TestClient deprecation.
- Historical Gemini dataset-v2 diagnostics calibration completed all 20 rows:
  candidate recall/Hit@5 0.667, MRR@5 0.583, strict answer-fact accuracy 0.500,
  direct-evidence citation rate 0.667, abstention precision/recall 1.000/1.000, total p95 10.043 s.
  The old dataset-v1 phrase score 0.417 remains historical and is not comparable.
- Sanitized diagnostics now report fact IDs, exact match, token-overlap ratios, missing fact IDs,
  qrel dense/sparse/RRF/rerank ranks and unexpected citation document IDs without answer/evidence.
- Phase 7.4 separates strict contiguous phrase accuracy (diagnostic) from deterministic typed fact
  accuracy (headline). Rescoring the existing sanitized output changes 6/12 strict matches to 8/12
  deterministic matches; it is derived evidence because raw answers are intentionally absent.
- Phase 7.4.1 closes the cross-document contamination gate without changing chunks, qrels, models,
  schemas, collections, or candidate budget. Fusion uses a weak role multiplier `0.10`; a separate
  rank-only post-rerank role prior uses multiplier `0.50`, offset `20`, and strong-and-weak
  confidence. It was selected by six bilingual intent folds from a sanitized Jina snapshot, not from
  qrels at runtime. The provider-free closure is candidate recall `12/12`, Hit@5 `11/12`, MRR@5
  `0.875`, wrong-document top-1 `0/12`, wrong-document top-5 `8/60 = 0.133`, EN Hit@5 `6/6`, VI
  Hit@5 `5/6`, and calibration 010 rank `6`; all contamination gates pass.
- Phase 7.5 freezes reranker budget `30`, batch size `8`, and runtime-default ONNX threads after a
  microbenchmark and three full repetitions of every calibration question. Rerank p95 is `6.996 s`
  and total p95 is `7.027 s`, a 47.8% improvement from the 13.399-second baseline with no quality
  regression. FastEmbed remains direct-pinned at `0.8.0`; no re-index or model/pooling change occurs.
- Before approval, typed calibration-v3 facts were an inactive human-review draft that preserved
  qrels/pages/phrases. Held-out remains sealed until the fresh calibration fact/citation/abstention
  gate passes.
- Calibration-v3 was subsequently frozen after explicit approval and Gemini 3.5 Flash Lite evaluated
  all 20 calibration rows. It failed the E2E release gate: deterministic fact accuracy `7/12 = 0.583`
  and wrong-document citations `2`, despite valid citation IDs `100%`, direct-evidence citations
  `0.917`, and abstention precision/recall `1.000`. Held-out was not executed. The evaluator now
  requires a dataset-specific provider token; a calibration token cannot authorize held-out.
- Phase 7 union runtime now supports same-document exact-normalized content deduplication before
  reranking. The setting defaults off globally and is enabled explicitly only by the Phase 7 E2E CLI,
  preserving legacy Phase 5/6 behavior.
- FastEmbed is now a direct `0.8.0` retrieval/dev constraint, matching the tested Python 3.11 runtime;
  Phase 7 does not switch to the warning-suggested 0.5.1 behavior or re-index the corpus.
- Fresh provider E2E is pending explicit data-egress approval. Held-out remains unrun because
  deterministic answer/citation/document gates have not all been demonstrated on the frozen
  Phase 7.4 runtime.

### Calibration-closure checkpoint (2026-08-11)

Status: `PARTIAL`; held-out: `BLOCKED_GOVERNANCE`.

- Calibration mode is sealed to `calibration-v3.jsonl`. It validates one split and obtains only the
  held-out SHA-256 from `phase-7-evaluation-manifest-v3.json`; it does not open `test.jsonl`.
- The E2E run identity now includes prompt/evaluator/runtime hashes, provider/model/base-host,
  reasoning effort, explicit Gemini temperature `0`, token/timeout/retry/store settings, Python and
  library versions, top-k, and correction count. A changed field invalidates the checkpoint.
- Typed fact evaluator v2 adds narrow, ASCII-only regular inflection for text facts and span-aware
  negation. It fixes generic `block`/`blocked` and `contact`/`contacts` cases without applying
  stemming/prefix rules to identifiers or numeric-unit facts. Strict phrase and token coverage remain
  diagnostics only.
- `QueryExecution` now distinguishes the full post-rerank ranking from actual generation evidence.
  Exact normalized content duplicated across different documents is represented once before top-k;
  the query-derived document role chooses provenance, and equivalent chunk/document IDs remain in
  diagnostics. Same-document duplicates and near-duplicates are not collapsed.
- Snapshot v2 was executed against real Qdrant/Jina with zero provider calls and zero held-out reads.
  The finite CE-rank/RRF-rank grid preserves candidate recall `12/12`, Hit@5 `11/12`, MRR@5 `0.875`,
  EN `6/6`, VI `5/6`, and wrong-document top-1 `0`. The active offset-20 profile has
  wrong-document actual evidence `7/60 = 0.117` after cross-document dedup.
- No grid profile moves calibration 010 from full rank 6 into actual evidence top 5. The single
  pre-registered `list_completeness_v1` fallback also fails because rank 5 has the same query-derived
  `MODE` identifier and a larger generic list-pair count. It is not activated; no query-specific or
  qrel-aware rule is added.
- Historical calibration 005 still cannot be reconstructed from v4 because raw answers were
  intentionally omitted. The separately approved `diagnose_phase7_calibration_005.py` run reused one
  retrieved/reranked evidence bundle for three provider attempts. All three completed, cited `S1`,
  and matched the expected fact with positive polarity. The current matcher/provider path therefore
  closes 005 without selecting a best attempt; the private raw file remains ignored.
- Three-run worst-case aggregation is implemented but not run. Full calibration requires a new token
  `APPROVE PHASE 7 CALIBRATION V5 STABILITY EGRESS`; each run must independently pass at least 11/12
  facts and all citation/abstention gates.
- Historical tracked documentation exposed held-out row content and the old CLI parsed both splits.
  Current documentation is metadata-only and the CLI refuses held-out execution, but history cannot
  be made unseen. A new access-controlled final set or an explicit reporting downgrade is required.
- No qrel, chunk, model, Qdrant collection, volume, public Query API, Docker image, or embedding
  behavior was changed. Exactly three approved provider generations were made for calibration 005;
  no held-out run, re-index, build, or prune occurred.
- Final canonical ingestion-container Python 3.11.15 validation: Ruff PASS; pytest
  `279 passed, 1 warning`; `git diff --check` and `docker compose config --quiet` PASS. The local
  `.venv` is Python 3.13.5 and also passes, but was not used as canonical or overwritten. The warning
  is the known Starlette/TestClient deprecation.

### Evaluation dataset

- Tối thiểu 20 answerable và 10 unanswerable questions.
- Bao gồm tiếng Việt và tiếng Anh.
- Answerable item có relevant chunk IDs, evidence-validation phrases, reviewed answer facts theo
  ngôn ngữ câu hỏi và citation expectations.
- Unanswerable item phải thực sự không có evidence trong indexed manual.

### Metrics

- Retrieval Hit@k, MRR và candidate recall.
- Reviewed answer-fact accuracy; evidence phrases không dùng để chấm generated answer.
- Citation validity và citation coverage.
- Abstention precision và recall.
- Dense, sparse, fusion, rerank, LLM và total latency.
- OpenAI input/output/cached token usage.

### Release gate

```text
Valid citation IDs: 100%
Reviewed answer-fact accuracy: >= 0.85
Abstention precision: >= 0.90
Abstention recall: >= 0.80
Critical direct-evidence top-5: 3/3
Unsupported citation IDs: 0
```

### Hardening

- Structured logs với request ID và stage latency.
- Không log secret hoặc full document text.
- Timeout và bounded retry cho Qdrant/OpenAI.
- Docker health checks cho API và Qdrant.
- API container chạy non-root.
- Dependency và Qdrant versions được pin theo compatible ranges.
- Migration/rollback procedure cho collection v2.
- CI chỉ chạy fake models và in-memory Qdrant.
- Real Qdrant, FastEmbed và OpenAI tests dùng integration marker riêng.
- Calibrate evidence/abstention policy from answerable and truly unanswerable held-out data.
- Resolve reranker CPU latency and non-commercial license before commercial deployment.

---

## 5. Ngoài phạm vi hiện tại

- Ingestion HTTP endpoint.
- Background ingestion workers hoặc queue.
- Database ngoài Qdrant.
- Multi-user authentication/authorization.
- Streaming answer.
- Multi-turn conversation memory.
- Agent hoặc tool-calling workflow.
- GPU support.
- OCR cho scanned PDF.
- Automatic destructive collection migration.

## 6. Definition of done chung

Một phase chỉ được xem là hoàn thành khi:

1. Code và public contracts đúng phạm vi phase.
2. Không phá vỡ behavior của phase trước.
3. Ruff pass.
4. Pytest pass và không cần internet, model thật hoặc API key.
5. Integration smoke tương ứng pass khi phase có external service/model.
6. README, CODEBASE và file plan này được cập nhật.
7. Metrics trước/sau được ghi lại; không tuyên bố cải thiện nếu chưa đo.
8. Limitations, migration và rollback được ghi rõ.

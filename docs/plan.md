# Industrial Technical Manual RAG — Project Plan

> Phase 3A.2 implementation snapshot — 2026-08-05
>
> Code, direct-evidence qrels, unit tests, dependency split, Docker targets and Compose profiles
> are implemented. The frozen retrieval-development set has 30 manually checked queries (15 VI / 15
> EN) against the 99-chunk batch-4 artifact with fingerprint
> `bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be`.
>
> Local static validation passed: `python -m pip install -e ".[dev]"`, Ruff, 57 pytest tests, and
> Compose config (default, development override, and `tools` profile). Docker/Qdrant integration
> then passed with Qdrant 1.18.3, two batch-4 re-indexes that stayed at 99 points, frozen-ID
> validation, and a direct-evidence baseline of Hit@1 `0.167`, Hit@3 `0.367`, Hit@5 `0.400`,
> MRR@5 `0.269`, MRR@20 `0.298`, p50 `15.14 ms`, p95 `37.77 ms`. The API image built at 544 MB,
> has no Docling import, and its health endpoint passed. Local `.venv` remains Anaconda Python
> 3.13.5 because no usable CPython 3.11 executable was discoverable; it was not overwritten.
> The ingestion image now includes the native `libxcb1`, `libgl1`, and `libglib2.0-0t64` runtime
> libraries required by Docling. Its 9.57 GB image built successfully and containerized batch-4
> ingestion produced 99 chunks/points. Only host Python 3.11 validation remains pending.

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
| Phase 3A.2 | Implementation complete; integration pending | Direct-evidence evaluation, 30-query development set, dependency split and Docker stabilization; real Docker/Qdrant validation pending engine availability |
| Phase 4 | Chưa bắt đầu | Sparse vectors, hybrid retrieval và explicit RRF |
| Phase 5 | Chưa bắt đầu | Multilingual cross-encoder reranking |
| Phase 3B | Tạm hoãn | Query API, LangChain, OpenAI, citations và abstention |
| Phase 6 | Chưa bắt đầu | End-to-end evaluation và production hardening |

Thứ tự triển khai đã chốt:

```text
Phase 3A.2 → Phase 4 → Phase 5 → Phase 3B → Phase 6
```

Phase 3B giữ tên theo roadmap lịch sử nhưng được triển khai sau Phase 4–5, vì retrieval phải
đủ tốt trước khi đưa evidence vào LLM.

---

## 3. Các phase đã hoàn thành

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

## 4. Các phase tiếp theo

## Phase 3A.2 — Dense baseline closure and Docker stabilization

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

## Phase 4 — Hybrid retrieval and RRF

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

---

## Phase 5 — Multilingual cross-encoder reranking

### Mục tiêu

Rerank hybrid candidate pool bằng query-document joint scoring để đưa evidence trực tiếp lên đầu.

### Thiết kế

- Dùng FastEmbed `TextCrossEncoder`.
- Model: `jinaai/jina-reranker-v2-base-multilingual`, có trong FastEmbed 0.8.
- Input gồm query và candidate text có heading breadcrumb.
- Rerank 20 hybrid candidates, trả final top 5.
- Không thay đổi chunk raw text hoặc payload.
- `RetrievalCandidate` bổ sung optional rerank score/rank.
- Giữ dense, sparse, RRF và rerank scores để benchmark/debug.
- Rerank score không được mô tả là probability.
- Model dùng persistent FastEmbed cache, không bake vào image.
- Reranker failure trả lỗi rõ ràng, không âm thầm fallback sang dense.

### Tests

- Candidate order theo fake cross-encoder scores.
- Deterministic ties.
- Candidate metadata và citation fields không bị mất.
- Empty candidate list và model trả sai số lượng scores.
- Reranker exception handling.
- Document filter vẫn được giữ từ retrieval stage.
- Unit tests không tải model thật.

### Acceptance gate

- Critical direct-evidence top-5 đạt 3/3.
- Hit@5 không giảm so với hybrid.
- Hit@1 hoặc MRR cải thiện ít nhất 10% tương đối so với hybrid.
- p95 hybrid + reranking dưới 1.5 giây trên CPU local.
- Real-model integration smoke pass trên 99 chunks.

---

## Phase 3B — Query API, LangChain, OpenAI, citations and abstention

Phase này chỉ bắt đầu khi Phase 4 và Phase 5 qua acceptance gate.

### Mục tiêu

Sinh câu trả lời grounded từ reranked evidence, trả citation kiểm chứng được và từ chối khi evidence
không đủ.

### Dependencies and provider

- `langchain-core` và `langchain-openai`.
- OpenAI Responses API thông qua LangChain.
- Default `OPENAI_MODEL=gpt-5.6-terra`; cho phép override bằng môi trường.
- Default reasoning effort: `low`.
- Không dùng agent, tools, LangChain retriever, OpenAI file search hoặc conversation memory.
- Retrieval, RRF, reranking, citations và evaluation vẫn là code explicit của project.

### API contract

Endpoint:

```text
POST /api/v1/query
```

Request:

```text
question: non-empty string
document_id: optional string
top_k: optional integer, 1–10, default 5
```

Response:

```text
answer: string
abstained: boolean
abstention_reason: optional string
citations: list[Citation]
```

Citation:

```text
chunk_id
document_id
filename
page_numbers
headings
excerpt
```

### Query flow

```text
validate request
→ hybrid retrieval
→ RRF
→ cross-encoder rerank
→ evidence gate
→ labeled evidence blocks
→ LangChain/OpenAI structured generation
→ citation ID validation
→ deterministic citation builder
→ QueryResponse
```

### Prompt, citation and abstention rules

- Answer bằng ngôn ngữ của câu hỏi và chỉ dùng supplied evidence blocks.
- Non-abstained answer phải cite ít nhất một source ID.
- Model output dùng typed structured output; không parse free-form citation text.
- Citation object build từ Qdrant payload, không nhận filename/page do LLM tự tạo.
- Excerpt lấy từ chunk text và bị giới hạn độ dài.
- Không gọi LLM nếu retrieval không có result hoặc evidence gate không đạt threshold đã tune.
- LLM có thể trả `insufficient_evidence`.
- Unknown source ID hoặc answer không có citation gây validation failure.
- Cho phép đúng một correction retry; nếu vẫn sai thì abstain với `citation_validation_failed`.
- Citation không được trỏ tới chunk ngoài retrieved context.

### HTTP behavior

- Invalid request: 422.
- Qdrant hoặc OpenAI unavailable: 503.
- Valid abstention: 200.
- Không log API key hoặc toàn bộ evidence text.

### Tests

- Request/response schema, empty question và invalid top-k.
- Grounded answer với valid citations.
- Retrieval-gate abstention không gọi LLM.
- LLM-declared abstention.
- Invalid citation correction retry và fallback abstention.
- Qdrant/OpenAI failure mapping.
- Fake LLM trong unit tests; không cần API key hoặc internet.

---

## Phase 6 — End-to-end evaluation and production hardening

### Evaluation dataset

- Tối thiểu 20 answerable và 10 unanswerable questions.
- Bao gồm tiếng Việt và tiếng Anh.
- Answerable item có relevant chunk IDs, expected phrases và citation expectations.
- Unanswerable item phải thực sự không có evidence trong indexed manual.

### Metrics

- Retrieval Hit@k, MRR và candidate recall.
- Answer key-phrase accuracy.
- Citation validity và citation coverage.
- Abstention precision và recall.
- Dense, sparse, fusion, rerank, LLM và total latency.
- OpenAI input/output/cached token usage.

### Release gate

```text
Valid citation IDs: 100%
Answer key-phrase accuracy: >= 0.85
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

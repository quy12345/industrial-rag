# Industrial RAG deep dive — hiểu kiến trúc, Qdrant và cách trình bày khi phỏng vấn

Tài liệu này giải thích repository từ góc nhìn của người cần tiếp quản, debug và trình bày lại trong
phỏng vấn. Nội dung bám theo code hiện tại, không mô tả những chức năng chưa tồn tại.

## 1. Project này thực sự đang làm gì?

Repository hiện có một RAG query path end-to-end ở mức correctness/demo, nhưng chưa production-ready:

1. Đọc PDF/DOCX bằng Docling.
2. Chia tài liệu thành các chunk có cấu trúc và metadata trang/heading.
3. Biến mỗi chunk thành dense vector và BM25 sparse vector.
4. Lưu vector cùng payload vào Qdrant.
5. Tìm candidate bằng dense, sparse hoặc hybrid RRF.
6. Có thể dùng multilingual cross-encoder để rerank candidate.
7. Gate evidence trước khi gửi dữ liệu ra provider.
8. Dùng OpenAI Responses hoặc Gemini OpenAI-compatible structured output để sinh answer và danh sách source IDs.
9. Validate source IDs và dựng citation từ trusted Qdrant payload.
10. Trả grounded answer hoặc explicit abstention qua `POST /api/v1/query`.
11. Đo retrieval/reranking bằng qrels trực tiếp và lưu artifacts JSON.

Cách trình bày chính xác khi phỏng vấn là:

> Đây là RAG tài liệu kỹ thuật có ingestion, immutable retrieval baselines, multilingual reranking,
> grounded structured generation và trusted citations. Tôi xây retrieval/evaluation trước rồi mới
> nối LLM, vì một câu trả lời trôi chảy không thể sửa evidence sai hoặc không được retrieve.

Đây là một quyết định kỹ thuật hợp lý: retrieval sai nhưng câu trả lời nghe trôi chảy là failure nguy
hiểm đối với tài liệu công nghiệp.

## 2. Qdrant là gì và mang ý nghĩa gì trong project?

### 2.1 Vai trò khái niệm

Qdrant là search database chuyên lưu và tìm kiếm vector. Trong project này, nó đóng vai trò
`retrieval index`: nhận vector của câu hỏi và trả về những chunk gần nhất hoặc phù hợp nhất.

Một Qdrant point có ba nhóm dữ liệu:

```text
Point
├── id: UUID ổn định
├── vectors
│   ├── dense: 384 số thực
│   └── sparse: indices + values, chỉ có ở collection v2
└── payload
    ├── chunk_id
    ├── document_id
    ├── filename
    ├── text
    ├── page_numbers
    ├── headings
    ├── content_type
    ├── source_path
    └── character_count
```

Vector dùng để tìm kiếm. Payload dùng để trả nội dung, filter theo tài liệu và chuẩn bị citation.

### 2.2 Qdrant làm gì?

- Lưu dense vector và sparse vector theo tên.
- Tính độ tương đồng cosine cho dense retrieval.
- Thực hiện sparse/IDF matching cho BM25 representation.
- Filter server-side theo `document_id`.
- Trả ranked points và payload.
- Persist dữ liệu qua Docker named volume `qdrant_storage`.

### 2.3 Qdrant không làm gì?

- Không đọc PDF.
- Không tự chia chunk.
- Không tự quyết định chunk nào là ground truth.
- Không chạy RRF trong implementation hiện tại; RRF được tính client-side bằng Python.
- Không chạy cross-encoder reranker.
- Không sinh câu trả lời.
- Không chứa model weights; model nằm trong `fastembed_cache`.
- Không chứa các JSON report trong `artifacts/`.

### 2.4 Vì sao không chỉ dùng file JSONL?

JSONL phù hợp để kiểm tra, freeze và tái tạo dữ liệu, nhưng không phải search index hiệu quả. Nếu chỉ
dùng JSONL, ứng dụng phải đọc tất cả 99 chunks và tự tính score cho từng query. Với nhiều tài liệu,
chi phí này tăng tuyến tính và khó filter/index đồng thời.

Qdrant cung cấp một abstraction cho vector search, sparse search, filtering và persistence. Ở quy mô
99 chunks, lợi ích hiệu năng chưa nổi bật; ý nghĩa lớn hơn là project đã có đúng data contract để mở
rộng lên nhiều manual mà không phải thay toàn bộ retrieval layer.

Một mental model nên nhớ:

| Thành phần | Vai trò |
|---|---|
| `data/raw/manual.pdf` | Tài liệu nguồn gốc |
| `artifacts/manual-batched.jsonl` | Frozen/reproducible chunk source |
| Qdrant | Derived online search index, có thể build lại |
| Metrics artifacts | Bằng chứng một lần benchmark |
| `fastembed_cache` | Model files, không phải dữ liệu nghiệp vụ |

Vì vậy Qdrant quan trọng đối với runtime search nhưng không nên là bản sao duy nhất của tài liệu.
Nếu Qdrant volume hỏng, project phải có khả năng tái tạo index từ raw/frozen source và manifests.

### 2.5 Hai collection có phải dữ liệu thừa không?

Hiện có hai collection:

| Collection | Nội dung | Mục đích |
|---|---|---|
| `industrial_manual_chunks` | dense vector `dense` | Frozen Phase 3 baseline, còn dùng cho union |
| `industrial_manual_chunks_v2` | dense `dense` + sparse `sparse` | Phase 4 sparse/hybrid retrieval |

Hai collection giống payload và stable point ID nhưng khác schema. Đây là duplication có chủ ý:

- không destructive-migrate baseline cũ;
- có thể regression-test dense v1;
- so sánh v1 và v2 công bằng;
- union Phase 5 lấy dense từ v1 và sparse từ v2;
- rollback dễ hơn.

Trong production lâu dài có thể hợp nhất sau migration được kiểm chứng. Ở giai đoạn benchmark hiện
tại, xóa v1 hoặc v2 sẽ làm mất khả năng so sánh và vi phạm frozen contract.

## 3. Bức tranh tổng thể

```text
OFFLINE / ON-DEMAND INGESTION

manual.pdf
    ↓ validate path, page range, batch plan
Docling DocumentConverter
    ↓ HierarchicalChunker
DocumentChunk[]
    ├── stable chunk_id
    ├── text
    ├── headings
    └── pages/metadata
    ↓
    ├── dense passage embedding ───────────────┐
    └── BM25 sparse embedding ────────────────┤
                                               ↓
                                  Qdrant v1 / v2 points

ONLINE QUERY API / SEARCH CLIs

question
    ├── query dense embedding → Qdrant dense top 20 ───┐
    └── query sparse embedding → Qdrant sparse top 20 ─┤
                                                        ├── sparse only
                                                        ├── RRF hybrid top 20
                                                        └── dense ∪ sparse
                                                                  ↓
                                                    cross-encoder rerank
                                                                  ↓
                                                     final ranked chunks
                                                                  ↓
                                               evidence gate + S1…Sn
                                                                  ↓
                                    provider structured grounded answer
                                                                  ↓
                                   validate IDs → trusted citations/abstain
```

Có hai loại flow cần phân biệt:

- Ingestion/indexing là offline hoặc on-demand, nặng và chỉ chạy khi tài liệu thay đổi.
- Retrieval/reranking/generation là query-time, chạy mỗi khi người dùng hỏi.

## 4. Data contract quan trọng nhất

### 4.1 `DocumentChunk`

`DocumentChunk` là output chuẩn hóa của ingestion. Tất cả retrieval strategy dùng cùng một chunk
contract. Các field chính:

- `chunk_id`: danh tính evidence ổn định;
- `document_id`: danh tính tài liệu dựa trên filename + file content;
- `text`: nội dung raw dùng để hiển thị/citation;
- `page_numbers`: trang nguồn;
- `headings`: breadcrumb cấu trúc;
- `content_type`: text/table/list/code/mixed/unknown;
- `metadata`: source path, chunk index, character count.

### 4.2 Vì sao stable ID quan trọng?

Metric không thể dựa vào thứ tự chunk vì re-index có thể đổi thứ tự. Project tạo:

```text
document_id = normalized filename + SHA-256 file content prefix
chunk_id    = hash(document_id, pages, headings, normalized text, duplicate occurrence)
point_id    = UUIDv5(namespace, chunk_id)
```

Kết quả:

- cùng file và cùng chunk content tạo cùng ID;
- re-index không nhân bản point;
- qrels có thể trỏ trực tiếp đến chunk;
- nếu content thay đổi, chunk nhận ID mới và point cũ trở thành stale;
- Qdrant nhận UUID hợp lệ nhưng project vẫn giữ `chunk_id` dễ đọc trong payload.

### 4.3 Ba model dữ liệu retrieval

| Model | Khi nào dùng | Điểm khác nhau |
|---|---|---|
| `DocumentChunk` | Sau ingestion, trước indexing | Chưa có score/rank |
| `RetrievedChunk` | Dense search v1 | Một `score` dense đơn giản |
| `RetrievalCandidate` | Sparse/hybrid/rerank | Giữ dense/sparse/RRF/rerank score và rank riêng |

Không nên gọi các score này là probability. Chúng chỉ là ranking signals và không có chung scale.

## 5. Luồng ingestion chi tiết

Code chính: `app/ingestion.py`.

### Bước 1 — Validate input

`validate_input_path` chỉ chấp nhận file tồn tại có extension `.pdf` hoặc `.docx`, không phân biệt
chữ hoa/thường.

### Bước 2 — Tạo document ID

`build_document_id` đọc file content, tính SHA-256 prefix và kết hợp slug filename. Vì ID phụ thuộc
content nên hai file cùng tên nhưng khác nội dung không bị coi là một document.

### Bước 3 — Chia PDF thành page batches

PDF có thể được xử lý theo batch 4 trang để giảm peak memory. Manual hiện tại tạo các range:

```text
1-4, 5-8, 9-12, 13-16, 17-20, 21-21
```

DOCX không hỗ trợ page range hoặc page batch vì khái niệm trang của DOCX không ổn định như PDF.

### Bước 4 — Docling conversion

`_convert_document` lazy-import Docling để base/API không cần cài Docling. Với PDF:

- OCR tắt;
- layout/table/OCR batch size đều để 1;
- chỉ chấp nhận `ConversionStatus.SUCCESS`;
- `PARTIAL_SUCCESS` bị từ chối để không index tài liệu thiếu trang.

Sau conversion, `HierarchicalChunker` tạo chunk dựa trên cấu trúc tài liệu thay vì cắt cứng theo số
ký tự.

### Bước 5 — Normalize

Project lấy text, page provenance, heading breadcrumb và content type từ Docling. Chunk rỗng bị bỏ.
Duplicate chunk content được phân biệt bằng `occurrence_index` nhưng vẫn deterministic.

### Bước 6 — JSONL preview

`write_chunks_jsonl` ghi UTF-8 theo cơ chế atomic replace: ghi file tạm cùng thư mục trước, chỉ
replace output khi serialization hoàn tất. Điều này tránh để lại JSONL dở nếu tiến trình lỗi.

### Hạn chế ingestion cần nhớ

- OCR scanned PDF chưa bật.
- Heading context không được nối lại qua page-batch boundary.
- Multi-page table có thể bị chia.
- Thay batch size có thể thay chunk set và metric.
- Docling rất nặng, vì vậy chỉ nằm trong ingestion image/extra.

## 6. Dense indexing và search

Code chính: `app/retrieval.py`.

### 6.1 Passage embedding

Mỗi chunk được format thành:

```text
Section: Heading > Subheading
Content:
raw chunk text
```

Model dùng `passage_embed` cho document và `query_embed` cho question. Đây là điểm thường được hỏi:
bi-encoder có thể có pathway/prompt khác nhau cho query và passage; dùng đúng API giữ contract của
model.

Model hiện tại:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
dimension = 384
distance = cosine
```

### 6.2 Dense point trong Qdrant

V1 point có dạng logic:

```python
PointStruct(
    id=uuid5(chunk_id),
    vector={"dense": [384 floats]},
    payload={...citation-ready metadata...},
)
```

Project không lưu absolute local path vào payload; nếu source path là absolute, nó thu về filename để
tránh leak đường dẫn máy.

### 6.3 Safe re-index

Thứ tự hiện tại:

1. Embed toàn bộ incoming chunks thành point batches trong memory.
2. Validate/create collection schema.
3. Đọc point IDs cũ theo từng document.
4. Upsert point mới với stable IDs.
5. Chỉ sau upsert mới xóa stale IDs của document đó.

Ưu điểm: embedding fail thì collection chưa bị xóa; re-index cùng input không tăng point count.

Nuance khi phỏng vấn: đây không phải distributed transaction hoàn toàn. Nếu một upsert batch giữa
chừng fail, code không rollback các batch mới đã thành công, nhưng nó chưa xóa stale points. Thiết kế
ưu tiên không làm mất dữ liệu cũ; muốn production-grade hơn có thể index vào staging collection rồi
atomically switch alias.

### 6.4 Dense query

`dense_search`:

1. Normalize và validate question.
2. Gọi `query_embed`.
3. Gọi Qdrant `query_points(... using="dense")`.
4. Optional filter `document_id` được thực hiện trong Qdrant, không filter sau khi retrieve.
5. Không kéo vector về; chỉ lấy payload và score.
6. Validate payload rồi map thành `RetrievedChunk`.

## 7. Sparse BM25 và hybrid RRF

Code chính: `app/hybrid_retrieval.py`.

### 7.1 Sparse/BM25 giải quyết vấn đề gì?

Dense retrieval mạnh ở paraphrase và cross-lingual semantic similarity, nhưng có thể bỏ sót mã thiết
bị, thuật ngữ chính xác, số và đơn vị. BM25 ưu tiên term overlap hiếm, nên thường mạnh trên manual kỹ
thuật.

BM25 về ý tưởng sử dụng term frequency, inverse document frequency và normalization theo độ dài:

```text
score(term, document)
  ≈ IDF(term) × TF saturation(k) × length normalization(b, |D| / avg_len)
```

Project dùng:

- model `Qdrant/bm25`;
- `k=1.2`, `b=0.75`;
- disable stemmer để không áp English stemming sai lên corpus tiếng Việt;
- `avg_len=72.838384`, tính bằng đúng tokenizer/preprocessing của FastEmbed trên 99 chunks.

Sparse vector chỉ chứa những term index xuất hiện và trọng số tương ứng, thay vì 384 phần tử dense.
Qdrant áp dụng IDF modifier trong collection v2.

### 7.2 Vì sao không cộng dense score và BM25 score?

Hai score khác scale và ý nghĩa. Cộng trực tiếp cần calibration và dễ để một strategy lấn át strategy
còn lại. Project dùng Reciprocal Rank Fusion:

```text
RRF(chunk) = Σ 1 / (60 + component_rank)
```

RRF dùng rank thay vì raw score. Nếu chunk xuất hiện ở cả dense và sparse, nó nhận hai contribution.
Tie-break deterministic theo best component rank rồi chunk ID.

### 7.3 Vì sao hybrid có thể kém sparse?

Fusion không bảo đảm tốt hơn từng component. Nếu dense đưa nhiều semantic candidate chưa đủ trực tiếp,
RRF có thể đẩy một sparse evidence tốt xuống dưới top 5. Benchmark thực tế xác nhận sparse Phase 4
Hit@5 `0.633`, còn hybrid chỉ `0.533`.

Đây là bài học phỏng vấn quan trọng:

> Hybrid là hypothesis cần benchmark, không phải keyword đảm bảo improvement.

## 8. Candidate audit và reranking

### 8.1 Candidate recall khác Hit@5

Trước khi rerank cần biết evidence có nằm trong pool hay không:

- Hit@5 hỏi evidence đã được xếp trong 5 kết quả đầu chưa.
- Candidate recall hỏi evidence có xuất hiện ở bất kỳ vị trí nào trong pool reranker nhìn thấy không.

Nếu evidence không có trong pool, reranker không thể tạo ra evidence mới.

`app/candidate_audit.py` tạo và kiểm tra bốn pool:

- dense top 20;
- sparse top 20;
- hybrid RRF top 20;
- dense top 20 union sparse top 20.

Module này không thừa so với evaluator: nó đo upper bound trước reranking và giải thích RRF demotion.
Nó cũng được `app/reranking.py` tái sử dụng để convert/deduplicate candidates.

### 8.2 Cross-encoder khác bi-encoder thế nào?

Dense bi-encoder embed query và passage riêng, sau đó so vector. Nó nhanh vì passage vector đã được
index trước.

Cross-encoder đọc query và candidate text cùng lúc:

```text
[query, candidate text] → relevance score
```

Nó hiểu tương tác token chi tiết hơn nhưng phải inference một lần cho từng candidate, nên chậm hơn.

Phase 5 dùng `jinaai/jina-reranker-v2-base-multilingual` qua FastEmbed `TextCrossEncoder`.
Model lazy-load, không download khi import và không bake vào image. License `CC-BY-NC-4.0` là giới
hạn quan trọng: benchmark/demo phi thương mại được, commercial deployment cần model/license khác.

### 8.3 Ba reranking strategy

| Strategy | Pool trước rerank | Kích thước |
|---|---|---:|
| `sparse` | sparse v2 top 20 | 7–20 trong benchmark |
| `hybrid` | dense+sparse v2 → RRF top 20 | 20 |
| `union` | dense v1 top 20 ∪ sparse v2 top 20 | 22–34 |

Union không truncate trước rerank để giữ candidate recall `0.933`. Candidate text là heading
breadcrumb, hai newline, rồi raw chunk text.

Reranker validate rất chặt:

- query không rỗng;
- chunk IDs unique;
- mỗi candidate có previous rank hợp lệ;
- output count bằng input count;
- candidate index đủ, unique và đúng range;
- score phải finite;
- model exception được wrap và không silent fallback.

Sort cuối:

```text
rerank score giảm dần
→ previous rank tăng dần
→ chunk ID tăng dần
```

### 8.4 Kết quả và cách diễn giải

| Strategy | Hit@5 | MRR@5 | Candidate recall | Warm p95 CPU |
|---|---:|---:|---:|---:|
| Sparse rerank | 0.733 | 0.529 | 0.867 | 9.880 s |
| Hybrid rerank | 0.767 | 0.546 | 0.867 | 8.466 s |
| Union rerank | 0.767 | 0.546 | 0.933 | 11.889 s |

Reranking cải thiện quality và đạt critical bilingual intents 3/3, nhưng latency fail mục tiêu 1.5
giây. Phase 6 chọn `union` làm accuracy-first API default và ghi rõ trade-off; sparse/no-rerank là
rollback khi latency quan trọng hơn quality.

Một câu trả lời phỏng vấn tốt:

> Cross-encoder cải thiện ranking nhưng inference CPU quá chậm. API demo chọn union vì accuracy,
> đồng thời có sparse rollback và không gọi hệ thống production-ready. Bước tiếp theo là profile,
> quantization/model nhỏ hơn hoặc batching/concurrency benchmark, đồng thời xử lý license thương mại.

### 8.5 Query API, evidence gate và trusted citations

`QueryService` giữ flow ngoài FastAPI route:

```text
request validation
→ lazy frozen runtime check
→ retrieve/rerank
→ cut final top_k
→ metadata/score evidence gate
→ S1…Sn untrusted evidence blocks
→ strict GeneratedAnswer
→ source-ID validation; tối đa một correction
→ citation metadata từ RetrievalCandidate
```

Điểm quan trọng để giải thích khi phỏng vấn:

- LLM chỉ được trả `answer`, `source_ids`, `insufficient_evidence`; nó không được tự tạo page hoặc
  filename.
- `S1` là label tạm cho request hiện tại. Backend giữ map `S1 → RetrievalCandidate` và chỉ map label
  đã cung cấp.
- No candidates/metadata invalid/score gate fail thì không gọi model. Threshold mặc định `None` vì
  chưa có held-out unanswerable set để calibrate.
- Unknown source ID hoặc grounded answer không citation bị reject. Correction retry dùng cùng
  evidence, không retrieve/rerank lần hai.
- Citation referential validity có nghĩa ID tồn tại trong evidence. Source có thực sự support từng
  claim là semantic correctness và phải đo ở Phase 7.
- Route chạy service sync trong threadpool. Model/retriever được lazy-cache, không khởi tạo theo mỗi
  request; health không cần API key hay model load.

## 9. Evaluation: project biết kết quả đúng bằng cách nào?

Code chính: `app/evaluation.py`; dataset: `data/eval/dense_smoke.jsonl`.

Mỗi query có:

- `question`;
- `language`;
- `relevant_chunk_ids`;
- `expected_phrases`;
- `expected_pages`;
- `category`;
- `critical`.

Ground truth duy nhất để tính Hit/MRR là `relevant_chunk_ids`:

```text
direct evidence rank = vị trí one-based đầu tiên có chunk_id thuộc qrels
```

Page và phrase chỉ dùng validate/diagnostic. Một chunk cùng trang nhưng không chứa direct evidence
không được tính hit.

### Metrics cần giải thích được

```text
Hit@5 = tỷ lệ query có relevant chunk trong top 5

RR(query) = 1 / first_relevant_rank
MRR@5     = trung bình RR, nhưng rank > 5 nhận 0

Candidate recall = tỷ lệ query có evidence trong toàn candidate pool

p50 = latency trung vị
p95 = 95% query có latency không vượt quá giá trị này
```

Development set hiện có 30 câu, 15 VI và 15 EN→VI. Nó dùng để phát triển/tuning, không phải held-out
Phase 7 final test set. Nói metric mà không nói đây là development set sẽ dễ gây hiểu nhầm.

### Vì sao artifact lớn?

Report lưu cả overall metrics và từng query, candidate IDs, ranks, pages, headings, scores, failure
class và latency stages. Điều này cho phép điều tra một metric thay vì chỉ có một con số tổng hợp.

## 10. Ý nghĩa từng module trong `app/`

| Module | Trách nhiệm | Ai gọi | Đánh giá audit |
|---|---|---|---|
| `app/__init__.py` | Đánh dấu/document package | Python import system | Giữ; không phải module thừa |
| `app/config.py` | Settings từ env/.env, validation và default contracts | API và mọi CLI | Giữ; single source of configuration |
| `app/models.py` | Pydantic contracts cho health/chunk/result/candidate/query/citation | Toàn bộ layers | Giữ; shared domain model |
| `app/main.py` | FastAPI app, health và query router | Uvicorn/API image | Giữ; composition root mỏng |
| `app/ingestion.py` | Docling conversion, page batching, stable IDs, normalization, JSONL | Preview/index CLIs | Giữ; ingestion core |
| `app/retrieval.py` | Dense model, Qdrant v1 schema/index/search/manifest | Dense, hybrid và rerank flows | Giữ; dense retrieval core |
| `app/hybrid_retrieval.py` | BM25, Qdrant v2, sparse search, RRF, manifest | Hybrid/audit/rerank CLIs | Giữ; hybrid core |
| `app/evaluation.py` | Typed qrels, frozen validation, direct metrics | Evaluators/audit/rerank | Giữ; correctness boundary |
| `app/candidate_audit.py` | Candidate coverage, union, RRF diagnosis | Audit và reranking | Giữ; không trùng evaluator |
| `app/reranking.py` | Lazy cross-encoder, pool preparation, rerank, latency/failures | Phase 5 CLIs và API runtime | Giữ; reranking core |
| `app/retrieval_runtime.py` | Frozen live-Qdrant checks, union/sparse composition, lazy cache | QueryService và Phase 5 wrapper | Giữ; production composition boundary |
| `app/generation.py` | Evidence blocks, prompt, strict Responses adapter và token usage | QueryService | Giữ; LLM boundary |
| `app/citations.py` | Source-ID validation và deterministic trusted citations | QueryService | Giữ; grounding boundary |
| `app/query_service.py` | Retrieve/gate/generate/retry/respond orchestration | Query route và smoke CLI | Giữ; application service |
| `app/api/query.py` | Threadpool handoff và HTTP error mapping | FastAPI | Giữ; route không chứa business logic |

Điểm dependency tốt: import/startup không tạo FastEmbed/OpenAI model và không import Docling. Health
endpoint vẫn chạy khi Docling hoặc API key không có; heavy runtime chỉ được dựng sau query hợp lệ.

## 11. Ý nghĩa từng script

Scripts là composition roots: chúng ghép các hàm trong `app/` thành command chạy thật. Business logic
nên ở `app/`, còn argument parsing, print và artifact output ở `scripts/`.

| Script | Dùng khi nào | Tạo/thay đổi gì |
|---|---|---|
| `ingest_preview.py` | Kiểm tra Docling chunks trước index | Có thể ghi JSONL artifact; không đụng Qdrant |
| `index_document.py` | Index dense v1 | Upsert v1 và ghi dense manifest |
| `index_hybrid.py` | Index dense+sparse v2 | Upsert v2 và ghi hybrid manifest |
| `search_dense.py` | Debug dense baseline | Read-only Qdrant query |
| `search_hybrid.py` | Debug sparse/hybrid | Read-only Qdrant query |
| `evaluate.py` | Benchmark dense/sparse/hybrid | Ghi baseline/regression JSON |
| `audit_candidate_pools.py` | Đo pre-rerank candidate coverage | Ghi candidate audit JSON |
| `generate_phase5_readiness.py` | Freeze handoff Phase 4.1 | Tổng hợp manifests, metrics, Git/Docker state |
| `rerank_runtime.py` | Dùng chung cách dựng/validate runtime Phase 5 | Không phải user CLI; composition helper |
| `search_reranked.py` | Debug một query qua reranker | Read-only Qdrant + model inference |
| `evaluate_reranking.py` | Benchmark ba rerank strategy | Ghi strategy reports và comparison |
| `validate_query_runtime.py` | Smoke frozen union/sparse runtime | Read-only Qdrant/model; không gọi OpenAI |
| `query_smoke.py` | Bounded real-provider smoke | Ghi sanitized Phase 6 JSON hoặc `not_run` |
| `validate_phase6.sh` | Canonical Python 3.11 one-shot check | Cài dev deps rồi chạy Ruff/pytest |

Ba search CLI nhìn giống nhau nhưng đang có giá trị chẩn đoán: mỗi CLI cô lập đúng contract của phase.
Query API tái sử dụng core functions qua `retrieval_runtime`; xóa các CLI sẽ làm regression và
interview demo khó quan sát hơn.

## 12. Tests đang bảo vệ điều gì?

| Test file | Boundary được bảo vệ |
|---|---|
| `test_health.py` | API scaffold hoạt động |
| `test_ingestion.py` | Stable IDs, batch ranges, Docling statuses, metadata, atomic JSONL |
| `test_retrieval.py` | Dense schema, vectors, payload, filters, re-index safety, manifests |
| `test_hybrid_retrieval.py` | Sparse IDF schema, BM25 length, v2 safety, RRF correctness |
| `test_evaluate.py` | Strict qrels, direct-evidence metrics, same-page false-positive prevention |
| `test_candidate_audit.py` | Union dedup, coverage và RRF diagnosis |
| `test_reranking.py` | Indexed model output, ordering, failures, pool construction, no eager model |
| `test_retrieval_runtime.py` | Lazy composition, frozen settings/hash, no retrieval/rerank fallback |
| `test_generation.py` | Evidence security/bounds, Responses kwargs, structured output, refusal/errors |
| `test_citations.py` | Referential validity, ordering, Unicode excerpt và trusted metadata |
| `test_query_service.py` | Gate, retry, abstention, timings, no sensitive logs |
| `test_query_api.py` | Request contract và sanitized HTTP mappings |

Default tests dùng fake embedding/cross-encoder/generator và local in-memory Qdrant. Chúng không tải
model, gọi Docker/Qdrant/provider thật hoặc cần API key. Phase 6 canonical Python 3.11 suite có 160
tests; provider/UTF-8 delta đưa local suite lên 162 và adapter Gemini đã được construct trong API
image Python 3.11. Real model/provider smokes là explicit integration commands.

## 13. Docker và dependency split

### Base/API dependencies

- FastAPI, Uvicorn, Pydantic settings.
- API target cài thêm retrieval và LLM dependencies.
- Không cài Docling.

### Retrieval dependencies

- `qdrant-client[fastembed]`.
- FastEmbed/ONNX phục vụ dense, sparse và cross-encoder.

### Ingestion dependencies

- Docling và các OS shared libraries cho PDF/image processing.
- Ingestion target kế thừa retrieval runtime.

### LLM dependencies

- `langchain-core` cho prompt/typed runnable contract.
- `langchain-openai` cho Responses API; không cài full LangChain meta-package.
- Chỉ API/dev nhận extra này; ingestion target không nhận.

### Compose services

| Service | Vai trò | Khởi động mặc định? |
|---|---|---|
| `qdrant` | Persistent vector/search database | Có |
| `api` | FastAPI health + grounded query runtime | Có |
| `ingestion` | Tool nặng chạy on-demand | Không; profile `tools` |

Named volumes:

- `qdrant_storage`: points, collection schema và indexes;
- `fastembed_cache`: model weights tải lúc runtime.

`data/raw` được bind read-only vào ingestion. Raw manual và artifacts không được bake vào image.

API image chứa retrieval + LLM dependencies vì query endpoint dùng cả hai, nhưng health không tạo
model. Shared FastEmbed volume chỉ được mount runtime; image inspection xác nhận không bake weights.

## 14. Audit code thừa và duplication

### 14.1 Item chưa nằm trên Phase 6 runtime path

| Item | Bằng chứng | Khuyến nghị |
|---|---|---|
| `rerank_candidate_strategy` | Phase 5 CLI vẫn bắt strategy rõ ràng; API dùng `retrieval_strategy` | Giữ compatibility rồi deprecate có tài liệu |
| `rerank_final_limit` | Phase 5 experiment setting; API dùng request `top_k` | Giữ compatibility hoặc bỏ trong cleanup riêng |

Không còn placeholder module nào trên query path. Hai setting cũ không điều khiển API; người vận hành
phải dùng `RETRIEVAL_STRATEGY`, `RERANK_ENABLED` và request `top_k`.

Các script `generate_phase5_readiness.py` và `audit_candidate_pools.py` không nằm trên query runtime
path sau khi Phase 5 hoàn tất. Chúng là phase-specific reproducibility tools, không phải dead code.
Có thể chuyển vào `scripts/archive/` hoặc một namespace `scripts/benchmarks/` khi roadmap ổn định,
nhưng hiện artifacts và walkthrough vẫn tham chiếu trực tiếp đến chúng.

### 14.2 Duplication có thể refactor

Các helper sau lặp ở nhiều script:

- `_write_json_atomic`;
- `_configure_output_encoding`;
- `_positive_int`;
- `_chunk_ids_by_document`;
- `_validate_frozen_ids`;
- package/runtime version collection.

Có thể gom thành `app/artifacts.py`, `app/runtime_validation.py` và `scripts/_cli_common.py`. Tuy nhiên
đây là cleanup, không phải correctness bug. Refactor chỉ nên làm kèm regression tests vì artifact
format và error messages đang là contract chẩn đoán.

`app/evaluation.py` và `app/reranking.py` cũng cùng có `_hit_rate`, `_mrr`, latency aggregation và
result summaries. Có thể tách metric primitives chung, nhưng hai evaluator có semantics khác nhau:

- retrieval evaluator đo một ranked list và một latency;
- rerank evaluator phân biệt pre-pool/final rank, failure class và stage latency.

Do đó không nên gộp toàn bộ evaluator thành một class lớn.

### 14.3 Architectural smell nhưng không phải unused code

`app/hybrid_retrieval.py` import nhiều private helper từ `app/retrieval.py`, ví dụ `_build_payload`,
`_document_filter`, `_batched` và `_scroll_document_point_ids`. Điều này cho thấy các helper thực chất
là shared indexing primitives nhưng đang mang tên private.

Refactor hợp lý:

```text
app/qdrant_store.py
├── payload mapping
├── document filter
├── point scrolling
└── safe upsert helpers

app/dense_retrieval.py
app/hybrid_retrieval.py
```

Không nên xóa helper; nên đổi ownership/public boundary ở phase cleanup.

### 14.4 Những thứ trông trùng nhưng phải giữ

- V1 và v2 collections: baseline preservation và union dependency.
- Dense và hybrid manifests: hai schema contract khác nhau.
- Dense/sparse/hybrid/rerank artifacts: historical evidence, không phải runtime source.
- Ba search CLI: strategy-specific smoke/debug.
- Candidate audit và evaluator: upper-bound coverage khác final ranking quality.
- `app/__init__.py`: package marker/documentation.
- Per-phase walkthrough docs: decision history và reproducibility.

## 15. Điểm mạnh kỹ thuật của project

1. Ground truth dùng stable direct-evidence chunk IDs, không dùng cùng trang để tính hit.
2. Dense baseline được giữ immutable khi thêm hybrid schema.
3. Stable UUIDv5 point IDs và document-scoped stale deletion.
4. Embedding hoàn tất trước stale deletion, giảm nguy cơ mất index khi model fail.
5. Query/passages dùng đúng FastEmbed API riêng.
6. Raw dense và BM25 score không bị cộng sai scale; fusion dùng RRF.
7. Candidate recall được đo trước khi rerank.
8. Reranker strict validation, deterministic ties, không silent fallback.
9. Unit tests offline và integration benchmark tách biệt.
10. Docling dependency không bị kéo vào API import path.
11. Runtime artifacts lưu đủ fingerprint, metrics, per-query failures và latency methodology.
12. Kết quả xấu được giữ trung thực: hybrid kém sparse và reranker fail latency vẫn được báo cáo.
13. Evidence gate chạy trước provider; source IDs được validate và citation metadata không do LLM tạo.
14. Missing key không làm hỏng health và không kích hoạt model download.

## 16. Điểm yếu và việc cần làm trước production

1. Query API đã có nhưng real Responses smoke chưa chạy vì thiếu API key.
2. Chỉ benchmark một PDF nghiên cứu tiếng Việt 21 trang và development set 30 câu; chưa phải corpus
   manual công nghiệp đại diện.
3. Chưa có held-out Phase 7 end-to-end evaluation hoặc calibrated unanswerable set.
4. OCR tắt; scanned PDF không hoạt động.
5. Page batching làm mất heading/table continuity qua boundary.
6. Re-index chưa có collection alias/staging transaction.
7. Phase 7.5 đưa Jina calibration p95 từ 13.399 s xuống 6.996 s bằng batch size 8 và ngân sách 30,
   nhưng CPU-only latency vẫn quá chậm cho production và license vẫn phi thương mại.
8. Chưa có authentication, rate limit, background jobs hoặc multi-tenant isolation.
9. API health chưa kiểm tra Qdrant/model readiness sâu.
10. Chưa có observability production: structured logs, traces, metrics exporter.
11. Hai Phase 5 rerank settings cũ không nằm trên API path và cần cleanup/deprecation sau.
12. Một số shared helpers đang lặp hoặc bị import dưới tên private.
13. Qdrant `depends_on` chưa phải readiness health gate; service có thể start trước khi Qdrant ready.
14. Chưa có concurrency/load benchmark hoặc memory budget cho model.

## 17. Cách đọc repository trong 90 phút

### 0–10 phút: contract và trạng thái

Đọc theo thứ tự:

```text
AGENTS.md
README.md
docs/plan.md
pyproject.toml
app/config.py
app/models.py
```

Mục tiêu: biết frozen data, dependency, collection names và phase hiện tại.

### 10–25 phút: ingestion

Đọc `app/ingestion.py`, rồi `scripts/ingest_preview.py` và `scripts/index_document.py`. Tự trả lời:

- Stable document/chunk ID được tạo từ gì?
- Tại sao batch PDF?
- Khi Docling partial success thì sao?
- Metadata nào đi vào Qdrant?

### 25–45 phút: storage và retrieval

Đọc `app/retrieval.py`, rồi `app/hybrid_retrieval.py`. Trace:

```text
DocumentChunk → embedding → PointStruct → Qdrant
question → query vector → query_points → payload model
```

Tự tính một ví dụ RRF với dense rank 2 và sparse rank 5:

```text
1/(60+2) + 1/(60+5)
```

### 45–60 phút: evaluation

Mở một dòng `data/eval/dense_smoke.jsonl`, tìm `relevant_chunk_ids` trong
`artifacts/manual-batched.jsonl`, rồi đọc `direct_evidence_rank` và `evaluate_cases`.

### 60–70 phút: reranking

Đọc `candidate_audit.py`, `reranking.py` và `rerank_runtime.py`. Phân biệt:

- candidate miss;
- reranker miss top 5;
- evidence hit;
- union coverage với final ranking.

### 70–82 phút: query/generation/citations

Đọc `retrieval_runtime.py`, `query_service.py`, `generation.py`, `citations.py`, rồi
`api/query.py`. Trace một valid source ID và một unknown source qua correction retry.

### 82–90 phút: integration và tests

Đọc Dockerfile/Compose, sau đó mapping mỗi module với test file tương ứng. Mở comparison artifact để
liên hệ code với số đo thật.

## 18. Cách trace một query khi có lỗi

Giả sử query trả answer hoặc evidence sai:

1. Xác nhận `document_id` filter đúng.
2. Chạy `search_dense.py`, ghi dense IDs/ranks.
3. Chạy `search_hybrid.py` ở sparse và hybrid, so component ranks.
4. Kiểm tra qrel ID có xuất hiện trong candidate pool không.
5. Nếu không xuất hiện: retrieval/candidate miss; reranker không thể sửa.
6. Nếu xuất hiện nhưng sau top 5: ranking/fusion/reranker issue.
7. Mở payload để kiểm tra pages/headings/text.
8. Kiểm tra manifest có khớp model/schema/frozen hash không.
9. Không dùng page match để tự kết luận hit.
10. Nếu candidate đúng nhưng answer sai, kiểm tra gate, evidence labels và structured source IDs.
11. Unknown/missing source phải bị correction hoặc abstain; page/filename phải đến từ payload.
12. Chạy evaluator/smoke để lưu failure trong artifact thay vì sửa qrel theo output.

## 19. Các câu hỏi phỏng vấn và câu trả lời gợi ý

### “Tại sao dùng Qdrant?”

> Tôi cần một vector/search database hỗ trợ named dense và sparse vectors, cosine search, IDF sparse
> modifier, payload filtering và persistence. JSONL được giữ làm reproducible frozen source, còn
> Qdrant là online retrieval index. Qdrant không parse tài liệu hay sinh answer.

### “Tại sao cần cả dense và BM25?”

> Dense bắt semantic/paraphrase và multilingual similarity; BM25 mạnh với exact term, model code,
> number và unit. Nhưng hybrid không mặc định tốt hơn: trên development set này sparse mạnh hơn RRF
> hybrid, nên tôi benchmark thay vì giả định.

### “Tại sao dùng RRF thay vì cộng score?”

> Cosine score và BM25 score khác scale. RRF fusion theo rank nên không cần calibration raw score.
> Công thức là tổng `1/(k+rank)` với rank one-based và k=60.

### “Tại sao reranker chậm hơn dense?”

> Dense bi-encoder index passage trước và chỉ embed query một lần. Cross-encoder đọc từng cặp
> query-candidate tại runtime, chất lượng cao hơn nhưng inference cost tăng theo pool size.

### “Làm sao re-index mà không duplicate?”

> Chunk ID content-based, Qdrant point ID là UUIDv5 của chunk ID. Upsert cùng input ghi đúng point cũ.
> Code embed trước, upsert point mới, rồi xóa stale IDs trong phạm vi document.

### “Làm sao đánh giá retrieval đúng?”

> Tôi dùng manually verified `relevant_chunk_ids`. Hit@k/MRR chỉ tính stable ID; phrase và page chỉ
> validate hoặc debug. Dataset hiện là development set, không được gọi là final held-out test.

### “Kết quả nào không đạt và bạn xử lý sao?”

> Phase 4 hybrid kém sparse ở top rank. Phase 5 reranker cải thiện Hit@5/MRR và critical 3/3 nhưng
> CPU p95 8–12 giây, vượt gate 1.5 giây. Phase 6 dùng union làm accuracy-first demo default nhưng
> giữ sparse/no-rerank rollback và không gọi đó là production-ready. Model còn có license
> non-commercial.

### “Nếu mở rộng lên hàng triệu chunks?”

> Tôi sẽ benchmark Qdrant index/search params, shard/replicate phù hợp, dùng payload index cho tenant
> và document filters, chuyển ingestion sang job workers, dùng staging collection + aliases, batch
> embedding, cache model, và load-test p95/p99. Qrels phải mở rộng theo nhiều manual/domain.

### “Điều gì xảy ra nếu Qdrant mất?”

> Search không hoạt động, nhưng index có thể tái tạo từ raw documents/frozen chunks và manifests.
> Production cần backup/snapshot Qdrant volume, readiness checks và documented restore procedure.

### “Project có phải production-ready RAG chưa?”

> Chưa. Query/generation/citation/abstention đã có, nhưng real provider chưa smoke vì thiếu key và
> vẫn thiếu
> held-out semantic citation/abstention evaluation, real industrial corpus, commercial reranker
> decision, latency optimization, auth và production observability.

## 20. Bài trình bày project trong 90 giây

> Tôi xây dựng RAG query pipeline cho tài liệu kỹ thuật song ngữ. PDF được
> Docling parse và hierarchical chunk, mỗi chunk có stable content-based ID, page và heading metadata.
> Tôi index dense multilingual vectors vào Qdrant v1; sau đó thêm collection v2 chứa cả dense và
> BM25 sparse vectors mà không phá baseline cũ. Dense và sparse được fusion client-side bằng RRF,
> không cộng raw scores khác scale. Evaluation dùng 30 manually verified direct-evidence qrels; chỉ
> chunk ID được tính Hit/MRR, page chỉ là diagnostics. Benchmark cho thấy sparse mạnh hơn hybrid,
> nên tôi không giả định hybrid luôn tốt. Tôi audit candidate recall rồi thử multilingual
> cross-encoder trên sparse, hybrid và union pools. Reranking đạt critical intents 3/3 và Hit@5 0.767,
> nhưng CPU p95 8–12 giây và model có license non-commercial. Query API dùng union accuracy-first,
> gate evidence, gọi generation provider với strict structured output, validate source IDs và dựng
> citation từ Qdrant payload; sparse/no-rerank là rollback. Offline/Docker correctness pass 160 tests,
> còn real OpenAI smoke chưa chạy vì thiếu key. Phase 7 phải đo semantic citations, abstention trên
> held-out set và giải quyết latency/license trước production.

## 21. Checklist tự kiểm tra kiến thức

Bạn nên trả lời được mà không mở code:

- Qdrant point gồm những phần nào?
- V1 và v2 khác nhau ra sao, vì sao chưa xóa v1?
- `document_id`, `chunk_id`, `point_id` được tạo thế nào?
- Passage và query embedding khác API nào?
- Dense, sparse, RRF và cross-encoder khác nhau ở đâu?
- Vì sao không cộng cosine với BM25 score?
- Candidate recall khác Hit@5 thế nào?
- Qrel nào mới được tính direct hit?
- Safe re-index bảo vệ được gì và chưa atomic ở điểm nào?
- Model cache khác Qdrant volume và artifacts thế nào?
- Module nào không nằm trên Phase 6 runtime path?
- LLM được phép điều khiển field nào, field citation nào phải do backend dựng?
- Evidence gate và correction retry ngăn failure nào?
- Vì sao project đã có complete query flow nhưng vẫn chưa production-ready?
- Metric nào đạt, latency/license nào chưa đạt?
- Bạn sẽ làm gì trước khi đưa project vào production?

Nếu trả lời trôi chảy các câu này và trace được một query qua scripts/artifacts, bạn đã hiểu phần lớn
kiến trúc hiện tại thay vì chỉ nhớ command chạy.

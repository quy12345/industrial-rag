

## 1. Tổng quan

Dự án là phần khung (scaffold) của một hệ thống **RAG (Retrieval-Augmented Generation)** cho tài liệu kỹ thuật công nghiệp (hướng dẫn vận hành/bảo trì dạng PDF, DOCX).

Ở giai đoạn hiện tại (**Phase 2 — ingestion preview**), dự án chỉ thực hiện:

- Chuyển đổi PDF/DOCX thành văn bản có cấu trúc bằng thư viện [Docling](https://github.com/docling-project/docling).
- Chunk văn bản theo cấu trúc (headings, đoạn, bảng...) bằng `HierarchicalChunker` của Docling.
- Chuẩn hóa metadata thành model Pydantic JSON-serializable (`DocumentChunk`).
- Cung cấp CLI preview và xuất JSONL.
- API chỉ có endpoint `/api/v1/health`.

Chưa có: embeddings, Qdrant, retrieval, gọi LLM, LangChain (đều nằm trong roadmap).

## 2. Cấu trúc thư mục

```text
app/                  Logic ứng dụng FastAPI + ingestion
  __init__.py         Khai báo package
  config.py           Cấu hình từ biến môi trường / .env
  models.py           Model Pydantic (request/response, chunk)
  main.py             Entry point FastAPI
  ingestion.py        Toàn bộ pipeline chuyển đổi + chunk + chuẩn hóa
  retrieval.py        Placeholder (chưa triển khai)
  generation.py       Placeholder (chưa triển khai)
scripts/
  ingest_preview.py   CLI chạy ingestion và in/xuất preview
  evaluate.py         Placeholder (chưa triển khai)
tests/
  test_health.py      Test endpoint health
  test_ingestion.py   Test logic ingestion + CLI
data/
  raw/                Thư mục chứa tài liệu đầu vào
  eval/               Thư mục dữ liệu đánh giá (rỗng)
artifacts/            JSONL preview, metrics, figures (đầu ra)
.github/workflows/ci.yml  CI: Ruff + pytest
Dockerfile            Image API container
docker-compose.yml    API + Qdrant (Qdrant chưa được dùng)
pyproject.toml        Metadata, dependencies, cấu hình Ruff/pytest
.env.example          Mẫu biến môi trường
```

## 3. Công nghệ sử dụng

| Thành phần | Vai trò |
|---|---|
| Python >= 3.11 | Ngôn ngữ |
| FastAPI | Web framework (API) |
| Uvicorn | ASGI server |
| Pydantic v2 + pydantic-settings | Model & cấu hình có kiểu |
| Docling (>=2.117,<2.118) | Chuyển đổi tài liệu + chunk theo cấu trúc |
| pypdfium2 | Đếm số trang PDF (phụ thuộc của Docling) |
| pytest | Test |
| Ruff | Lint |
| Docker / docker-compose | Container hóa API + Qdrant |

## 4. Chi tiết từng module

### 4.1 `app/config.py` — Cấu hình

- Lớp `Settings(BaseSettings)` đọc cấu hình từ biến môi trường và file `.env`.
- Các trường: `app_name`, `app_version`, `environment`, `api_prefix` (mặc định `/api/v1`), `qdrant_url`, `qdrant_port`.
- `model_config` cấu hình `env_file=".env"`, `extra="ignore"` (bỏ qua biến lạ).
- Hàm `get_settings()` dùng `@lru_cache` để chỉ khởi tạo cấu hình **một lần** cho toàn tiến trình.

### 4.2 `app/models.py` — Model dữ liệu

- `HealthResponse`: phản hồi endpoint health với `status` là literal `"ok"`.
- `DocumentChunk`: biểu diễn một chunk đã chuẩn hóa, gồm:
  - `chunk_id`: ID duy nhất, ổn định.
  - `document_id`: ID của tài liệu gốc.
  - `filename`, `text`: tên file và nội dung văn bản.
  - `page_numbers`: danh sách trang nguồn (từ provenance Docling).
  - `headings`: "breadcrumb" tiêu đề tới chunk.
  - `content_type`: nhãn nội dung bảo thủ (`text`, `table`, `list`, `code`, `mixed`, `unknown`).
  - `metadata`: dict phụ (`source_path`, `file_extension`, `chunk_index`, `character_count`).

### 4.3 `app/main.py` — FastAPI entry point

- Tạo ứng dụng `FastAPI(title=..., version=...)`.
- Khởi tạo `settings = get_settings()` ở module level.
- Endpoint `GET {api_prefix}/health` trả về `HealthResponse(status="ok", service="industrial-rag", version=...)`.

### 4.4 `app/ingestion.py` — Pipeline ingestion (module chính)

Toàn bộ luồng: **kiểm tra đầu vào → xác định khoảng trang/batch → chuyển đổi Docling → chunk → chuẩn hóa → (tùy chọn) xuất JSONL**.

Các hàm công khai:

- `validate_input_path(file_path)`: kiểm tra tồn tại, là file, đuôi nằm trong `SUPPORTED_EXTENSIONS = (".pdf", ".docx")`. Ném `IngestionError` nếu sai.
- `build_document_id(file_path)`: tạo ID ổn định từ tên file (chuẩn hóa Unicode NFKD, loại bỏ ký tự lạ, slug hóa) + 12 ký tự đầu của SHA-256 nội dung file. Ví dụ: `motor-drive-manual-a4f832bd71c2`. Cùng nội dung → cùng ID; đổi nội dung → đổi ID.
- `build_chunk_id(document_id, page_numbers, chunk_index)`: dạng `{document_id}_p{trang_đầu}_c{index:04d}`; nếu không có trang → `punknown`.
- `build_page_batches(start_page, end_page, batch_size)`: chia khoảng trang (inclusive) thành các batch theo trang (page-aligned). Ví dụ `(1,21,8)` → `[(1,8),(9,16),(17,21)]`.
- `get_pdf_page_count(file_path)`: đếm trang PDF bằng `pypdfium2`, tự đóng tài liệu trong `finally`.
- `ingest_document(file_path, *, page_range=None, batch_size=None)`: hàm chính:
  1. Validate đầu vào.
  2. Giải quyết danh sách khoảng chuyển đổi qua `_resolve_conversion_ranges`.
  3. Tính `document_id`.
  4. Với mỗi khoảng: `_convert_document` → `_append_normalized_chunks`; sau đó `del` + `gc.collect()` để giảm bộ nhớ.
  5. Nếu không có chunk nào → ném `IngestionError`.
- `write_chunks_jsonl(output_path, chunks)`: ghi JSONL UTF-8 **nguyên tử** — ghi vào file tạm trong cùng thư mục rồi `replace` lên file đích; nếu lỗi thì xóa file tạm, giữ nguyên file cũ. Đảm bảo không có artifact "nửa vời".

Các hàm nội bộ (`_` prefix):

- `_resolve_conversion_ranges`: chỉ cho phép `page_range`/`batch_size` với PDF (DOCX → `[None]`); kiểm tra batch > 0, validate range so với số trang thực tế.
- `_validate_page_range`: kiểm tra `start >= 1`, `end >= start`, `end <= page_count`.
- `_convert_document`: tạo `DocumentConverter`:
  - PDF: dùng `PdfPipelineOptions(do_ocr=False, ocr_batch_size=1, layout_batch_size=1, table_batch_size=1)` để tiết kiệm bộ nhớ (PDF kỹ thuật số có text layer, không cần OCR). Truyền `page_range` nếu có.
  - DOCX: converter mặc định.
  - Kết quả được `_validate_conversion_result`, rồi `list(HierarchicalChunker().chunk(...))`.
  - Mọi lỗi không phải `IngestionError` được bọc lại với ngữ cảnh khoảng trang.
- `_validate_conversion_result`: **chỉ chấp nhận `ConversionStatus.SUCCESS`**. `PARTIAL_SUCCESS`/`FAILURE` bị từ chối (tránh xuất output không hoàn chỉnh), kèm chi tiết lỗi.
- `_conversion_error_details` + `_compact_page_numbers` + `_format_page_span`: tổng hợp trang lỗi thành dải gọn (`9-12, 15`) và tối đa 3 thông điệp lỗi duy nhất.
- `_append_normalized_chunks`: với mỗi chunk thô:
  - Bỏ chunk text rỗng.
  - Trích `page_numbers` (sorted, unique) từ provenance các item.
  - Trích `headings` (breadcrumb, bỏ trùng liên tiếp).
  - `chunk_index` được đánh **toàn tài liệu** (liên tục qua các batch).
  - Metadata: `source_path`, `file_extension`, `chunk_index`, `character_count`.
- `_extract_text` / `_extract_page_numbers` / `_extract_headings` / `_doc_items` / `_label_name`: các helper truy xuất thuộc tính của chunk Docling một cách an toàn (dùng `getattr`, `_as_sequence` để chống thiếu dữ liệu).
- `_infer_content_type`: heuristic bảo thủ:
  - Có nhãn `table` → `table` (nếu chỉ mỗi bảng) hoặc `mixed`.
  - Có `code` → `code` hoặc `mixed`.
  - Chỉ gồm `list`/`list_item` → `list`.
  - Chỉ gồm các nhãn văn bản (paragraph, heading, caption, equation...) → `text`.
  - Không có nhãn hoặc khác → `unknown`.
- `_source_path`: trả về đường dẫn ổn định không phụ thuộc máy — nếu là đường dẫn tuyệt đối trong thư mục làm việc thì đổi thành tương đối.

### 4.5 `app/retrieval.py`, `app/generation.py` — Placeholder

Chỉ có docstring, dành cho Phase 3+ (retrieval và sinh câu trả lời).

## 5. Scripts

### 5.1 `scripts/ingest_preview.py` — CLI chính

Chạy: `python scripts/ingest_preview.py data/raw/manual.pdf [options]`

- `main(argv)`: cấu hình UTF-8 cho stdout/stderr, parse args, gọi `ingest_document`, in summary + preview, tùy chọn ghi JSONL. Lỗi `IngestionError`/`OSError` → in ra stderr, trả về code 1.
- Các tham số CLI:
  - `input` (bắt buộc): đường dẫn PDF/DOCX.
  - `--limit N` (mặc định 10): số chunk in preview; `0` = không in chi tiết chunk.
  - `--preview-chars N` (mặc định 500): giới hạn ký tự text mỗi preview.
  - `--output PATH`: ghi JSONL (tự tạo thư mục cha).
  - `--page-start` / `--page-end`: phạm vi trang (inclusive), **phải dùng cùng nhau**.
  - `--batch-size N`: tách PDF thành các lần chuyển đổi tuần tự để giảm bộ nhớ đỉnh.
- `_parse_page_range`: báo lỗi nếu chỉ có một trong hai giới hạn trang.
- `_build_pdf_plan`: dựng kế hoạch batch để hiển thị (chỉ cho PDF).
- `_print_summary`: in document ID, tổng chunk, trang đại diện, phạm vi trang, số batch, phân bố content type.
- `_print_previews`: in thông tin chi tiết từng chunk.
- `_non_negative_int` / `_positive_int`: validator tham số.

### 5.2 `scripts/evaluate.py` — Placeholder

Chỉ có docstring, dành cho Phase 4 (đánh giá retrieval/answer).

## 6. Tests

### 6.1 `tests/test_health.py`

- Dùng `TestClient` của FastAPI kiểm tra `GET /api/v1/health` trả về 200 và đúng payload.

### 6.2 `tests/test_ingestion.py`

- `build_document_id`: tính xác định, phụ thuộc nội dung, phân biệt file khác nội dung, không chứa khoảng trắng, prefix đúng.
- `validate_input_path`: chấp nhận đuôi hoa thường (`.PDF`, `.DOCX`); báo lỗi khi thiếu file, là thư mục, hoặc đuôi không hỗ trợ.
- `build_chunk_id`: dùng trang nhỏ nhất, dùng `unknown` khi không có trang, khác nhau khi index khác.
- `build_page_batches`: các ca hợp lệ và bất hợp lệ (start < 1, end < start, batch ≤ 0).
- `_validate_conversion_result`: chấp nhận `SUCCESS`; từ chối `PARTIAL_SUCCESS`/`FAILURE` kèm chi tiết (trang lỗi, thông điệp).
- `write_chunks_jsonl`: giữ nguyên Unicode, đủ 8 trường; không ghi đè file cũ khi serialization lỗi, không để sót file `.tmp`.
- `ingest_document` (mocked `_convert_document`): chuẩn hóa metadata (text strip, page unique, headings dedupe, content type `table`, chunk_id, character_count).
- `ingest_document` (mocked theo batch): các batch `(1,8),(9,16),(17,21)` được gọi đúng, `chunk_index` toàn cục, chunk_id duy nhất.
- Các lỗi: page range vượt số trang; `batch_size` bị từ chối với DOCX.
- CLI: thiếu một trong hai page bound → exit 1; ingestion fail → không ghi file output.

## 7. Hạ tầng

### 7.1 `Dockerfile`

- Base `python:3.11-slim`.
- Tắt bytecode (`PYTHONDONTWRITEBYTECODE`), bật output ngay (`PYTHONUNBUFFERED`).
- Tạo user không root `appuser`, copy `pyproject.toml` + `README.md` + `app/`, `pip install .` (không cache).
- `CMD` chạy `uvicorn app.main:app` cổng 8000.

### 7.2 `docker-compose.yml`

- Dịch vụ `api`: build từ Dockerfile, map cổng 8000, đọc `.env` (không bắt buộc), `depends_on: qdrant`, mount thư mục hiện tại vào `/app`, chạy uvicorn `--reload` (dev).
- Dịch vụ `qdrant`: image `qdrant/qdrant:latest`, map cổng 6333/6334, volume `qdrant_storage`.
- Lưu ý: ở Phase 2 ứng dụng **chưa** kết nối Qdrant.

### 7.3 `.github/workflows/ci.yml`

- Chạy trên push/PR, OS `ubuntu-latest`, Python 3.11.
- Các bước: checkout → setup Python → upgrade pip → `pip install -e ".[dev]"` → `ruff check .` → `pytest`.

## 8. Cấu hình biến môi trường (`.env.example`)

| Biến | Mặc định | Mô tả |
|---|---|---|
| `APP_NAME` | Industrial Technical Manual RAG | Tên ứng dụng |
| `APP_VERSION` | 0.1.0 | Phiên bản |
| `ENVIRONMENT` | development | Môi trường chạy |
| `API_PREFIX` | /api/v1 | Tiền tố API |
| `QDRANT_URL` | http://qdrant | URL Qdrant (chưa dùng) |
| `QDRANT_PORT` | 6333 | Cổng Qdrant (chưa dùng) |

## 9. Luồng dữ liệu tổng quan

```text
data/raw/manual.pdf (.docx)
        │
        ▼
validate_input_path ──► extension check
        │
        ▼
build_document_id  (slug + sha256[:12])
        │
        ▼
_resolve_conversion_ranges  (page_range / batch_size → list of ranges)
        │
        ▼  (loop từng range)
_convert_document
   ├─ PdfPipelineOptions(do_ocr=False, batch=1)  [PDF]
   ├─ DocumentConverter.convert(page_range=...)
   └─ _validate_conversion_result (chỉ chấp nhận SUCCESS)
        │
        ▼
HierarchicalChunker().chunk(dl_doc)
        │
        ▼
_append_normalized_chunks  (text/page/headings/content_type/metadata → DocumentChunk)
        │
        ▼
ingest_document → list[DocumentChunk]
        │
        ├── CLI in summary + preview  (scripts/ingest_preview.py)
        └── write_chunks_jsonl (nguyên tử, UTF-8) → artifacts/*.jsonl
```

## 10. Những quyết định thiết kế đáng chú ý

1. **ID xác định dựa trên nội dung**: cùng nội dung → cùng document_id, cho phép tái chạy ingestion idempotent.
2. **Chỉ chấp nhận SUCCESS**: từ chối `PARTIAL_SUCCESS`/`FAILURE` để không tạo artifact không hoàn chỉnh; JSONL ghi nguyên tử bằng file tạm + `replace`.
3. **Batching theo trang để giảm bộ nhớ**: mỗi batch tạo converter mới và `gc.collect()` sau mỗi vòng; hạn chế tăng bộ nhớ đỉnh của Docling trên PDF lớn. Hạn chế: không nối heading context/table đa trang qua ranh giới batch.
4. **Content type bảo thủ**: chỉ gán `table`/`list`/`code` khi có bằng chứng nhãn Docling; ngược lại dùng `text`/`mixed`/`unknown`.
5. **`extra="ignore"` + `@lru_cache`**: cấu hình linh hoạt, khởi tạo một lần.
6. **Metadata không phụ thuộc máy**: `_source_path` đổi đường dẫn tuyệt đối trong workspace thành tương đối.

## 11. Roadmap (từ README)

1. Thêm embeddings và nạp vào Qdrant.
2. Thêm dense/hybrid retrieval.
3. Thêm reranking, sinh câu trả lời bằng LLM, và citations.
4. Thêm đánh giá retrieval và câu trả lời.

Các module `app/retrieval.py`, `app/generation.py`, `scripts/evaluate.py` là chỗ dành sẵn cho các phase này.

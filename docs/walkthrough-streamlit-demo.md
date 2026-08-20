# Streamlit demo walkthrough

## Mục đích

Streamlit là giao diện demo mỏng cho runtime Phase 7 ATV320. Nó chỉ gọi HTTP API và không import,
khởi tạo hoặc truy cập trực tiếp Qdrant, FastEmbed, Jina hay Gemini.

```text
Browser :8501
  -> Streamlit UI
  -> FastAPI POST /api/v1/query :8000
  -> Qdrant dense + BM25 sparse retrieval
  -> Jina multilingual reranking
  -> evidence gate
  -> Gemini structured generation
  -> validated citations
```

Hai manual đang được demo là ATV320 Installation Manual và ATV320 Programming Manual. Phase 7 dùng
hai collection frozen chứa 2.753 chunks; UI không upload hoặc re-index tài liệu.

## Cấu hình

Provider key chỉ nằm trong `.env` của API. Không đưa Gemini/OpenAI key vào Streamlit hoặc trình
duyệt.

```text
GENERATION_PROVIDER=gemini
GEMINI_API_KEY=YOUR_KEY
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_REASONING_EFFORT=minimal
GEMINI_TEMPERATURE=0
OPENAI_MAX_OUTPUT_TOKENS=800
OPENAI_TIMEOUT_SECONDS=60
OPENAI_MAX_RETRIES=1
OPENAI_STORE=false
```

Compose đặt `RETRIEVAL_PROFILE=phase7` cho API. Profile này chọn nguyên khối frozen contract:

```text
dense collection     industrial_manual_phase7_dense_v1
hybrid collection    industrial_manual_phase7_hybrid_v1
chunks               2753
dense/sparse limits  60/40
RRF k                40
reranker candidates  30
Jina batch size      8
exact-content dedup  enabled
```

UI nhận ba biến riêng:

```text
RAG_API_URL=http://api:8000/api/v1
RAG_API_TIMEOUT_SECONDS=180
RAG_API_AUTH_TOKEN=
```

`RAG_API_AUTH_TOKEN` chỉ cần khi API bật `API_AUTH_KEY`. Token được thêm vào request server-side;
không render lên trang. UI không tự retry `POST /query`, tránh gọi Gemini hai lần.

## Chạy bằng Docker

Tại repository root:

```powershell
docker compose --progress plain build api ui
docker compose up -d qdrant api ui
docker compose ps
```

Không cần activate `.venv` và không cần build ingestion image. Mở:

```text
http://localhost:8501
```

Kiểm tra từng lớp:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
Invoke-RestMethod http://localhost:8000/api/v1/ready
Invoke-WebRequest -UseBasicParsing http://localhost:8501/_stcore/health
```

`/health` chỉ chứng minh FastAPI đang sống. `/ready` đọc Qdrant và fail closed nếu collection count,
chunk hash hoặc frozen runtime contract không khớp Phase 7. Streamlit health chỉ chứng minh process UI
đang phục vụ HTTP.

## Chạy UI local

API/Qdrant vẫn có thể chạy bằng Docker, còn Streamlit chạy trong Python 3.11 local:

```powershell
python -m pip install -e ".[ui]"
$env:RAG_API_URL="http://localhost:8000/api/v1"
python -m streamlit run ui/streamlit_app.py
```

Nếu API bật bearer auth:

```powershell
$env:RAG_API_AUTH_TOKEN="YOUR_API_AUTH_KEY"
```

## Cách dùng giao diện

- Chọn tất cả manual, Installation Manual hoặc Programming Manual ở sidebar.
- Chọn `top_k` từ 1 đến 10; mặc định là 5 evidence chunks sau reranking/dedup.
- Nhập câu hỏi tiếng Việt hoặc tiếng Anh. Mỗi câu hỏi là request độc lập; lịch sử không được gửi lại
  cho API.
- Mở từng citation để xem filename, page, heading breadcrumb, raw excerpt, chunk ID và document ID.
- Nút xóa lịch sử chỉ xóa tối đa 20 lượt đang lưu trong Streamlit session.

Answer dùng Markdown an toàn; excerpt dùng text rendering, không thực thi HTML. Khi evidence thiếu,
provider từ chối hoặc citation validation thất bại, giao diện hiển thị abstention và reason thay vì
tạo câu trả lời không có nguồn.

## Lỗi thường gặp

| Hiện tượng | Ý nghĩa và cách kiểm tra |
|---|---|
| API offline | Chạy `docker compose up -d qdrant api` và kiểm tra `/health`. |
| Phase 7 corpus not ready | Kiểm tra `/ready`, Qdrant volume và đúng hai collection 2.753 points; không re-index tự động. |
| API authentication failed | `RAG_API_AUTH_TOKEN` phải bằng `API_AUTH_KEY` của API. |
| Gemini not configured | Đặt provider key trong `.env`, rồi recreate API; không đặt key ở UI. |
| Reranker unavailable | Kiểm tra shared `fastembed_cache` và API logs; first query có thể tải/khởi tạo model. |
| Provider timeout | UI chờ tối đa 180 giây, API provider timeout mặc định 60 giây; POST không tự retry. |
| Câu trả lời đầu chậm | Dense model và Jina được lazy-load; CPU reranking là bottleneck đã biết. |

Xem log có giới hạn:

```powershell
docker compose logs --tail 100 api ui
```

Code không log API key, full question, prompt, answer hoặc evidence. Không thêm các dữ liệu này vào
debug logging khi mở rộng demo.

## Dừng an toàn

```powershell
docker compose stop ui api qdrant
```

Lệnh này giữ Qdrant volume, collections, model cache, image và container. Không dùng `docker system
prune` hoặc `docker volume prune`.

## Giới hạn

- Streamlit là frontend demo bind vào localhost, không phải production frontend.
- Chưa có auth riêng, TLS, reverse proxy, rate limit hoặc multi-user persistence.
- Jina reranker có license CC-BY-NC-4.0; chưa được duyệt cho commercial deployment.
- Held-out v2 cuối giữ nguyên: Hit@5 `0.800`, deterministic fact accuracy `0.786`, citation validity
  `1.000`, total p95 `12.62 s`. UI không được dùng để tune hoặc thay đổi benchmark này.
- UI chỉ hiển thị citations đã được API validate; semantic claim support vẫn phụ thuộc quality của
  retrieval và generation.

## Validation record — 2026-08-19

```text
Python                         3.11.16 in API/UI images
Ruff                           PASS
pytest                         308 passed, 1 known third-party warning
git diff --check               PASS
docker compose config --quiet  PASS
API image build                PASS
UI image build                 PASS — 134.9 s final rebuild
API /health                    200 / ok
API /ready                     200 / ok
Streamlit /_stcore/health      200 / ok
Phase 7 dense collection       2753 points / frozen hash PASS
Phase 7 hybrid collection      2753 points / frozen hash PASS
VI Installation filter         PASS — 29 candidates, one document
EN Programming filter          PASS — 30 candidates, one document
API image size                 150,553,757 bytes
UI image size                  190,268,831 bytes
```

The real retrieval smoke was provider-free. Its observed reranker times were 15.01 s on the first
VI request and 7.17 s on the following EN request. Four final Streamlit-to-Gemini demo queries were
not sent because provider egress for those new questions and retrieved excerpts requires separate
explicit approval. Offline client/UI tests cover normal answers, abstentions, citations and error
rendering; this is not presented as a real-provider UI PASS.

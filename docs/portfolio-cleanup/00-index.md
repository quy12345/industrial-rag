# Vòng 1 — Portfolio Cleanup

> **Baseline sau R00 — 2026-08-22:** chỉ hai manual ATV320 thuộc Phase 7 được dùng cho runtime và
> portfolio hiện hành. `Settings` từ chối `phase6`; Phase 6 tools nằm trong
> `scripts/archive/phase6/`. Hai legacy Qdrant collections, destructive-write guards và Git history
> vẫn được bảo toàn. Các đoạn được ghi rõ là pre-R00 snapshot chỉ còn giá trị audit lịch sử.
>
> **Round 1 complete — 2026-08-24:** R01–R07 đã được triển khai và validate theo từng module.
> Audit bên dưới được giữ làm bằng chứng “before”; code và tài liệu module là trạng thái “after”.

## Trạng thái

Tài liệu này là cổng vào của audit, kiến trúc đích và completion record Vòng 1. Audit gốc ngày
2026-08-20 là read-only; R00 sau đó thực hiện baseline transition đã duyệt; R01–R07 đã hoàn tất
refactor giữ behavior của active Phase 7 runtime.

## Mục tiêu

Vòng 1 là refactor giữ nguyên behavior cho prototype Industrial Technical Manual RAG đang chạy
end-to-end. Kết quả mong muốn là một modular monolith dễ đọc và dễ giải thích khi phỏng vấn:

- dependency direction rõ từ inbound adapters đến application và domain/ports;
- Qdrant, Docling, FastEmbed/Jina và provider SDK nằm sau infrastructure adapters;
- production runtime không import evaluation/research utilities;
- một nguồn cấu hình canonical, với frozen retrieval profiles được xem là immutable contracts;
- FastAPI, CLI và Streamlit mỏng;
- script vận hành, script đánh giá và script lịch sử được phân loại;
- characterization tests khóa behavior trước mỗi thay đổi có rủi ro;
- README và module documentation phản ánh code thực tế;
- Git history tuyến tính, trung thực và có thể review theo từng responsibility.

Không viết lại repository từ đầu. Mỗi lượt implementation sau audit chỉ được làm một module đã
được duyệt hoặc một vertical slice nhỏ của module đó.

## Phạm vi Vòng 1

Bao gồm:

- bổ sung characterization tests;
- làm rõ package/module ownership;
- tách application orchestration khỏi framework và SDK;
- cô lập evaluation khỏi production import graph;
- hợp nhất ownership của configuration và composition;
- làm mỏng API, CLI, Streamlit;
- đổi tên hoặc di chuyển bằng compatibility shim có kế hoạch loại bỏ rõ ràng;
- archive script một lần sau khi xác nhận không còn là operational contract;
- sửa tài liệu và liên kết sai;
- giữ nguyên mọi runtime, data và public contract đã đóng băng.

## Ngoài phạm vi

Các việc sau thuộc **Deferred to Round 2**, không được lẫn vào structural refactor:

- thay embedding/reranker/generation model, pooling hoặc dimension;
- tune dense, sparse, BM25, RRF, query expansion, role prior hay reranker;
- thay candidate budgets, `top_k`, evidence threshold hoặc answer behavior;
- tối ưu latency, CPU/GPU, model cache hay Docker image lớn;
- đổi provider hoặc provider policy;
- re-index, thay Qdrant schema/collection hoặc tạo migration;
- sửa frozen chunks, expected facts, qrels hay expected answers;
- tune theo held-out v2 hoặc tạo benchmark v3;
- production deployment, microservices, message queue hoặc DI framework.

## Baseline được dùng trong kế hoạch

Repository có một active runtime contract và một historical archive contract:

1. **Phase 6 historical archive contract**: `manual-77d5dae4c2c5`, 99 chunks, chunk-ID hash
   `bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be`, dense/hybrid
   collections v1/v2 và 30 direct-evidence qrels.
2. **Phase 7 active Compose/demo contract**: hai manual ATV320, 2.753 chunks, chunk-ID hash
   `2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d`, hai collection
   Phase 7 và frozen retrieval/reranking profile trong source.

`Settings`, `.env.example` và Docker Compose đều chọn Phase 7. Historical Phase 6 identity không còn
trong production resolver; archive tools không phải supported runtime entrypoints.

Held-out v2 đã bị lộ và chỉ còn là regression benchmark. Audit không đọc raw held-out question,
answer hay evidence, không chạy evaluation và không dùng kết quả đó để đề xuất tuning.

## Thứ tự đọc

1. [01-current-state-audit.md](01-current-state-audit.md) — code đang hoạt động như thế nào,
   baseline, coverage, technical debt và bảng quyết định.
2. [02-target-architecture.md](02-target-architecture.md) — dependency direction, package tree,
   ownership và các invariants.
3. [03-round-1-roadmap.md](03-round-1-roadmap.md) — bảy module, vertical slices, tests, Git plan,
   acceptance criteria và stop conditions.

## Cách audit

Audit đã kiểm tra Git state, import graph, runtime entry points, Python/dependency configuration,
source, tests, scripts, Docker/Compose, README, walkthrough và benchmark metadata. Local `.env`, raw
manual PDFs, model caches, ignored private benchmark payload và raw held-out nội dung không được mở.
Không có network call, provider call, model download, Qdrant mutation, re-index hoặc Docker build.

Các con số metric trong tài liệu lịch sử chỉ được ghi nhận là **historical claims** của repository.
Lượt này không xác nhận lại chúng vì không chạy benchmark hay test suite.

## Kết luận ngắn

Round 1 đã tạo modular ownership rõ hơn, composition root canonical, production/evaluation boundary,
thin inbound adapters và script lifecycle có kiểm soát. Characterization tests giữ frozen runtime và
public behavior trong khi code được di chuyển theo vertical slices. Historical workflows được archive
thay vì xóa; README hiện tập trung vào active two-manual runtime. Không có thuật toán hoặc frozen data
nào được thay đổi để đạt kết quả này.

## Trạng thái sau closure

- R00–R07: `COMPLETE`.
- Phase 7 là active runtime; Phase 6 chỉ là historical archive.
- Eight top-level scripts là supported command surface; completed one-offs nằm trong archive.
- Round 2 chưa được bắt đầu và cần scope/approval riêng.

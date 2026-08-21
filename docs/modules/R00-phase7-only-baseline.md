# R00 — Phase 7-only baseline

## 1. Goal and scope

R00 chuyển product focus từ hai corpus sang duy nhất corpus gồm hai manual kỹ thuật ATV320. Module
được người dùng duyệt ngày 2026-08-21 và là prerequisite trước roadmap R01–R07.

Slice R00A đã chọn Phase 7 làm default runtime, khóa default mới bằng tests, và loại local source
`data/raw/manual.pdf`. Slice R00B chuyển hai supported smoke CLIs sang Phase 7 và bổ sung offline
contract tests. Phase 6 sau đó được chuyển dần vào historical archive trước khi bị loại khỏi
production profile selection.

Slice R00C archive ba CLI reranking Phase 5/6 khỏi operational script surface. Các CLI lịch sử vẫn
reproducible qua namespace `scripts.archive.phase6`, nhưng không còn bị hiểu là công cụ của runtime
Phase 7.

Slice R00D archive chuỗi Phase 4 retrieval evaluation, candidate-pool audit và Phase 5 readiness
handoff vào cùng namespace. Parser defaults, report contracts và fail-closed behavior được khóa bằng
offline characterization tests trước các slice cleanup tiếp theo.

Slice R00E archive hai direct-search adapters và validation shell của Phase 6. Phase 7 tiếp tục dùng
`QueryService`/`build_query_retriever`; các adapters lịch sử không được đổi thành đường query mới.

Slice R00F archive hai mutating indexing adapters của Phase 3/4. Phase 7 indexing không import các
CLI này mà dùng shared `app.retrieval`/`app.hybrid_retrieval` functions qua
`scripts/index_phase7_corpus.py`.

Slice R00G1 làm `phase7` thành giá trị runtime profile duy nhất, loại Phase 6 corpus identity khỏi
`app.retrieval_runtime`, và buộc mọi runtime builder nhận explicit frozen contract. Historical
reranking giữ contract riêng trong archive.

Slice R00H đồng bộ AGENTS, README, codebase guide và bốn planning documents. Pre-R00 audit cùng
Phase 3–6 walkthroughs được giữ làm historical records, nhưng canonical docs không còn trình bày
chúng như active runtime.

## 2. Position in the system

```text
environment/default Settings
  -> resolve_retrieval_runtime
  -> Phase 7 frozen contract
  -> readiness and query composition
  -> two ATV320 Qdrant collections
```

R00 không thay retrieval algorithm. Nó thay corpus/profile được chọn khi không có explicit override.

## 3. Relevant background concepts

A frozen retrieval profile là một nhóm atomic gồm corpus identity, collection names, models và
ranking parameters. Chỉ đổi `retrieval_profile`; resolver tiếp tục thay toàn bộ mutable-looking
fields bằng contract tương ứng để tránh cấu hình lai.

Historical archive nghĩa là code cũ còn được lưu để đọc hoặc reproduce có kiểm soát, nhưng không còn
là supported runtime profile, product target hoặc nguồn cho development/evaluation mới.

## 4. Input, output and contracts

Input của selection là `Settings.retrieval_profile`. Giá trị hợp lệ duy nhất là `phase7`; explicit
`phase6` bị Pydantic validation từ chối trước runtime composition.

Output Phase 7 giữ nguyên:

```text
documents:          2 ATV320 manuals
chunks:             2753
chunk-ID hash:      2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d
dense collection:   industrial_manual_phase7_dense_v1
hybrid collection:  industrial_manual_phase7_hybrid_v1
```

## 5. Step-by-step data flow

1. `Settings()` đọc env hoặc dùng default `phase7`.
2. `resolve_retrieval_runtime` chọn `PHASE7_RETRIEVAL_CONTRACT`.
3. Resolver ghi đè collection/model/budget fields bằng frozen values.
4. FastAPI readiness validate hai Phase 7 collections mà không load model/provider.
5. Query composition dùng cùng resolved settings/contract.

## 6. Responsibilities of changed files

| File | Responsibility trong các slice R00 |
|---|---|
| `app/config.py` | Chỉ chấp nhận Phase 7 và dùng active contract values làm defaults |
| `app/retrieval_runtime.py` | Sở hữu duy nhất active Phase 7 contract; yêu cầu explicit contract ở builders |
| `.env.example` | Thể hiện đúng active Phase 7 collections và frozen retrieval values |
| `tests/test_retrieval_runtime.py` | Khóa atomic Phase 7 resolution và từ chối Phase 6 profile |
| `tests/test_hybrid_retrieval.py` | Khóa Phase 7 collection default nhưng giữ generic manifest round-trip |
| `tests/test_health.py` | Khóa readiness default vào hai Phase 7 collections |
| `AGENTS.md` | Ghi approved scope, safety và thứ tự R00 trước R01 |
| `docs/portfolio-cleanup/00-index.md` | Cảnh báo planning cũ bị quyết định R00 supersede |
| `docs/portfolio-cleanup/01-current-state-audit.md` | Giữ audit snapshot và thêm post-R00 delta |
| `docs/portfolio-cleanup/02-target-architecture.md` | Khóa target vào một active Phase 7 contract |
| `docs/portfolio-cleanup/03-round-1-roadmap.md` | Cập nhật R01–R07 prerequisites sau R00 |
| `docs/CODEBASE.md` | Phản ánh active script/runtime surface sau archive |
| `data/raw/manual.pdf` | Local ignored Phase 6 source đã được xóa theo yêu cầu |
| `scripts/validate_query_runtime.py` | Resolve active Phase 7 contract trước khi build real retriever |
| `scripts/query_smoke.py` | Dùng hai ATV320 document IDs và Phase 7 sanitized artifact identity |
| `tests/test_phase7_operational_smoke.py` | Khóa CLI defaults, corpus IDs và provider-free missing-key path |
| `scripts/archive/phase6/rerank_runtime.py` | Sở hữu và truyền explicit historical Phase 6 rerank contract |
| `scripts/archive/phase6/search_reranked.py` | Giữ interactive historical reranked search ngoài active script surface |
| `scripts/archive/phase6/evaluate_reranking.py` | Giữ historical Phase 5 evaluation và comparison generation |
| `scripts/archive/phase6/README.md` | Ghi rõ purpose, invocation và safety boundary của archive |
| `tests/test_reranking.py` | Trỏ CLI characterization tests vào namespace archive mới |
| `scripts/archive/phase6/evaluate.py` | Giữ historical dense/sparse/hybrid retrieval evaluation |
| `scripts/archive/phase6/audit_candidate_pools.py` | Giữ historical candidate coverage audit |
| `scripts/archive/phase6/generate_phase5_readiness.py` | Giữ historical Phase 4-to-5 handoff generator |
| `tests/test_evaluate.py` | Trỏ evaluation CLI parser test vào namespace archive |
| `tests/test_phase6_archived_cli.py` | Khóa defaults và lỗi trước external runtime access của archive CLIs |
| `scripts/archive/phase6/search_dense.py` | Giữ direct dense search adapter dùng legacy settings |
| `scripts/archive/phase6/search_hybrid.py` | Giữ hybrid search adapter khóa vào frozen 99-chunk artifact |
| `scripts/archive/phase6/validate_phase6.sh` | Giữ historical dependency-install/test recipe nhưng không thực thi |
| `scripts/archive/phase6/index_document.py` | Giữ unsupported legacy dense indexing adapter |
| `scripts/archive/phase6/index_hybrid.py` | Giữ unsupported legacy hybrid indexing adapter và frozen guard |
| `README.md` | Đánh dấu các evaluation/reranking section là historical và dùng archive paths |

## 7. Important symbols

- `Settings.retrieval_profile`: active profile selector; default hiện là `phase7`.
- `resolve_retrieval_runtime`: atomic resolver, không thay thuật toán.
- `PHASE7_RETRIEVAL_CONTRACT`: source contract cho active corpus.
- `ARCHIVED_PHASE6_RETRIEVAL_CONTRACT`: historical identity chỉ tồn tại trong archive, không thuộc
  production runtime.

## 8. Before-and-after structure

Before:

```text
Settings default phase6
Compose explicit phase7
manual.pdf present locally
```

After current R00 slices:

```text
Settings default phase7
.env.example phase7
Compose phase7
manual.pdf absent locally
retrieval/query smoke CLIs target Phase 7
Phase 5/6 reranking CLIs live under scripts/archive/phase6
Phase 4 evaluation and Phase 5 handoff CLIs live under scripts/archive/phase6
Phase 6 direct-search and validation entrypoints live under scripts/archive/phase6
Phase 3/4 mutating indexing entrypoints live under scripts/archive/phase6
Settings accepts only phase7
production runtime contains only PHASE7_RETRIEVAL_CONTRACT
```

## 9. Design decisions and trade-offs

Legacy collections không bị xóa vì chúng là persisted historical evidence và người dùng yêu cầu giữ
an toàn. Phase 6 tools được archive theo slices trước khi production compatibility bị loại bỏ, giúp
mỗi bước có characterization tests mà không duy trì thêm một active runtime branch.

Các generic Settings defaults không phải corpus identity (`rerank_batch_size=16`, content dedup tắt,
`bm25_avg_len=None`) được giữ để tránh thay đổi public/tooling behavior. Phase 7 resolver tiếp tục
đóng băng runtime tương ứng ở batch 8, content dedup bật và BM25 average length của active corpus.

Các filename `manual.pdf` trong unit-test fixtures không đại diện raw source và được giữ lại; chúng
kiểm tra generic PDF behavior.

Ngày 2026-08-22, người dùng quyết định giữ `generate_phase5_readiness` ở trạng thái unsupported
historical archive. Vì vậy R00 ghi nhận lỗi `EvaluationError` bị rò ra ngoài nhưng không mở bug-fix,
không thêm compatibility guarantee và không xem CLI này là public product surface.

## 10. Tests and protected behavior

- Runtime default phải resolve Phase 7 atomically, kể cả khi caller truyền conflicting collection và
  budget values.
- Readiness mặc định phải validate đúng hai Phase 7 collections và chunk count 2753.
- Explicit Phase 6 phải bị configuration validation từ chối.
- Runtime builders phải nhận explicit contract; không còn implicit Phase 6 default.
- Generic hybrid manifest round-trip vẫn phải chấp nhận caller-supplied BM25 average length.
- Model/provider vẫn không được load trong readiness.
- Retrieval smoke phải truyền resolved Phase 7 contract vào builder thay vì dùng default argument
  lịch sử, và phải từ chối explicit Phase 6 trước khi construct retriever.
- Query smoke phải từ chối explicit Phase 6, chỉ dùng hai ATV320 document IDs, và không construct
  service/provider khi API key vắng mặt.
- Archived reranking parsers, manifest gates và runtime validation phải giữ nguyên behavior sau khi
  đổi namespace.
- Import archived CLI modules không được khởi tạo hoặc tải cross-encoder.
- Archived retrieval evaluation, candidate audit và readiness parsers phải giữ nguyên frozen input,
  output và strategy defaults.
- Missing local inputs của retrieval evaluation và candidate audit phải trả lỗi trước khi model hoặc
  Qdrant client được tạo.
- Dense/hybrid search parser defaults và bounded preview behavior phải giữ nguyên sau khi move.
- Hybrid search phải từ chối missing frozen chunks trước model/Qdrant access.
- Dense/hybrid indexing parser defaults và paired page-range validation phải giữ nguyên sau move.
- `index_phase7_corpus.py` phải tiếp tục phụ thuộc shared app functions, không phụ thuộc archive.

## 11. Commands and expected results

Focused commands dự kiến:

```text
python -m pytest -q tests/test_retrieval_runtime.py tests/test_health.py
python -m pytest -q tests/test_phase7_operational_smoke.py
python -m pytest -q tests/test_reranking.py
python -m pytest -q tests/test_evaluate.py tests/test_phase6_archived_cli.py
python -m ruff check app/config.py tests/test_retrieval_runtime.py tests/test_health.py
python -m ruff check .
python -m pytest -q
git diff --check
```

Host workspace không có Python 3.11 (`py` không tồn tại và `.venv` là Python 3.13.5). Full offline
pytest vì vậy được chạy thêm trong image ingestion hiện có dùng Python 3.11.15: repository và bốn
package pytest thuần Python được mount read-only, network và plugin autoload bị tắt, không cài hoặc
tải dependency. Cả Python 3.11.15 và host Python 3.13.5 đều PASS 323 tests với cùng một warning.
Focused và full Ruff cũng PASS; Ruff đọc `target-version = "py311"` từ `pyproject.toml`.

## 12. Small usage example

```python
from app.config import Settings
from app.retrieval_runtime import PHASE7_RETRIEVAL_CONTRACT, resolve_retrieval_runtime

settings, contract = resolve_retrieval_runtime(Settings())
assert contract is PHASE7_RETRIEVAL_CONTRACT
assert settings.qdrant_collection == "industrial_manual_phase7_dense_v1"
```

## 13. Common failures and debugging

- Nếu readiness tìm legacy collections, kiểm tra `RETRIEVAL_PROFILE` trong process environment.
- Nếu pytest bị ảnh hưởng local `.env`, kiểm tra `tests/conftest.py` vẫn disable env-file loading.
- Historical CLI cần legacy collections phải dùng explicit historical configuration theo archive
  notice; production `RETRIEVAL_PROFILE` không còn cung cấp compatibility.
- `RETRIEVAL_PROFILE=phase6` bị Settings validation từ chối. Hai smoke CLIs vẫn giữ defensive guard
  để không fallback nếu nhận malformed/injected settings object.
- Không sửa collection contents để giải quyết profile mismatch.

## 14. Current limitations

- Historical benchmark artifacts và Phase 3–6 walkthroughs được giữ nguyên làm evidence; chúng
  không phải active runbooks và không được dùng làm product claims mới.
- `app.phase7.PROTECTED_COLLECTIONS` cố ý giữ legacy collection names làm destructive-write guard;
  `app.evaluation.EvaluationCase` còn historical default document ID và sẽ được xử lý khi cô lập
  evaluation, không phải trong runtime-profile slice này.
- Historical `generate_phase5_readiness.main` không bắt `EvaluationError` từ frozen-chunk loader.
  CLI đã được người dùng xác nhận là unsupported archive; không có bug-fix dự kiến trong R00.
- Archived dense search không resolve frozen profile và archived hybrid search phụ thuộc trực tiếp
  `manual-batched.jsonl`; cả hai là unsupported historical adapters, không phải Phase 7 contracts.
- Archived indexing adapters có thể mutate Qdrant và historical manifests; chúng unsupported và
  không được chạy trong R00.
- Phase 3–6 walkthroughs giữ namespace command tại thời điểm lịch sử và không phải runbook hiện hành;
  archive README là nguồn invocation hiện tại cho mục đích forensic/reproduction có kiểm soát.
- Archived tools ngoài reranking vẫn đọc explicit legacy settings từ môi trường nếu được chạy; chúng
  là unsupported và không được dùng với product `.env.example`.

## 15. Self-check questions

1. Vì sao không xóa hai legacy Qdrant collections?
2. Default profile khác frozen profile contract ở điểm nào?
3. Vì sao generic test fixture tên `manual.pdf` không phải legacy corpus dependency?
4. Vì sao historical contract phải nằm trong archive thay vì production runtime?

## 16. Interview summary

Project ban đầu giữ Phase 6 làm local default để backward compatibility trong khi demo dùng Phase 7.
Khi product scope được thu hẹp về hai ATV320 manuals, R00 chuyển default một cách explicit và testable,
giữ persisted data an toàn, rồi archive compatibility theo slices thay vì xóa hàng loạt.

## 17. Validation results and proposed commit

Kết quả R00A–R00H:

```text
R00A/R00B focused Ruff                    PASS
R00C archived-script Ruff                 PASS
R00D archived-script Ruff                 PASS
R00E archived-search Ruff                 PASS
R00F archived-indexing Ruff               PASS
R00G1 Phase 7-only runtime Ruff           PASS
R00H canonical documentation checks       PASS
Full Ruff                                 PASS
Archived source/content comparison        PASS
Archived path and symbol scan             PASS
Operational smoke legacy-reference scan  PASS
Default/profile constants                PASS
manual.pdf absent; two ATV320 PDFs exist PASS
Changed Markdown local links              PASS
git diff --check                          PASS
R00G1 focused pytest on Python 3.13.5     PASS — 87 tests, 1 warning
Full pytest on Python 3.13.5              PASS — 323 tests, 1 warning
Python 3.11.15 syntax/core import         PASS — 103 files
Full pytest on Python 3.11.15             PASS — 323 tests, 1 warning
```

Các pytest PASS ở trên tương ứng đúng các command đã chạy; không suy diễn kết quả Python 3.11,
metric hay Qdrant state. Không có provider call, model download, re-index hoặc Qdrant mutation.
Conventional Commit đề xuất:

```text
refactor: make phase 7 the active corpus baseline
chore: archive phase 5 reranking utilities
chore: archive phase 4 retrieval evaluation tools
chore: archive phase 6 search utilities
chore: archive phase 6 indexing utilities
refactor: make phase 7 the only runtime profile
docs: close the phase 7-only baseline transition
```

Không commit trong slice nếu người dùng chưa yêu cầu.

## 18. Status

`COMPLETE` — R00A–R00G1 đã chuyển active product surface sang Phase 7, archive unsupported Phase
3–6 CLIs và loại production Phase 6 profile/contract. R00H đã đồng bộ canonical documentation và
toàn bộ validation bắt buộc, gồm full offline pytest trên target Python 3.11.15.

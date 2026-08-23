# R05 — Grounded query application boundaries

## 1. Goal and scope

R05 separates provider-neutral generation contracts, prompt policy, provider infrastructure, evidence
and citation policies, and query orchestration without changing grounded-answer behavior.

R05A, R05B, R05C1, and R05C2 are implemented. R05A moves structured generation records and the generator port to the domain,
moves immutable prompt text and deterministic evidence rendering to the application layer, and keeps
`app.generation` as a compatibility facade. R05B moves the concrete LangChain implementation to
infrastructure, gives it the provider-neutral canonical name `LangChainStructuredGenerator`, and
injects prompt policy from the composition root.
R05C1 moves deterministic evidence selection, cross-document exact-duplicate handling, and the
pre-generation evidence gate into one domain policy owner. Existing top-level imports remain valid.
R05C2 moves referential answer validation and trusted citation construction into a second domain
policy owner without changing correction feedback or public citation metadata.

R05A does not change prompt text, evidence selection, provider settings, generation retries, citation
validation, abstention, retrieval, or ranking.

## 2. Position in the system

```text
retrieved candidates
        ↓
app.domain.evidence selection and gate
        ↓
app.application.generation_prompt.format_evidence
        ↓ EvidenceBundle
app.domain.generation.AnswerGenerator
        ↓ GenerationResult
citation validation and trusted citation construction
        ↓
QueryResponse
```

The prompt policy consumes only already-selected candidates. The generation port exposes no
LangChain, OpenAI, Gemini, FastAPI, or Qdrant type.

```text
app.bootstrap
├── app.application.generation_prompt
└── app.infrastructure.generation.langchain_structured
                    ↓ implements
          app.domain.generation.AnswerGenerator
```

## 3. Relevant background concepts

A port is a provider-neutral interface owned by the code that needs the capability. `QueryService`
needs structured generation, so it depends on `AnswerGenerator`; provider adapters implement that
contract at the edge.

An `EvidenceBundle` contains rendered untrusted text and the authoritative mapping from ephemeral
labels such as `S1` to retrieved candidates. The model may return labels, but it never controls
citation metadata.

Prompt bytes are runtime behavior. Even punctuation or newline changes can affect provider output and
benchmark provenance, so R05A protects both prompt templates with SHA-256 characterization tests.

## 4. Input, output, and contracts

`format_evidence(candidates, max_chars)` accepts an ordered non-empty candidate sequence and returns:

- deterministic labels in candidate order;
- exact source metadata headers;
- content inside `<untrusted_document>` boundaries;
- a rendered string no longer than `max_chars`;
- the original candidate objects in `source_map`.

`AnswerGenerator.generate(...)` accepts a question, one `EvidenceBundle`, and optional validation
feedback. It returns `GenerationResult`, containing a validated `GeneratedAnswer` and optional
normalized `TokenUsage`.

`GeneratedAnswer` still forbids extra fields and retains exactly `answer`, `source_ids`, and
`insufficient_evidence`. Source IDs remain untrusted until citation validation.

Compatibility imports from `app.generation` resolve to the canonical records, port, prompt constants,
and evidence formatter. `LangChainStructuredGenerator` is the canonical concrete adapter.
`LangChainOpenAIGenerator` remains at the old path as a compatibility subclass with the historical
constructor; it injects the same frozen prompt policy automatically.

`select_evidence_candidates(...)` preserves input rank except when exact content occurs in multiple
documents. Query-role policy then chooses one representative and retains sanitized duplicate
provenance. `EvidenceGate.evaluate(...)` returns one of the established pre-generation reasons and
never performs provider or evaluation work.

`validate_generated_answer(...)` treats model-returned source IDs as untrusted labels, preserves their
first-occurrence order, and returns `ValidatedGeneration` only after referential checks.
`build_citations(...)` constructs public metadata exclusively from retrieved candidates in the
authoritative source map.

## 5. Step-by-step data flow

1. Domain evidence policy validates selector inputs and derives query role.
2. Exact cross-document duplicates collapse to one deterministic representative before `top_k`.
3. The evidence gate validates candidate metadata, requested document, finite score, and threshold.
4. `format_evidence` assigns `S1`, `S2`, and subsequent labels in existing candidate order.
5. Trusted candidate metadata forms fixed source headers.
6. Raw candidate text is wrapped as untrusted document content.
7. When necessary, the available content budget is distributed deterministically across sources.
8. `EvidenceBundle` retains the exact label-to-candidate mapping.
9. The adapter builds messages from frozen system and human templates.
10. The composition root injects those templates and the correction builder into infrastructure.
11. Infrastructure lazily constructs one LangChain structured-output model.
12. The generator returns provider-native structured output mapped to `GeneratedAnswer`.
13. Query orchestration validates model-returned labels against `EvidenceBundle.source_map`.
14. Public citations are built only from the authoritative retrieved candidates.

## 6. Responsibilities of changed files

- [`app/domain/generation.py`](../../app/domain/generation.py) owns provider-neutral generation DTOs,
  usage records, evidence bundle contract, and `AnswerGenerator` port.
- [`app/domain/evidence.py`](../../app/domain/evidence.py) owns selection errors/results, duplicate
  diagnostics, exact-content selection, metadata validation, and pre-generation gate decisions.
- [`app/domain/citations.py`](../../app/domain/citations.py) owns referential answer validation,
  validation-error ordering, source/chunk deduplication, and trusted citation construction.
- [`app/application/generation_prompt.py`](../../app/application/generation_prompt.py) owns immutable
  prompt templates, correction feedback text, evidence rendering, and truncation policy.
- [`app/generation.py`](../../app/generation.py) compatibility-exports canonical contracts/policy and
  preserves the historical adapter constructor through a small compatibility subclass.
- [`app/infrastructure/generation/__init__.py`](../../app/infrastructure/generation/__init__.py)
  identifies the outbound structured-generation adapter package without eager imports.
- [`app/infrastructure/generation/langchain_structured.py`](../../app/infrastructure/generation/langchain_structured.py)
  owns lazy SDK imports, provider configuration, structured parsing, normalized usage, and sanitized
  provider error mapping.
- [`app/bootstrap.py`](../../app/bootstrap.py) composes the canonical adapter with application-owned
  prompt policy and constructs the canonical domain `EvidenceGate`.
- [`app/query_service.py`](../../app/query_service.py) consumes the canonical generator port, usage
  record, application evidence formatter, and domain evidence policy while preserving old gate exports.
- [`app/evidence_selection.py`](../../app/evidence_selection.py) is a compatibility facade whose
  public selection symbols are direct aliases to the domain owner.
- [`app/citations.py`](../../app/citations.py) is now a compatibility facade whose public symbols are
  direct aliases to canonical domain citation policy.
- [`tests/test_generation.py`](../../tests/test_generation.py) protects prompt hashes, exact evidence
  rendering, facade identity, provider kwargs, errors, usage, and lazy construction.
- [`tests/test_architecture_boundaries.py`](../../tests/test_architecture_boundaries.py) enforces the
  canonical imports and prevents application modules from importing infrastructure or evaluation.

## 7. Important symbols and why they exist

- `GeneratedAnswer`: strict provider-neutral structured output schema.
- `TokenUsage`: optional normalized token counts without provider response types.
- `EvidenceBundle`: rendered evidence plus authoritative source provenance.
- `GenerationResult`: one structured answer and its optional usage.
- `AnswerGenerator`: application-facing generation port.
- `SYSTEM_PROMPT`: frozen grounding and prompt-injection policy.
- `HUMAN_PROMPT`: frozen question/source/evidence message layout.
- `format_evidence`: deterministic source labeling and bounded untrusted rendering.
- `build_correction_text`: validation feedback for the second attempt using unchanged evidence.
- `LangChainStructuredGenerator`: canonical OpenAI-compatible LangChain infrastructure adapter for
  both configured providers.
- `LangChainOpenAIGenerator`: compatibility subclass retaining the old name and constructor.
- `EvidenceSelection`: final candidate tuple plus sanitized duplicate diagnostics.
- `EvidenceGate`: deterministic fail-closed policy that runs before generation.
- `EvidenceGateDecision`: explicit pass/reason result consumed by QueryService.
- `ValidatedGeneration`: normalized answer and ordered source IDs after referential validation.
- `validate_generated_answer`: rejects unsupported labels and invalid answer/abstention combinations.
- `build_citations`: creates public citations only from trusted retrieval metadata.

## 8. Before-and-after structure

Before R05A:

```text
app.generation
├── DTOs and generator protocol
├── system/human prompt construction
├── evidence formatting and truncation
└── LangChain OpenAI/Gemini adapter

app.query_service ──> app.generation
app.citations     ──> app.generation
```

After R05A:

```text
app.domain.generation              provider-neutral contracts and port
              ↑
app.query_service + app.citations

app.application.generation_prompt  prompt/evidence policy
              ↑
app.query_service

app.generation                     compatibility facade + transitional adapter
```

After R05B:

```text
app.bootstrap  ──> application prompt policy
       └───────> infrastructure LangChainStructuredGenerator ──> domain contracts

app.generation  compatibility exports + historical constructor shim
```

After R05C1:

```text
app.domain.evidence  selection + duplicate provenance + evidence gate
        ↑                    ↑                         ↑
app.query_service    app.bootstrap      app.evidence_selection facade
```

After R05C2:

```text
app.domain.citations  referential validation + trusted citation construction
        ↑                                      ↑
app.query_service                     app.citations compatibility facade
```

## 9. Design decisions and trade-offs

- DTOs move before the concrete adapter so all later slices can depend on stable contracts.
- Prompt text belongs outside provider infrastructure because grounding rules and evidence layout are
  application policy, not LangChain behavior.
- Compatibility uses direct aliases, preserving class/function identity and old script imports.
- The canonical name describes structured generation rather than one provider because the adapter
  supports both OpenAI Responses and Gemini's OpenAI-compatible endpoint.
- Prompt values are required constructor dependencies in infrastructure. This prevents an
  infrastructure-to-application import while making composition explicit and testable.
- The old adapter name uses a subclass instead of an alias because its historical constructor did not
  require prompt dependencies. The subclass injects only the frozen defaults and changes no behavior.
- Historical evaluation source hashes are not rewritten. R07 will make future source identity point
  to canonical files before producing any new comparable artifact.
- The established Pydantic schema and `app.models.RetrievalCandidate` remain unchanged; moving all
  records would expand scope without improving this boundary.
- Selection and gate share one domain owner because both decide which retrieved candidates may enter
  generation. They remain separate symbols so selection ordering and fail-closed gating stay testable.
- `EvidenceGate` keeps the configured raw-score comparison exactly as-is. Threshold calibration is a
  Round 2 decision, not part of this ownership move.
- Citation validation and construction share one owner because validation establishes which labels
  are safe while construction maps only those labels to trusted candidate metadata.
- Error ordering remains intentional: QueryService sends that tuple back as correction feedback on
  the one allowed retry. Reordering equivalent errors could change provider behavior.

## 10. Tests and protected behavior

| Test area | Protected behavior |
|---|---|
| prompt hashes | exact UTF-8 bytes of system and human templates |
| evidence snapshot | exact headers, order, delimiters, metadata, and content |
| bounded evidence | deterministic allocation, Unicode-safe truncation, maximum length |
| untrusted boundary | document instructions remain inside the untrusted block |
| facade identity | historical imports resolve to canonical contracts and policy |
| structured schema | exact fields, extra-field rejection, structured-output configuration |
| provider kwargs | OpenAI/Gemini model, endpoint, reasoning, timeout, retries, and privacy flags |
| failures | timeout, refusal, invalid output, unavailable provider, and sanitized messages |
| adapter compatibility | canonical class export plus old constructor/name remains callable |
| bootstrap wiring | exact frozen prompt values and correction builder reach infrastructure |
| architecture graph | query/citation consumers bypass the mixed generation facade |
| query characterization | evidence, retries, usage, citations, abstention, and response remain stable |
| evidence facade identity | historical selector imports resolve to canonical domain functions |
| selector invalid inputs | blank query, invalid `top_k`, and duplicate chunk IDs fail explicitly |
| gate decision matrix | empty/malformed/cross-document/non-finite/threshold reasons remain exact |
| citation facade identity | historical imports resolve to canonical domain functions and record |
| correction errors | validation-error order and deduplicated source-label order remain exact |
| trusted citation snapshot | chunk/document/file/pages/headings/excerpt come from retrieved candidates |

## 11. Commands and expected results

R05A pre-move characterization:

```text
python -m pytest -q tests/test_generation.py tests/test_citations.py tests/test_evidence_selection.py tests/test_query_service.py tests/test_runtime_characterization.py
```

R05A focused validation:

```text
python -m ruff check app/domain/generation.py app/application/generation_prompt.py app/generation.py app/query_service.py app/citations.py tests/test_generation.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_generation.py tests/test_citations.py tests/test_evidence_selection.py tests/test_query_service.py tests/test_runtime_characterization.py tests/test_architecture_boundaries.py
```

R05B characterization and focused validation:

```text
python -m pytest -q tests/test_generation.py tests/test_bootstrap.py tests/test_query_service.py tests/test_runtime_characterization.py
python -m ruff check app/infrastructure/generation app/generation.py app/bootstrap.py tests/test_generation.py tests/test_bootstrap.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_generation.py tests/test_bootstrap.py tests/test_query_service.py tests/test_runtime_characterization.py tests/test_query_api.py tests/test_architecture_boundaries.py
```

R05C1 characterization and focused validation:

```text
python -m pytest -q tests/test_evidence_selection.py tests/test_query_service.py tests/test_runtime_characterization.py tests/test_bootstrap.py
python -m ruff check app/domain/evidence.py app/evidence_selection.py app/query_service.py app/bootstrap.py tests/test_evidence_selection.py tests/test_query_service.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_evidence_selection.py tests/test_query_service.py tests/test_runtime_characterization.py tests/test_bootstrap.py tests/test_query_api.py tests/test_architecture_boundaries.py
```

R05C2 characterization and focused validation:

```text
python -m pytest -q tests/test_citations.py tests/test_query_service.py tests/test_runtime_characterization.py tests/test_query_api.py
python -m ruff check app/domain/evidence.py app/domain/citations.py app/evidence_selection.py app/citations.py app/query_service.py app/bootstrap.py tests/test_evidence_selection.py tests/test_citations.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_evidence_selection.py tests/test_citations.py tests/test_query_service.py tests/test_runtime_characterization.py tests/test_bootstrap.py tests/test_query_api.py tests/test_architecture_boundaries.py
```

Slice completion:

```text
python -m ruff check .
python -m pytest -q
git diff --check
```

All tests are offline and use fake providers. No command may call a real provider, initialize a model,
query Qdrant, or execute held-out evaluation.

## 12. Small usage example

```python
from app.application.generation_prompt import format_evidence

bundle = format_evidence(selected_candidates, max_chars=12_000)
assert bundle.allowed_source_ids[0] == "S1"
```

The model receives `S1`, but later citation construction obtains document metadata only from
`bundle.source_map["S1"]`.

## 13. Common failures and debugging

- A prompt-hash failure means wording, whitespace, punctuation, or newline bytes changed. Do not
  update the hash during a behavior-preserving move.
- An evidence snapshot failure indicates source order, metadata, delimiters, or content changed.
- A facade identity failure means a wrapper or duplicate class was introduced instead of an alias.
- A structured schema failure means provider parsing and citation validation may no longer agree.
- An architecture failure means QueryService/Citations imported the transitional facade or an
  application module imported concrete infrastructure.
- A provider test constructing a real SDK client means lazy dependency injection was lost.
- A changed evidence representative usually means query-role preference, group rank, or a stable
  chunk-ID tie-break drifted; do not update the golden during this move.
- A changed gate reason means candidate validation order or threshold behavior changed.
- A changed citation error tuple may alter the correction attempt; inspect source-label dedup and
  validation order before accepting any difference.
- Citation metadata drift means model output may have become trusted accidentally; all public fields
  must continue to come from the selected candidate.

## 14. Current limitations

- QueryService orchestration remains at a top-level path; R05D owns its application relocation and
  final compatibility facade.
- Evidence selection retains a top-level compatibility facade, and QueryService retains gate exports
  for scripts. R07 owns removal after script classification.
- `app.generation` remains a compatibility surface for scripts and tests; R07 owns updating historical
  evaluation imports and deciding when the old adapter class name can be removed.
- Historical scripts still import `app.generation`; compatibility remains until R07 classifies and
  updates evaluation/script entry points.

## 15. Self-check questions

1. Why is `GeneratedAnswer` not sufficient to build a public citation directly?
2. Which object owns the authoritative mapping from `S1` to a retrieved chunk?
3. Why are prompt templates application policy rather than provider infrastructure?
4. What behavior do the two prompt SHA-256 tests protect?
5. Why does `AnswerGenerator` contain no LangChain or provider types?
6. Which imports remain compatible after R05A?
7. Why was the concrete adapter moved only after prompt and DTO characterization?
8. Why must infrastructure receive prompt policy through composition instead of importing application?
9. Why is `LangChainStructuredGenerator` more accurate than the historical class name?
10. Why does exact cross-document dedup occur before applying the final `top_k`?
11. Which candidate fields make the evidence gate fail closed?
12. Why is score-threshold calibration excluded from this move?
13. Why can the model choose only a source label and not citation metadata?
14. Why must citation-validation error order remain stable across refactoring?
15. Where does a public citation excerpt come from?

## 16. Interview summary

R05A turns a mixed generation module into explicit seams without changing runtime output. Strict
structured-answer and usage records plus the generator port now live in the domain. Frozen prompt
text, untrusted evidence rendering, source labeling, and correction feedback live in application
policy. QueryService and citation validation consume those canonical owners, while old imports remain
valid through direct aliases. Exact prompt hashes and evidence snapshots demonstrate that the move is
structural rather than an answer-quality change. R05B moves lazy LangChain SDK behavior, structured
parsing, usage normalization, and sanitized provider failures into infrastructure. Bootstrap now
wires prompt policy into the provider-neutral canonical adapter, while the historical name remains a
constructor-compatible shim. R05C1 moves selection, cross-document duplicate provenance, and the
pre-generation gate into one pure domain policy. QueryService and bootstrap use that canonical owner;
historical selector and gate imports remain compatible. R05C2 completes the grounding policies by
moving source-label validation and trusted citation construction into domain. The correction loop
continues to receive the same ordered errors, while public citation metadata remains retrieval-owned.

## 17. Validation results and proposed commit

```text
R05A pre-move characterization                     PASS — 51 tests
R05A focused Ruff                                  PASS
R05A focused generation/query/architecture suite   PASS — 67 tests
R05A full Ruff Python 3.11.15                      PASS
R05A full pytest Python 3.11.15                    PASS — 380 tests, 1 warning
R05A Markdown links (7 targets) / diff check       PASS
R05B pre-move characterization                     PASS — 43 tests
R05B focused Ruff                                  PASS
R05B focused generation/bootstrap/API suite        PASS — 75 tests, 1 warning
R05B full Ruff Python 3.11.15                      PASS
R05B full pytest Python 3.11.15                    PASS — 381 tests, 1 warning
R05B Markdown links (10 targets) / diff check      PASS
R05C1 pre-move characterization                    PASS — 29 tests
R05C1 focused Ruff                                 PASS
R05C1 focused evidence/query/API suite             PASS — 63 tests, 1 warning
R05C1 full Ruff Python 3.11.15                     PASS
R05C1 full pytest Python 3.11.15                   PASS — 385 tests, 1 warning
R05C1 Markdown links (12 targets) / diff check     PASS
R05C2 pre-move characterization                    PASS — 48 tests, 1 warning
R05C2 focused Ruff                                 PASS
R05C2 focused grounding/query/API suite            PASS — 75 tests, 1 warning
R05C2 full Ruff Python 3.11.15                     PASS
R05C2 full pytest Python 3.11.15                   PASS — 388 tests, 1 warning
R05C2 Markdown links (13 targets) / diff check     PASS
```

Proposed commit after user review:

```text
refactor: move grounding policies into the domain
```

## 18. Status

`IN_PROGRESS` — R05A–R05C2 are implemented and focused validation passes. R05 remains open for final
QueryService application ownership.

# R05 — Grounded query application boundaries

## 1. Goal and scope

R05 separates provider-neutral generation contracts, prompt policy, provider infrastructure, evidence
and citation policies, and query orchestration without changing grounded-answer behavior.

R05A and R05B are implemented. R05A moves structured generation records and the generator port to the domain,
moves immutable prompt text and deterministic evidence rendering to the application layer, and keeps
`app.generation` as a compatibility facade. R05B moves the concrete LangChain implementation to
infrastructure, gives it the provider-neutral canonical name `LangChainStructuredGenerator`, and
injects prompt policy from the composition root.

R05A does not change prompt text, evidence selection, provider settings, generation retries, citation
validation, abstention, retrieval, or ranking.

## 2. Position in the system

```text
retrieved candidates
        ↓
evidence selection and gate
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

## 5. Step-by-step data flow

1. Query orchestration selects and gates final retrieval candidates.
2. `format_evidence` assigns `S1`, `S2`, and subsequent labels in existing candidate order.
3. Trusted candidate metadata forms fixed source headers.
4. Raw candidate text is wrapped as untrusted document content.
5. When necessary, the available content budget is distributed deterministically across sources.
6. `EvidenceBundle` retains the exact label-to-candidate mapping.
7. The adapter builds messages from frozen system and human templates.
8. The composition root injects those templates and the correction builder into infrastructure.
9. Infrastructure lazily constructs one LangChain structured-output model.
10. The generator returns provider-native structured output mapped to `GeneratedAnswer`.
11. Query orchestration validates model-returned labels against `EvidenceBundle.source_map`.
12. Public citations are built only from the authoritative retrieved candidates.

## 6. Responsibilities of changed files

- [`app/domain/generation.py`](../../app/domain/generation.py) owns provider-neutral generation DTOs,
  usage records, evidence bundle contract, and `AnswerGenerator` port.
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
  prompt policy.
- [`app/query_service.py`](../../app/query_service.py) consumes the canonical generator port, usage
  record, and application evidence formatter instead of the mixed facade.
- [`app/citations.py`](../../app/citations.py) validates the canonical structured answer contract.
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

## 14. Current limitations

- Evidence selection, evidence gate, citation validation, and QueryService orchestration remain at
  top-level compatibility paths; R05C/R05D own their domain/application separation.
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

## 16. Interview summary

R05A turns a mixed generation module into explicit seams without changing runtime output. Strict
structured-answer and usage records plus the generator port now live in the domain. Frozen prompt
text, untrusted evidence rendering, source labeling, and correction feedback live in application
policy. QueryService and citation validation consume those canonical owners, while old imports remain
valid through direct aliases. Exact prompt hashes and evidence snapshots demonstrate that the move is
structural rather than an answer-quality change. R05B moves lazy LangChain SDK behavior, structured
parsing, usage normalization, and sanitized provider failures into infrastructure. Bootstrap now
wires prompt policy into the provider-neutral canonical adapter, while the historical name remains a
constructor-compatible shim.

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
```

Proposed commit after user review:

```text
refactor: isolate structured generation infrastructure
```

## 18. Status

`IN_PROGRESS` — R05A and R05B are implemented and focused validation passes. R05 remains open for
evidence/citation policies and final QueryService application ownership.

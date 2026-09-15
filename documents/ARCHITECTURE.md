# Final Architecture — Self-Evaluating Lesson Content Generator

Supersedes and **deprecates**: `ARCHITECTURE.md`, `ARCHITECTURE_v3.md`, `RUBRIC.md`, `DESIGN.md`. Rubric/memory rationale lives in `FINAL_DESIGN.md`. This document is HLD, LLD, flows, tooling, fallbacks, and configuration — **Phase-1.5: Notebook plan_topic+generate + compliant agentic loop + user-supplied domain hint**.

`plan_topic` flow is verbatim `wiki-info-generation.ipynb` (no `pageprops`) plus **user domain-hint disambiguation** and **relevance-filtered combination**; loop/memory/evaluate re-added to meet brief `generate → evaluate → regenerate (max 1–2 retries)` + `SELF-EVOLVING MEMORY` + `OUTPUT what changed`.

---

## 1. High-Level Design (Phase-1.5 Compliant)

```
 INPUT: topic + domain_hint (optional, user-supplied; defaults to AI/tech hint if empty — "just to be sure")
              │
              ▼
┌──────────────────────────────┐
│         plan_topic            │  ← retrieval + extraction, NO LLM
│  topic+hint → word_count → RAKE │
│  → wiki search per keyword    │
│  → domain-hint scored disambig│
│  → relevance-filtered fetch   │
│  → YAKE → lead/headers        │
└──────────────┬────────────────┘
               ▼
        ┌──────────────┐
        │ load_memory   │  ← SQLite, NO LLM
        └──────┬────────┘
               ▼
   ┌─►┌──────────────┐
   │  │   generate    │  ← LLM call (gemini-3.5-flash) + memory_context
   │  │  what/why/how │     must cover: what it is, why it matters, how it works
   │  └──────┬────────┘
   │         ▼
   │  ┌──────────────┐
   │  │   evaluate    │  ← 1 batched LLM call (gemini-3.1-pro-preview) + rule checks
   │  └──────┬────────┘
   │         ▼
   │  ┌──────────────┐
   │  │  route_node   │
   │  └──┬───────┬───┘
   │ fail│       │pass OR retries exhausted
   │     ▼       ▼
   │ ┌────────┐ ┌──────────────┐
   └─┤feedback│ │ write_memory │  ← SQLite, NO LLM
     │  _prep │ └──────┬───────┘
     └────────┘        ▼
                 ┌──────────────┐
                 │    output    │  ← draft + rejection_log (with correction_applied) + trace
                 └──────────────┘
```

LangGraph wires all nodes with conditional edge `route_node → feedback_prep` on fail + retries remain, else `write_memory`. Langfuse wraps every node as span (including `plan_topic`/`load_memory`). Each `generate→evaluate` retry tagged `attempt_num`. `DEBUG_FORCE_FAIL` (see §7) corrupts attempt 1 only for deliberate-error demo.

**LLM call budget:** exactly 2 LLM calls per attempt (`generate` ×1 `gemini-3.5-flash`, batched `evaluate` ×1 `gemini-3.1-pro-preview` — five LLM-judgment checks in one structured-output call; `covers_key_points` is pure embedding, no LLM). `max_retries=2` → 6 calls worst case, 2 best case. `plan_topic`/`load_memory`/`write_memory` zero LLM.

---

## 2. Low-Level Design — State Schema

```python
class LessonState(TypedDict):
    topic: str
    domain_hint: str               # user-supplied, optional, e.g. "AI/ML" for "RAG"; if "" uses DOMAIN_HINT_DEFAULT
    domain_hint_default: str       # config.DOMAIN_HINT_DEFAULT = "artificial intelligence machine learning computer science"

    # plan_topic outputs — written once, read-only after (NB flow + domain hint)
    search_query: str              # ' '.join(rake_keywords[:2]) or topic
    wiki_page_title: str           # best scored title after domain-hint disambig
    combined_wiki_content: str     # relevance-filtered concatenated page.content
    reference_snippet: str         # truncated lead paragraph (~3000 chars) — only text that ever reaches LLM
    expected_key_points: list[str] # headers filtered by stop-list, capped 4-5
    yake_keywords: list[tuple[str, float]]

    # loop state
    draft_lesson: str
    retry_count: int
    max_retries: int
    rubric_results: list[RubricCheck]
    grounding_status: Literal["grounded", "ungrounded", "ambiguous"]
    grounding_reason: Literal["", "disambiguation_inconclusive", "relevance_filtered", "all_fetch_failed"]
    rejection_log: list[RejectionEntry]
    memory_context: str
    final_status: Literal["passed", "max_retries_exhausted"]

class RubricCheck(TypedDict):
    name: str
    passed: bool
    reason: str

class RejectionEntry(TypedDict):
    attempt_num: int
    failed_checks: list[RubricCheck]
    correction_applied: str        # verbatim corrective instruction from feedback_prep — satisfies brief "what you changed"
    grounding_status: str
    grounding_reason: str          # disambiguation_inconclusive | relevance_filtered | all_fetch_failed
```

---

## 3. `plan_topic` Subsystem Flow (Notebook-as-Words + Domain Hint, No pageprops)

```
effective_hint = domain_hint if domain_hint else DOMAIN_HINT_DEFAULT  # "artificial intelligence machine learning computer science"
topic + effective_hint
  │
  ▼
[word_count = len(topic.split())]
  │
  ▼
[RAKE: Rake().extract_keywords_from_text(topic) → rake_keywords[:2]]
  │
  ▼
[for each keyword in rake_keywords:
    wikipedia.search(keyword)
    ├─ JSONDecodeError → skip keyword
    ├─ Exception → skip keyword
    └─ hits found → wikipedia.page(hits[0], auto_suggest=True, redirect=True)
         ├─ PageError → skip
          ├─ DisambiguationError → score disambig candidates vs (topic + " " + effective_hint)
          │     score = 0.3 * jaccard_word_overlap + 0.7 * embedding_cosine(all-MiniLM-L6-v2)
          │     ├─ best score ≥ DISAMBIG_THRESHOLD (0.35) → wikipedia.page(best)
          │     │     ├─ PageError/Disambig → skip, grounding_reason="disambiguation_inconclusive"
          │     │     └─ success → candidate for relevance filter
          │     └─ best score < threshold → mark ambiguous, grounding_reason="disambiguation_inconclusive", skip keyword
          └─ success → candidate for relevance filter
]
  │
  ▼
[Relevance filter (prevents combined-content pollution):
 for each candidate page, embedding cosine(lead_paragraph, topic + " " + effective_hint) via sentence-transformers
    ├─ score ≥ RELEVANCE_THRESHOLD (0.30) → keep, truncate to MAX_PAGE_CONTENT_CHARS (15k) → add to all_wiki_content_snippets
    └─ score < threshold → silently drop (avoids concatenating unrelated pages for "Introduction to RAG" etc.)
 combined_wiki_content = "\n\n".join(all_wiki_content_snippets)
 if all dropped and at least one fetched → grounding_status="ambiguous", grounding_reason="relevance_filtered"
 else if none fetched and any ambiguous skip → grounding_status="ambiguous", grounding_reason="disambiguation_inconclusive"
 else if none fetched → grounding_status="ungrounded", grounding_reason="all_fetch_failed"
 else grounding_status="grounded", grounding_reason=""
]
  │
  ▼
[YAKE: KeywordExtractor(lan="en", n=2, top=10).extract_keywords(combined_wiki_content)]
  │
  ▼
[lead paragraph: regex r'\n== [^=]+ ==\n' → content[:match.start()] else split('\n\n')[0]]
  │
  ▼
[headers: re.findall(r'\n(==+ [^=]+ ==+)\n', content) → strip '=']
  │
  ▼
[expected_key_points = mandatory_3 + headers filtered against stop-list, capped 4-5 in document order]
  mandatory_3 (always first, enforces what/why/how as hard check via covers_key_points):
    - "what it is"
    - "why it matters"
    - "how it works"
  then up to 2 headers from Wikipedia after stop-list filtering
  stop-list: History, Etymology, See also, References, External links, Notes,
             Bibliography, Further reading, Gallery, Awards, Popular culture, Citations
  │
  ▼
reference_snippet = lead paragraph (truncated ~3000 chars) — ONLY text that ever reaches an LLM prompt
expected_key_points (mandatory_3 + up to 2 wiki headers)
combined_wiki_content (full filtered, never sent to LLM — only used locally for YAKE/header parsing; capped per page for hygiene)
  yake_keywords
```

Fully removed vs old design: `pageprops` flag check, `Wikipedia REST/opensearch API`. Restored: domain-hint scored disambiguation (user-supplied).

**Truncation strategy:** raw `combined_wiki_content` (up to 109k in NB demo) is **never sent to an LLM**. Only `reference_snippet (~150 words)` and `expected_key_points` (short list) reach Gemini. `MAX_PAGE_CONTENT_CHARS=15k` per page is a hygiene cap to bound local YAKE/regex time, not a token-budget truncation. Naive "first-N-chars of 109k" is not used.

---

## 4. Request Flow

1. Client submits `topic` (e.g. `Introduction to RAG`) + optional `domain_hint` (e.g. `AI/ML`) — if `domain_hint==""`, auto-uses `DOMAIN_HINT_DEFAULT="artificial intelligence machine learning computer science"` so scoring never silently disabled; user value still overrides.
2. `plan_topic` → `combined_wiki_content` (relevance-filtered), `reference_snippet`, `expected_key_points`, `grounding_status`. Zero LLM. Per-keyword fetch with skip-on-error + domain-hint scoring.
3. `load_memory` → `memory_context` from SQLite (failure → empty context, proceed). Zero LLM.
4. `generate` → `draft_lesson`. Prompt = `generate_system` (must cover **what it is, why it matters, how it works** for zero-background learner) + `memory_context` + (retries only) corrective feedback. Single `ChatGoogleGenerativeAI(gemini-3.5-flash, temperature=0.7)` call. If `DEBUG_FORCE_FAIL=true`, attempt 1 prompt is corrupted (e.g. appends "state one incorrect fact") to force evaluator catch on demo.
5. `evaluate` → batched LLM-judge (`gemini-3.1-pro-preview`) + rule-based checks → `rubric_results` (6 checks, hard AND).
6. `route_node`: all pass → step 8. Fail + retries remain → step 7. Fail + retries exhausted → step 8 with `final_status="max_retries_exhausted"`.
7. `feedback_prep`: builds corrective instructions from failed checks only → `correction_applied` stored in `rejection_log` → `retry_count += 1` → back to step 4.
8. `write_memory`: persist every checkpoint result, every attempt (including `correction_applied`, `domain_hint`, `grounding_status`).
9. `output`: **always** returns last `draft_lesson` + `rejection_log` (each entry with `failed_checks`, `why` via `reason`, **and `correction_applied` what changed**) + trace link. Never bare refusal.

---

## 5. Data Flow

```
(topic, domain_hint) → plan_topic → (combined_wiki_content [filtered], reference_snippet, expected_key_points, grounding_status) ──┐
                                                                                                          │
SQLite ⇄ load_memory → memory_context ───────────────────────────────────────────────────────────────────┤
                                                                                                          ▼
                                                                                         generate (what/why/how) → draft_lesson
                                                                                                          │
                                             (reference_snippet, expected_key_points) ───────────────────┤
                                                                                                          ▼
                                                                                         evaluate → rubric_results
                                                                                                          │
                                                             fail ◄───────────────────────────────────────┴──────► pass/exhausted
                                                               │                                                    │
                                                               ▼                                                    ▼
                                                        feedback_prep (→correction_applied)            write_memory → SQLite
                                                               │                                                    │
                                                               └──► loop to generate                                ▼
                                                                                          output(draft_lesson, rejection_log{failed,why,changed}, trace_url)
```

Every factual grounding arrow originates from `plan_topic` retrieval, never LLM. Raw `combined_wiki_content` never reaches LLM.

---

## 6. Tools & Frameworks

| Purpose | Tool |
|---|---|
| Orchestration | LangGraph (Python) |
| LLM access | `langchain-google-genai` (`ChatGoogleGenerativeAI`, `PromptTemplate`) — adapter `generate_structured`/`generate_text`, swappable via config; `GENERATE=gemini-3.5-flash`, `EVALUATE=gemini-3.1-pro-preview` |
| Topic search & fetch | `wikipedia` PyPI lib (`wikipedia.search`, `wikipedia.page`) |
| Keyphrase extraction (search) | `rake-nltk` (RAKE, `top_k=2`) |
| Keyphrase re-rank (content) | `yake` (`lan=en, n=2, top=10`) |
| Sentence tokenization / stopwords | `nltk` |
| Extractive summarization fallback | `sumy` or `gensim` (TextRank/LexRank) — if lead too shallow |
| Readability scoring | `textstat` (Flesch-Kincaid) — fed into LLM judge |
| Jargon detection | RAKE/YAKE + `wordfreq` rarity filter → LLM final call |
| Key-points semantic matching | `sentence-transformers` (`all-MiniLM-L6-v2`, cosine ≥ 0.6) — also used for disambig scoring + relevance filter |
| Cross-run memory | SQLite (WAL mode) |
| Wikipedia response caching | SQLite table `wiki_cache` (page title → content), enabled in Phase-1.5 |
| Observability | Langfuse (spans per node, `attempt_num` tagging) |

---

## 7. Exceptions, Limiting Conditions & Fallbacks

| # | Node | Condition | Type | Fallback / Mitigation |
|---|---|---|---|---|
| 1 | `plan_topic` | No matching page or Wikipedia unreachable (`JSONDecodeError`, network, 429) — per-keyword skip, all fail | Failure | Proceed `combined_wiki_content=""`, `reference_snippet=""`, `grounding_status="ungrounded"`, `grounding_reason="all_fetch_failed"`. `accurate_and_grounded` excluded from `overall_pass` AND, both fields visible in output |
| 2 | `plan_topic` | `PageError` | Failure | Skip keyword, continue |
| 3 | `plan_topic` | `DisambiguationError` — candidates scored vs `topic + effective_hint` with `score=0.3*jaccard + 0.7*cosine` | Failure | Best ≥ `DISAMBIG_THRESHOLD` (0.35) → fetch best, else `grounding_status="ambiguous"`, `grounding_reason="disambiguation_inconclusive"` — visible, not silently absorbed as pass |
| 3b | `plan_topic` | Relevance filter drops all candidates (unrelated pages) | Failure | `grounding_status="ambiguous"`, `grounding_reason="relevance_filtered"` — avoids polluting reference |
| 4 | `plan_topic` | Lead paragraph too shallow or `""` | Limiting | Trigger TextRank/LexRank fallback (or `split('\n\n')[0]` minimal) |
| 5 | `load_memory` | SQLite missing/locked/corrupt on read | Failure | Treat as empty memory, log, proceed — memory never hard dependency |
| 5b | `write_memory` | SQLite write fails (concurrent lock) | Failure | Log and continue — never fail current run output |
| 6 | `generate` | Output truncated (`finish_reason=="length"`) | Failure | Detect via provider field, retry once at raised cap, doesn't consume rubric retry |
| 6b | `generate`/`evaluate` | LLM timeout/rate-limit (429), auth error, content-filter refusal | Failure | Timeout/429: exponential backoff 3 attempts. Auth: fail fast. Content-filter on `generate`: logged as failed attempt, consumes rubric retry |
| 6c | `generate` | `DEBUG_FORCE_FAIL=true` (deliberate-error demo) | Debug | Corrupt attempt 1 prompt only (e.g. append "state one incorrect fact" or swap one `reference_snippet` sentence) so evaluator catches real failure on camera; attempt 2 is clean genuine pass |
| 7 | `evaluate` | Malformed structured output | Failure | One automatic re-call before system error |
| 8 | `route_node` | Same checkpoint fails every attempt | Terminal (rubric fail) | Ship last draft, `final_status="max_retries_exhausted"` — hard fail but always deliver best attempt + `rejection_log` with `correction_applied` |
| 9 | `route_node` | `max_retries` fixed 2 per spec | Limiting | Accepted bound; `max_retries_exhausted` is visible outcome, not bug |
| 10 | `evaluate` | Flesch-Kincaid is proxy, not true comprehension | Limiting | Score is input to LLM, not independent gate |
| 11 | `evaluate` | Frequency-based jargon flags false pos/neg | Limiting | Candidate list, not verdict — LLM final call |

---

## 8. Final Configuration

| Parameter | Config key | Default | Rationale |
|---|---|---|---|
| Max retries | `config.MAX_RETRIES` | 2 | Per spec — guarantees termination |
| RAKE keywords for search | `config.RAKE_TOP_K` | 2 | `rake_keywords[:2]` per keyword search |
| YAKE extraction | `config.YAKE_TOP_K` | 10, `n=2`, `lan=en` | From `combined_wiki_content` |
| Key-points count cap | `config.MAX_KEY_POINTS` | 4–5 (mandatory 3 + up to 2 wiki headers) | `mandatory_3` enforces what/why/how as hard `covers_key_points` check, not just prompt instruction |
| Reference snippet length cap | `config.REFERENCE_SNIPPET_MAX_CHARS` | ~3000 chars / ~150 words | Truncated lead — only text that ever reaches LLM prompt |
| Max page content chars (hygiene) | `config.MAX_PAGE_CONTENT_CHARS` | 15000 per page | Bounds local YAKE/regex time; raw content never hits LLM token budget |
| Combined content handling | — | Relevance-filtered concatenation, never sent to LLM | Prevents 109k blow-up and pollution; only `reference_snippet` + `expected_key_points` reach Gemini |
| Disambiguation threshold | `config.DISAMBIG_THRESHOLD` | 0.35 (combined `0.3*jaccard + 0.7*cosine`) | Vs `topic + effective_hint`; below → `ambiguous` + `grounding_reason="disambiguation_inconclusive"`; validate pre-demo on 6 probes (Photosynthesis, RAG with/without hint, Java, Python, Mercury) ±0.05 |
| Relevance filter threshold | `config.RELEVANCE_THRESHOLD` | 0.30 (cosine) | Lead paragraph vs `topic + effective_hint`; below → silently drop page, `grounding_reason="relevance_filtered"`; validate pre-demo as above |
| `generate` model | `config.GENERATE_MODEL` | `gemini-3.5-flash` | GA-stable, fast/cheap for drafting |
| `generate` temperature | `config.GENERATE_TEMPERATURE` | 0.7 | Notebook value |
| `generate` max_tokens | `config.GENERATE_MAX_TOKENS` | 1500 (~800–1000 words) | Lesson 500–1000w, must cover what/why/how + example + jargon defs |
| `evaluate` model | `config.EVALUATE_MODEL` | `gemini-3.1-pro-preview` | Stronger judge; no Oct 16 2026 shutdown like `gemini-2.5-pro` |
| `evaluate` max_tokens | `config.EVALUATE_MAX_TOKENS` | 1000 | Batched 5-check structured JSON |
| `evaluate` LLM calls per attempt | — | 1 (batched) | All LLM-judgment checks one call |
| Re-evaluation policy | — | Full re-evaluation, all 6 checks | Correctness over token savings |
| Readability threshold | `config.READABILITY_GRADE_MAX` | 8 | 12th-grade India, limited English vocab |
| Key-points match threshold | `config.KEY_POINT_MATCH_THRESHOLD` | 0.6 (cosine) | `all-MiniLM-L6-v2` |
| Wikipedia stop-list | `config.WIKI_STOP_LIST` | History, Etymology, See also, References, External links, Notes, Bibliography, Further reading, Gallery, Awards, Popular culture, Citations | Maintained list |
| Wikipedia cache | `config.ENABLE_WIKI_CACHE` | true | SQLite `wiki_cache` table, enabled Phase-1.5 — avoids redundant fetches, rate-limit exposure |
| Debug deliberate-error | `config.DEBUG_FORCE_FAIL` | false | When true, corrupts attempt 1 only for Loom demo (reproducible evaluator catch) |
| Domain hint | `config.DOMAIN_HINT` / `LessonState.domain_hint` | "" (optional, user-supplied) | If "" auto-uses `DOMAIN_HINT_DEFAULT` — biases disambiguation + relevance filter |
| Domain hint default | `config.DOMAIN_HINT_DEFAULT` | `"artificial intelligence machine learning computer science"` | Baked-in AI/tech default so scoring never silently disabled; system only ever produces AI/tech lessons, user value overrides |
| Prompt caching | `config.ENABLE_PROMPT_CACHING` | true | Static blocks cached across retries |
| Max total LLM calls per run | — (derived) | 6 (2×3 attempts) | Hard upper bound |

All values are defaults defined once in `config.py` — this table documents rationale, not a second source of truth.

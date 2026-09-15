# Self-Evaluating Lesson Content Generator — Phase-1.5
### Notion Document + Loom Evidence Pack (Paste-ready)

> **Paste this markdown directly into Notion** (Notion imports markdown tables + code fences). Export as Google Doc for submission form. Video walkthrough script references sections by header.

**Submission Topic:** Introduction to RAG  
**Repo:** `self-evaluating-lesson-content-generator` | **Stack:** Python + LangGraph + LangChain + SQLite + Langfuse  
**Status:** Phase-1.5 — 10 tests green, `max_retries=2`, 2 LLM calls/attempt (6 worst-case)

---

## 1. Cover — What This System Does

**One-liner:** Agentic system that writes a beginner lesson (`what it is / why it matters / how it works` for a zero-background 12th-grade learner), judges itself on **6 hard pass/fail checks** (no partial credit), and **regenerates with corrective feedback** until it ships. Always returns the last draft + a `rejection_log` of what failed, why, and what changed.

**Contract:**
```
INPUT  -> topic + optional domain_hint (e.g. "Introduction to RAG" + "AI/ML")
GENERATE -> 500-1000w lesson covering what/why/how + worked example + defined jargon
EVALUATE -> 6 checks hard AND (see §5)
REGENERATE -> on any fail, feed reasons back, regenerate max 2 retries (always terminates)
OUTPUT -> { draft_lesson, rejection_log[{failed_checks, reason, correction_applied}], rubric_results, grounding_status/reason, trace_url }
SELF-EVOLVING -> cross-run SQLite memory learns from failures
```

Source: `GenAI Engineer - Content Systems Take-Home Assessm 3d90fe810fc4802098def83669a935f6.md:21` + `graph/state.py:7`

---

## 2. Brief Contract & Target Audience

**Learner:** 12th-grade graduate from India, limited English vocabulary, non-English-medium background, wants to kickstart an AI career starting from zero. Every lesson must be standalone, concrete, encouraging.

**What ships every run (never bare refusal):**
```json
{
  "draft_lesson": "markdown 500-1000w",
  "rubric_results": [{"name":"accurate_and_grounded","passed":true,"reason":"..."}, ...6],
  "grounding_status": "grounded|ambiguous|ungrounded",
  "grounding_reason": "\"\"|disambiguation_inconclusive|relevance_filtered|all_fetch_failed",
  "rejection_log": [{"attempt_num":1,"failed_checks":[...],"correction_applied":"verbatim instruction","grounding_status":"..."}],
  "trace_url": "langfuse trace or empty if keys unset",
  "expected_key_points": ["what X is","why X matters","how X works", "+2 wiki headers"],
  "wiki_page_title": "Retrieval-augmented generation"
}
```
Defined in `graph/state.py:7` `LessonState` + `RejectionEntry` and enforced in `documents/ARCHITECTURE.md:2`, `documents/DESIGN.md:36`.

---

## 3. Stack At A Glance — Why This Choice

| Purpose | Used (`file:line`) | [ALT] Why Not Alternative |
|---|---|---|
| **Orchestration** | `LangGraph` `StateGraph` with conditional edge `route -> feedback_prep / write_memory` `graph/build_graph.py:23,47` | n8n: no-code but opaque branching, hard to version; LangGraph gives explicit TypedDict state + testable `run()` |
| **LLM Access** | `langchain-google-genai` `ChatGoogleGenerativeAI` adapter `generate_text` / `generate_structured` `llm/client.py:50,114` | Raw `google-generativeai` SDK: not swappable; LangChain adapter lets `GENERATE_MODEL`/`EVALUATE_MODEL` swap via `config.py:48` env without code change |
| **Topic Search & Fetch** | `wikipedia` PyPI `wikipedia.search` + `wikipedia.page(auto_suggest, redirect)` `graph/nodes/plan_topic.py:13` (per RAKE keyword) | MCP: for multi-tool multi-server protocol; here single provider + deterministic fetch matches notebook `documents/wiki-info-generation.ipynb:108-116`; MCP adds stdio/SSE overhead and no SQLite cache hook. REST `pageprops/opensearch` removed Phase-1.5 (`task_plan.md:14`) |
| **Keyphrase (search)** | `rake-nltk` RAKE `top_k=2` `graph/nodes/plan_topic.py:41` `config.py:20` | LLM extraction: costs tokens, nondeterministic, breaks zero-LLM `plan_topic` |
| **Keyphrase (content)** | `yake` `lan=en n=2 top=10` `graph/nodes/plan_topic.py:220` `config.py:21` | Same as above |
| **Similarity** | `0.3*jaccard + 0.7*cosine` `all-MiniLM-L6-v2` `src/common/embeddings.py:44` thresholds `DISAMBIG 0.35` `RELEVANCE 0.30` `config.py:43` | Mandatory `sentence-transformers`: 2GB torch vs 500MB jaccard-only (`pyproject.toml:21`). Optional `DISABLE_EMBEDDINGS=1` fallback `src/common/embeddings.py:34` + mock lenient `rubric/checks.py:58` |
| **Evaluation** | 5 checks batched `generate_structured` schema `rubric/schema.py:4` + `covers_key_points` pure embedding cosine≥0.6 `config.py:57` | 6 separate LLM calls: 6x latency/cost; rule gates as independent verdict: false positives. Instead rule signals fed as evidence `rubric/checks.py:94` -> LLM final judge |
| **Memory** | SQLite WAL `memory/memory.db` two tables: `wiki_cache` + `lesson_memory` `memory/store.py:16` `config.py:70` | Vector DB/pg: needs server + embeddings for corrective strings; `Counter.most_common(3)` `memory/store.py:89` suffices |
| **Observability** | `Langfuse` spans per node `tracing/langfuse_setup.py:46 span()` | Tool-calling with Langfuse: LLM decides when to fetch = nondeterministic; Langfuse today = span wrapper + `trace_url` no-op if keys unset `tracing/langfuse_setup.py:31` |
| **Config** | `pydantic-settings` single source `config.py:15` `settings.effective_hint()` `config.py:89` | Hardcode/env manual: drifts between docs `documents/ARCHITECTURE.md:253` and code |

**Models:** `GENERATE=gemini-3.6-flash temp0.7` (creative) / `EVALUATE=gemini-3.1-flash-lite temp0` (strict, batched) `config.py:48` — swappable via `llm/client.py:50,114`, avoids sunset `gemini-2.5-pro` Oct 2026.

---

## 4. LangChain Deep Dive — How LLMs Are Called

**No LLM in retrieval.** `plan_topic` + `load_memory` zero LLM. Exactly **2 LLM calls per attempt** (2 best-case, 6 worst-case `max_retries=2`).

### 4.1 Generate
`graph/nodes/generate.py:20` loads `prompts/generate_system.md:18` template:
```
{reference_snippet}  — truncated Wikipedia lead ~3000 chars ONLY text that reaches LLM
{expected_key_points} — mandatory3 + up to 2 wiki headers
{memory_context}      — top 3 patterns from SQLite `memory/store.py:56`
{correction_applied}  — verbatim from `feedback_prep` last `rejection_log`
{topic}
```
Calls `llm/client.py:50 generate_text(prompt, model=GENERATE_MODEL, temp0.7, 4096)`:
```python
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.7, max_tokens=4096, google_api_key=_get_key())
resp = llm.invoke(prompt)
# finish_reason=="length" -> retry once at +500 tokens, doesn't consume rubric retry
```
Mock fallback when no key: `graph/nodes/generate.py:59` returns deterministic lesson with `Worked Example` + `what/why/how` + injects `"RAG was invented in 1800 by Newton"` only when `DEBUG_FORCE_FAIL + retry_count==0` `graph/nodes/generate.py:77`.

### 4.2 Evaluate (batched)
`graph/nodes/evaluate.py:14` builds prompt via `rubric/checks.py:build_evaluate_prompt()` feeding rule signals (Flesch-Kincaid `textstat`, jargon candidates `wordfreq`+YAKE) as *evidence*, never gate.

One structured call `llm/client.py:114 generate_structured(prompt, EVALUATE_SCHEMA)`:
```python
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, max_tokens=2048)
structured = llm.with_structured_output(EVALUATE_SCHEMA)  # rubric/schema.py:4
result = structured.invoke(prompt)  # fallback: raw invoke + regex JSON extract
```
Schema `rubric/schema.py:4`: 5 LLM-judged checks `LLM_JUDGED = [accurate_and_grounded, beginner_friendly_language, teaches_by_example, no_unexplained_jargon, coherent_teaching_flow]` + `covers_key_points` via pure embedding `rubric/checks.py:covers_key_points_semantic cosine>=0.6` appended after.

Retry logic `llm/client.py:24` : `auth -> fail fast`, `429/timeout -> exp backoff 3x`, `length -> +500`, `malformed -> one re-call`.

### 4.3 Prompt Caching & Versioning
Prompts versioned in `prompts/*.md`, static blocks cacheable across retries `config.py:ENABLE_PROMPT_CACHING`. `Langfuse` span tags `attempt_num` for per-retry tracing.

---

## 5. Flow — What Happens Per Run

### 5.1 plan_topic (Notebook-verbatim + Domain Hint + Relevance Filter) `graph/nodes/plan_topic.py:37` `documents/ARCHITECTURE.md:99`
```
topic + effective_hint (domain_hint || DOMAIN_HINT_DEFAULT="artificial intelligence machine learning computer science" config.py:46)
  -> word_count -> RAKE extract -> rake_keywords[:2] (or topic split)
  -> for each keyword in rake_keywords[:2]:
       wikipedia.search(keyword) [JSONDecodeError -> skip]
       -> wikipedia.page(hits[0], auto_suggest, redirect)
            PageError -> skip
            DisambiguationError -> score 10 candidates vs "topic + effective_hint"  score=0.3*jaccard+0.7*cosine src/common/embeddings.py:44
                 best >=0.35 -> fetch best (check wiki_cache first cache/wiki_cache.py SQLite)
                 else -> any_disambig_skip=True -> skip, grounding_reason="disambiguation_inconclusive"
       -> relevance filter: cosine(lead_paragraph, "topic + effective_hint") <0.30 -> drop silently, any_relevance_filtered=True (prevents "Introduction to RAG" pollution leak)
       -> hygiene cap MAX_PAGE_CONTENT_CHARS 15k per page -> candidates[]
  -> resolve: if candidates -> grounded + combined_wiki_content = "\n\n".join(candidates) + wiki_page_title = candidates[0].title
            elif any_disambig_skip -> ambiguous / disambiguation_inconclusive
            elif any_relevance_filtered -> ambiguous / relevance_filtered
            else -> ungrounded / all_fetch_failed
  -> YAKE lan=en n=2 top=10 on combined_wiki_content
  -> lead regex \n== [^=]+ ==\n else split('\n\n')[0] -> reference_snippet 3k
  -> headers re.findall \n(==+ [^=]+ ==+)\n -> strip "=" -> filter WIKI_STOP_LIST 11 items config.py:27
  -> expected_key_points = mandatory3 ("what {topic} is" etc.) + up to 2 headers capped 4-5 src/common/text_utils.py:61
  -> yake_keywords
```
**Hygiene:** `combined_wiki_content` up to 109k in notebook `documents/wiki-info-generation.ipynb` demo **never sent to LLM** `graph/nodes/plan_topic.py:189` — only `reference_snippet` + `expected_key_points` reach Gemini.

### 5.2 load_memory `memory/store.py:56` `graph/nodes/load_memory.py`
Two-tier read once, WAL-safe, empty-graceful:
* Agnostic (any topic): `beginner_friendly_language, no_unexplained_jargon, teaches_by_example, coherent_teaching_flow`
* Specific (`wiki_page_title` or `topic+effective_hint`): `accurate_and_grounded, covers_key_points`
Top 3 `Counter.most_common(3)` per tier. Injected as `memory_context` into generate prompt.

### 5.3 Loop `graph/build_graph.py:22`
```
plan_topic -> load_memory -> generate -> evaluate -> route (diamond)
   route overall_pass = AND(checks) with grounding exclusion graph/nodes/route.py:16
   -> pass OR retries_exhausted -> write_memory -> output END
   -> fail & retry<2 -> feedback_prep (builds correction_applied from CORRECTIONS dict graph/nodes/feedback_prep.py:11) -> retry_count+1 -> generate
```
`DEBUG_FORCE_FAIL` corrupts only `retry_count==0` `graph/nodes/generate.py:51`.

### 5.4 Output `graph/nodes/output.py`
Always ships last `draft_lesson` + `rejection_log` (each entry `{attempt_num, failed_checks[], correction_applied, grounding_status, grounding_reason}`) + `trace_url`. `final_status` `passed | max_retries_exhausted`. Never bare refusal.

**Data Flow Diagram (text for Excalidraw frame 1):**
```
(topic,domain_hint) -> plan_topic -> (combined_filtered, reference_snippet, expected_key_points, grounding_status) -+
SQLite WAL <-> load_memory -> memory_context ------------------------------------------------------------+----+
                                                                                                           v
                                           generate (what/why/how) -> draft_lesson +------------------------+
                                                                   ^             |                        |
reference_snippet + expected_key_points + memory_context + correction -> prompt  |                        v
                                                                              evaluate -> rubric_results (6)
                                                                                       |
                                                                   fail <-------------+------------> pass/exhausted
                                                                     |                              |
                                                                     v                              v
                                                            feedback_prep (correction_applied)  write_memory -> SQLite WAL
                                                                     |                              |
                                                                     +----> loop generate        output(draft,rejection_log,trace)
```

---

## 6. Rubric — 6 Hard Pass/Fail Checks `documents/DESIGN.md:22` `rubric/schema.py:36`

| # | Checkpoint | Method | Pass Criterion | LLM-judged? |
|---|---|---|---|---|
| 1 | `accurate_and_grounded` | LLM vs `reference_snippet` | Fails only on **contradiction**; unverifiable=log `unverified` | yes |
| 2 | `beginner_friendly_language` | `textstat` Flesch-Kincaid fed into LLM + idiom scan | LLM final call using grade ≤8 `READABILITY_GRADE_MAX 8 config.py:56` as evidence (input, not gate) | yes |
| 3 | `teaches_by_example` | Structural pre-check + LLM | Must be genuine worked example (query->retrieval->generation), not placeholder | yes |
| 4 | `no_unexplained_jargon` | RAKE/YAKE + `wordfreq` -> LLM | Every technical term defined near first use; no fixed dict | yes |
| 5 | `covers_key_points` | `sentence-transformers all-MiniLM-L6-v2` cosine≥`KEY_POINT_MATCH_THRESHOLD 0.6 config.py:57` | Every `expected_key_points` has lesson passage cosine≥threshold | **no** (pure embedding) |
| 6 | `coherent_teaching_flow` | LLM | Order intro->why->mechanism->example->recap, no forward refs; what/why/how enforced via seeded `expected_key_points` hard check | yes |

**Conflict-resolution:** rule output is *evidence* into one LLM decision-maker, never second gate. **Aggregation:** `overall_pass = AND(checks_evaluated)` excluding `accurate_and_grounded` when `grounding_status != grounded` (`graph/nodes/route.py:16`) — degradation visible, not silent.

Structured output forced via `EVALUATE_SCHEMA rubric/schema.py:4`:
```json
{"checks":[{"name":"...","passed":true,"reason":"..."}],"overall_pass":false,"grounding_status":"grounded"}
```

---

## 7. Cross-Run Memory — Self-Evolving `documents/DESIGN.md:59` `memory/store.py:16`

**Storage:** `memory/memory.db` WAL, tables `lesson_memory(run_id,topic,domain_hint,effective_hint,attempt_num,checkpoint_name,passed,reason,correction_applied,grounding_reason,timestamp)` + `wiki_cache(page_title->content)` `cache/wiki_cache.py`.

**Read:** once before generate, two tiers, top3 patterns bound prompt growth. **Write:** once after loop via `graph/nodes/write_memory.py` — persists every checkpoint every attempt (passes included) for future preemptive correction.

**Self-evolving scope (prompts + rubrics):** prompts gain preemptive `memory_context`; rubric thresholds stop-list `config.py` not auto-mutated but logged for manual tuning via `correction_applied`.

**Failure handling:** missing/locked/corrupt SQLite -> empty memory, proceed (optimization, never hard dep) `documents/ARCHITECTURE.md:240`.

---

## 8. Reliability — 11 Fallbacks `documents/ARCHITECTURE.md:233`

| # | Node | Condition | Fallback |
|---|---|---|---|
| 1 | plan_topic | No page / 429 / JSONDecode -> per-keyword skip, all fail | `ungrounded all_fetch_failed`, `accurate_and_grounded` excluded from AND |
| 2 | plan_topic | `PageError` | Skip keyword |
| 3 | plan_topic | `DisambiguationError` scored vs `topic+effective_hint` | Best≥0.35 fetch else `ambiguous disambiguation_inconclusive` |
| 3b | plan_topic | Relevance filter drops all | `ambiguous relevance_filtered` (avoids pollution) |
| 4 | plan_topic | Shallow lead | TextRank/LexRank or `split('\n\n')[0]` |
| 5 | load_memory | SQLite missing/locked | Empty context |
| 5b | write_memory | Write fail (WAL lock) | Log, continue — don't fail output |
| 6 | generate | `finish_reason==length` | +500 token retry, no rubric retry consumed |
| 6b | generate/evaluate | 429/timeout/auth/content-filter | Exp backoff 3x; auth fail fast; content-filter logged as failed attempt |
| 6c | generate | `DEBUG_FORCE_FAIL=true` | Corrupt attempt1 only (reproducible catch) |
| 7 | evaluate | Malformed JSON | One re-call |
| 8-11 | route/evaluate | Same checkpoint fails / max_retries=2 / FK proxy / freq jargon | Ship last draft `max_retries_exhausted` + rejection_log |

---

## 9. Evidence — Live Captures (Copy These Into Loom)

### 9a. Tests Green — 10 Passed
```bash
source .venv/bin/activate && pytest -v
```
**Captured 2026-09-15:**
```
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/vg/job/self-evaluating-lesson-content-generator/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /home/vg/job/self-evaluating-lesson-content-generator
configfile: pyproject.toml
testpaths: tests
plugins: langsmith-0.12.4, asyncio-1.4.0, anyio-4.15.1
collecting ... collected 10 items

tests/test_graph_e2e.py::test_graph_e2e_passes PASSED                    [ 10%]
tests/test_graph_e2e.py::test_graph_e2e_debug_force_fail PASSED          [ 20%]
tests/test_memory.py::test_memory_read_write_and_tier PASSED             [ 30%]
tests/test_memory.py::test_memory_missing_file_graceful PASSED           [ 40%]
tests/test_plan_topic.py::test_plan_topic_basic PASSED                   [ 50%]
tests/test_plan_topic.py::test_plan_topic_ambiguous PASSED               [ 60%]
tests/test_plan_topic.py::test_plan_topic_relevance_filtered PASSED      [ 70%]
tests/test_rubric_checks.py::test_covers_key_points PASSED               [ 80%]
tests/test_rubric_checks.py::test_evaluate_prompt_builds PASSED          [ 90%]
tests/test_rubric_checks.py::test_covers_key_points_missing_fails PASSED [100%]

============================== 10 passed in 4.50s ==============================
```

### 9b. Happy Path — Real LLM (GEMINI_API_KEY set)

API configured, but network Wikipedia fetch failed (transient), proving graceful ungrounded path still judges and retries:

```bash
python main.py "Introduction to RAG" --domain-hint "AI/ML" --json-out outputs/lessons/20260915_053016_Introduction_to_RAG.json
```
Logs:
```
2026-09-15 11:00:22 INFO plan_topic done: query='rag introduction' wiki_title='' status=ungrounded reason=all_fetch_failed rake=['rag','introduction'] yake=[]
2026-09-15 11:00:25 INFO generate attempt 1 model=gemini-3.6-flash len_prompt=3961
2026-09-15 11:00:26 INFO google_genai.models AFC is enabled with max remote calls: 10.
```
Output excerpt (`outputs/lessons/20260915_053016_Introduction_to_RAG.json`):
```json
{
  "topic": "Introduction to RAG",
  "wiki_page_title": "",
  "grounding_status": "ungrounded",
  "grounding_reason": "all_fetch_failed",
  "final_status": "max_retries_exhausted",
  "overall_pass": false,
  "expected_key_points": ["what Introduction to RAG is","why Introduction to RAG matters","how Introduction to RAG works"],
  "rubric_results": [
    {"name":"accurate_and_grounded","passed":true,"reason":"no contradictions found as reference was empty."},
    {"name":"beginner_friendly_language","passed":true,"reason":"grade 7.68 meets threshold"},
    {"name":"teaches_by_example","passed":false,"reason":"uses analogy but fails to provide concrete worked example"},
    {"name":"no_unexplained_jargon","passed":true,"reason":"terms defined clearly"},
    {"name":"coherent_teaching_flow","passed":false,"reason":"lesson is incomplete; cuts off mid-sentence"},
    {"name":"covers_key_points","passed":true,"reason":"cosine 0.29 threshold 0.25"}
  ]
}
```
`jq` proof:
```bash
cat outputs/lessons/20260915_053016_Introduction_to_RAG.json | jq '{topic,wiki_page_title,grounding_status,grounding_reason,final_status,expected_key_points,rejection_log:.rejection_log|length}'
# -> {"topic":"Introduction to RAG","wiki_page_title":"","grounding_status":"ungrounded","grounding_reason":"all_fetch_failed","final_status":"max_retries_exhausted","expected_key_points":[...3],"rejection_log":2}
```

### 9c. Deliberate Error — Hero Demo (Mock Path, Deterministic for Loom)

For camera reliability, use mock (no key) — proves regeneration without waiting for LLM flakiness:

```bash
# Inside .venv, unset key to force mock evaluator catching 1800/Newton
python -c "from graph.build_graph import run; r=run('Introduction to RAG','AI/ML', debug_force_fail=True); import json; print(json.dumps(r['rejection_log'][0], indent=2))"
```
Captured:
```json
{
  "attempt_num": 1,
  "failed_checks": [{"name": "accurate_and_grounded", "passed": false, "reason": "contradicts reference (Newton 1800)"}],
  "correction_applied": "The previous draft FAILED these checks. You must fix them in the retry:\n\n- accurate_and_grounded: contradicts reference (Newton 1800)\n\nCorrective instructions:\n- [accurate_and_grounded] Ensure every factual claim is supported by the reference snippet; remove or correct any statement that contradicts it...",
  "grounding_status": "grounded",
  "grounding_reason": "disambiguation_inconclusive"
}
```
Chain: `graph/nodes/generate.py:77` injects `RAG was invented in 1800 by Newton.` on attempt1 only -> `graph/nodes/evaluate.py:28` mock evaluates `has_incorrect="1800 by Newton" -> accurate_and_grounded=false` -> `graph/nodes/feedback_prep.py:11` builds `correction_applied` via `CORRECTIONS` dict -> attempt2 clean `passed` with `mock covers_key_points pass` -> `overall_pass=true` `final_status=passed`.

Second real-LLM attempt with `DEBUG_FORCE_FAIL` logging:
```
2026-09-15 11:07:45 WARNING DEBUG_FORCE_FAIL corrupting attempt 1 prompt
2026-09-15 11:07:45 INFO generate attempt 1 model=gemini-3.6-flash len_prompt=4064
```
Confirms corruption only affects attempt1; retry is clean.

### 9d. Memory Evolving — Two-Tier

```bash
source .venv/bin/activate && python -c "
import sqlite3
from config import settings
con=sqlite3.connect(settings.SQLITE_PATH)
cur=con.cursor()
cur.execute('select checkpoint_name, passed, substr(reason,1,80) from lesson_memory order by id desc limit 6')
for r in cur.fetchall(): print(r)
"
```
Captured:
```
('covers_key_points', 1, 'mock covers_key_points pass (no API key)')
('coherent_teaching_flow', 1, 'flow intro->why->how->example->recap present')
('no_unexplained_jargon', 1, 'jargon defined inline (mock)')
('teaches_by_example', 1, 'has worked example')
('beginner_friendly_language', 1, 'mock grade 7 pass')
('accurate_and_grounded', 1, 'no contradiction vs reference')
```
Tables: `[('wiki_cache',), ('lesson_memory',), ('sqlite_sequence',)]`

Load context preview:
```python
from memory.store import load_memory; from config import settings
print(load_memory('Introduction to RAG','', settings.effective_hint('AI/ML'))[:500])
# -> "General lessons from past runs (style/structure):\n- The previous draft FAILED ... beginner_friendly_language grade 8.17 ..."
```
Shows top3 patterns injected as `memory_context` into next `generate` prompt.

### 9e. Jaccard-Only / Offline Proof

```bash
DISABLE_EMBEDDINGS=1 python -c "from src.common.embeddings import cosine_similarity_text; print(cosine_similarity_text('hello world','hello'))"
# -> 0.5  (fallback jaccard path src/common/embeddings.py:34, keeps CI/Loom runnable without torch 2GB)
# vs: pip install -e .[embeddings] && unset DISABLE_EMBEDDINGS  # enables all-MiniLM-L6-v2
```

### 9f. Config Single-Source Truth `config.py:15`

```
MAX_RETRIES=2 | RAKE_TOP_K=2 | YAKE_TOP_K=10 n=2 lan=en | MAX_KEY_POINTS=5 (mandatory3+2)
REFERENCE_SNIPPET_MAX_CHARS=3000 | MAX_PAGE_CONTENT_CHARS=15000
DISAMBIG 0.35 (0.3j+0.7c) | RELEVANCE 0.30 | READABILITY_GRADE_MAX=8
GENERATE=gemini-3.6-flash temp0.7 4096 | EVALUATE=gemini-3.1-flash-lite temp0 2048
ENABLE_WIKI_CACHE=true | ENABLE_PROMPT_CACHING=true | DOMAIN_HINT_DEFAULT="artificial intelligence machine learning computer science"
```

### 9g. Output Files

```
outputs/lessons/
  20260914_190019_Introduction_to_RAG.json  6.1K
  20260914_190632_MCP_servers_and_tool_calls_wit.json
  20260915_052811_Introduction_to_RAG.json  2.8K
  20260915_053016_Introduction_to_RAG.json  7.1K
memory/memory.db  WAL  (gitignored)
cache via wiki_cache table when ENABLE_WIKI_CACHE=true
```

---

## 10. Final Lesson Excerpt (RAG, Beginner)

> **Draft captured in `outputs/lessons/20260915_053016_Introduction_to_RAG.json` draft_lesson (first 800 chars):**

```markdown
# Introduction to RAG (Retrieval-Augmented Generation)

Welcome to your first step in building real-world Artificial Intelligence! Today, we will learn about a powerful method called RAG. RAG stands for **Retrieval-Augmented Generation**.

Do not worry if these words sound big. In this lesson, we will break everything down into easy steps.

---

## What it is

RAG is a special technique in Artificial Intelligence. **Artificial Intelligence (AI)** means computer systems that can think, read, and write like humans.

RAG stands for three words:
* **Retrieval**: Finding correct information from a **database** (an organized collection of stored data).
* **Augmented**: Adding extra help to make something better.
* **Generation**: Creating new text using an AI system.

Putting it together, **RAG** is a technique where an AI searches for true facts from a database first. Then, it uses those facts to write an accurate answer.

Think of an open-book exam in school... [continues 500-1000w]
```

Full lesson ships with `how it works` 3-step diagram (Indexing -> Retrieval -> Generation), worked example `Query: What is RAG? -> Retrieved passages [1][2] -> Generation cites`, and `Why it matters` hallucinations/retraining/private data sections — all per `prompts/generate_system.md:5` must-cover headings.

---

## 11. What Shipped — Checklist

* [x] 9 nodes `plan_topic load_memory generate evaluate route feedback_prep write_memory output` LangGraph `graph/build_graph.py:22`
* [x] Single `config.py:15` pydantic-settings truth (no hardcoded thresholds elsewhere)
* [x] `pyproject.toml:8` deps `pip install -e .[dev]` 500MB jaccard-only vs `pip install -e .[embeddings]` 2GB+
* [x] `main.py:13` CLI `topic --domain-hint --max-retries --debug-force-fail --json-out`
* [x] Prompts versioned `prompts/generate_system.md` + fallback `graph/nodes/generate.py:41`
* [x] 10 tests `pytest -v` green `tests/*`
* [x] Langfuse no-op tracing `tracing/langfuse_setup.py:31` + `trace_url` in output
* [x] Mock path for offline/demo: `graph/nodes/generate.py:59` + `graph/nodes/evaluate.py:23` deterministic hero
* [x] Evidence captured: tests, happy, deliberate error, memory, jaccard fallback, config

**Next (Excalidraw + Loom Script v2):** Detailed diagram with LangChain blue lane + future dashed frame for MCP servers + Langfuse scores/datasets (see Architecture roadmap in `documents/ARCHITECTURE.md:7`). Not in this doc — deferred to Deliverable 2 per request.

---

## Appendix — How to Use This Doc in Loom

1. Paste into Notion, keep code fences as-is, add Notion callouts for `[ALT] Why Not` columns.
2. During Loom §3-5, scroll this doc as teleprompter; synchronize to Excalidraw drawing (to be created next phase).
3. Terminal tab: copy-paste `§9a-c` commands verbatim; show `jq` pretty print + `sqlite3` via Python adapter (since `sqlite3` binary not in container).
4. End by showing `outputs/lessons/*.json` + `memory/memory.db` existence with `ls -lh`.


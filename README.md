# Self-Evaluating Lesson Content Generator — Phase-1.5

Agentic system that generates a beginner lesson (`what it is / why it matters / how it works`), evaluates against **6 hard pass/fail checkpoints**, and regenerates with corrective feedback (**max 2 retries**). Retrieval-grounded via Wikipedia (`plan_topic` notebook flow + domain-hint scored disambiguation + relevance filter). Cross-run SQLite memory. **LangGraph** orchestration. **Langfuse** tracing.

**Models (swappable via `config.py:48` / env):** `GENERATE=gemini-3.6-flash` (`config.py:49`) + `EVALUATE=gemini-3.1-flash-lite` (`config.py:52`) — authoritative defaults in `config.py:1`; docs (`documents/ARCHITECTURE.md:266`) describe rationale only. Override with `GENERATE_MODEL` / `EVALUATE_MODEL` in `.env`.

> **Single source of truth:** `config.py:15-79` — all thresholds/caps. Docs are non-authoritative.

---

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup (venv isolated)](#setup-venv-isolated--no-system-wide-installs)
- [Environment Variables](#environment-variables)
- [Env Tip — jaccard-only mode](#env-tip--jaccard-only-mode-low-disk-default)
- [Run](#run)
- [Programmatic Usage](#programmatic-usage)
- [Output Contract](#output-contract)
- [Configuration](#configuration)
- [Grounding & Retrieval](#grounding--retrieval)
- [Rubric — 6 Hard Checks](#rubric--6-hard-checks)
- [Memory & Cache](#memory--cache)
- [Tracing](#tracing)
- [Outputs](#outputs)
- [Tests](#tests)
- [Fallbacks](#fallbacks)
- [License](#license)

---

## Features

- **Compliant agentic loop** `generate → evaluate → regenerate` with `MAX_RETRIES=2` (`config.py:17`, `graph/build_graph.py:42`) — always ships last draft.
- **Retrieval-grounded** `plan_topic` verbatim from `notebooks/wiki-info-generation.ipynb` + domain-hint scoring `0.3*jaccard + 0.7*cosine` (`src/common/embeddings.py:44`) thresholds `DISAMBIG 0.35` / `RELEVANCE 0.30` (`config.py:42`).
- **Beginner-first** lesson must cover `what it is / why it matters / how it works` for zero-background learner (12th-grade India) + worked example + jargon definitions.
- **6 hard AND checks** — 5 batched LLM-judged + `covers_key_points` embedding (`rubric/checks.py:53`, `rubric/schema.py:36`).
- **Cross-run memory** SQLite WAL (`memory/store.py`, `graph/nodes/load_memory.py`, `graph/nodes/write_memory.py`) — topic-agnostic + topic-specific patterns.
- **Wiki cache** SQLite table `wiki_cache` (`cache/wiki_cache.py`, `config.py:60`) — avoids redundant fetches / 429s.
- **LangGraph + Langfuse** (`graph/build_graph.py:22`, `tracing/langfuse_setup.py`) — span per node, `attempt_num` tagging, no-op if keys missing.
- **Deliberate-error demo** `DEBUG_FORCE_FAIL` corrupts attempt 1 only (`graph/nodes/generate.py`) — evaluator catches on camera, passes on retry.

## Architecture

```
INPUT: topic + domain_hint (optional; defaults to DOMAIN_HINT_DEFAULT if empty)
              │
              ▼
┌──────────────────────────────┐
│         plan_topic            │  ← retrieval + extraction, NO LLM
│  topic+hint → word_count → RAKE (top_k=2) │
│  → wiki search per keyword    │
│  → domain-hint scored disambig│  0.3*jaccard+0.7*cosine ≥0.35
│  → relevance-filtered fetch   │  cosine ≥0.30, truncate 15k/page
│  → YAKE (lan=en, n=2, top=10) → lead/headers │
└──────────────┬────────────────┘
               ▼
        ┌──────────────┐
        │ load_memory   │  ← SQLite, NO LLM
        └──────┬────────┘
               ▼
   ┌─►┌──────────────┐
   │  │   generate    │  ← LLM call (gemini-3.6-flash, temp 0.7) + memory_context
   │  │  what/why/how │     must cover: what it is, why it matters, how it works
   │  └──────┬────────┘
   │         ▼
   │  ┌──────────────┐
   │  │   evaluate    │  ← 1 batched LLM call (gemini-3.1-flash-lite) + rule signals
   │  └──────┬────────┘     textstat + wordfreq → LLM final call
   │         ▼
   │  ┌──────────────┐
   │  │  route_node   │  conditional edge (graph/build_graph.py:47)
   │  └──┬───────┬───┘
   │ fail│       │pass OR retries exhausted
   │     ▼       ▼
   │ ┌────────┐ ┌──────────────┐
   └─┤feedback│ │ write_memory │  ← SQLite, NO LLM
     │  _prep │ └──────┬───────┘  persists correction_applied + grounding
     └────────┘        ▼
                 ┌──────────────┐
                 │    output    │  ← draft + rejection_log{failed,why,changed} + trace
                 └──────────────┘
```

*Flow:* `plan_topic (RAKE→wiki per-keyword→disambig 0.3*jaccard+0.7*cosine threshold 0.35→relevance cosine 0.30→YAKE→lead/headers→mandatory_3) → load_memory → generate ⇄ evaluate → route → feedback_prep (correction_applied) → write_memory → output` — see `documents/ARCHITECTURE.md:11` and `documents/REPO_STRUCTURE.md:67`.

**LLM budget:** exactly **2 calls / attempt** (`generate` ×1 + batched `evaluate` ×1; `covers_key_points` is pure embedding) → 2 best case, 6 worst case (`max_retries=2`). `plan_topic`/`load_memory`/`write_memory` zero LLM.

**State:** `graph/state.py:21` — `LessonState` with `topic`, `domain_hint`/`effective_hint`, `wiki_page_title`, `reference_snippet` (~3000 chars, only text reaching LLM), `expected_key_points` (mandatory 3 + up to 2 wiki headers), `grounding_status/reason`, `rejection_log[].correction_applied`, `rubric_results`, `final_status`.

## Project Structure

```
lesson-content-generator/
├── README.md
├── pyproject.toml                # deps: rake-nltk, yake, wikipedia, textstat, wordfreq, langchain-google-genai, langgraph, langfuse, pydantic-settings
├── config.py                     # SINGLE source of truth (pydantic-settings)
├── .env.example                  # template for GEMINI_API_KEY, LANGFUSE, DOMAIN_HINT, DEBUG_FORCE_FAIL
├── main.py                       # CLI entrypoint (topic + --domain-hint + --max-retries + --debug-force-fail + --json-out)
├── notebooks/wiki-info-generation.ipynb
├── prompts/
│   ├── generate_system.md
│   ├── evaluate_system.md
│   └── feedback_template.md
├── graph/
│   ├── state.py
│   ├── build_graph.py
│   └── nodes/{plan_topic,load_memory,generate,evaluate,route,feedback_prep,write_memory,output}.py
├── llm/client.py                 # adapter generate_text/generate_structured + retry/backoff
├── rubric/{schema,checks}.py     # 6 checkpoints, embedding threshold
├── src/common/{decorators,embeddings,text_utils,logging_setup,errors}.py
├── memory/{store.py,memory.db}   # SQLite WAL (gitignored)
├── cache/wiki_cache.py           # wiki_cache table
├── tracing/langfuse_setup.py
├── tests/{test_plan_topic,test_rubric_checks,test_memory,test_graph_e2e}.py
├── outputs/lessons/*.json        # per-run lessons (gitignored)
├── output_analysis/              # timestamped ANALYSIS/LESSON mirrors (see output_analysis/README.md:1)
└── documents/{ARCHITECTURE,DESIGN,REPO_STRUCTURE}.md
```

See `documents/REPO_STRUCTURE.md:1` for full tree.

## Setup (venv isolated — no system-wide installs)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]          # jaccard-only, no sentence-transformers/torch (~500MB vs 2GB+)
# For semantic embeddings (optional):
pip install -e .[embeddings]   # adds sentence-transformers>=3.0 (pyproject.toml:21)

# NLTK data reused from ~/nltk_data (preserved, world-writable warning harmless):
python -c "import nltk; print(nltk.data.path)"
# If missing:
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

cp .env.example .env  # then fill GEMINI_API_KEY / GOOGLE_API_KEY, optional LANGFUSE keys, DOMAIN_HINT
# All deps are inside .venv — system site-packages were purged + pip cache cleared
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | **yes** for real LLM | `""` (`config.py:74`) | Alias; `config.py:86` `effective_api_key()` picks first set. Without it, `covers_key_points` uses lenient mock threshold `0.25` (`rubric/checks.py:58`) and lessons are mock-tagged. |
| `LANGFUSE_SECRET_KEY` | no | `""` | Tracing — no-op if unset (`tracing/langfuse_setup.py`) |
| `LANGFUSE_PUBLIC_KEY` | no | `""` | |
| `LANGFUSE_HOST` | no | `https://cloud.langfuse.com` (`config.py:78`) | |
| `DOMAIN_HINT` | no | `""` → `DOMAIN_HINT_DEFAULT` | User-supplied hint e.g. `AI/ML`; empty uses default per `config.py:89` |
| `DOMAIN_HINT_DEFAULT` | no | `artificial intelligence machine learning computer science` (`config.py:46`) | Baked-in AI/tech bias so scoring never silently disabled |
| `DEBUG_FORCE_FAIL` | no | `false` (`config.py:62`) | Corrupts attempt 1 only for Loom demo |
| `GENERATE_MODEL` | no | `gemini-3.6-flash` (`config.py:49`) | Override generate model |
| `EVALUATE_MODEL` | no | `gemini-3.1-flash-lite` (`config.py:52`) | Override evaluate model |
| `MAX_RETRIES` | no | `2` (`config.py:17`) | Guarantees termination |
| `DISABLE_EMBEDDINGS` | no | unset | `1/true/yes` → jaccard fallback (`src/common/embeddings.py:8`), threshold `0.25` |
| `ENABLE_WIKI_CACHE` | no | `true` (`config.py:60`) | SQLite `wiki_cache` table |
| `OUTPUT_ANALYSIS_MODE` | no | `always_new` (`config.py:65`) | `always_new`=timestamp every run (test), `once`=single overwrite |

See `.env.example:1` for template.

## Env tip — jaccard-only mode (low-disk, default)

```bash
# Embeddings disabled → 0.3*jaccard+0.7*cosine collapses to jaccard, covers_key_points uses lenient mock when GEMINI_API_KEY unset
export DISABLE_EMBEDDINGS=1
# To enable semantic mode later: pip install -e .[embeddings] && unset DISABLE_EMBEDDINGS

# Quick checks
DISABLE_EMBEDDINGS=1 python -c "from src.common.embeddings import cosine_similarity_text; print(cosine_similarity_text('hello world','hello'))"
DISABLE_EMBEDDINGS=1 python main.py "Photosynthesis" --json-out /tmp/demo.json
```

## Run

```bash
# Basic (uses DOMAIN_HINT_DEFAULT if no hint → AI/tech bias, never silently disabled)
python main.py "Introduction to RAG" --domain-hint "AI/ML"

# Without hint (still grounded via default)
python main.py "Photosynthesis"

# Deliberate-error demo for Loom (corrupts attempt 1 → evaluator catches → pass on retry)
python main.py "Introduction to RAG" --domain-hint "AI/ML" --debug-force-fail

# Custom retries
python main.py "MCP Servers with tool calling" --max-retries 1

# JSON output
python main.py "Introduction to RAG" --json-out outputs/lessons/run.json

# Using default topic (Introduction to RAG) with no args
python main.py
```

CLI args from `main.py:15`: `topic` (positional, default `Introduction to RAG`), `--domain-hint`, `--max-retries` (default `config.py:17`), `--debug-force-fail`, `--json-out`.

## Programmatic Usage

```python
from graph.build_graph import run

result = run("Introduction to RAG", domain_hint="AI/ML")
print(result["draft_lesson"][:500])
print(result["grounding_status"], result["grounding_reason"])
print(result["final_status"], result["overall_pass"])
print(result["rejection_log"])  # each entry: {attempt_num, failed_checks, correction_applied}

# With custom retries and debug flag
result = run("Photosynthesis", domain_hint="", max_retries=1, debug_force_fail=False)
```

## Output Contract

Every run returns (also printed by `main.py:25` + `graph/nodes/output.py`):

```json
{
  "topic": "Introduction to RAG",
  "wiki_page_title": "Retrieval-augmented generation",
  "grounding_status": "grounded | ambiguous | ungrounded",
  "grounding_reason": " | disambiguation_inconclusive | relevance_filtered | all_fetch_failed",
  "final_status": "passed | max_retries_exhausted",
  "overall_pass": true,
  "draft_lesson": "# ... markdown 500-1000 words, what/why/how ...",
  "rubric_results": [{"name": "accurate_and_grounded", "passed": true, "reason": "..."}],
  "rejection_log": [{"attempt_num": 1, "failed_checks": [...], "correction_applied": "Fix ...", "grounding_status": "..."}],
  "trace_url": "https://cloud.langfuse.com/... or '' if no keys",
  "generated_at": "2026-09-15T06:17:11+00:00"
}
```

*Always* ships last `draft_lesson` + `rejection_log` (each entry with `failed_checks`, `reason` as why, and `correction_applied` as what changed) + `trace_url`. Never bare refusal.

## Configuration

All thresholds in `config.py:15-79` (single source; docs `documents/ARCHITECTURE.md:253` only rationale). Env overrides via `pydantic-settings` (`config.py:80`).

| Parameter | Config key | Default | Rationale |
|---|---|---|---|
| Max retries | `MAX_RETRIES` | `2` | Per spec — guarantees termination |
| RAKE keywords | `RAKE_TOP_K` | `2` | `rake_keywords[:2]` per keyword search |
| YAKE extraction | `YAKE_TOP_K` / `YAKE_N` / `YAKE_LAN` | `10` / `2` / `en` | From `combined_wiki_content` |
| Key-points cap | `MAX_KEY_POINTS` | `5` (mandatory 3 + up to 2 wiki headers) | `mandatory_3` enforces what/why/how via `covers_key_points` |
| Reference snippet | `REFERENCE_SNIPPET_MAX_CHARS` | `3000` (~150 words) | Only text reaching LLM prompt |
| Max page content | `MAX_PAGE_CONTENT_CHARS` | `15000` / page | Hygiene cap for local YAKE/regex; raw never hits LLM |
| Disambiguation | `DISAMBIG_THRESHOLD` | `0.35` (`0.3*jaccard+0.7*cosine`) | Vs `topic + effective_hint`; below → `ambiguous/disambiguation_inconclusive` |
| Relevance filter | `RELEVANCE_THRESHOLD` | `0.30` (cosine) | Lead vs `topic + effective_hint`; below → drop page |
| Generate model | `GENERATE_MODEL` | `gemini-3.6-flash` | Drafting; swappable via env |
| Generate temp/tokens | `GENERATE_TEMPERATURE` / `GENERATE_MAX_TOKENS` | `0.7` / `4096` | Lesson 500-1000w |
| Evaluate model | `EVALUATE_MODEL` | `gemini-3.1-flash-lite` | Batched judge |
| Evaluate tokens | `EVALUATE_MAX_TOKENS` | `2048` | Batched 5-check JSON |
| Readability | `READABILITY_GRADE_MAX` | `8` | Flesch-Kincaid, 12th-grade India |
| Key-point match | `KEY_POINT_MATCH_THRESHOLD` | `0.6` (cosine) / `0.25` jaccard fallback | `all-MiniLM-L6-v2` or jaccard |
| Wiki stop-list | `WIKI_STOP_LIST` | `History, Etymology, See also, ...` (11 items) | Filtered from headers |
| Wiki cache | `ENABLE_WIKI_CACHE` | `true` | SQLite `wiki_cache` table |
| Debug | `DEBUG_FORCE_FAIL` | `false` | Corrupts attempt 1 only |

See `documents/ARCHITECTURE.md:253` for full table.

## Grounding & Retrieval

`graph/nodes/plan_topic.py` — `effective_hint = domain_hint if domain_hint else DOMAIN_HINT_DEFAULT`:

1. `RAKE` → `rake_keywords[:2]` → per-keyword `wikipedia.search` → `wikipedia.page` (auto_suggest, redirect).
2. `DisambiguationError` → score candidates vs `topic + " " + effective_hint` with `combined_disambig_score` (`src/common/embeddings.py:44`) `0.3*jaccard+0.7*cosine`; best ≥ `0.35` → fetch, else `ambiguous/disambiguation_inconclusive`.
3. **Relevance filter** — cosine `lead_paragraph` vs `topic + effective_hint` ≥ `0.30` → keep truncated `15k`; else drop. All dropped → `ambiguous/relevance_filtered`; none fetched → `ungrounded/all_fetch_failed`; else `grounded`.
4. `YAKE` → `yake_keywords`; lead paragraph via regex `\n== [^=]+ ==\n` else `\n\n` split; headers via `re.findall(r'\n(==+ [^=]+ ==+)\n')` filtered by `WIKI_STOP_LIST` → `expected_key_points = mandatory_3 + headers` capped `4-5`.
5. `reference_snippet` = lead truncated `~3000` chars — **only text reaching LLM**. `combined_wiki_content` never sent to LLM (hygiene).

## Rubric — 6 Hard Checks

`rubric/schema.py:36` `CHECKPOINT_NAMES` + `rubric/checks.py:1` (hard AND; `grounding_status` excluded from pass when `ungrounded` per fallbacks). `evaluate` batches 5 LLM-judged in one call (`rubric/checks.py:87`), `covers_key_points` is pure embedding (`rubric/checks.py:53`).

| # | Check | Signal | LLM vs Rule |
|---|---|---|---|
| 1 | `accurate_and_grounded` | Contradiction vs `reference_snippet` only; unverifiable ≠ fail | LLM (lenient if `grounding_status!=grounded`) |
| 2 | `beginner_friendly_language` | Flesch-Kincaid ≤8 (`textstat`, `rubric/checks.py:11`) + idiom check | LLM (grade as input) |
| 3 | `teaches_by_example` | Regex marker + genuine worked example (query→passages or numbers) | LLM |
| 4 | `no_unexplained_jargon` | `wordfreq` zipf <3.5 (`rubric/checks.py:24`) + RAKE candidates → LLM arbitrates + checks inline defs | LLM |
| 5 | `covers_key_points` | Embedding cosine per `expected_key_points` ≥0.6 (or ≥0.25 jaccard fallback `rubric/checks.py:58`) | Pure embedding, no LLM |
| 6 | `coherent_teaching_flow` | Order intro→why→mechanism→example→recap, no forward refs | LLM |

## Memory & Cache

- **Cross-run memory** `memory/store.py` — SQLite WAL `memory/memory.db` (`config.py:69`), schema `lesson_memory`, two-tier load (`graph/nodes/load_memory.py`): topic-agnostic + topic-specific by `wiki_page_title`/`effective_hint`, top 3 patterns injected as `memory_context` into `generate`. Graceful empty on missing/locked.
- **Wiki cache** `cache/wiki_cache.py` — table `wiki_cache` (`config.py:70`) keyed by page title, enabled (`config.py:60`), avoids redundant fetches / rate-limit.
- **Write** `graph/nodes/write_memory.py` persists every checkpoint + `correction_applied` + `domain_hint` + `grounding_status`; write failure logged, never fails run.

## Tracing

`Tracing` via `tracing/langfuse_setup.py` — span per node (`@with_langfuse_span`), `attempt_num` tagging, no-op if `LANGFUSE_*` unset (`documents/ARCHITECTURE.md:51`). `trace_url` in output.

## Outputs

- `outputs/lessons/*.json` — per-run lesson + `rejection_log` (gitignored; keep `.gitkeep` if needed)
- `output_analysis/` — timestamped mirrors (`output_analysis/README.md:1`):
  - `ANALYSIS_YYYYMMDD_HHMMSS.md` — deep dive vs requirements (every run)
  - `LESSON_YYYYMMDD_HHMMSS.md` + `.json` — only when `overall_pass && final_status=="passed"`
  - `ALL_RUNS_SUMMARY_*.md` — scan of `outputs/lessons/*.json` history
  - `LESSON_LATEST.md` — symlink/copy of latest passed lesson
  - Toggle via `OUTPUT_ANALYSIS_MODE` (`config.py:65`): `always_new` (test) vs `once` (single overwrite)
- `memory/memory.db` — SQLite WAL (`config.py:69`) — `*.db`, `*.db-wal`, `*.db-shm` gitignored
- `cache` via same DB `wiki_cache` table when `ENABLE_WIKI_CACHE=true`

## Tests

```bash
pytest -v                          # all 10 tests, jaccard-only by default (DISABLE_EMBEDDINGS=1 in tests)
pytest tests/test_graph_e2e.py -v  # includes DEBUG_FORCE_FAIL e2e (fail→correct→pass)
pytest tests/test_plan_topic.py -v # per-keyword search, disambig scoring, relevance filter, stop-list
pytest tests/test_rubric_checks.py -v
pytest tests/test_memory.py -v

# Lint / type
ruff check . && black --check . && mypy .
```

Tests set `DISABLE_EMBEDDINGS=1` (`tests/test_plan_topic.py:3`) to avoid heavy model in CI. Mock LLM paths used when `GEMINI_API_KEY` unset.

## Fallbacks

| Node | Condition | Fallback |
|---|---|---|
| `plan_topic` | `JSONDecodeError` / network / 429 / `PageError` | Skip keyword, continue |
| `plan_topic` | `DisambiguationError` best <0.35 | `ambiguous/disambiguation_inconclusive`, skip |
| `plan_topic` | Relevance <0.30 for all candidates | `ambiguous/relevance_filtered` (avoids pollution) |
| `plan_topic` | Lead too shallow / `""` | `split('\n\n')[0]` minimal + TextRank fallback |
| `plan_topic` | All fetch failed | `ungrounded/all_fetch_failed`; `accurate_and_grounded` excluded from AND, fields visible |
| `load_memory` | SQLite missing/locked/corrupt | Empty context, log, proceed |
| `write_memory` | Write lock | Log and continue |
| `generate` | `finish_reason=="length"` | Retry once at raised cap, no rubric retry consumed |
| `generate`/`evaluate` | Timeout / 429 | Exponential backoff 3 attempts; auth fail-fast |
| `evaluate` | Malformed structured JSON | One auto re-call |
| `route` | Same check fails every attempt | Ship last draft, `final_status="max_retries_exhausted"` |
| `generate` | `DEBUG_FORCE_FAIL=true` | Corrupt attempt 1 prompt only, attempt 2 clean |

See `documents/ARCHITECTURE.md:231` for full table.

## License

MIT — see `pyproject.toml:7`.

---

**Docs:** `documents/ARCHITECTURE.md` (HLD/LLD/flows), `documents/DESIGN.md` (rubric/memory rationale), `documents/REPO_STRUCTURE.md` (layout). Raw `combined_wiki_content` (up to 109k) never reaches LLM — only `reference_snippet` + `expected_key_points`.

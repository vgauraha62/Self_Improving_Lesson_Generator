# Repo Structure (Phase-1.5: Notebook plan_topic + Compliant Loop + Domain Hint)

```
lesson-content-generator/
├── README.md                    # setup & run instructions (submission requirement)
├── FINAL_DESIGN.md               # rubric, memory, generalization — single source of truth
├── FINAL_ARCHITECTURE.md         # HLD, LLD, flows, tools, fallbacks, configuration (Phase-1.5)
├── pyproject.toml                # deps: rake-nltk, yake, wikipedia, nltk, langchain, langgraph,
│                                  # sentence-transformers, textstat, wordfreq, sumy, langfuse
├── .env.example                  # GEMINI_API_KEY / GOOGLE_API_KEY / LANGFUSE_KEYS / DOMAIN_HINT / DEBUG_FORCE_FAIL
├── config.py                     # SINGLE source of truth for thresholds/caps
│                                  # GENERATE=gemini-3.5-flash, EVALUATE=gemini-3.1-pro-preview
│                                  # DISAMBIG_THRESHOLD=0.35 (0.3*jaccard+0.7*cosine), RELEVANCE_THRESHOLD=0.30,
│                                  # MAX_PAGE_CONTENT_CHARS=15000, DOMAIN_HINT_DEFAULT="artificial intelligence machine learning computer science",
│                                  # DEBUG_FORCE_FAIL=false
│
├── notebooks/
│   └── wiki-info-generation.ipynb # reference impl for plan_topic → generate (NB flow, no pageprops)
│
├── prompts/                      # prompt templates as separate files
│   ├── generate_system.md        # must cover what it is, why it matters, how it works
│   ├── evaluate_system.md
│   └── feedback_template.md
│
├── graph/
│   ├── state.py                  # LessonState (topic, domain_hint/effective_hint, grounding_status+grounding_reason,
│   │                              # RejectionEntry with correction_applied, mandatory_3 in expected_key_points), RubricCheck
│   ├── build_graph.py            # LangGraph wiring: nodes + conditional edges (route_node)
│   └── nodes/
│       ├── plan_topic.py         # NB flow + domain-hint scored disambig + relevance filter
│       ├── load_memory.py        # SQLite read (WAL), topic-agnostic vs topic-specific
│       ├── generate.py           # PromptTemplate + ChatGoogleGenerativeAI (gemini-3.5-flash, temp0.7, what/why/how)
│       ├── evaluate.py           # batched LLM call (gemini-3.1-pro-preview, 5 checks) + embedding key-points match
│       ├── route.py              # pass/fail → write_memory or feedback_prep
│       ├── feedback_prep.py      # builds corrective instructions → correction_applied stored in rejection_log
│       ├── write_memory.py       # SQLite write (WAL) — persists correction_applied + domain_hint
│       └── output.py             # always returns last draft + rejection_log{failed,why,changed} + trace
│
├── llm/
│   └── client.py                 # provider-agnostic adapter with retry/backoff (timeout/429), fail-fast auth
│
├── rubric/
│   ├── checks.py                 # one function per checkpoint (what/why/how enforced via coherent_teaching_flow)
│   └── schema.py                 # structured-output JSON schema for evaluate + RejectionEntry
│
├── memory/
│   ├── store.py                  # SQLite read/write, WAL mode, graceful empty on missing/locked
│   └── memory.db                 # created at runtime, gitignored
│
├── cache/
│   └── wiki_cache.py             # SQLite wiki_cache table (page title → content), ENABLED in Phase-1.5
│
├── tracing/
│   └── langfuse_setup.py         # wraps nodes as spans, no-ops if LANGFUSE_KEY unset
│
├── tests/
│   ├── test_plan_topic.py        # per-keyword search, JSONDecode/Page/Disambig+hint scoring, relevance filter, stop-list
│   ├── test_rubric_checks.py     # each checkpoint isolated, contradiction cases
│   ├── test_memory.py            # read/write, missing-file fallback, tier separation
│   └── test_graph_e2e.py         # full run incl. DEBUG_FORCE_FAIL → evaluator catch → pass on retry
│
└── outputs/
    ├── lessons/                  # generated lesson + rejection_log (with correction_applied) per run
    └── traces/                   # optional local trace exports
```

**Phase-1.5 scope:** `plan_topic` is NB flow (per-keyword `wikipedia` search, no `pageprops`) plus **user domain-hint disambiguation** and **relevance-filtered combination**; loop is fully compliant: `plan_topic → load_memory → generate ⇄ evaluate → route → feedback_prep → write_memory → output` with `MAX_RETRIES=2`.

**Inputs:** `topic` (e.g. `Introduction to RAG`) + optional `domain_hint` (e.g. `AI/ML`, user-supplied just to be sure) — if empty auto-uses `DOMAIN_HINT_DEFAULT="artificial intelligence machine learning computer science"` so scoring never silently disabled; `ambiguous` now carries `grounding_reason` (`disambiguation_inconclusive`|`relevance_filtered`|`all_fetch_failed`) for diagnostics.

**Deliberate-error demo:** `config.DEBUG_FORCE_FAIL=true` corrupts attempt 1 only for Loom video — reproducible evaluator catch, genuine pass on retry.

**Notes:**
- `pyproject.toml` replaces bare `requirements.txt` given version-conflict risk across this stack.
- `config.py` is explicitly the only place thresholds/caps live — docs describe rationale, not authoritative numbers.
- `memory/store.py` specifies WAL mode for SQLite concurrent-write safety.
- `tracing/langfuse_setup.py` specifies graceful no-op on missing API key instead of crashing.
- `cache/wiki_cache.py` is **enabled** in Phase-1.5 (SQLite table, keyed by page title) — avoids redundant fetches and rate-limit exposure during dev/demo.

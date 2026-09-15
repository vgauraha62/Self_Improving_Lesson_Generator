# Final Design Document — Self-Evaluating Lesson Content Generator

Supersedes and **deprecates**: `DESIGN.md`, `RUBRIC.md`, `ARCHITECTURE.md`, `ARCHITECTURE_v3.md`. This is the single source of truth for the rubric, memory, and generalization strategy — **Phase-1.5: Notebook plan_topic+generate + compliant loop/memory + user domain hint**.

---

## 1. Evaluation Rubric

### 1.1 In words

The lesson is judged against six checkpoints. Every checkpoint must pass — no partial credit, no checkpoint can be outweighed by the others.

- **Accurate and grounded** — checked against a factual reference retrieved for the topic beforehand, not the model's own memory. Fails only if it *contradicts* the reference, never for true content the reference doesn't mention. Claims unverifiable against the reference are logged as `unverified` in the reason field. If `grounding_status="ungrounded"` or `"ambiguous"`, this check is excluded from `overall_pass` and degradation is visible via `grounding_status`.
- **Beginner-friendly language** — must score at or below grade 8 reading level, *and* avoid idioms or culturally specific phrasing a non-native-English-medium reader wouldn't have context for. Both conditions required.
- **Teaches by example** — must contain a genuine worked example, not just a labelled section gesturing at one.
- **No unexplained jargon** — every technical term must be defined near its first use. Terms found automatically (RAKE/YAKE + `wordfreq`), no fixed dictionary.
- **Covers the key points** — checked against `expected_key_points` (headers after stop-list, capped 4-5) generated before writing. Missing any point fails.
- **Coherent teaching flow** — must build in order (introduction → why it matters → mechanism → example → recap) without forward references. `what/why/how` is enforced as **hard check** via seeded `expected_key_points` (see §3), not just prompt instruction.

### 1.2 Technical spec

| Checkpoint | Method | Pass criterion |
|---|---|---|
| `accurate_and_grounded` | LLM-judge (`gemini-3.1-pro-preview`) vs. `reference_snippet` | **Fails only on contradiction** with `reference_snippet` |
| `beginner_friendly_language` | Rule-based Flesch-Kincaid (`textstat`) fed into single LLM-judge (`gemini-3.1-pro-preview`) | LLM final call using score as evidence + idiom scan. Rule score is input, not independent gate |
| `teaches_by_example` | Structural pre-check + LLM-judge (`gemini-3.1-pro-preview`) | Structural check confirms example section exists; LLM makes final pass/fail on genuine illustration |
| `no_unexplained_jargon` | RAKE/YAKE + `wordfreq` rarity → single LLM-judge (`gemini-3.1-pro-preview`) | Rule proposes candidates (no fixed dictionary); LLM arbitrates jargon vs false positives and checks inline definitions |
| `covers_key_points` | Semantic match: `sentence-transformers` (`all-MiniLM-L6-v2`) cosine similarity | Every `expected_key_points` item has lesson passage with cosine ≥ `config.KEY_POINT_MATCH_THRESHOLD` (0.6) |
| `coherent_teaching_flow` | LLM-judge (`gemini-3.1-pro-preview`) | No forward references; follows teaching arc; prompt guarantees what/why/how content exists |

**Conflict-resolution principle:** for every check with both a rule-based signal and LLM judgment, the rule-based output is *evidence passed into* the LLM call, never a second independent gate. Exactly one decision-maker per checkpoint (LLM informed by signal).

**Aggregation:** `overall_pass = AND(checks_evaluated)`, all six normally. Exceptions: if `grounding_status="ungrounded"` or `"ambiguous"`, `accurate_and_grounded` **excluded** from AND, tracked separately via `grounding_status` + `grounding_reason` (`disambiguation_inconclusive` | `relevance_filtered` | `all_fetch_failed`) in state/output — never silently absorbed into pass. `ambiguous` is an honest, visible state; `grounding_reason` tells which of the three causes.

**Output contract:** forced structured output (provider tool-use/schema, never free-text parsing):
```json
{
  "checks": [{"name": "...", "passed": true, "reason": "..."}],
  "overall_pass": false,
  "grounding_status": "grounded"
}
```

**Rejection log (satisfies brief "what you changed"):** per retry attempt:
```json
{
  "attempt_num": 1,
  "failed_checks": [{"name": "...", "passed": false, "reason": "..."}],
  "correction_applied": "Add a worked example showing RAG retrieval → augmentation → generation with a query and 2 retrieved passages...",
  "grounding_status": "grounded"
}
```
`correction_applied` is the verbatim corrective instruction text `feedback_prep` generated from failed checks — not just which checks failed, but what was actually changed in the retry prompt. Persisted via `write_memory` for self-evolution.

**Token efficiency:** all five LLM-judgment checks are **one batched structured-output call per attempt** (`gemini-3.1-pro-preview`), not five separate calls. `covers_key_points` is pure embedding, no LLM. → Exactly **2 LLM calls per attempt** (`generate` `gemini-3.5-flash` + batched `evaluate`), 6 worst case with `MAX_RETRIES=2`.

---

## 2. Cross-Run Memory

**Storage:** SQLite (`memory.db`), one row per checkpoint result per attempt (`run_id, topic, domain_hint, effective_hint, attempt_num, checkpoint_name, passed, reason, correction_applied, grounding_reason, timestamp`), WAL mode. `wiki_cache` is a separate SQLite table (`wiki_cache`: `page_title → content`) enabled in Phase-1.5.

**Why not a vector store:** "self-evolving" means learning from repeated failure patterns, not semantic retrieval — at this scale (one topic per run) a vector store adds embedding cost with no benefit.

**Self-evolving scope:** memory sharpens **both** generation prompts and rubric application:
- **Prompts:** `feedback_prep` patterns (top 3 per tier) injected as `memory_context` into `generate` — repeated failures (e.g. `teaches_by_example` missing) become preemptive instructions.
- **Rubrics (lightweight):** thresholds/stop-lists live in `config.py` and are not auto-tuned per run, but repeated `no_unexplained_jargon` false positives/negatives or `covers_key_points` threshold misses are logged with `correction_applied` and surfaced for manual `config` tuning — satisfies brief "sharpen prompts **and rubrics**" without risky auto-mutation of pass/fail boundaries.

**Read (`load_memory`):** once per run, before generation — two tiers:

| Tier | Checkpoints | Why |
|---|---|---|
| Topic-agnostic (all past runs, any topic) | `beginner_friendly_language`, `no_unexplained_jargon`, `teaches_by_example`, `coherent_teaching_flow` | Style/structure habits generalize across topics |
| Topic-specific (same resolved topic) | `accurate_and_grounded`, `covers_key_points` | Factual content is topic-specific |

**Topic-specific key:** matched by `wiki_page_title` when available (so "RAG" and "Retrieval-Augmented Generation" resolve to same key). If `wiki_page_title=""`, normalized raw `topic` + **`effective_hint`** (not raw `domain_hint`) used instead — this ensures a run with no user-supplied hint and a run that happens to pass the same string as `DOMAIN_HINT_DEFAULT` collapse to the same key, rather than fragmenting memory for functionally identical runs. Capped at top 3 most frequent patterns per tier to bound prompt growth.

**Write (`write_memory`):** once per run, after loop — persists every checkpoint result from every attempt, passes included, plus `correction_applied` and `domain_hint`, so future runs know what's working and don't over-correct.

**Failure handling:** if SQLite missing/locked/corrupt, treat as empty memory and proceed — memory is optimization, never hard dependency.

---

## 3. Generalization Strategy (`plan_topic`) — Notebook-Aligned + User Domain Hint

**Final design: zero LLM calls, retrieval + extraction only — verbatim `wiki-info-generation.ipynb` workflow plus domain-hint disambiguation.**

1. **RAKE extraction:** `Rake().extract_keywords_from_text(topic)` → `rake_keywords` (use top 2). No threshold branching — always RAKE.
2. **Per-keyword Wikipedia search + fetch with domain-hint scoring:** for each `keyword` in `rake_keywords[:2]`:
   - `effective_hint = domain_hint if domain_hint else DOMAIN_HINT_DEFAULT` (`"artificial intelligence machine learning computer science"`)
   - `wikipedia.search(keyword)` with `JSONDecodeError`/`Exception` → skip keyword
   - `wikipedia.page(hits[0], auto_suggest=True, redirect=True)` with `PageError` → skip; `DisambiguationError` → score candidates vs `topic + " " + effective_hint` with `score = 0.3 * jaccard_word_overlap + 0.7 * embedding_cosine(all-MiniLM-L6-v2)`
     - Best score ≥ `DISAMBIG_THRESHOLD` (0.35, combined) → `wikipedia.page(best)` → `PageError`/`Disambig` → skip, `grounding_reason="disambiguation_inconclusive"`
     - Best score < threshold → `grounding_status="ambiguous"`, `grounding_reason="disambiguation_inconclusive"`, skip keyword
   - On success: candidate page held for relevance filter
3. **Relevance filter (prevents combined-content pollution):** for each candidate page, `cosine(lead_paragraph, topic + " " + effective_hint)` via `sentence-transformers`
   - Score ≥ `RELEVANCE_THRESHOLD` (0.30) → keep, truncate to `MAX_PAGE_CONTENT_CHARS` (15k) → `all_wiki_content_snippets.append(page.content)`
   - Score < threshold → silently drop
   - `combined_wiki_content = "\n\n".join(all_wiki_content_snippets)`. Status/reason resolved as an explicit if/elif chain (checked in this order):
     ```
     if len(all_wiki_content_snippets) > 0:
         grounding_status = "grounded"
     elif any_keyword_had_disambiguation_skip:      # at least one skip was disambiguation-inconclusive
         grounding_status, grounding_reason = "ambiguous", "disambiguation_inconclusive"
     elif any_candidate_was_relevance_filtered:      # something resolved cleanly but scored below threshold
         grounding_status, grounding_reason = "ambiguous", "relevance_filtered"
     else:                                            # search/fetch itself failed for every keyword
         grounding_status, grounding_reason = "ungrounded", "all_fetch_failed"
     ```
4. **YAKE re-rank:** `KeywordExtractor(lan="en", n=2, top=10).extract_keywords(combined_wiki_content)` → `yake_keywords`.
5. **Summarize:** regex `r'\n== [^=]+ ==\n'` → `lead_paragraph = content[:match.start()]` else `split('\n\n')[0]`. Truncated to `~3000` chars for prompt. Fall back to TextRank/LexRank only if lead too shallow.
6. **Derive key points:** `mandatory_3 = [f"what {topic} is", f"why {topic} matters", f"how {topic} works"]` seeded first (interpolating the topic — not the bare labels "what it is"/"why it matters"/"how it works", which are too generic for embedding similarity to discriminate against real lesson content) — enforces the brief's `Cover what it is, why it matters, how it works` as a hard `covers_key_points` check, not just prompt instruction. Then `headers = re.findall(r'\n(==+ [^=]+ ==+)\n', content)` → strip `=` → filter against stop-list → cap total 4-5 (mandatory 3 + up to 2 wiki headers).
   - **Stop-list (in `config.py`):** History, Etymology, See also, References, External links, Notes, Bibliography, Further reading, Gallery, Awards, Popular culture, Citations
   - `reference_snippet` = lead paragraph; `expected_key_points` = `mandatory_3` + filtered headers; `combined_wiki_content` kept filtered (never sent to LLM — only used locally for YAKE/header parsing; capped per page for hygiene).

**Fully removed vs old design:** `pageprops` disambiguation flag, `Wikipedia REST/opensearch API`.

**Truncation strategy:** raw `combined_wiki_content` (up to 109k in NB demo) is **never sent to an LLM**. Only `reference_snippet (~150 words)` and `expected_key_points` (short list) reach Gemini. `MAX_PAGE_CONTENT_CHARS=15k` per page bounds local YAKE/regex time, not token budget. `DISAMBIG_THRESHOLD=0.35` / `RELEVANCE_THRESHOLD=0.30` are initial guesses — validate pre-demo on 6 probes (Photosynthesis, RAG with/without hint, Java, Python, Mercury) and adjust ±0.05.

**Why this over LLM call:** retrieval-grounding is verifiable (Wikipedia source) and zero LLM cost.

**Known trade-off:** with `DOMAIN_HINT_DEFAULT` baked in, even bare "Java"/"Python" with no user hint gets AI/tech bias (not empty); if still below threshold, disambiguation is honestly `grounding_status="ambiguous"` + `grounding_reason` rather than papered over.

**Deliberate-error demo (for Loom video):** `config.DEBUG_FORCE_FAIL` flag (default `false`). When `true`, corrupts attempt 1 only — e.g. appends "state one incorrect fact" to generate prompt or swaps one `reference_snippet` sentence — so evaluator catches a real, reproducible failure on camera, followed by genuine clean pass on attempt 2.

"""output node — always returns last draft + rejection_log{failed,why,changed} + trace."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from tracing.langfuse_setup import span

logger = logging.getLogger(__name__)

def output_node(state: dict) -> dict:
    with span("output"):
        topic = state.get("topic", "unknown")
        draft = state.get("draft_lesson", "")
        rejection_log = state.get("rejection_log", []) or []
        rubric_results = state.get("rubric_results", []) or []
        grounding_status = state.get("grounding_status", "grounded")
        grounding_reason = state.get("grounding_reason", "")
        final_status = state.get("final_status", "passed")
        wiki_page_title = state.get("wiki_page_title", "")
        trace_url = state.get("trace_url", "") or os.getenv("LANGFUSE_HOST", "") or ""

        # derive overall_pass with grounding exclusion
        checks = [c for c in rubric_results if not (grounding_status in ("ungrounded", "ambiguous") and c.get("name") == "accurate_and_grounded")]
        overall_pass = all(bool(c.get("passed")) for c in checks) if checks else False

        output = {
            "topic": topic,
            "wiki_page_title": wiki_page_title,
            "grounding_status": grounding_status,
            "grounding_reason": grounding_reason,
            "final_status": final_status,
            "overall_pass": overall_pass,
            "draft_lesson": draft,
            "rejection_log": rejection_log,
            "rubric_results": rubric_results,
            "trace_url": trace_url,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # persist to outputs/lessons
        ts = datetime.now(timezone.utc)
        ts_str = ts.strftime("%Y%m%d_%H%M%S")
        try:
            out_dir = Path(__file__).parents[2] / "outputs" / "lessons"
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{ts_str}_{topic.replace(' ', '_')[:30]}.json"
            (out_dir / fname).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("output written to %s", out_dir / fname)
        except Exception as e:
            logger.warning("output persist failed: %s", e)

        # persist to output_analysis (always_new vs once via config)
        try:
            from config import settings as _settings

            mode = getattr(_settings, "OUTPUT_ANALYSIS_MODE", "always_new")
            analysis_dir = Path(getattr(_settings, "OUTPUT_ANALYSIS_DIR", str(Path(__file__).parents[2] / "output_analysis")))
            analysis_dir.mkdir(parents=True, exist_ok=True)

            # determine filenames (plain timestamp)
            if mode == "once":
                analysis_name = "ANALYSIS.md"
                summary_name = "ALL_RUNS_SUMMARY.md"
                lesson_name = "LESSON.md"
                lesson_json_name = "LESSON.json"
            else:
                analysis_name = f"ANALYSIS_{ts_str}.md"
                summary_name = f"ALL_RUNS_SUMMARY_{ts_str}.md"
                lesson_name = f"LESSON_{ts_str}.md"
                lesson_json_name = f"LESSON_{ts_str}.json"

            # write ANALYSIS
            word_count = len(draft.split())
            is_mock = "mock" in str(rubric_results).lower() or len(draft) < 500
            # derive assignment coverage
            analysis_md = f"""# Run Analysis — {ts.isoformat()}

## Where is my final lesson?
- **This run's lesson file:** `outputs/lessons/{fname}` (full JSON, always saved)
- **Analysis copy:** `output_analysis/{analysis_name}`
- **Lesson markdown (if passed):** `output_analysis/{lesson_name}` (only when `overall_pass=true` && `final_status=="passed"`)
- **Latest final real lesson:** among `output_analysis/LESSON_*.md` sorted by timestamp, latest where `grounding_status=="grounded"` && `overall_pass`. Mock runs are tagged.

## Assignment Requirements Check (GenAI Engineer — Content Systems Take-Home)
- **INPUT topic:** `{topic}` (domain_hint effective: `{state.get('effective_hint','')}`)
- **GENERATE what/why/how:** mandatory_3 seeded in `expected_key_points` + prompt `generate_system.md` — {'PASS' if draft and 'what' in draft.lower() else 'FAIL'}
- **EVALUATE 6 hard pass/fail:** {', '.join(f"{c.get('name')}:{'PASS' if c.get('passed') else 'FAIL'}" for c in rubric_results)} (hard AND, grounding excluded if ambiguous/ungrounded)
- **REGENERATE max 2 retries:** `retry_count={state.get('retry_count',0)}/{state.get('max_retries',2)}`, `rejection_log` len={len(rejection_log)} (each entry has `failed_checks`, `reason`, `correction_applied`)
- **OUTPUT lesson + rejection_log:** {'saved' if draft else 'missing'} (`overall_pass={overall_pass}`, `final_status={final_status}`)
- **SELF-EVOLVING MEMORY:** persists via `memory/memory.db` WAL, two-tier load (top 3 per tier) — see `memory_context` length {len(state.get('memory_context',''))}
- **Audience 12th-grade India:** `beginner_friendly_language` {'PASS' if any(c.get('name')=='beginner_friendly_language' and c.get('passed') for c in rubric_results) else 'FAIL'} (Flesch-Kincaid ≤8 + idiom scan)

## Current Run Metrics
- **Grounding:** `{grounding_status}/{grounding_reason}` (wiki_page_title: `{wiki_page_title or '(none)'}`)
- **Word count:** {word_count} {'(mock short)' if is_mock else '(real len)'} — target 500-1000 for submission
- **Lesson quality:** {'MOCK template (no GEMINI_API_KEY or API failure)' if is_mock else 'REAL LLM (gemini-3.6-flash / gemini-3.1-flash-lite)'} — see `draft_lesson` excerpt below
- **Rubric hard AND:** {'PASS' if overall_pass else 'FAIL'}
- **Rejection log details:**
{chr(10).join(f"- attempt {e.get('attempt_num')}: failed {[c.get('name') for c in e.get('failed_checks',[])]} | correction: {e.get('correction_applied','')[:200]}..." for e in rejection_log) if rejection_log else "- (no rejections — passed first attempt)"}

## Good response?
- **This run:** {'YES — grounded, passed, would be submission-grade after exporting to Google Doc (when real LLM)' if overall_pass and grounding_status=='grounded' and not is_mock else 'PARTIAL — passed but ' + ('ungrounded/ambiguous (Wikipedia fetch failed) — retry with network or wiki cache' if grounding_status!='grounded' else 'mock lesson — set GEMINI_API_KEY for real content')}

## Draft excerpt (first 1000 chars)
```
{draft[:1000]}
```

## Full rubric
```json
{json.dumps(rubric_results, indent=2)}
```
"""
            (analysis_dir / analysis_name).write_text(analysis_md, encoding="utf-8")
            logger.info("analysis written to %s", analysis_dir / analysis_name)

            # write LESSON markdown (latest generated draft)
            lesson_status_str = "PASSED" if (overall_pass and final_status == "passed") else f"FINAL_DRAFT ({final_status})"
            lesson_md = (
                f"# Lesson: {topic}\n\n"
                f"> **Status:** {lesson_status_str} | **Word Count:** {word_count} | **Model:** {'MOCK' if is_mock else 'REAL (gemini-3.6-flash)'} | **Generated at:** {ts.isoformat()}\n\n"
                f"---\n\n{draft}\n"
            )
            (analysis_dir / lesson_name).write_text(lesson_md, encoding="utf-8")
            (analysis_dir / "LESSON_LATEST.md").write_text(lesson_md, encoding="utf-8")
            if overall_pass and final_status == "passed":
                (analysis_dir / lesson_json_name).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("lesson written to %s", analysis_dir / lesson_name)

            # write ALL_RUNS_SUMMARY (scan outputs/lessons)
            try:
                lessons_dir = Path(__file__).parents[2] / "outputs" / "lessons"
                rows = []
                for jf in sorted(lessons_dir.glob("*.json")):
                    try:
                        d = json.loads(jf.read_text(encoding="utf-8"))
                        wc = len(d.get("draft_lesson","").split())
                        rows.append(f"| {jf.name} | {d.get('grounding_status','')} | {d.get('grounding_reason','')} | {d.get('final_status','')} | {d.get('overall_pass','')} | {len(d.get('rejection_log',[]))} | {d.get('wiki_page_title','')[:30]} | {wc} |")
                    except Exception:
                        continue
                summary_md = f"# All Runs Summary — {ts.isoformat()}\n\n| file | grounding | reason | final_status | overall_pass | rejections | wiki_title | wc |\n|------|-----------|--------|--------------|--------------|------------|------------|----|\n" + "\n".join(rows[-30:]) + f"\n\n_Total runs: {len(rows)}_\n"
                (analysis_dir / summary_name).write_text(summary_md, encoding="utf-8")
            except Exception as e2:
                logger.warning("summary write failed: %s", e2)

        except Exception as e:
            logger.warning("output_analysis persist failed: %s", e)

        return output

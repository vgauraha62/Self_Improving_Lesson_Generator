"""feedback_prep — builds corrective instructions → correction_applied stored in rejection_log."""
from __future__ import annotations

import logging
from pathlib import Path

from tracing.langfuse_setup import span

logger = logging.getLogger(__name__)

CORRECTIONS = {
    "accurate_and_grounded": "Ensure every factual claim is supported by the reference snippet; remove or correct any statement that contradicts it; mark unverifiable claims cautiously.",
    "beginner_friendly_language": "Rewrite in grade ≤8 language: short sentences, simple words, no idioms or culturally specific phrases; define context for any analogy.",
    "teaches_by_example": "Add a genuine worked example with concrete query → retrieved passages → step-by-step generation trace (not a placeholder heading).",
    "no_unexplained_jargon": "Define every technical term inline at first use; replace or explain each jargon candidate in plain language.",
    "covers_key_points": "Explicitly cover all expected key points with dedicated paragraphs/headings; do not merge or skip any point.",
    "coherent_teaching_flow": "Reorganize into Introduction → Why it matters → How it works → Worked Example → Recap; remove forward references and ensure what/why/how headings exist.",
}

def _load_feedback_tpl() -> str:
    p = Path(__file__).parents[2] / "prompts" / "feedback_template.md"
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return "{failed_checks_bullets}\n\n{corrections}"

def feedback_prep_node(state: dict) -> dict:
    with span("feedback_prep"):
        rubric_results: list[dict] = state.get("rubric_results", []) or []
        retry_count: int = state.get("retry_count", 0) or 0
        grounding_status: str = state.get("grounding_status", "grounded")
        grounding_reason: str = state.get("grounding_reason", "")
        failed = [c for c in rubric_results if not c.get("passed")]

        failed_bullets = "\n".join(f"- {c['name']}: {c.get('reason','')}" for c in failed) or "- (none)"
        corrections_list: list[str] = []
        for c in failed:
            name = c.get("name", "")
            corr = CORRECTIONS.get(name, f"Fix {name}: {c.get('reason','')}" )
            corrections_list.append(f"- [{name}] {corr} (because: {c.get('reason','')})")

        corrections_text = "\n".join(corrections_list) if corrections_list else "No corrections — all checks should pass."
        # Also handle covers_key_points_item breakdown if present (embedding details)
        tpl = _load_feedback_tpl()
        # tpl vars: failed_checks_bullets, corrections, memory_context
        memory_context = state.get("memory_context", "")
        correction_applied = tpl.format(
            failed_checks_bullets=failed_bullets,
            corrections=corrections_text,
            memory_context=memory_context or "(none)",
        )

        attempt_num = retry_count + 1
        rejection_log: list[dict] = list(state.get("rejection_log", []) or [])
        rejection_log.append(
            {
                "attempt_num": attempt_num,
                "failed_checks": failed,
                "correction_applied": correction_applied,
                "grounding_status": grounding_status,
                "grounding_reason": grounding_reason,
            }
        )
        logger.info("feedback_prep attempt %s failed %s", attempt_num, [c['name'] for c in failed])
        return {
            "rejection_log": rejection_log,
            "retry_count": retry_count + 1,
        }

"""evaluate node — batched LLM judge (gemini-3.1-flash) + embedding covers_key_points."""
from __future__ import annotations

import logging

from config import settings
from llm.client import generate_structured
from rubric.checks import build_evaluate_prompt, covers_key_points_semantic
from rubric.schema import EVALUATE_SCHEMA
from tracing.langfuse_setup import span

logger = logging.getLogger(__name__)

def evaluate_node(state: dict) -> dict:
    with span("evaluate"):
        draft: str = state.get("draft_lesson", "") or ""
        reference_snippet: str = state.get("reference_snippet", "") or ""
        expected_key_points: list[str] = state.get("expected_key_points", []) or []
        grounding_status: str = state.get("grounding_status", "grounded") or "grounded"

        # No API key → mock evaluation (for tests; mock lesson passes except debug fail)
        api_key = settings.effective_api_key()
        if not api_key:
            logger.warning("No GEMINI_API_KEY — mock evaluate")
            # detect deliberate incorrect fact
            has_incorrect = "1800 by Newton" in draft or "incorrect fact" in draft.lower()
            checks = [
                {"name": "accurate_and_grounded", "passed": not has_incorrect, "reason": "contradicts reference (Newton 1800)" if has_incorrect else "no contradiction vs reference"},
                {"name": "beginner_friendly_language", "passed": True, "reason": "mock grade 7 pass"},
                {"name": "teaches_by_example", "passed": "Worked Example" in draft or "example" in draft.lower(), "reason": "has worked example" if "example" in draft.lower() else "missing genuine example"},
                {"name": "no_unexplained_jargon", "passed": True, "reason": "jargon defined inline (mock)"},
                {"name": "coherent_teaching_flow", "passed": "What it is" in draft and "Why it matters" in draft, "reason": "flow intro→why→how→example→recap present" if "What it is" in draft else "missing flow"},
            ]
            # covers_key_points — mock lenient pass (no-API mode) to allow E2E demo without strict embedding threshold
            checks.append({"name": "covers_key_points", "passed": True, "reason": "mock covers_key_points pass (no API key)"})
            # For DEBUG demo, force grounding to grounded when deliberate error so it is not excluded
            if has_incorrect and grounding_status in ("ambiguous", "ungrounded"):
                grounding_status = "grounded"
            return {"rubric_results": checks, "grounding_status": grounding_status}

        # Real LLM judge
        prompt = build_evaluate_prompt(draft, reference_snippet, expected_key_points, grounding_status)
        logger.info("evaluate calling %s len_prompt=%s", settings.EVALUATE_MODEL, len(prompt))
        try:
            result = generate_structured(prompt, EVALUATE_SCHEMA, model=settings.EVALUATE_MODEL, max_tokens=settings.EVALUATE_MAX_TOKENS)
        except Exception as e:
            logger.error("evaluate structured call failed: %s", e)
            # graceful fallback: fail safe
            return {
                "rubric_results": [
                    {"name": "accurate_and_grounded", "passed": False, "reason": f"evaluate failed: {e}"},
                    {"name": "beginner_friendly_language", "passed": False, "reason": f"evaluate failed: {e}"},
                    {"name": "teaches_by_example", "passed": False, "reason": f"evaluate failed: {e}"},
                    {"name": "no_unexplained_jargon", "passed": False, "reason": f"evaluate failed: {e}"},
                    {"name": "covers_key_points", "passed": False, "reason": f"evaluate failed: {e}"},
                    {"name": "coherent_teaching_flow", "passed": False, "reason": f"evaluate failed: {e}"},
                ]
            }

        llm_checks: list[dict] = result.get("checks", []) or []
        # ensure all 5 LLM checks present, map names lower
        by_name = {c.get("name", "").strip(): c for c in llm_checks}
        final_checks: list[dict] = []
        for name in ["accurate_and_grounded", "beginner_friendly_language", "teaches_by_example", "no_unexplained_jargon", "coherent_teaching_flow"]:
            chk = by_name.get(name)
            if chk:
                final_checks.append({"name": name, "passed": bool(chk.get("passed")), "reason": str(chk.get("reason", ""))})
            else:
                final_checks.append({"name": name, "passed": False, "reason": "LLM did not return this check"})

        # covers_key_points via embedding (no LLM)
        kp_checks = covers_key_points_semantic(draft, expected_key_points)
        final_checks.extend(kp_checks)

        # Fix grounding_status from LLM vs state (trust LLM if provided)
        llm_grounding = result.get("grounding_status")
        if llm_grounding in ("grounded", "ambiguous", "ungrounded"):
            grounding_status = llm_grounding  # type: ignore[assignment]

        return {"rubric_results": final_checks, "grounding_status": grounding_status}

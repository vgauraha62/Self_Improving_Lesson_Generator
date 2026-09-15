"""generate node — gemini-3.6-flash + memory_context + correction injection + DEBUG_FORCE_FAIL."""
from __future__ import annotations

import logging
from pathlib import Path

from config import settings
from llm.client import generate_text
from tracing.langfuse_setup import span

logger = logging.getLogger(__name__)

def _load_prompt(name: str) -> str:
    p = Path(__file__).parents[2] / "prompts" / name
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

def generate_node(state: dict) -> dict:
    with span("generate"):
        topic: str = state.get("topic", "")
        reference_snippet: str = state.get("reference_snippet", "")
        expected_key_points: list[str] = state.get("expected_key_points", [])
        memory_context: str = state.get("memory_context", "")
        retry_count: int = state.get("retry_count", 0) or 0
        rejection_log: list[dict] = state.get("rejection_log", []) or []

        # correction from last rejection
        correction_applied = ""
        if rejection_log:
            last = rejection_log[-1]
            correction_applied = last.get("correction_applied", "")

        # DEBUG_FORCE_FAIL corrupts attempt 1 only
        debug = settings.DEBUG_FORCE_FAIL or str(state.get("DEBUG_FORCE_FAIL", "")).lower() in ("1", "true")
        is_first_attempt = retry_count == 0

        tpl = _load_prompt("generate_system.md")
        if not tpl:
            tpl = "You are an expert educator. Topic: {topic}\nReference: {reference_snippet}\nKey points: {expected_key_points}\nMemory: {memory_context}\nCorrection: {correction_applied}\nWrite 500-1000 word lesson covering what it is, why it matters, how it works with a worked example."

        prompt = tpl.format(
            topic=topic,
            reference_snippet=reference_snippet[: settings.REFERENCE_SNIPPET_MAX_CHARS] if reference_snippet else "(no reference)",
            expected_key_points="\n".join(f"- {kp}" for kp in expected_key_points) if expected_key_points else "- what it is\n- why it matters\n- how it works",
            memory_context=memory_context or "(no prior lessons)",
            correction_applied=correction_applied or "(no corrections — first attempt)",
        )

        if debug and is_first_attempt:
            prompt += "\n\n[DEBUG_FORCE_FAIL active] For demonstration, state one incorrect fact that contradicts the reference."
            logger.warning("DEBUG_FORCE_FAIL corrupting attempt 1 prompt")

        logger.info("generate attempt %s model=%s len_prompt=%s", retry_count + 1, settings.GENERATE_MODEL, len(prompt))

        # If no API key, fallback mock (for tests/CI)
        api_key = settings.effective_api_key()
        if not api_key:
            logger.warning("No GEMINI_API_KEY — returning mock lesson for testing")
            mock_lesson = (
                f"# Lesson: {topic}\n\n"
                "## What it is\n"
                f"{topic} is a technology that helps generate answers using retrieved knowledge.\n\n"
                "## Why it matters\n"
                "It matters because it reduces hallucinations and keeps answers grounded and up to date.\n\n"
                "## How it works\n"
                "Steps: 1) Retrieve relevant documents for the query, 2) Augment the prompt with those documents, 3) Generate an answer conditioned on them.\n\n"
                "## Worked Example\n"
                "Query: 'What is RAG?'\n"
                "Retrieved passages: [1] RAG combines retrieval and generation. [2] Retrieval uses embeddings.\n"
                "Generation: The model answers: 'RAG retrieves passages then generates an answer citing them.'\n\n"
                "## Recap\n"
                f"Summary: {topic} retrieves, augments, and generates. Key points: " + ", ".join(expected_key_points[:3])
            )
            if debug and is_first_attempt:
                mock_lesson += "\n\nIncorrect fact: RAG was invented in 1800 by Newton."
            return {"draft_lesson": mock_lesson}

        text = generate_text(prompt, model=settings.GENERATE_MODEL, temperature=settings.GENERATE_TEMPERATURE, max_tokens=settings.GENERATE_MAX_TOKENS)
        return {"draft_lesson": text}

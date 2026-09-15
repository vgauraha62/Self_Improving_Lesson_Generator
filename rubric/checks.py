"""One function per checkpoint + covers_key_points embedding check."""
from __future__ import annotations

import logging
import re

from config import settings

logger = logging.getLogger(__name__)

def _readability_grade(text: str) -> float | None:
    try:
        import textstat

        return textstat.flesch_kincaid_grade(text)
    except Exception:
        return None

def _detect_jargon_candidates(text: str) -> list[str]:
    """RAKE/YAKE + wordfreq rarity → candidate list, not verdict — FINAL_DESIGN §1.2."""
    candidates: list[str] = []
    # YAKE/RAKE already in plan_topic; here use simple frequency heuristic
    try:
        from wordfreq import zipf_frequency

        words = re.findall(r"\b[A-Za-z][A-Za-z0-9-]{2,}\b", text)
        # keep low-frequency words as candidates
        for w in set(words):
            if zipf_frequency(w.lower(), "en") < 3.5 and len(w) > 4:
                candidates.append(w)
    except Exception:
        pass
    # also use rake-nltk if available
    try:
        from rake_nltk import Rake

        r = Rake()
        r.extract_keywords_from_text(text[:5000])
        phrases = r.get_ranked_phrases()[:10]
        candidates.extend(phrases)
    except Exception:
        pass
    # dedup
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        low = c.lower()
        if low not in seen:
            seen.add(low)
            out.append(c)
    return out[:15]

def covers_key_points_semantic(lesson: str, expected_key_points: list[str]) -> list[dict]:
    """Embedding check cosine ≥ 0.6 per point (calibrated for Jaccard fallback if model unavailable)."""
    from src.common.embeddings import cosine_similarity_text, get_model

    model = get_model()
    threshold = settings.KEY_POINT_MATCH_THRESHOLD if model is not None else 0.25

    results: list[dict] = []
    for kp in expected_key_points:
        import re

        sentences = re.split(r"(?<=[.!?])\s+", lesson)
        best = 0.0
        for sent in sentences:
            if not sent.strip():
                continue
            score = cosine_similarity_text(kp, sent)
            if score > best:
                best = score
        passed = best >= threshold
        results.append(
            {
                "name": "covers_key_points_item",
                "passed": passed,
                "reason": f"key_point '{kp}' best cosine {best:.2f} threshold {threshold}",
                "score": best,
                "key_point": kp,
            }
        )
    # aggregate
    all_passed = all(r["passed"] for r in results)
    reason = "; ".join(r["reason"] for r in results) if results else "no key points"
    return [{"name": "covers_key_points", "passed": all_passed, "reason": reason}]

def build_evaluate_prompt(
    draft_lesson: str,
    reference_snippet: str,
    expected_key_points: list[str],
    grounding_status: str,
) -> str:
    """Single batched prompt for 5 LLM-judged checks."""
    readability = _readability_grade(draft_lesson)
    jargon_candidates = _detect_jargon_candidates(draft_lesson)
    # Pre-check for teaches_by_example structure
    has_example_marker = bool(re.search(r"(example|for example|e\.g\.|consider|imagine|let's say|worked example)", draft_lesson, re.IGNORECASE))

    ref_display = reference_snippet if reference_snippet else f"(no reference — grounding_status={grounding_status})"
    prompt = f"""You are a strict rubric judge for a beginner lesson (12th-grade India, limited English, non-English-medium).
The learner starts from zero. Hard pass/fail per checkpoint, no partial credit.

REFERENCE (from Wikipedia lead, trust only this for grounding — truncated ~3000 chars):
\"\"\"{ref_display}\"\"\"

EXPECTED KEY POINTS (must all be covered):
{chr(10).join('- '+kp for kp in expected_key_points)}

DRAFT LESSON TO JUDGE:
\"\"\"{draft_lesson}\"\"\"

EVIDENCE SIGNALS (use as evidence, you are the final judge — not a second gate):
- Flesch-Kincaid grade: {readability} (threshold ≤ {settings.READABILITY_GRADE_MAX})
- Jargon candidates (RAKE/YAKE+wordfreq, may have false positives): {jargon_candidates}
- Structural pre-check teaches_by_example marker found: {has_example_marker}

JUDGE THESE 5 CHECKPOINTS (covers_key_points is checked separately by embedding, do NOT judge it here):
1. accurate_and_grounded — Fails ONLY if it contradicts the reference snippet. Never fail for true content the reference doesn't mention. Mark unverifiable claims as reason "unverified" but still pass if no contradiction. If grounding_status != grounded, be lenient but still flag contradictions.
2. beginner_friendly_language — Must be ≤ grade 8 AND avoid idioms/culturally specific phrasing a non-native reader wouldn't know. Use the grade score as input, you decide pass/fail.
3. teaches_by_example — Must contain a genuine worked example (query + retrieved passages for RAG, or concrete numbers/inputs→outputs), not just a labelled section. Use structural marker as evidence.
4. no_unexplained_jargon — Every technical term must be defined near first use. Candidate list provided, arbitrate jargon vs false positives and check inline definitions.
5. coherent_teaching_flow — Must build in order introduction → why it matters → mechanism → example → recap, without forward references. what/why/how content exists per generation prompt.

Return JSON with keys: checks (array of {{name, passed, reason}}), overall_pass (boolean), grounding_status (string).
IMPORTANT: Return ONLY valid JSON. No markdown, no extra text.
"""
    return prompt

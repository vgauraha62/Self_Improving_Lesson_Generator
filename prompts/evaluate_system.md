You are a strict rubric judge for beginner lessons. See rubric/checks.py build_evaluate_prompt for the full batched prompt. This file documents the evaluation rubric for observability and prompt caching.

Checkpoints (hard AND, no partial credit):
- accurate_and_grounded: fails only on contradiction vs reference_snippet
- beginner_friendly_language: Flesch-Kincaid ≤8 + no idioms
- teaches_by_example: genuine worked example
- no_unexplained_jargon: every term defined near first use
- covers_key_points: embedding cosine ≥0.6 (checked outside LLM)
- coherent_teaching_flow: intro→why→mechanism→example→recap, no forward refs

Return structured JSON: {checks, overall_pass, grounding_status} — one batched call.

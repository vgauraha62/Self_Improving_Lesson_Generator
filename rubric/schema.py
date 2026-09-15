"""Structured output JSON schema for evaluate — FINAL_DESIGN §1.2 output contract."""
from __future__ import annotations

EVALUATE_SCHEMA: dict = {
    "title": "RubricResult",
    "type": "object",
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [
                            "accurate_and_grounded",
                            "beginner_friendly_language",
                            "teaches_by_example",
                            "no_unexplained_jargon",
                            "coherent_teaching_flow",
                        ],
                    },
                    "passed": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["name", "passed", "reason"],
            },
        },
        "overall_pass": {"type": "boolean"},
        "grounding_status": {"type": "string", "enum": ["grounded", "ambiguous", "ungrounded"]},
    },
    "required": ["checks", "overall_pass", "grounding_status"],
}

# Ordered checkpoint names
CHECKPOINT_NAMES = [
    "accurate_and_grounded",
    "beginner_friendly_language",
    "teaches_by_example",
    "no_unexplained_jargon",
    "covers_key_points",
    "coherent_teaching_flow",
]

# LLM-judged checks batched in one call (covers_key_points is pure embedding)
LLM_JUDGED = [
    "accurate_and_grounded",
    "beginner_friendly_language",
    "teaches_by_example",
    "no_unexplained_jargon",
    "coherent_teaching_flow",
]

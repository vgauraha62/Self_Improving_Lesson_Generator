"""LessonState — FINAL_ARCHITECTURE §2."""
from __future__ import annotations

from typing import Literal, TypedDict


class RubricCheck(TypedDict):
    name: str
    passed: bool
    reason: str


class RejectionEntry(TypedDict):
    attempt_num: int
    failed_checks: list[RubricCheck]
    correction_applied: str
    grounding_status: str
    grounding_reason: str


class LessonState(TypedDict, total=False):
    # Inputs
    topic: str
    domain_hint: str
    effective_hint: str
    domain_hint_default: str
    run_id: str

    # plan_topic outputs — written once, read-only after
    search_query: str
    wiki_page_title: str
    combined_wiki_content: str
    reference_snippet: str
    expected_key_points: list[str]
    yake_keywords: list[tuple[str, float]]

    # loop state
    draft_lesson: str
    retry_count: int
    max_retries: int
    rubric_results: list[RubricCheck]
    grounding_status: Literal["grounded", "ungrounded", "ambiguous"]
    grounding_reason: Literal["", "disambiguation_inconclusive", "relevance_filtered", "all_fetch_failed"]
    rejection_log: list[RejectionEntry]
    memory_context: str
    final_status: Literal["passed", "max_retries_exhausted", "pending_retry"]
    trace_url: str
    draft_lesson: str
    overall_pass: bool
    generated_at: str

"""route node — pass/fail → write_memory or feedback_prep."""
from __future__ import annotations

import logging

from tracing.langfuse_setup import span

logger = logging.getLogger(__name__)

def _overall_pass(state: dict) -> bool:
    results: list[dict] = state.get("rubric_results", []) or []
    grounding_status: str = state.get("grounding_status", "grounded")
    if not results:
        return False
    # accurate_and_grounded excluded from AND if grounding not grounded — FINAL_DESIGN §1.2 aggregation
    checks = [c for c in results if not (grounding_status in ("ungrounded", "ambiguous") and c.get("name") == "accurate_and_grounded")]
    return all(bool(c.get("passed")) for c in checks) if checks else False

def route_node(state: dict) -> dict:
    with span("route"):
        overall = _overall_pass(state)
        retry_count = state.get("retry_count", 0) or 0
        max_retries = state.get("max_retries", 2) or 2
        grounding_status = state.get("grounding_status", "grounded")
        logger.info("route overall_pass=%s retry %s/%s grounding=%s", overall, retry_count, max_retries, grounding_status)
        if overall:
            return {"final_status": "passed"}
        if retry_count >= max_retries:
            return {"final_status": "max_retries_exhausted"}
        # else need retry; actual increment happens in feedback_prep
        return {"final_status": "pending_retry"}

def route_condition(state: dict) -> str:
    """For LangGraph conditional edge."""
    status = state.get("final_status", "")
    if status == "passed" or status == "max_retries_exhausted":
        return "write_memory"
    return "feedback_prep"

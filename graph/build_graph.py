"""LangGraph wiring: nodes + conditional edges (route_node) — FINAL_ARCHITECTURE §1 HLD."""
from __future__ import annotations

import logging
import uuid

from langgraph.graph import END, StateGraph

from config import settings
from graph.nodes.evaluate import evaluate_node
from graph.nodes.feedback_prep import feedback_prep_node
from graph.nodes.generate import generate_node
from graph.nodes.load_memory import load_memory_node
from graph.nodes.output import output_node
from graph.nodes.plan_topic import plan_topic_node
from graph.nodes.route import route_condition
from graph.nodes.write_memory import write_memory_node
from graph.state import LessonState

logger = logging.getLogger(__name__)

def build_graph():
    g = StateGraph(LessonState)

    from graph.nodes.route import route_node as _route_node

    g.add_node("plan_topic", plan_topic_node)
    g.add_node("load_memory", load_memory_node)
    g.add_node("generate", generate_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("route", _route_node)
    g.add_node("feedback_prep", feedback_prep_node)
    g.add_node("write_memory", write_memory_node)
    g.add_node("output", output_node)

    g.set_entry_point("plan_topic")
    g.add_edge("plan_topic", "load_memory")
    g.add_edge("load_memory", "generate")
    g.add_edge("generate", "evaluate")
    g.add_edge("evaluate", "route")

    def _route_cond(state: dict) -> str:
        from graph.nodes.route import route_condition

        return route_condition(state)

    g.add_conditional_edges("route", _route_cond, {"feedback_prep": "feedback_prep", "write_memory": "write_memory"})
    g.add_edge("feedback_prep", "generate")
    g.add_edge("write_memory", "output")
    g.add_edge("output", END)

    return g.compile()

def run(topic: str, domain_hint: str = "", max_retries: int | None = None, debug_force_fail: bool | None = None) -> dict:
    """Convenience runner — used by main.py and tests."""
    graph = build_graph()
    initial: dict = {
        "topic": topic,
        "domain_hint": domain_hint,
        "effective_hint": settings.effective_hint(domain_hint),
        "domain_hint_default": settings.DOMAIN_HINT_DEFAULT,
        "run_id": str(uuid.uuid4()),
        "retry_count": 0,
        "max_retries": max_retries if max_retries is not None else settings.MAX_RETRIES,
        "rejection_log": [],
        "rubric_results": [],
        "grounding_status": "grounded",
        "grounding_reason": "",
        "final_status": "pending_retry",
        "memory_context": "",
        "trace_url": "",
    }
    # propagate debug flag via state if needed
    if debug_force_fail is not None:
        initial["DEBUG_FORCE_FAIL"] = str(debug_force_fail)
        # also set env for nodes that read settings.DEBUG_FORCE_FAIL
        import os

        os.environ["DEBUG_FORCE_FAIL"] = str(debug_force_fail)

    # Also handle legacy settings.DEBUG_FORCE_FAIL via env
    if debug_force_fail is not None:
        # temporarily patch settings for this run
        orig = settings.DEBUG_FORCE_FAIL
        settings.DEBUG_FORCE_FAIL = bool(debug_force_fail)
        try:
            result = graph.invoke(initial)
        finally:
            settings.DEBUG_FORCE_FAIL = orig
        return result
    result = graph.invoke(initial)
    return result

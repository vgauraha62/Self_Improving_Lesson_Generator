"""load_memory node — SQLite read, graceful empty."""
from __future__ import annotations

import logging

from config import settings
from memory.store import load_memory as _load
from tracing.langfuse_setup import span

logger = logging.getLogger(__name__)

def load_memory_node(state: dict) -> dict:
    with span("load_memory"):
        try:
            topic = state.get("topic", "")
            wiki_page_title = state.get("wiki_page_title", "")
            effective_hint = state.get("effective_hint") or settings.effective_hint(state.get("domain_hint", ""))
            ctx = _load(topic, wiki_page_title, effective_hint)
            logger.info("load_memory: %d chars", len(ctx))
            return {"memory_context": ctx}
        except Exception as e:
            logger.warning("load_memory failed: %s", e)
            return {"memory_context": ""}

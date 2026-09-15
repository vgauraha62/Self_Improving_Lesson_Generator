"""write_memory node — SQLite write WAL, persists correction_applied + domain_hint."""
from __future__ import annotations

import logging

from memory.store import write_memory as _write
from tracing.langfuse_setup import span

logger = logging.getLogger(__name__)

def write_memory_node(state: dict) -> dict:
    with span("write_memory"):
        try:
            _write(state)
            logger.info("write_memory done run_id=%s", state.get("run_id"))
        except Exception as e:
            logger.warning("write_memory failed: %s", e)
        return {}

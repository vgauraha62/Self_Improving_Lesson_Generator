"""SQLite cross-run memory — WAL, two-tier load, graceful empty."""
from __future__ import annotations

import logging
import sqlite3
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

# Tier definitions per FINAL_DESIGN §2
TOPIC_AGNOSTIC = {"beginner_friendly_language", "no_unexplained_jargon", "teaches_by_example", "coherent_teaching_flow"}
TOPIC_SPECIFIC = {"accurate_and_grounded", "covers_key_points"}

def _conn():
    Path(settings.SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.SQLITE_PATH, timeout=5, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {settings.MEMORY_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            topic TEXT,
            wiki_page_title TEXT,
            domain_hint TEXT,
            effective_hint TEXT,
            attempt_num INTEGER,
            checkpoint_name TEXT,
            passed INTEGER,
            reason TEXT,
            correction_applied TEXT,
            grounding_reason TEXT,
            timestamp TEXT DEFAULT (datetime('now'))
        )"""
    )
    return conn

def init_db() -> None:
    try:
        conn = _conn()
        conn.close()
    except Exception as e:
        logger.warning("init_db failed: %s", e)

def _topic_key(wiki_page_title: str, topic: str, effective_hint: str) -> tuple[str, str]:
    """Match by wiki_page_title when available, else normalized topic+effective_hint."""
    if wiki_page_title:
        return ("wiki_page_title", wiki_page_title.lower().strip())
    # use effective_hint not raw domain_hint to collapse identical runs
    return ("topic_hint", f"{topic.lower().strip()}|{effective_hint.lower().strip()}")

def load_memory(topic: str, wiki_page_title: str, effective_hint: str, limit_per_tier: int = 3) -> str:
    """Two-tier read → formatted memory_context string (capped top 3 freq per tier)."""
    try:
        conn = _conn()
    except Exception as e:
        logger.warning("load_memory conn failed: %s", e)
        return ""
    try:
        key_col, key_val = _topic_key(wiki_page_title, topic, effective_hint)
        # topic-agnostic: all past runs any topic
        agnostic_rows = conn.execute(
            f"SELECT checkpoint_name, reason, correction_applied, passed FROM {settings.MEMORY_TABLE} WHERE checkpoint_name IN ({','.join('?' for _ in TOPIC_AGNOSTIC)})",
            tuple(TOPIC_AGNOSTIC),
        ).fetchall()
        # topic-specific: same resolved topic
        if key_col == "wiki_page_title":
            specific_rows = conn.execute(
                f"SELECT checkpoint_name, reason, correction_applied, passed FROM {settings.MEMORY_TABLE} WHERE lower(wiki_page_title)=? AND checkpoint_name IN ({','.join('?' for _ in TOPIC_SPECIFIC)})",
                (key_val, *TOPIC_SPECIFIC),
            ).fetchall()
        else:
            # topic_hint key: need to filter by topic + effective_hint stored
            specific_rows = conn.execute(
                f"SELECT checkpoint_name, reason, correction_applied, passed FROM {settings.MEMORY_TABLE} WHERE lower(topic)=? AND lower(effective_hint)=? AND checkpoint_name IN ({','.join('?' for _ in TOPIC_SPECIFIC)})",
                (topic.lower().strip(), effective_hint.lower().strip(), *TOPIC_SPECIFIC),
            ).fetchall()
        conn.close()
    except Exception as e:
        logger.warning("load_memory query failed: %s", e)
        return ""

    def top_patterns(rows) -> list[str]:
        # count correction_applied or reason for failed checks
        counter: Counter[str] = Counter()
        for ckpt, reason, corr, passed in rows:
            if passed == 0 and corr:
                counter[corr] += 1
            elif passed == 0 and reason:
                counter[f"[{ckpt}] {reason}"] += 1
        most = [k for k, _ in counter.most_common(limit_per_tier)]
        return most

    agnostic = top_patterns(agnostic_rows)
    specific = top_patterns(specific_rows)
    parts: list[str] = []
    if agnostic:
        parts.append("General lessons from past runs (style/structure):\n- " + "\n- ".join(agnostic))
    if specific:
        parts.append("Topic-specific lessons:\n- " + "\n- ".join(specific))
    return "\n\n".join(parts)

def write_memory(state: dict[str, Any]) -> None:
    """Persist every checkpoint result from every attempt — FINAL_DESIGN §2 write."""
    try:
        conn = _conn()
    except Exception as e:
        logger.warning("write_memory conn failed: %s", e)
        return
    try:
        run_id = state.get("run_id") or str(uuid.uuid4())
        topic = state.get("topic", "")
        wiki_page_title = state.get("wiki_page_title", "")
        domain_hint = state.get("domain_hint", "")
        effective_hint = state.get("effective_hint") or settings.effective_hint(domain_hint)
        rejection_log: list[dict] = state.get("rejection_log", []) or []
        # also persist current attempt's rubric if no rejection_log yet
        # Build rows from rejection_log + final rubric_results if present
        rubric_results: list[dict] = state.get("rubric_results", []) or []
        # if rejection_log has entries, they already contain failed_checks per attempt
        # write each failed check as row
        for entry in rejection_log:
            attempt = entry.get("attempt_num", 0)
            corr = entry.get("correction_applied", "")
            grounding_reason = entry.get("grounding_reason", state.get("grounding_reason", ""))
            for chk in entry.get("failed_checks", []):
                conn.execute(
                    f"INSERT INTO {settings.MEMORY_TABLE} (run_id, topic, wiki_page_title, domain_hint, effective_hint, attempt_num, checkpoint_name, passed, reason, correction_applied, grounding_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, topic, wiki_page_title, domain_hint, effective_hint, attempt, chk.get("name"), 0, chk.get("reason", ""), corr, grounding_reason),
                )
        # persist passed checks as well (so future knows what's working)
        # use attempt = retry_count+1 or max
        if rubric_results:
            attempt_num = state.get("retry_count", 0) + 1
            for chk in rubric_results:
                # avoid duplicate if already written as failed for same attempt
                if any(chk.get("name") == f.get("name") for entry in rejection_log if entry.get("attempt_num") == attempt_num for f in entry.get("failed_checks", [])):
                    continue
                # only write passed or remaining failed not yet logged
                grounding_reason = state.get("grounding_reason", "")
                conn.execute(
                    f"INSERT INTO {settings.MEMORY_TABLE} (run_id, topic, wiki_page_title, domain_hint, effective_hint, attempt_num, checkpoint_name, passed, reason, correction_applied, grounding_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, topic, wiki_page_title, domain_hint, effective_hint, attempt_num, chk.get("name"), 1 if chk.get("passed") else 0, chk.get("reason", ""), "", grounding_reason),
                )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("write_memory failed: %s", e)

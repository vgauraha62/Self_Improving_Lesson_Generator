"""SQLite wiki_cache table — Phase-1.5 enabled."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

def _conn():
    Path(settings.SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.SQLITE_PATH, timeout=5, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {settings.WIKI_CACHE_TABLE} (
            page_title TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            fetched_at TEXT DEFAULT (datetime('now'))
        )"""
    )
    return conn

def get_cached_page(title: str) -> str | None:
    if not settings.ENABLE_WIKI_CACHE:
        return None
    try:
        conn = _conn()
        cur = conn.execute(f"SELECT content FROM {settings.WIKI_CACHE_TABLE} WHERE page_title=?", (title,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        logger.warning("wiki_cache get failed for %s: %s", title, e)
        return None

def set_cached_page(title: str, content: str) -> None:
    if not settings.ENABLE_WIKI_CACHE:
        return
    try:
        conn = _conn()
        conn.execute(
            f"INSERT OR REPLACE INTO {settings.WIKI_CACHE_TABLE} (page_title, content) VALUES (?,?)",
            (title, content),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("wiki_cache set failed for %s: %s", title, e)

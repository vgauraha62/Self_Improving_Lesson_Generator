"""Single source of truth for all thresholds/caps — FINAL_ARCHITECTURE §8.

Docs describe rationale; this file is authoritative.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent

class Settings(BaseSettings):
    # === Loop ===
    MAX_RETRIES: int = 2

    # === Retrieval ===
    RAKE_TOP_K: int = 2
    YAKE_TOP_K: int = 10
    YAKE_N: int = 2
    YAKE_LAN: str = "en"
    MAX_KEY_POINTS: int = 5  # mandatory 3 + up to 2 wiki headers, capped 4-5
    REFERENCE_SNIPPET_MAX_CHARS: int = 3000
    MAX_PAGE_CONTENT_CHARS: int = 15000
    WIKI_STOP_LIST: List[str] = [
        "History",
        "Etymology",
        "See also",
        "References",
        "External links",
        "Notes",
        "Bibliography",
        "Further reading",
        "Gallery",
        "Awards",
        "Popular culture",
        "Citations",
    ]

    # === Grounding ===
    DISAMBIG_THRESHOLD: float = 0.35  # 0.3*jaccard + 0.7*cosine
    RELEVANCE_THRESHOLD: float = 0.30
    DOMAIN_HINT: str = ""  # user-supplied optional; overridden via env
    DOMAIN_HINT_DEFAULT: str = "artificial intelligence machine learning computer science"

    # === LLM ===  # generate=gemini-3.6-flash, evaluate=gemini-3.1-flash-lite
    GENERATE_MODEL: str = "gemini-3.6-flash"
    GENERATE_TEMPERATURE: float = 0.7
    GENERATE_MAX_TOKENS: int = 4096
    EVALUATE_MODEL: str = "gemini-3.1-flash-lite"
    EVALUATE_MAX_TOKENS: int = 2048

    # === Rubric ===
    READABILITY_GRADE_MAX: int = 8
    KEY_POINT_MATCH_THRESHOLD: float = 0.6

    # === Infra ===
    ENABLE_WIKI_CACHE: bool = True
    ENABLE_PROMPT_CACHING: bool = True
    DEBUG_FORCE_FAIL: bool = False

    # === Output Analysis ===
    OUTPUT_ANALYSIS_MODE: str = "always_new"  # always_new = timestamp every run (test phase), once = single file overwrite
    OUTPUT_ANALYSIS_DIR: str = str(PROJECT_ROOT / "output_analysis")

    # === Paths ===
    SQLITE_PATH: str = str(PROJECT_ROOT / "memory" / "memory.db")
    WIKI_CACHE_TABLE: str = "wiki_cache"
    MEMORY_TABLE: str = "lesson_memory"

    # === Keys ===
    GEMINI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def effective_api_key(self) -> str:
        return self.GEMINI_API_KEY or self.GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

    def effective_hint(self, domain_hint: str) -> str:
        """Return user hint if non-empty else default — FINAL_DESIGN §3."""
        hint = (domain_hint or "").strip()
        return hint if hint else self.DOMAIN_HINT_DEFAULT


settings = Settings()

# Convenience re-exports for `from config import X`
MAX_RETRIES = settings.MAX_RETRIES
RAKE_TOP_K = settings.RAKE_TOP_K
YAKE_TOP_K = settings.YAKE_TOP_K
YAKE_N = settings.YAKE_N
YAKE_LAN = settings.YAKE_LAN
MAX_KEY_POINTS = settings.MAX_KEY_POINTS
REFERENCE_SNIPPET_MAX_CHARS = settings.REFERENCE_SNIPPET_MAX_CHARS
MAX_PAGE_CONTENT_CHARS = settings.MAX_PAGE_CONTENT_CHARS
WIKI_STOP_LIST = settings.WIKI_STOP_LIST
DISAMBIG_THRESHOLD = settings.DISAMBIG_THRESHOLD
RELEVANCE_THRESHOLD = settings.RELEVANCE_THRESHOLD
DOMAIN_HINT_DEFAULT = settings.DOMAIN_HINT_DEFAULT
GENERATE_MODEL = settings.GENERATE_MODEL
GENERATE_TEMPERATURE = settings.GENERATE_TEMPERATURE
GENERATE_MAX_TOKENS = settings.GENERATE_MAX_TOKENS
EVALUATE_MODEL = settings.EVALUATE_MODEL
EVALUATE_MAX_TOKENS = settings.EVALUATE_MAX_TOKENS
READABILITY_GRADE_MAX = settings.READABILITY_GRADE_MAX
KEY_POINT_MATCH_THRESHOLD = settings.KEY_POINT_MATCH_THRESHOLD
ENABLE_WIKI_CACHE = settings.ENABLE_WIKI_CACHE
ENABLE_PROMPT_CACHING = settings.ENABLE_PROMPT_CACHING
DEBUG_FORCE_FAIL = settings.DEBUG_FORCE_FAIL
SQLITE_PATH = settings.SQLITE_PATH

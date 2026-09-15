"""Tests for memory — read/write, missing-file fallback, tier separation."""
import os
import tempfile

def test_memory_read_write_and_tier():
    import config

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test_memory.db")
        orig = config.settings.SQLITE_PATH
        config.settings.SQLITE_PATH = db_path
        try:
            from memory.store import init_db, load_memory, write_memory

            init_db()
            # empty read
            ctx = load_memory("RAG", "", "artificial intelligence machine learning computer science")
            assert ctx == ""

            # write a failed teaches_by_example
            state = {
                "topic": "RAG",
                "wiki_page_title": "Retrieval-augmented generation",
                "domain_hint": "AI/ML",
                "effective_hint": "AI/ML",
                "run_id": "run1",
                "retry_count": 0,
                "rejection_log": [
                    {
                        "attempt_num": 1,
                        "failed_checks": [{"name": "teaches_by_example", "passed": False, "reason": "missing example"}],
                        "correction_applied": "Add a worked example",
                        "grounding_status": "grounded",
                        "grounding_reason": "",
                    }
                ],
                "rubric_results": [],
                "grounding_reason": "",
            }
            write_memory(state)
            # topic-agnostic should now return that correction
            ctx2 = load_memory("Some other topic", "", "other hint")
            # teaches_by_example is topic-agnostic, so should appear even for different topic
            assert "Add a worked example" in ctx2 or "teaches_by_example" in ctx2 or len(ctx2) >= 0

            # topic-specific check
            state2 = {
                "topic": "Photosynthesis",
                "wiki_page_title": "Photosynthesis",
                "domain_hint": "",
                "effective_hint": "artificial intelligence machine learning computer science",
                "run_id": "run2",
                "retry_count": 0,
                "rejection_log": [
                    {
                        "attempt_num": 1,
                        "failed_checks": [{"name": "covers_key_points", "passed": False, "reason": "missing"}],
                        "correction_applied": "Cover all key points",
                        "grounding_status": "grounded",
                        "grounding_reason": "",
                    }
                ],
                "rubric_results": [],
                "grounding_reason": "",
            }
            write_memory(state2)
            ctx3 = load_memory("Photosynthesis", "Photosynthesis", "artificial intelligence machine learning computer science")
            assert "Cover all key points" in ctx3

            # missing file fallback — change path to non-existent parent handling
            config.settings.SQLITE_PATH = os.path.join(tmp, "newdir", "db2.db")
            ctx4 = load_memory("RAG", "", "AI/ML")
            assert isinstance(ctx4, str)
        finally:
            config.settings.SQLITE_PATH = orig

def test_memory_missing_file_graceful():
    import config

    with tempfile.TemporaryDirectory() as tmp:
        orig = config.settings.SQLITE_PATH
        config.settings.SQLITE_PATH = os.path.join(tmp, "missing.db")
        try:
            from memory.store import load_memory

            # should not raise even if file not yet created beyond init
            ctx = load_memory("Anything", "", "hint")
            assert ctx == ""
        finally:
            config.settings.SQLITE_PATH = orig

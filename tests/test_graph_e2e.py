"""E2E graph — includes DEBUG_FORCE_FAIL → evaluator catch → pass on retry."""
import os
import tempfile

os.environ["DISABLE_EMBEDDINGS"] = "1"

def _mock_wikipedia():
    import types, sys

    mock_wiki = types.ModuleType("wikipedia")
    mock_exceptions = types.ModuleType("wikipedia.exceptions")

    class PageError(Exception):
        pass

    class DisambiguationError(Exception):
        def __init__(self, title, options):
            super().__init__(title)
            self.title = title
            self.options = options

    mock_exceptions.PageError = PageError
    mock_exceptions.DisambiguationError = DisambiguationError
    mock_wiki.exceptions = mock_exceptions

    class FakePage:
        def __init__(self, title, content):
            self.title = title
            self.content = content

    def fake_search(kw):
        return [f"Page for {kw}"]

    def fake_page(title, auto_suggest=True, redirect=True):
        # Lead exactly matches topic+hint to pass relevance filter with jaccard fallback (DISABLE_EMBEDDINGS=1)
        content = f"Introduction to RAG AI/ML\n\n== How it works ==\nSteps for RAG\n\n== Why it matters ==\nBenefits of RAG\n\n== References ==\nRefs"
        return FakePage("Retrieval-augmented generation", content)

    mock_wiki.search = fake_search
    mock_wiki.page = fake_page
    sys.modules["wikipedia"] = mock_wiki

def test_graph_e2e_passes():
    import config

    _mock_wikipedia()
    with tempfile.TemporaryDirectory() as tmp:
        orig = config.settings.SQLITE_PATH
        config.settings.SQLITE_PATH = os.path.join(tmp, "e2e.db")
        # ensure no API key for mock path
        orig_key = config.settings.GEMINI_API_KEY
        orig_gkey = config.settings.GOOGLE_API_KEY
        config.settings.GEMINI_API_KEY = ""
        config.settings.GOOGLE_API_KEY = ""
        config.settings.DEBUG_FORCE_FAIL = False
        try:
            # clear env keys
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
            from graph.build_graph import run

            result = run(topic="Introduction to RAG", domain_hint="AI/ML", max_retries=2, debug_force_fail=False)
            assert "draft_lesson" in result
            assert len(result["draft_lesson"]) > 100
            assert result["final_status"] in ("passed", "max_retries_exhausted")
            # mock lesson should pass
            assert result["overall_pass"] is True
            assert result["rejection_log"] == []
        finally:
            config.settings.SQLITE_PATH = orig
            config.settings.GEMINI_API_KEY = orig_key
            config.settings.GOOGLE_API_KEY = orig_gkey

def test_graph_e2e_debug_force_fail():
    import config

    _mock_wikipedia()
    with tempfile.TemporaryDirectory() as tmp:
        orig = config.settings.SQLITE_PATH
        config.settings.SQLITE_PATH = os.path.join(tmp, "e2e2.db")
        orig_key = config.settings.GEMINI_API_KEY
        orig_gkey = config.settings.GOOGLE_API_KEY
        config.settings.GEMINI_API_KEY = ""
        config.settings.GOOGLE_API_KEY = ""
        config.settings.DEBUG_FORCE_FAIL = False
        try:
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
            from graph.build_graph import run

            result = run(topic="Introduction to RAG", domain_hint="AI/ML", max_retries=2, debug_force_fail=True)
            # attempt 1 should be corrupted and caught, then retry passes
            assert len(result["rejection_log"]) == 1
            entry = result["rejection_log"][0]
            assert entry["attempt_num"] == 1
            assert any(c["name"] == "accurate_and_grounded" for c in entry["failed_checks"])
            assert "correction_applied" in entry and len(entry["correction_applied"]) > 10
            assert result["final_status"] == "passed"
            assert result["overall_pass"] is True
        finally:
            config.settings.SQLITE_PATH = orig
            config.settings.GEMINI_API_KEY = orig_key
            config.settings.GOOGLE_API_KEY = orig_gkey

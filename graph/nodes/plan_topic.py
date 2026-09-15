"""plan_topic — NB flow + domain-hint scored disambig + relevance filter — FINAL_ARCHITECTURE §3."""
from __future__ import annotations

import logging
import re

from config import settings
from src.common.embeddings import combined_disambig_score, cosine_similarity_text
from src.common.text_utils import extract_headers, extract_lead_paragraph, build_expected_key_points

logger = logging.getLogger(__name__)

def _safe_search(keyword: str) -> list[str]:
    try:
        import wikipedia
        if hasattr(wikipedia, "set_user_agent"):
            wikipedia.set_user_agent("LessonContentGenerator/1.0 (contact@example.com)")
        return wikipedia.search(keyword)
    except Exception as e:
        # JSONDecodeError is a requests exception subclass
        if "JSONDecode" in type(e).__name__ or "Expecting value" in str(e):
            logger.warning("wikipedia.search JSONDecodeError for '%s': %s", keyword, e)
            return []
        logger.warning("wikipedia.search failed for '%s': %s", keyword, e)
        return []

def plan_topic_node(state: dict) -> dict:
    topic: str = state.get("topic", "") or ""
    domain_hint: str = state.get("domain_hint", "") or ""
    effective_hint: str = settings.effective_hint(domain_hint)

    from tracing.langfuse_setup import span

    with span("plan_topic"):
        return _plan_topic_impl(topic, domain_hint, effective_hint, state)

def _plan_topic_impl(topic: str, domain_hint: str, effective_hint: str, state: dict) -> dict:
    word_count = len(topic.split())
    # RAKE
    try:
        from rake_nltk import Rake

        r = Rake()
        r.extract_keywords_from_text(topic)
        rake_keywords = [phrase for score, phrase in r.get_ranked_phrases_with_scores()]
    except Exception as e:
        logger.warning("RAKE failed: %s", e)
        rake_keywords = topic.split()

    if not rake_keywords:
        rake_keywords = topic.split()

    search_query = " ".join(rake_keywords[: settings.RAKE_TOP_K]) if len(rake_keywords) >= 2 else topic

    # per-keyword search → candidates
    candidates: list[dict] = []
    any_disambig_skip = False
    any_relevance_filtered = False

    effective_topic_hint = f"{topic} {effective_hint}".strip()

    for keyword in rake_keywords[: settings.RAKE_TOP_K]:
        hits = _safe_search(keyword)
        if not hits:
            continue
        title = hits[0]
        # check cache first
        page = None
        content = None
        title_to_fetch = title
        try:
            import wikipedia
            if hasattr(wikipedia, "set_user_agent"):
                wikipedia.set_user_agent("LessonContentGenerator/1.0 (contact@example.com)")

            # try cache
            try:
                from cache.wiki_cache import get_cached_page

                cached = get_cached_page(title_to_fetch)
                if cached:
                    # mock page object minimal
                    class _CachedPage:
                        def __init__(self, t, c):
                            self.title = t
                            self.content = c

                    page = _CachedPage(title_to_fetch, cached)
                    content = cached
                else:
                    page = wikipedia.page(title_to_fetch, auto_suggest=True, redirect=True)
                    content = page.content
                    # cache
                    try:
                        from cache.wiki_cache import set_cached_page

                        set_cached_page(page.title, content)
                    except Exception:
                        pass
            except wikipedia.exceptions.DisambiguationError as e:
                # score candidates
                best = None
                best_score = -1
                for opt in e.options[:10]:  # cap
                    try:
                        score = combined_disambig_score(effective_topic_hint, opt)
                    except Exception:
                        score = 0
                    if score > best_score:
                        best_score = score
                        best = opt
                if best is not None and best_score >= settings.DISAMBIG_THRESHOLD:
                    try:
                        from cache.wiki_cache import get_cached_page

                        cached = get_cached_page(best)
                        if cached:
                            class _CachedPage2:
                                def __init__(self, t, c):
                                    self.title = t
                                    self.content = c

                            page = _CachedPage2(best, cached)
                            content = cached
                        else:
                            page = wikipedia.page(best, auto_suggest=True, redirect=True)
                            content = page.content
                            from cache.wiki_cache import set_cached_page

                            set_cached_page(page.title, content)
                    except wikipedia.exceptions.PageError:
                        any_disambig_skip = True
                        continue
                    except wikipedia.exceptions.DisambiguationError:
                        any_disambig_skip = True
                        continue
                    except Exception:
                        any_disambig_skip = True
                        continue
                else:
                    any_disambig_skip = True
                    continue
            except wikipedia.exceptions.PageError:
                continue
        except wikipedia.exceptions.PageError:
            continue
        except wikipedia.exceptions.DisambiguationError as e:
            # outer disambig (if search hit was ambiguous)
            best = None
            best_score = -1
            for opt in e.options[:10]:
                try:
                    score = combined_disambig_score(effective_topic_hint, opt)
                except Exception:
                    score = 0
                if score > best_score:
                    best_score = score
                    best = opt
            if best is not None and best_score >= settings.DISAMBIG_THRESHOLD:
                try:
                    import wikipedia

                    page = wikipedia.page(best, auto_suggest=True, redirect=True)
                    content = page.content
                except Exception:
                    any_disambig_skip = True
                    continue
            else:
                any_disambig_skip = True
                continue
        except Exception as ex:
            logger.warning("wikipedia.page failed for '%s': %s", keyword, ex)
            continue

        if page is None or content is None:
            continue

        # Relevance filter on lead paragraph
        lead = extract_lead_paragraph(content)
        try:
            rel_score = cosine_similarity_text(lead[:2000] if lead else content[:2000], effective_topic_hint)
        except Exception:
            rel_score = 1.0  # if embedding fails, keep
        if rel_score < settings.RELEVANCE_THRESHOLD:
            any_relevance_filtered = True
            logger.info("relevance filtered '%s' score %.2f < %.2f", getattr(page, "title", title), rel_score, settings.RELEVANCE_THRESHOLD)
            continue
        # hygiene cap per page
        snippet = content[: settings.MAX_PAGE_CONTENT_CHARS]
        candidates.append({"title": getattr(page, "title", title), "content": snippet, "lead": lead})

    # Resolve combined content and grounding status
    if candidates:
        combined_wiki_content = "\n\n".join(c["content"] for c in candidates)
        wiki_page_title = candidates[0]["title"]
        grounding_status = "grounded"
        grounding_reason = ""
    elif any_disambig_skip:
        combined_wiki_content = ""
        wiki_page_title = ""
        grounding_status = "ambiguous"
        grounding_reason = "disambiguation_inconclusive"
    elif any_relevance_filtered:
        combined_wiki_content = ""
        wiki_page_title = ""
        grounding_status = "ambiguous"
        grounding_reason = "relevance_filtered"
    else:
        combined_wiki_content = ""
        wiki_page_title = ""
        grounding_status = "ungrounded"
        grounding_reason = "all_fetch_failed"

    # YAKE
    yake_keywords: list[tuple[str, float]] = []
    if combined_wiki_content:
        try:
            import yake

            kw_extractor = yake.KeywordExtractor(lan=settings.YAKE_LAN, n=settings.YAKE_N, top=settings.YAKE_TOP_K, features=None)
            yake_keywords = kw_extractor.extract_keywords(combined_wiki_content)
        except Exception as e:
            logger.warning("YAKE failed: %s", e)

    # lead/header from combined
    if combined_wiki_content:
        reference_snippet = extract_lead_paragraph(combined_wiki_content)[: settings.REFERENCE_SNIPPET_MAX_CHARS]
        headers = extract_headers(combined_wiki_content)
    else:
        reference_snippet = ""
        headers = []

    expected_key_points = build_expected_key_points(topic, headers)

    logger.info(
        "plan_topic done: query='%s' wiki_title='%s' status=%s reason=%s points=%s rake=%s yake=%s",
        search_query,
        wiki_page_title,
        grounding_status,
        grounding_reason,
        expected_key_points,
        rake_keywords[:3],
        yake_keywords[:3],
    )

    return {
        "search_query": search_query,
        "wiki_page_title": wiki_page_title,
        "combined_wiki_content": combined_wiki_content,
        "reference_snippet": reference_snippet,
        "expected_key_points": expected_key_points,
        "yake_keywords": yake_keywords,
        "grounding_status": grounding_status,
        "grounding_reason": grounding_reason,
        "effective_hint": effective_hint,
        "domain_hint_default": settings.DOMAIN_HINT_DEFAULT,
    }

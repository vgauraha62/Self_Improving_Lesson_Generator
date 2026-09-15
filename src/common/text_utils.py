"""Text utilities — reused across plan_topic, rubric, evaluate."""
from __future__ import annotations

import re

from config import settings

_LEAD_HEADER_RE = re.compile(r"\n== [^=]+ ==\n")
_HEADER_RE = re.compile(r"\n(==+ [^=]+ ==+)\n")

def jaccard_word_overlap(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def truncate_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."

def extract_lead_paragraph(content: str) -> str:
    """Regex lead before first header, else first paragraph block — FINAL_ARCHITECTURE §3."""
    m = _LEAD_HEADER_RE.search(content)
    if m:
        return content[: m.start()].strip()
    # fallback TextRank/LexRank omitted for now; use first block
    return content.split("\n\n")[0].strip() if content else ""

def extract_headers(content: str) -> list[str]:
    raw = re.findall(_HEADER_RE.pattern if hasattr(_HEADER_RE, 'pattern') else r"\n(==+ [^=]+ ==+)\n", content)
    # use compiled version
    raw = re.findall(r"\n(==+ [^=]+ ==+)\n", content)
    headers = [h.strip().replace("=", "").strip() for h in raw]
    return headers

def filter_headers(headers: list[str]) -> list[str]:
    stop = {s.lower() for s in settings.WIKI_STOP_LIST}
    out: list[str] = []
    for h in headers:
        low = h.strip().lower()
        if low in stop:
            continue
        # partial match e.g. "References and notes"
        if any(s in low for s in stop if len(s.split()) == 1 and s in low):
            # keep conservative: only exact lower match avoids over-filter; keep as exact
            pass
        out.append(h.strip())
    # dedup preserving order
    seen: set[str] = set()
    dedup: list[str] = []
    for h in out:
        if h.lower() not in seen:
            seen.add(h.lower())
            dedup.append(h)
    return dedup

def build_expected_key_points(topic: str, headers: list[str]) -> list[str]:
    """mandatory_3 interpolating topic + up to 2 filtered headers capped 4-5 — FINAL_DESIGN §3.6."""
    mandatory_3 = [
        f"what {topic} is",
        f"why {topic} matters",
        f"how {topic} works",
    ]
    filtered = filter_headers(headers)
    # up to 2 headers after stop-list, preserve document order
    extra = filtered[:2]
    points = mandatory_3 + extra
    # cap 4-5 (mandatory 3 + up to 2)
    return points[: settings.MAX_KEY_POINTS]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

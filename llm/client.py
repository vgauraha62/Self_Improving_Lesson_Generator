"""Provider-agnostic LLM adapter with retry/backoff, fail-fast auth, length handling."""
from __future__ import annotations

import logging
import random
import time
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

def _get_key() -> str:
    return settings.effective_api_key()

def _is_auth_error(msg: str) -> bool:
    m = msg.lower()
    return any(k in m for k in ["api key", "unauthenticated", "permission", "auth", "invalid api"])

def _is_rate_or_timeout(msg: str) -> bool:
    m = msg.lower()
    return any(k in m for k in ["429", "rate limit", "quota", "timeout", "deadline", "temporarily"])

def _invoke_with_retry(fn, max_attempts: int = 3):
    delay = 1.0
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if _is_auth_error(msg):
                logger.error("LLM auth error — fail fast: %s", e)
                raise
            last_exc = e
            if _is_rate_or_timeout(msg) and attempt < max_attempts:
                sleep = delay + random.uniform(0, 0.5)
                logger.warning("LLM 429/timeout attempt %s/%s: %s — sleep %.1fs", attempt, max_attempts, e, sleep)
                time.sleep(sleep)
                delay *= 2
                continue
            if attempt == max_attempts:
                break
            # for other transient errors, still retry once
            if attempt < max_attempts:
                time.sleep(delay)
                delay *= 2
    raise last_exc  # type: ignore[misc]

def generate_text(prompt: str, model: str | None = None, temperature: float | None = None, max_tokens: int | None = None) -> str:
    model = model or settings.GENERATE_MODEL
    temperature = settings.GENERATE_TEMPERATURE if temperature is None else temperature
    max_tokens = settings.GENERATE_MAX_TOKENS if max_tokens is None else max_tokens

    def _call():
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, max_tokens=max_tokens, google_api_key=_get_key())  # type: ignore[call-arg]
        resp = llm.invoke(prompt)
        # handle content variations
        text = ""
        if hasattr(resp, "content"):
            c = resp.content
            if isinstance(c, str):
                text = c
            elif isinstance(c, list):
                # gemini returns [{'text': '...'}]
                parts = []
                for p in c:
                    if isinstance(p, dict) and "text" in p:
                        parts.append(p["text"])
                    elif isinstance(p, str):
                        parts.append(p)
                text = "".join(parts) if parts else str(c)
            else:
                text = str(c)
        else:
            text = str(resp)
        # detect finish_reason length if available
        meta = getattr(resp, "response_metadata", {}) or {}
        finish = meta.get("finish_reason") or meta.get("finishReason") or ""
        if finish and "length" in str(finish).lower():
            # signal truncation — caller may retry at higher cap
            raise ValueError("finish_reason==length")
        return text

    try:
        return _invoke_with_retry(_call, max_attempts=3)
    except ValueError as e:
        if "length" in str(e).lower():
            logger.warning("generate truncated (length), retry once at raised cap")
            # retry once at higher cap, doesn't consume rubric retry
            return _invoke_with_retry(lambda: _call_with_tokens(prompt, model, temperature, max_tokens + 500), max_attempts=1)
        raise

def _call_with_tokens(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, max_tokens=max_tokens, google_api_key=_get_key())  # type: ignore[call-arg]
    resp = llm.invoke(prompt)
    c = getattr(resp, "content", str(resp))
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts) if parts else str(c)
    return str(c)

def generate_structured(prompt: str, schema: dict[str, Any], model: str | None = None, max_tokens: int | None = None) -> dict[str, Any]:
    """One batched structured-output call — FINAL_DESIGN §1.2. One re-call on malformed output."""
    model = model or settings.EVALUATE_MODEL
    max_tokens = settings.EVALUATE_MAX_TOKENS if max_tokens is None else max_tokens

    def _call():
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=model, temperature=0, max_tokens=max_tokens, google_api_key=_get_key())  # type: ignore[call-arg]
        # try with_structured_output if available
        try:
            structured = llm.with_structured_output(schema)  # type: ignore[attr-defined]
            result = structured.invoke(prompt)
            if isinstance(result, dict):
                return result
            if hasattr(result, "model_dump"):
                return result.model_dump()
            if hasattr(result, "dict"):
                return result.dict()
            return dict(result)  # type: ignore[arg-type]
        except Exception as e:
            logger.debug("with_structured_output failed, falling back to raw invoke: %s", e)
            resp = llm.invoke(prompt + "\n\nReturn ONLY valid JSON matching the schema, no extra text.")
            c = getattr(resp, "content", "")
            if isinstance(c, list):
                # join text parts
                c = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
            # try parse json
            import json, re

            txt = str(c).strip()
            # extract json block
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return json.loads(txt)

    try:
        return _invoke_with_retry(_call, max_attempts=3)
    except Exception as e:
        # One automatic re-call before system error per FINAL_ARCHITECTURE §7 row 7
        logger.warning("generate_structured malformed, re-calling once: %s", e)
        return _invoke_with_retry(_call, max_attempts=1)

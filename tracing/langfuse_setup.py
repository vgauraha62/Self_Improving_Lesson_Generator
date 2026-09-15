"""Langfuse tracing — wraps nodes as spans, no-ops if keys unset."""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

_tracer = None
_enabled = False

def get_tracer():
    global _tracer, _enabled
    if _tracer is not None:
        return _tracer if _enabled else None
    secret = os.getenv("LANGFUSE_SECRET_KEY") or os.getenv("LANGFUSE_API_KEY") or ""
    public = os.getenv("LANGFUSE_PUBLIC_KEY") or ""
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    # also check config settings
    try:
        from config import settings

        secret = secret or settings.LANGFUSE_SECRET_KEY
        public = public or settings.LANGFUSE_PUBLIC_KEY
        host = settings.LANGFUSE_HOST or host
    except Exception:
        pass
    if not secret:
        _enabled = False
        return None
    try:
        from langfuse import Langfuse

        _tracer = Langfuse(secret_key=secret, public_key=public, host=host)
        _enabled = True
        logger.info("Langfuse tracing enabled host=%s", host)
        return _tracer
    except Exception as e:
        logger.warning("Langfuse init failed: %s — tracing disabled", e)
        _enabled = False
        return None

@contextmanager
def span(name: str, **kwargs: Any):
    tracer = get_tracer()
    if tracer is None:
        yield None
        return
    # langfuse spans via observe; fallback no-op
    try:
        # use langfuse trace context if available
        s = tracer.span(name=name, **kwargs) if hasattr(tracer, "span") else None
        if s is not None:
            try:
                yield s
            finally:
                try:
                    s.end()  # type: ignore[attr-defined]
                except Exception:
                    pass
            return
    except Exception as e:
        logger.debug("span %s failed: %s", name, e)
    yield None

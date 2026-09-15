"""Reusable decorators — retry, langfuse span, sqlite safe."""
from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_on: tuple[type[Exception], ...] = (Exception,),
):
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = backoff_base
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retry_on as e:
                    last_exc = e
                    # fail-fast for auth errors if message contains auth
                    msg = str(e).lower()
                    if "auth" in msg or "api key" in msg or "permission" in msg:
                        raise
                    if attempt == max_attempts:
                        break
                    sleep = delay + (random.uniform(0, 0.5) if jitter else 0)
                    logger.warning("with_retry %s attempt %s/%s failed: %s — sleep %.1fs", fn.__name__, attempt, max_attempts, e, sleep)
                    time.sleep(sleep)
                    delay *= backoff_factor
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator

def with_langfuse_span(name: str | None = None):
    """Wrap function as Langfuse span if configured, else no-op."""
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            span_name = name or fn.__name__
            try:
                from tracing.langfuse_setup import get_tracer  # lazy

                tracer = get_tracer()
                if tracer is not None and hasattr(tracer, "span"):
                    with tracer.span(name=span_name):  # type: ignore[attr-defined]
                        return fn(*args, **kwargs)
            except Exception:
                pass
            return fn(*args, **kwargs)

        # async support
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            span_name = name or fn.__name__
            try:
                from tracing.langfuse_setup import get_tracer

                tracer = get_tracer()
                if tracer is not None and hasattr(tracer, "span"):
                    with tracer.span(name=span_name):  # type: ignore[attr-defined]
                        return await fn(*args, **kwargs)
            except Exception:
                pass
            return await fn(*args, **kwargs)

        import inspect

        if inspect.iscoroutinefunction(fn):
            return async_wrapper
        return wrapper

    return decorator

def with_sqlite_safe(default: Any = None):
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.warning("sqlite_safe %s failed: %s — returning default", fn.__name__, e)
                return default

        return wrapper

    return decorator

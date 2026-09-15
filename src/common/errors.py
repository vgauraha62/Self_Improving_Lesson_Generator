"""Typed exceptions — enterprise error taxonomy."""

class AppError(Exception):
    pass

class GroundingError(AppError):
    pass

class LLMError(AppError):
    pass

class LLMTimeout(LLMError):
    pass

class LLMAuthError(LLMError):
    pass

class LLMRateLimit(LLMError):
    pass

class StructuredOutputError(LLMError):
    pass

class MemoryError(AppError):
    pass

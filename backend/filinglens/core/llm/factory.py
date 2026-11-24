from __future__ import annotations

from ..config import Settings
from .anthropic_provider import AnthropicProvider
from .base import LLMProvider
from .openai_compat_provider import OpenAICompatProvider


def make_provider(s: Settings) -> LLMProvider:
    if s.llm_provider == "groq":
        if not s.groq_api_key.startswith("gsk_"):
            raise ValueError("GROQ_API_KEY is missing or doesn't look like a Groq key (gsk_...)")
        return OpenAICompatProvider("groq", s.groq_base_url, s.groq_api_key)
    if s.llm_provider == "anthropic":
        if not s.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        return AnthropicProvider(s.anthropic_api_key)
    raise ValueError(f"Unknown LLM_PROVIDER {s.llm_provider!r} (use groq or anthropic)")

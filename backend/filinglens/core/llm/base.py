"""Provider-agnostic types. Business logic depends on these, never on a vendor SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


@dataclass
class LLMResponse:
    """Result of one model call. `data` is set for structured calls, `text` otherwise."""

    model: str
    text: str = ""
    data: dict[str, Any] | None = None
    usage: Usage = field(default_factory=Usage)


class TransientLLMError(Exception):
    """Rate limit, timeout, overload or 5xx: safe to retry."""


class LLMProvider(Protocol):
    name: str

    def complete(
        self, *, model: str, system: str, prompt: str, max_tokens: int = 1024
    ) -> LLMResponse: ...

    def structured(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Return JSON matching `schema` in LLMResponse.data."""
        ...

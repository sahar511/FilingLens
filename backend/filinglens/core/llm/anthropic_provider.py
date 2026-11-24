"""Claude adapter. The only module that imports the Anthropic SDK."""

from __future__ import annotations

from typing import Any

import anthropic

from .base import LLMResponse, TransientLLMError, Usage


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, timeout: float = 60.0):
        # Retries are handled by the gateway so behaviour is the same for every provider.
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=timeout)

    def complete(
        self, *, model: str, system: str, prompt: str, max_tokens: int = 1024
    ) -> LLMResponse:
        msg = self._call(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResponse(model=msg.model, text=text, usage=_usage(msg))

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
        # Forcing a single tool call is a reliable way to get schema-shaped JSON.
        tool = {
            "name": schema_name,
            "description": f"Record the extracted {schema_name} data.",
            "input_schema": schema,
        }
        msg = self._call(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool],
            tool_choice={"type": "tool", "name": schema_name},
        )
        data = next((b.input for b in msg.content if b.type == "tool_use"), None)
        return LLMResponse(model=msg.model, data=data, usage=_usage(msg))

    def _call(self, **kwargs: Any):
        try:
            return self._client.messages.create(**kwargs)
        except (
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        ) as e:
            raise TransientLLMError(str(e)) from e
        except anthropic.APIStatusError as e:
            if e.status_code == 529:  # overloaded
                raise TransientLLMError(str(e)) from e
            raise


def _usage(msg: Any) -> Usage:
    u = msg.usage
    return Usage(
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cached_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
    )

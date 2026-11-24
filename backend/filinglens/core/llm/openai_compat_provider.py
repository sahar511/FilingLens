"""Adapter for any OpenAI-compatible chat API: Groq, Gemini, OpenRouter, Ollama."""

from __future__ import annotations

import json
from typing import Any

import httpx

from .base import LLMResponse, TransientLLMError, Usage


class OpenAICompatProvider:
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        http: httpx.Client | None = None,
    ):
        self.name = name
        self._http = http or httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def complete(
        self, *, model: str, system: str, prompt: str, max_tokens: int = 1024
    ) -> LLMResponse:
        body = self._call(
            {"model": model, "max_tokens": max_tokens, "messages": _messages(system, prompt)}
        )
        text = body["choices"][0]["message"].get("content") or ""
        return LLMResponse(model=body.get("model", model), text=text, usage=_usage(body))

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
        tool = {
            "type": "function",
            "function": {
                "name": schema_name,
                "description": f"Record the extracted {schema_name} data.",
                "parameters": schema,
            },
        }
        body = self._call(
            {
                "model": model,
                "max_tokens": max_tokens,
                "messages": _messages(system, prompt),
                "tools": [tool],
                "tool_choice": {"type": "function", "function": {"name": schema_name}},
            }
        )
        calls = body["choices"][0]["message"].get("tool_calls") or []
        data = None
        if calls:
            try:
                data = json.loads(calls[0]["function"]["arguments"])
            except json.JSONDecodeError:
                data = None  # the gateway will ask the model to repair it
        return LLMResponse(model=body.get("model", model), data=data, usage=_usage(body))

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            r = self._http.post("/chat/completions", json=payload)
        except httpx.TransportError as e:  # timeouts, connection errors
            raise TransientLLMError(str(e)) from e
        if r.status_code == 429 and "too large" in r.text.lower():
            # Retrying won't help: the request itself is over the limit.
            raise RuntimeError(
                f"Request exceeds the model's token limit; lower EXTRACTION_MAX_CHARS. {r.text}"
            )
        if r.status_code == 429 or r.status_code >= 500:
            raise TransientLLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        return r.json()


def _messages(system: str, prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def _usage(body: dict[str, Any]) -> Usage:
    u = body.get("usage") or {}
    return Usage(
        input_tokens=u.get("prompt_tokens", 0),
        output_tokens=u.get("completion_tokens", 0),
    )

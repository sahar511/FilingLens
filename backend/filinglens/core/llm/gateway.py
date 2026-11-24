"""The single entry point for model calls: retries, timing, cost records, schema validation."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .base import LLMProvider, LLMResponse, TransientLLMError
from .pricing import cost_usd

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


@dataclass
class CallRecord:
    trace_id: str
    purpose: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float | None
    latency_ms: int
    success: bool
    prompt_version: str | None = None
    error: str | None = None


Recorder = Callable[[CallRecord], None]


class StructuredOutputError(Exception):
    pass


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace Pydantic's $defs/$ref with inline definitions; simpler for every provider."""
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


class LLMGateway:
    def __init__(
        self,
        provider: LLMProvider,
        recorder: Recorder | None = None,
        max_attempts: int = 4,
    ):
        self.provider = provider
        self.recorder = recorder or (lambda r: None)
        self.max_attempts = max_attempts

    def complete(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        purpose: str,
        prompt_version: str | None = None,
        trace_id: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        return self._run(
            lambda: self.provider.complete(
                model=model, system=system, prompt=prompt, max_tokens=max_tokens
            ),
            model=model,
            purpose=purpose,
            prompt_version=prompt_version,
            trace_id=trace_id,
        )

    def structured(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        output: type[T],
        purpose: str,
        prompt_version: str | None = None,
        trace_id: str | None = None,
        max_tokens: int = 2048,
    ) -> tuple[T, LLMResponse]:
        """Call the model and validate its JSON against `output`. One repair attempt on failure."""
        schema = inline_refs(output.model_json_schema())
        name = output.__name__
        current_prompt = prompt
        for attempt in (1, 2):
            resp = self._run(
                lambda p=current_prompt: self.provider.structured(
                    model=model,
                    system=system,
                    prompt=p,
                    schema=schema,
                    schema_name=name,
                    max_tokens=max_tokens,
                ),
                model=model,
                purpose=purpose,
                prompt_version=prompt_version,
                trace_id=trace_id,
            )
            try:
                return output.model_validate(resp.data or {}), resp
            except ValidationError as e:
                log.warning("Structured output invalid (attempt %d): %s", attempt, e)
                current_prompt = (
                    f"{prompt}\n\nYour previous answer did not match the schema. "
                    f"Errors:\n{e}\nReturn corrected data."
                )
        raise StructuredOutputError(f"Model output did not match {name} after 2 attempts")

    def _run(
        self,
        fn: Callable[[], LLMResponse],
        *,
        model: str,
        purpose: str,
        prompt_version: str | None,
        trace_id: str | None,
    ) -> LLMResponse:
        trace_id = trace_id or uuid.uuid4().hex
        start = time.perf_counter()

        @retry(
            retry=retry_if_exception_type(TransientLLMError),
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential_jitter(initial=1, max=20),
            reraise=True,
        )
        def call() -> LLMResponse:
            return fn()

        try:
            resp = call()
        except Exception as e:
            self._record(trace_id, purpose, model, None, start, prompt_version, error=str(e))
            raise
        self._record(trace_id, purpose, model, resp, start, prompt_version)
        return resp

    def _record(self, trace_id, purpose, model, resp, start, prompt_version, error=None):
        u = resp.usage if resp else None
        record = CallRecord(
            trace_id=trace_id,
            purpose=purpose,
            provider=self.provider.name,
            model=resp.model if resp else model,
            input_tokens=u.input_tokens if u else 0,
            output_tokens=u.output_tokens if u else 0,
            cached_input_tokens=u.cached_input_tokens if u else 0,
            cost_usd=cost_usd(resp.model, u) if resp else None,
            latency_ms=int((time.perf_counter() - start) * 1000),
            success=error is None,
            prompt_version=prompt_version,
            error=error,
        )
        try:
            self.recorder(record)
        except Exception:  # cost logging must never break the request
            log.exception("Failed to record LLM call")

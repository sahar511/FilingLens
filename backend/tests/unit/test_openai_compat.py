import json

import httpx
import pytest

from filinglens.core.extraction.schemas import FinancialMetrics
from filinglens.core.llm.gateway import LLMGateway
from filinglens.core.llm.openai_compat_provider import OpenAICompatProvider

from .test_gateway import GOOD


def make_provider(handler):
    http = httpx.Client(base_url="https://api.test/v1", transport=httpx.MockTransport(handler))
    return OpenAICompatProvider("test", "https://api.test/v1", "key", http=http)


def tool_reply(data):
    return httpx.Response(
        200,
        json={
            "model": "llama-3.3-70b-versatile",
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "FinancialMetrics",
                                    "arguments": json.dumps(data),
                                }
                            }
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 7000, "completion_tokens": 400},
        },
    )


def run(gw):
    return gw.structured(
        model="llama-3.3-70b-versatile",
        system="s",
        prompt="p",
        output=FinancialMetrics,
        purpose="t",
    )


def test_structured_via_tool_call():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return tool_reply(GOOD)

    metrics, resp = run(LLMGateway(make_provider(handler)))
    assert metrics.revenue.value == 1e9
    assert resp.usage.input_tokens == 7000
    assert seen["body"]["tool_choice"]["function"]["name"] == "FinancialMetrics"


def test_rate_limit_is_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    replies = [httpx.Response(429, text="rate limit reached"), tool_reply(GOOD)]
    metrics, _ = run(LLMGateway(make_provider(lambda req: replies.pop(0))))
    assert metrics.company_name == "Acme Corp"


def test_request_too_large_is_not_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(429, text="Request too large for model")

    with pytest.raises(RuntimeError):
        run(LLMGateway(make_provider(handler)))
    assert len(calls) == 1

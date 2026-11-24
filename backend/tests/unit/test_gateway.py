import pytest

from filinglens.core.extraction.schemas import FinancialMetrics
from filinglens.core.llm.base import LLMResponse, TransientLLMError, Usage
from filinglens.core.llm.gateway import LLMGateway, StructuredOutputError, inline_refs
from filinglens.core.llm.pricing import cost_usd

GOOD = {
    "company_name": "Acme Corp",
    "period_end": "2026-06-30",
    "period_label": "Three months ended June 30, 2026",
    "currency": "USD",
    "revenue": {"value": 1_000_000_000, "evidence": "Total net sales 1,000"},
    "gross_profit": {"value": 400_000_000},
    "operating_income": {"value": 200_000_000},
    "net_income": {"value": 150_000_000},
    "diluted_eps": {"value": 1.23},
}


class FakeProvider:
    name = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def _next(self):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def complete(self, **kw):
        return self._next()

    def structured(self, **kw):
        return self._next()


def resp(data):
    return LLMResponse(
        model="claude-haiku-5-5", data=data, usage=Usage(input_tokens=10_000, output_tokens=500)
    )


def make(responses):
    records = []
    gw = LLMGateway(FakeProvider(responses), recorder=records.append, max_attempts=3)
    return gw, records


def test_structured_success_records_cost():
    gw, records = make([resp(GOOD)])
    metrics, _ = gw.structured(
        model="claude-haiku-5-5", system="s", prompt="p", output=FinancialMetrics, purpose="t"
    )
    assert metrics.diluted_eps.value == 1.23
    assert len(records) == 1 and records[0].success
    assert records[0].cost_usd == pytest.approx((10_000 * 0.10 + 500 * 0.50) / 1e6)


def test_retries_transient_errors(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    gw, records = make([TransientLLMError("429"), TransientLLMError("529"), resp(GOOD)])
    metrics, _ = gw.structured(
        model="m", system="s", prompt="p", output=FinancialMetrics, purpose="t"
    )
    assert metrics.company_name == "Acme Corp"
    assert gw.provider.calls == 3


def test_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    gw, records = make([TransientLLMError("x")] * 3)
    with pytest.raises(TransientLLMError):
        gw.complete(model="m", system="s", prompt="p", purpose="t")
    assert records[-1].success is False


def test_repairs_invalid_output_once():
    bad = {"company_name": "Acme"}  # missing required fields
    gw, records = make([resp(bad), resp(GOOD)])
    metrics, _ = gw.structured(
        model="m", system="s", prompt="p", output=FinancialMetrics, purpose="t"
    )
    assert metrics.revenue.value == 1_000_000_000
    assert len(records) == 2


def test_fails_after_two_invalid_outputs():
    gw, _ = make([resp({}), resp({})])
    with pytest.raises(StructuredOutputError):
        gw.structured(model="m", system="s", prompt="p", output=FinancialMetrics, purpose="t")


def test_recorder_failure_does_not_break_call():
    def boom(_):
        raise RuntimeError("db down")

    gw = LLMGateway(FakeProvider([resp(GOOD)]), recorder=boom)
    gw.structured(model="m", system="s", prompt="p", output=FinancialMetrics, purpose="t")


def test_inline_refs_removes_defs():
    schema = inline_refs(FinancialMetrics.model_json_schema())
    assert "$defs" not in str(schema) and "$ref" not in str(schema)
    assert "value" in schema["properties"]["revenue"]["properties"]


def test_unknown_model_cost_is_none():
    assert cost_usd("some-other-model", Usage(1, 1)) is None

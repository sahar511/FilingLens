from __future__ import annotations

from ..ingest.edgar import CompanyRef, FilingRef
from ..ingest.parse import html_to_text, select_financials
from ..llm.gateway import LLMGateway
from ..prompts import load_prompt
from .schemas import FinancialMetrics

PROMPT_NAME = "extract_metrics_v1"


def extract_metrics(
    gateway: LLMGateway,
    *,
    model: str,
    company: CompanyRef,
    filing: FilingRef,
    html: str,
    max_chars: int,
) -> tuple[FinancialMetrics, str]:
    """Extract headline metrics from a filing. Returns (metrics, model actually used)."""
    prompt = load_prompt(PROMPT_NAME)
    text = select_financials(html_to_text(html), max_chars=max_chars)
    metrics, resp = gateway.structured(
        model=model,
        system=prompt.system,
        prompt=prompt.render(
            form=filing.form,
            company=company.name,
            period_end=str(filing.period_end or "unknown"),
            text=text,
        ),
        output=FinancialMetrics,
        purpose="extract_metrics",
        prompt_version=prompt.version,
    )
    return metrics, resp.model

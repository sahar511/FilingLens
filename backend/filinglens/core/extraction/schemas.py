from __future__ import annotations

from pydantic import BaseModel, Field


class Metric(BaseModel):
    value: float | None = Field(
        description="Value in full units (convert 'in millions'/'in thousands'). "
        "Null if not reported in the text."
    )
    evidence: str | None = Field(
        default=None,
        description="Short excerpt (under 25 words) from the text that shows this value.",
    )


class FinancialMetrics(BaseModel):
    """Headline figures for the most recent quarter (10-Q) or fiscal year (10-K)."""

    company_name: str
    period_end: str = Field(description="Period end date as YYYY-MM-DD")
    period_label: str = Field(description="e.g. 'Three months ended June 28, 2026'")
    currency: str = Field(default="USD")
    revenue: Metric
    gross_profit: Metric
    operating_income: Metric
    net_income: Metric
    diluted_eps: Metric = Field(description="Diluted earnings per share, in currency per share")
    notes: str | None = Field(
        default=None, description="Anything ambiguous, e.g. restatements or unusual items."
    )


METRIC_FIELDS = ["revenue", "gross_profit", "operating_income", "net_income", "diluted_eps"]

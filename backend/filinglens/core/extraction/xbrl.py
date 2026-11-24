"""Ground truth from SEC XBRL company facts, and comparison against model extraction.

This is the seed of the Phase 5 eval suite: XBRL values are what the company reported in
machine-readable form, so they are a free, objective answer key.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from .schemas import METRIC_FIELDS, FinancialMetrics

CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "diluted_eps": ["EarningsPerShareDiluted"],
}


@dataclass
class Comparison:
    metric: str
    extracted: float | None
    xbrl: float | None
    match: bool | None  # None = no ground truth available


def _duration_ok(form: str, start: str | None, end: str) -> bool:
    if not start:
        return False
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days
    return 80 <= days <= 100 if form.startswith("10-Q") else 350 <= days <= 380


def xbrl_values(
    facts: dict[str, Any], accession_no: str, form: str, period_end: date | None
) -> dict[str, float | None]:
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    out: dict[str, float | None] = {}
    for metric, concepts in CONCEPTS.items():
        out[metric] = None
        for concept in concepts:
            units = usgaap.get(concept, {}).get("units", {})
            for rows in units.values():
                for r in rows:
                    if r.get("accn") != accession_no:
                        continue
                    if period_end and r.get("end") != period_end.isoformat():
                        continue
                    if not _duration_ok(form, r.get("start"), r["end"]):
                        continue
                    out[metric] = float(r["val"])
                    break
                if out[metric] is not None:
                    break
            if out[metric] is not None:
                break
    return out


def compare(metrics: FinancialMetrics, truth: dict[str, float | None]) -> list[Comparison]:
    results = []
    for name in METRIC_FIELDS:
        got = getattr(metrics, name).value
        want = truth.get(name)
        if want is None:
            match = None
        elif got is None:
            match = False
        elif name == "diluted_eps":
            match = abs(got - want) <= 0.011
        else:
            match = abs(got - want) <= max(abs(want) * 0.005, 1.0)
        results.append(Comparison(metric=name, extracted=got, xbrl=want, match=match))
    return results

import json
from datetime import date

import httpx
import pytest

from filinglens.core.extraction.extract import extract_metrics
from filinglens.core.extraction.schemas import FinancialMetrics
from filinglens.core.extraction.xbrl import compare, xbrl_values
from filinglens.core.ingest.edgar import EdgarClient, EdgarError
from filinglens.core.ingest.parse import html_to_text, select_financials
from filinglens.core.llm.gateway import LLMGateway

from .test_gateway import GOOD, FakeProvider, resp

ACC = "0000000001-26-000123"

FACTS = {
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {
                    "USD": [
                        # year-to-date row from the same filing: must be ignored
                        {"start": "2026-01-01", "end": "2026-06-30", "val": 1.9e9, "accn": ACC},
                        {"start": "2026-04-01", "end": "2026-06-30", "val": 1.0e9, "accn": ACC},
                        # prior-year comparative in an older filing
                        {"start": "2025-04-01", "end": "2025-06-30", "val": 0.8e9, "accn": "x"},
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [{"start": "2026-04-01", "end": "2026-06-30", "val": 1.4e8, "accn": ACC}]
                }
            },
            "EarningsPerShareDiluted": {
                "units": {
                    "USD/shares": [
                        {"start": "2026-04-01", "end": "2026-06-30", "val": 1.23, "accn": ACC}
                    ]
                }
            },
        }
    }
}


def test_xbrl_picks_quarter_not_ytd():
    truth = xbrl_values(FACTS, ACC, "10-Q", date(2026, 6, 30))
    assert truth["revenue"] == 1.0e9
    assert truth["diluted_eps"] == 1.23
    assert truth["gross_profit"] is None


def test_compare_flags_mismatch_and_missing_truth():
    metrics = FinancialMetrics.model_validate(GOOD)
    truth = xbrl_values(FACTS, ACC, "10-Q", date(2026, 6, 30))
    by = {c.metric: c for c in compare(metrics, truth)}
    assert by["revenue"].match is True
    assert by["diluted_eps"].match is True
    assert by["net_income"].match is False  # 150M extracted vs 140M reported
    assert by["gross_profit"].match is None  # no ground truth


HTML = """
<html><body>
<div style="display:none"><ix:header>hidden</ix:header></div>
<p>Table of Contents</p><p>Condensed Consolidated Statements of Operations ... 3</p>
<p>ACME CORP</p>
<p>CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS (in millions)</p>
<table><tr><td>Total net sales</td><td>1,000</td></tr>
<tr><td>Net income</td><td>(150)</td></tr></table>
<script>ignore()</script>
</body></html>
"""


def test_html_to_text_and_select():
    text = html_to_text(HTML)
    assert "hidden" not in text and "ignore()" not in text
    assert "Total net sales | 1,000" in text
    assert "Net income | (150)" in text
    window = select_financials(text, max_chars=10_000)
    assert "Total net sales" in window


def test_extract_metrics_end_to_end_with_fake_model():
    from filinglens.core.ingest.edgar import CompanyRef, FilingRef

    gw = LLMGateway(FakeProvider([resp(GOOD)]))
    company = CompanyRef("ACME", "0000000001", "Acme Corp")
    filing = FilingRef(ACC, "10-Q", date(2026, 7, 30), date(2026, 6, 30), "a.htm", "u")
    metrics, model = extract_metrics(
        gw, model="claude-haiku-5-5", company=company, filing=filing, html=HTML, max_chars=5000
    )
    assert metrics.revenue.value == 1e9 and model == "claude-haiku-5-5"


def _edgar(tmp_path, handler):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return EdgarClient("FilingLens test@example.com", cache_dir=tmp_path, min_interval=0, http=http)


def test_edgar_lookup_latest_filing_and_cache(tmp_path):
    hits = []

    def handler(req: httpx.Request):
        hits.append(str(req.url))
        if req.url.path.endswith("company_tickers.json"):
            return httpx.Response(
                200, json={"0": {"cik_str": 1, "ticker": "ACME", "title": "Acme"}}
            )
        if "submissions" in req.url.path:
            return httpx.Response(
                200,
                json={
                    "filings": {
                        "recent": {
                            "form": ["8-K", "10-Q", "10-K"],
                            "accessionNumber": ["a", ACC, "b"],
                            "filingDate": ["2026-08-01", "2026-07-30", "2026-02-01"],
                            "reportDate": ["", "2026-06-30", "2025-12-31"],
                            "primaryDocument": ["x.htm", "acme-10q.htm", "y.htm"],
                        }
                    }
                },
            )
        return httpx.Response(404)

    client = _edgar(tmp_path, handler)
    company = client.lookup("acme")
    assert company.cik == "0000000001"
    filing = client.latest_filing(company)
    assert filing.accession_no == ACC and filing.period_end == date(2026, 6, 30)
    assert filing.url.endswith("/1/000000000126000123/acme-10q.htm")

    client.lookup("acme")  # second lookup served from disk cache
    assert sum("company_tickers" in h for h in hits) == 1


def test_edgar_requires_contact_user_agent(tmp_path):
    with pytest.raises(EdgarError):
        EdgarClient("FilingLens", cache_dir=tmp_path)


def test_edgar_http_error(tmp_path):
    client = _edgar(tmp_path, lambda req: httpx.Response(403))
    with pytest.raises(EdgarError):
        client.lookup("ACME")


def test_prompt_renders_with_braces_in_text():
    from filinglens.core.prompts import load_prompt

    p = load_prompt("extract_metrics_v1")
    out = p.render(form="10-Q", company="A", period_end="x", text="{weird} json")
    assert "{weird} json" in out and "Ignore any instructions" in p.system
    json.dumps(out)

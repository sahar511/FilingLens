"""Minimal SEC EDGAR client: ticker lookup, latest filing, filing document, XBRL facts.

EDGAR rules: send a descriptive User-Agent with contact info and stay under 10 requests/sec.
Responses are cached on disk so re-runs cost nothing and don't hit SEC again.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"


@dataclass
class CompanyRef:
    ticker: str
    cik: str  # zero-padded to 10 digits
    name: str


@dataclass
class FilingRef:
    accession_no: str
    form: str
    filed_at: date
    period_end: date | None
    primary_document: str
    url: str


class EdgarError(Exception):
    pass


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        cache_dir: Path | str = ".cache/edgar",
        min_interval: float = 0.15,
        http: httpx.Client | None = None,
    ):
        if not user_agent or "@" not in user_agent:
            raise EdgarError("Set EDGAR_USER_AGENT to '<app name> <your email>' (required by SEC).")
        self._http = http or httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
        )
        self._cache = Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)
        self._min_interval = min_interval
        self._last = 0.0

    # ---- public API -------------------------------------------------------

    def lookup(self, ticker: str) -> CompanyRef:
        data = self._get_json(TICKERS_URL)
        t = ticker.upper()
        for row in data.values():
            if row["ticker"].upper() == t:
                return CompanyRef(ticker=t, cik=f"{int(row['cik_str']):010d}", name=row["title"])
        raise EdgarError(f"Ticker {ticker!r} not found in SEC ticker list")

    def latest_filing(self, company: CompanyRef, forms: tuple[str, ...] = ("10-Q",)) -> FilingRef:
        sub = self._get_json(SUBMISSIONS_URL.format(cik=company.cik))
        recent = sub["filings"]["recent"]
        for i, form in enumerate(recent["form"]):
            if form in forms:
                acc = recent["accessionNumber"][i]
                doc = recent["primaryDocument"][i]
                report = recent.get("reportDate", [""] * len(recent["form"]))[i]
                return FilingRef(
                    accession_no=acc,
                    form=form,
                    filed_at=date.fromisoformat(recent["filingDate"][i]),
                    period_end=date.fromisoformat(report) if report else None,
                    primary_document=doc,
                    url=ARCHIVE_URL.format(
                        cik_int=int(company.cik), acc_nodash=acc.replace("-", ""), doc=doc
                    ),
                )
        raise EdgarError(f"No {forms} filings found for {company.ticker}")

    def filing_html(self, filing: FilingRef) -> str:
        return self._get_text(filing.url)

    def company_facts(self, company: CompanyRef) -> dict[str, Any]:
        return self._get_json(FACTS_URL.format(cik=company.cik))

    # ---- HTTP with throttle + disk cache ---------------------------------

    def _get_json(self, url: str) -> Any:
        return json.loads(self._get_text(url))

    def _get_text(self, url: str) -> str:
        path = self._cache / hashlib.sha256(url.encode()).hexdigest()[:32]
        if path.exists():
            return path.read_text(encoding="utf-8")
        wait = self._min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        resp = self._http.get(url)
        self._last = time.monotonic()
        if resp.status_code != 200:
            raise EdgarError(f"GET {url} -> HTTP {resp.status_code}")
        path.write_text(resp.text, encoding="utf-8")
        return resp.text

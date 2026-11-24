"""Small helpers for finding or creating rows, shared by every command that saves data."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..ingest.edgar import CompanyRef, FilingRef
from .models import Company, Filing


def get_or_create_company(db: Session, ref: CompanyRef) -> Company:
    company = db.scalar(select(Company).where(Company.ticker == ref.ticker))
    if company is None:
        company = Company(ticker=ref.ticker, cik=ref.cik, name=ref.name)
        db.add(company)
        db.flush()  # sends the INSERT now so company.id is filled in
    return company


def get_or_create_filing(db: Session, company: Company, ref: FilingRef) -> Filing:
    filing = db.scalar(select(Filing).where(Filing.accession_no == ref.accession_no))
    if filing is None:
        filing = Filing(
            company_id=company.id,
            accession_no=ref.accession_no,
            form=ref.form,
            period_end=ref.period_end,
            filed_at=ref.filed_at,
            url=ref.url,
        )
        db.add(filing)
        db.flush()
    return filing

"""Runs against a real Postgres with pgvector. Skipped unless TEST_DATABASE_URL is set:

    docker compose exec db createdb -U filinglens filinglens_test
    TEST_DATABASE_URL=postgresql+psycopg://filinglens:filinglens@localhost:5432/filinglens_test \
        pytest -q tests/integration
"""

import os
from datetime import date

import pytest
from sqlalchemy import func, select

from filinglens.core.db.models import Base, FilingChunk
from filinglens.core.db.session import init_db, make_sessionmaker
from filinglens.core.ingest.edgar import CompanyRef, FilingRef
from filinglens.core.retrieval.store import index_filing, search_chunks

from ..unit.test_retrieval import FakeEmbedder

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")

FILL = "general business text about operations. " * 15


def html(*sections: tuple[str, str]) -> str:
    body = "".join(f"<p>{head}</p><p>{text} {FILL}</p>" for head, text in sections)
    return f"<html><body>{body}</body></html>"


ACME_HTML = html(
    ("PART I", ""),
    ("Item 1. Financial Statements", "Total net sales grew this quarter."),
    ("PART II", ""),
    ("Item 1. Legal Proceedings", "Antitrust litigation and lawsuits are pending."),
    ("Item 1A. Risk Factors", "Supply chain disruption in China could hurt production."),
)
BETA_HTML = html(
    ("PART II", ""),
    ("Item 1A. Risk Factors", "Cloud competition from other providers is intense."),
)
ACME = CompanyRef("ACME", "0000000001", "Acme Corp")
BETA = CompanyRef("BETA", "0000000002", "Beta Inc")
ACME_Q = FilingRef("acc-1", "10-Q", date(2026, 7, 30), date(2026, 6, 30), "a.htm", "u1")
BETA_Q = FilingRef("acc-2", "10-Q", date(2026, 7, 30), date(2026, 6, 30), "b.htm", "u2")


@pytest.fixture
def sm():
    sm = make_sessionmaker(URL)
    init_db(sm)
    Base.metadata.drop_all(sm.kw["bind"])  # start every test from empty tables
    init_db(sm)
    return sm


def test_index_then_search_across_companies(sm):
    e = FakeEmbedder()
    with sm.begin() as db:
        assert index_filing(db, e, ACME, ACME_Q, ACME_HTML).chunks == 3
        index_filing(db, e, BETA, BETA_Q, BETA_HTML)

    with sm() as db:
        hits = search_chunks(db, e.embed_query("China supply chain risk"), k=3)
        assert hits[0].ticker == "ACME" and hits[0].section_title == "Risk Factors"
        assert hits[0].score >= hits[1].score

        hits = search_chunks(db, e.embed_query("cloud competition"), k=1)
        assert hits[0].ticker == "BETA"

        only_acme = search_chunks(db, e.embed_query("cloud competition"), k=5, ticker="acme")
        assert {h.ticker for h in only_acme} == {"ACME"}


def test_reindex_skips_or_replaces_without_duplicates(sm):
    e = FakeEmbedder()
    with sm.begin() as db:
        index_filing(db, e, ACME, ACME_Q, ACME_HTML)
    with sm.begin() as db:
        assert index_filing(db, e, ACME, ACME_Q, ACME_HTML).skipped is True
    with sm.begin() as db:
        assert index_filing(db, e, ACME, ACME_Q, ACME_HTML, force=True).skipped is False
    with sm() as db:
        assert db.scalar(select(func.count()).select_from(FilingChunk)) == 3

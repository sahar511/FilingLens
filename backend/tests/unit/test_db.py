from sqlalchemy import select

from filinglens.core.db.models import LLMCall
from filinglens.core.db.session import db_recorder, init_db, make_sessionmaker
from filinglens.core.extraction.schemas import FinancialMetrics
from filinglens.core.llm.gateway import LLMGateway

from .test_gateway import GOOD, FakeProvider, resp


def test_gateway_writes_llm_calls_to_db():
    sm = make_sessionmaker("sqlite://")
    init_db(sm)
    gw = LLMGateway(FakeProvider([resp(GOOD)]), recorder=db_recorder(sm))
    gw.structured(
        model="claude-haiku-5-5",
        system="s",
        prompt="p",
        output=FinancialMetrics,
        purpose="extract_metrics",
        prompt_version="v1",
    )
    with sm() as db:
        row = db.scalars(select(LLMCall)).one()
    assert row.purpose == "extract_metrics" and row.cost_usd > 0 and row.prompt_version == "v1"

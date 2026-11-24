from __future__ import annotations

from dataclasses import asdict

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from ..llm.gateway import CallRecord
from .models import Base, LLMCall


def make_sessionmaker(database_url: str) -> sessionmaker[Session]:
    engine = create_engine(database_url, pool_pre_ping=True)
    return sessionmaker(engine, expire_on_commit=False)


def init_db(sm: sessionmaker[Session]) -> None:
    """Create tables (and the pgvector extension on Postgres). Missing tables are added;
    existing ones are left alone."""
    engine = sm.kw["bind"]
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


def db_recorder(sm: sessionmaker[Session]):
    """Gateway recorder that writes each model call to llm_calls."""

    def record(r: CallRecord) -> None:
        with sm.begin() as s:
            s.add(LLMCall(**asdict(r)))

    return record

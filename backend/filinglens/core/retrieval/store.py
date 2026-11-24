"""Save a filing's chunks and embeddings in Postgres, and search them with pgvector."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ..db.models import Company, Filing, FilingChunk
from ..db.repo import get_or_create_company, get_or_create_filing
from ..ingest.chunking import chunk_filing
from ..ingest.edgar import CompanyRef, FilingRef
from ..ingest.parse import html_to_text
from ..ingest.sections import split_sections
from .embeddings import Embedder
from .search import text_for_embedding


@dataclass
class IndexResult:
    chunks: int
    skipped: bool  # True if the filing was already indexed with this model


def index_filing(
    db: Session,
    embedder: Embedder,
    company_ref: CompanyRef,
    filing_ref: FilingRef,
    html: str,
    force: bool = False,
) -> IndexResult:
    company = get_or_create_company(db, company_ref)
    filing = get_or_create_filing(db, company, filing_ref)

    existing = db.scalar(
        select(func.count())
        .select_from(FilingChunk)
        .where(FilingChunk.filing_id == filing.id, FilingChunk.embed_model == embedder.name)
    )
    if existing and not force:
        return IndexResult(chunks=existing, skipped=True)

    chunks = chunk_filing(split_sections(html_to_text(html)))
    vectors = embedder.embed_passages([text_for_embedding(c) for c in chunks])

    # Replace any old chunks for this filing, so re-indexing never creates duplicates.
    db.execute(delete(FilingChunk).where(FilingChunk.filing_id == filing.id))
    db.add_all(
        FilingChunk(
            filing_id=filing.id,
            section_key=c.section_key,
            section_title=c.section_title[:255],
            chunk_index=c.index,
            text=c.text,
            embedding=v,
            embed_model=embedder.name,
        )
        for c, v in zip(chunks, vectors, strict=True)
    )
    return IndexResult(chunks=len(chunks), skipped=False)


@dataclass
class StoredHit:
    ticker: str
    form: str
    period_end: date | None
    section_key: str
    section_title: str
    chunk_index: int
    text: str
    url: str
    score: float  # cosine similarity, 1 = same meaning


def search_chunks(
    db: Session,
    query_vec: np.ndarray,
    k: int = 5,
    ticker: str | None = None,
    embed_model: str | None = None,
) -> list[StoredHit]:
    distance = FilingChunk.embedding.cosine_distance(query_vec)  # 0 = identical, 2 = opposite
    stmt = (
        select(FilingChunk, Filing, Company, distance.label("distance"))
        .join(Filing, FilingChunk.filing_id == Filing.id)
        .join(Company, Filing.company_id == Company.id)
        .order_by(distance)
        .limit(k)
    )
    if ticker:
        stmt = stmt.where(Company.ticker == ticker.upper())
    if embed_model:
        stmt = stmt.where(FilingChunk.embed_model == embed_model)

    return [
        StoredHit(
            ticker=company.ticker,
            form=filing.form,
            period_end=filing.period_end,
            section_key=chunk.section_key,
            section_title=chunk.section_title,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            url=filing.url,
            score=1 - dist,
        )
        for chunk, filing, company, dist in db.execute(stmt).all()
    ]

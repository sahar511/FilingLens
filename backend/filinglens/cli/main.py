"""FilingLens CLI.

filinglens init-db
filinglens extract AAPL
filinglens extract MSFT --form 10-K
filinglens costs
"""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

from ..core.config import get_settings
from ..core.db.models import Company, ExtractedMetric, Filing, LLMCall
from ..core.db.session import db_recorder, init_db, make_sessionmaker
from ..core.extraction.extract import PROMPT_NAME, extract_metrics
from ..core.extraction.schemas import METRIC_FIELDS
from ..core.extraction.xbrl import compare, xbrl_values
from ..core.ingest.chunking import chunk_section
from ..core.ingest.edgar import EdgarClient
from ..core.ingest.parse import html_to_text
from ..core.ingest.sections import split_sections
from ..core.llm.factory import make_provider
from ..core.llm.gateway import CallRecord, LLMGateway
from ..core.rag.answer import answer_question
from ..core.retrieval.embeddings import FastEmbedEmbedder
from ..core.retrieval.store import index_filing, search_chunks

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()
logging.basicConfig(level=logging.WARNING)


def _fmt(v: float | None, metric: str) -> str:
    if v is None:
        return "-"
    return f"{v:,.2f}" if metric == "diluted_eps" else f"{v:,.0f}"


@app.command("init-db")
def init_db_cmd() -> None:
    """Create database tables."""
    init_db(make_sessionmaker(get_settings().database_url))
    console.print("[green]Tables created.[/green]")


@app.command()
def extract(
    ticker: str,
    form: str = typer.Option("10-Q", help="10-Q or 10-K"),
    save: bool = typer.Option(True, help="Save results and cost records to the database"),
) -> None:
    """Fetch the latest filing, extract headline metrics, and check them against XBRL."""
    s = get_settings()

    calls: list[CallRecord] = []
    sm = None
    if save:
        sm = make_sessionmaker(s.database_url)
        try:
            init_db(sm)
        except OperationalError:
            console.print(
                "[yellow]Database not reachable; continuing without saving. "
                "Run `docker compose up -d` to enable it.[/yellow]"
            )
            sm = None
    record_db = db_recorder(sm) if sm else None

    def recorder(r: CallRecord) -> None:
        calls.append(r)
        if record_db:
            record_db(r)

    edgar = EdgarClient(s.edgar_user_agent)
    try:
        provider = make_provider(s)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    gateway = LLMGateway(provider, recorder=recorder)

    with console.status(f"Looking up {ticker.upper()} on EDGAR..."):
        company = edgar.lookup(ticker)
        filing = edgar.latest_filing(company, forms=(form,))
        html = edgar.filing_html(filing)
        facts = edgar.company_facts(company)
    console.print(
        f"[bold]{company.name}[/bold] {filing.form} for period ending {filing.period_end} "
        f"(filed {filing.filed_at})\n{filing.url}"
    )

    with console.status("Extracting metrics with the model..."):
        metrics, model_used = extract_metrics(
            gateway,
            model=s.extraction_model,
            company=company,
            filing=filing,
            html=html,
            max_chars=s.extraction_max_chars,
        )

    truth = xbrl_values(facts, filing.accession_no, filing.form, filing.period_end)
    results = compare(metrics, truth)

    table = Table(title=f"{metrics.period_label} ({metrics.currency})")
    table.add_column("Metric")
    table.add_column("Extracted", justify="right")
    table.add_column("XBRL", justify="right")
    table.add_column("Match")
    for r in results:
        mark = {True: "[green]yes[/green]", False: "[red]no[/red]", None: "[dim]n/a[/dim]"}
        table.add_row(r.metric, _fmt(r.extracted, r.metric), _fmt(r.xbrl, r.metric), mark[r.match])
    console.print(table)

    checked = [r for r in results if r.match is not None]
    if checked:
        ok = sum(r.match for r in checked)
        console.print(f"Accuracy vs XBRL: {ok}/{len(checked)}")
    if metrics.notes:
        console.print(f"[dim]Model notes: {metrics.notes}[/dim]")

    cost = sum(c.cost_usd or 0 for c in calls)
    tokens_in = sum(c.input_tokens for c in calls)
    tokens_out = sum(c.output_tokens for c in calls)
    console.print(
        f"[dim]{len(calls)} model call(s), {tokens_in:,} in / {tokens_out:,} out tokens, "
        f"${cost:.4f} ({model_used})[/dim]"
    )

    if sm:
        _save(sm, company, filing, metrics, model_used)
        console.print("[dim]Saved to database.[/dim]")


def _save(sm, company, filing, metrics, model_used) -> None:
    with sm.begin() as db:
        c = db.scalar(select(Company).where(Company.ticker == company.ticker))
        if not c:
            c = Company(ticker=company.ticker, cik=company.cik, name=company.name)
            db.add(c)
            db.flush()
        f = db.scalar(select(Filing).where(Filing.accession_no == filing.accession_no))
        if not f:
            f = Filing(
                company_id=c.id,
                accession_no=filing.accession_no,
                form=filing.form,
                period_end=filing.period_end,
                filed_at=filing.filed_at,
                url=filing.url,
            )
            db.add(f)
            db.flush()
        f.status = "done"
        for name in METRIC_FIELDS:
            m = getattr(metrics, name)
            existing = db.scalar(
                select(ExtractedMetric).where(
                    ExtractedMetric.filing_id == f.id,
                    ExtractedMetric.metric == name,
                    ExtractedMetric.prompt_version == PROMPT_NAME,
                )
            )
            row = existing or ExtractedMetric(
                filing_id=f.id, metric=name, prompt_version=PROMPT_NAME
            )
            row.value = m.value
            row.unit = "USD/share" if name == "diluted_eps" else metrics.currency
            row.evidence = m.evidence
            row.model = model_used
            db.add(row)


@app.command()
def costs() -> None:
    """Show LLM spend recorded in the database, by purpose and model."""
    sm = make_sessionmaker(get_settings().database_url)
    with sm() as db:
        rows = db.execute(
            select(
                LLMCall.purpose,
                LLMCall.model,
                func.count(),
                func.sum(LLMCall.input_tokens),
                func.sum(LLMCall.output_tokens),
                func.sum(LLMCall.cost_usd),
                func.avg(LLMCall.latency_ms),
            ).group_by(LLMCall.purpose, LLMCall.model)
        ).all()
    table = Table(title="LLM usage")
    for col in ["Purpose", "Model", "Calls", "Tokens in", "Tokens out", "Cost (USD)", "Avg ms"]:
        table.add_column(col)
    for p, m, n, ti, to, c, ms in rows:
        table.add_row(p, m, str(n), f"{ti:,}", f"{to:,}", f"{c or 0:.4f}", f"{ms:.0f}")
    console.print(table)


@app.command()
def ask(
    question: str,
    system: str = typer.Option("You are a helpful, concise assistant.", help="System prompt"),
) -> None:
    """Send one question straight to the AI model and print the answer."""
    s = get_settings()
    try:
        provider = make_provider(s)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    calls: list[CallRecord] = []
    gateway = LLMGateway(provider, recorder=calls.append)
    resp = gateway.complete(
        model=s.extraction_model, system=system, prompt=question, purpose="ask", max_tokens=2048
    )
    console.print(resp.text or "[yellow](empty answer: try a higher max_tokens)[/yellow]")
    c = calls[-1]
    console.print(
        f"[dim]{c.input_tokens} in / {c.output_tokens} out tokens, "
        f"{c.latency_ms} ms ({c.model})[/dim]"
    )


@app.command()
def sections(
    ticker: str,
    form: str = typer.Option("10-Q", help="10-Q or 10-K"),
    show: str | None = typer.Option(
        None, help="Print the first chunk of a section, e.g. 'Part II - Item 1A'"
    ),
) -> None:
    """Show a filing's sections and how many chunks each becomes (no AI calls)."""
    s = get_settings()
    edgar = EdgarClient(s.edgar_user_agent)
    company = edgar.lookup(ticker)
    filing = edgar.latest_filing(company, forms=(form,))
    secs = split_sections(html_to_text(edgar.filing_html(filing)))

    table = Table(title=f"{company.name} {filing.form} ({filing.period_end})")
    for col in ["Section", "Title", "Chars", "Chunks"]:
        table.add_column(col)
    total = 0
    for sec in secs:
        chunks = chunk_section(sec)
        total += len(chunks)
        table.add_row(sec.key, sec.title[:50], f"{len(sec.text):,}", str(len(chunks)))
        if show and sec.key.lower() == show.lower():
            console.print(f"[bold]{sec.key} chunk 0:[/bold]\n{chunks[0].text}\n")
    console.print(table)
    console.print(f"{len(secs)} sections, {total} chunks")


@app.command()
def index(
    ticker: str,
    form: str = typer.Option("10-Q", help="10-Q or 10-K"),
    force: bool = typer.Option(False, help="Re-index even if already done"),
) -> None:
    """Split, embed and store a filing's chunks in the database (once per filing)."""
    s = get_settings()
    sm = make_sessionmaker(s.database_url)
    init_db(sm)
    edgar = EdgarClient(s.edgar_user_agent)
    company = edgar.lookup(ticker)
    filing = edgar.latest_filing(company, forms=(form,))
    html = edgar.filing_html(filing)

    with console.status(f"Indexing {company.name} {filing.form} ({filing.period_end})..."):
        embedder = FastEmbedEmbedder()
        with sm.begin() as db:
            result = index_filing(db, embedder, company, filing, html, force=force)

    if result.skipped:
        console.print(f"Already indexed ({result.chunks} chunks). Use --force to redo it.")
    else:
        console.print(f"[green]Indexed {result.chunks} chunks[/green] with {embedder.name}.")


@app.command()
def search(
    query: str,
    ticker: str | None = typer.Option(None, help="Only search this company"),
    k: int = typer.Option(5, help="How many results to show"),
) -> None:
    """Search all indexed filings for passages closest in meaning to QUERY."""
    s = get_settings()
    sm = make_sessionmaker(s.database_url)
    with console.status("Searching..."):
        embedder = FastEmbedEmbedder()
        query_vec = embedder.embed_query(query)
        with sm() as db:
            hits = search_chunks(db, query_vec, k=k, ticker=ticker, embed_model=embedder.name)

    if not hits:
        console.print("No results. Index a filing first, e.g. [bold]filinglens index AAPL[/bold]")
        return
    for rank, h in enumerate(hits, start=1):
        console.print(
            f"[bold]{rank}. {h.ticker} {h.form} ({h.period_end}) · {h.section_key}[/bold]"
            f" ({h.section_title[:40]}), chunk {h.chunk_index}   [dim]score {h.score:.3f}[/dim]"
        )
        console.print(h.text[:300].replace("\n", " ") + "...\n")


@app.command()
def answer(
    question: str,
    ticker: str | None = typer.Option(None, help="Only use this company's filings"),
    k: int = typer.Option(5, help="How many passages to give the model"),
) -> None:
    """Answer QUESTION from indexed filings, with verified citations."""
    s = get_settings()
    try:
        provider = make_provider(s)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    sm = make_sessionmaker(s.database_url)
    calls: list[CallRecord] = []
    record_db = db_recorder(sm)

    def recorder(r: CallRecord) -> None:
        calls.append(r)
        record_db(r)

    with console.status("Finding relevant passages..."):
        embedder = FastEmbedEmbedder()
        query_vec = embedder.embed_query(question)
        with sm() as db:
            hits = search_chunks(db, query_vec, k=k, ticker=ticker, embed_model=embedder.name)

    with console.status(f"Writing an answer from {len(hits)} passages..."):
        result = answer_question(
            LLMGateway(provider, recorder=recorder),
            model=s.extraction_model,
            question=question,
            hits=hits,
        )

    console.print(f"[bold]Q: {escape(question)}[/bold]\n")
    if not result.answer.answerable:
        console.print("[yellow]The indexed filings don't answer this.[/yellow]")
        if result.answer.note:
            console.print(escape(result.answer.note))
    for c in result.checked:
        cites = escape(" ".join(f"[{sid}]" for sid in c.claim.source_ids))
        line = f"• {escape(c.claim.text)} {cites}"
        if not c.ok:
            line += f"  [red](unverified: {escape('; '.join(c.problems))})[/red]"
        console.print(line)

    if result.cited_ids:
        console.print("\n[bold]Sources[/bold]")
        for sid in result.cited_ids:
            h = result.sources[sid]
            console.print(
                escape(f"[{sid}] {h.ticker} {h.form} ({h.period_end}), {h.section_key}")
                + f" {escape(h.section_title[:50])}\n     [dim]{h.url}[/dim]"
            )

    ok = sum(c.ok for c in result.checked)
    tokens = sum(c.input_tokens + c.output_tokens for c in calls)
    console.print(f"\n[dim]{ok}/{len(result.checked)} claims verified · {tokens:,} tokens[/dim]")


if __name__ == "__main__":
    app()

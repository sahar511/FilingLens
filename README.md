# FilingLens

Ask questions about public companies and get answers from their SEC filings, with every
claim cited and checked against the source text.

> For research and education only. Not investment advice.

```
$ filinglens answer "Is Apple facing any lawsuits?" --ticker AAPL

• Apple is subject to civil antitrust lawsuits in the United States alleging monopolization of smartphone markets. 
[S2]
• Apple is also involved in various civil litigation matters filed in state and federal courts in the U.S. alleging 
antitrust violations. [S5]

Sources
[S2] AAPL 10-Q (2026-06-27), Part II - Item 1A Risk Factors
     https://www.sec.gov/Archives/edgar/data/...
[S5] AAPL 10-Q (2026-06-27), Part II - Item 1 Legal Proceedings
     https://www.sec.gov/Archives/edgar/...

2/2 claims verified · 2,197 tokens
```

## How it works

```
SEC EDGAR → parse HTML → split into filing sections → ~600-token chunks
          → local embeddings (bge-small) → Postgres + pgvector (HNSW index)

Question → embed → top-k chunks → LLM answers with source IDs + exact quotes
         → code verifies every citation and quote → answer with sources
```

- **Grounded answers:** the model may only use the retrieved passages and must refuse when
  they don't contain the answer.
- **Verified citations:** each claim includes a word-for-word quote, checked in code
  against the cited passage. Unsupported or invented citations are flagged.
- **Metric extraction:** pulls revenue, margins, net income, and EPS into a validated
  schema and checks them against the company's official XBRL data.
- **LLM gateway:** one place for every model call: provider-agnostic, retries that honor
  `retry-after`, schema validation with one repair attempt, and token/cost/latency logging.

## Tech stack

Python · PostgreSQL + pgvector · SQLAlchemy · Pydantic · fastembed (bge-small-en-v1.5) ·
Groq API (gpt-oss-120b), Claude-compatible · Typer · pytest · GitHub Actions

## Quick start

Requires Python 3.11+, Docker, and a free [Groq](https://console.groq.com) API key.

```bash
cp .env.example .env      # add GROQ_API_KEY and EDGAR_USER_AGENT ("FilingLens you@email.com")
docker compose up -d      # Postgres with pgvector

cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

filinglens index AAPL
filinglens answer "What are Apple's main supply chain risks?" --ticker AAPL
```

## Commands

| Command | What it does |
|---|---|
| `filinglens extract AAPL` | Extract headline metrics and compare them with XBRL |
| `filinglens index AAPL` | Split, embed, and store a filing (once per filing) |
| `filinglens search "question"` | Find the most relevant passages across indexed filings |
| `filinglens answer "question"` | Answer with verified citations |
| `filinglens eval` | Score retrieval (and answers with `--answers`) on a labeled question set |
| `filinglens sections AAPL` | Show how a filing splits into sections and chunks |
| `filinglens costs` | Token usage and cost per feature |

## Evaluation

| Metric | Result |
|---|---|
| Metric extraction vs XBRL | X/Y across 5 companies |
| Retrieval hit@5 | X% (N labeled questions) |
| Retrieval MRR | X.XX |
| Claims verified | X% |
| Refused unanswerable questions | X% |

Results from each run are saved in `backend/evals/results/`.

## Tests

```bash
cd backend && pytest -q    # unit tests: fake model and HTTP responses, no API key needed

# integration tests against real Postgres + pgvector
docker compose exec db createdb -U filinglens filinglens_test
TEST_DATABASE_URL=postgresql+psycopg://filinglens:filinglens@localhost:5432/filinglens_test \
  pytest -q tests/integration
```

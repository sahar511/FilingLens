"""Answer a question from retrieved filing chunks, with citations the code can verify."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..llm.gateway import LLMGateway
from ..prompts import load_prompt
from ..retrieval.store import StoredHit

PROMPT_NAME = "answer_v1"


# ---- what the model must return ------------------------------------------------------


class Claim(BaseModel):
    text: str = Field(description="One factual sentence that answers part of the question.")
    source_ids: list[str] = Field(description="IDs of the sources supporting it, e.g. ['S1'].")
    quote: str = Field(
        description="5 to 25 words copied exactly from one of the cited sources, "
        "showing the claim is supported."
    )


class CitedAnswer(BaseModel):
    answerable: bool = Field(description="False if the sources do not contain the answer.")
    claims: list[Claim] = Field(default_factory=list)
    note: str | None = Field(default=None, description="If not answerable, what is missing.")


# ---- building the prompt -------------------------------------------------------------


def _attr(value: object) -> str:
    return str(value).replace('"', "'")  # keep attribute values from breaking the tag


def format_sources(hits: list[StoredHit]) -> tuple[str, dict[str, StoredHit]]:
    """Give each chunk an ID (S1, S2, ...) and wrap it in a labeled <source> tag."""
    sources = {f"S{i}": h for i, h in enumerate(hits, start=1)}
    blocks = []
    for sid, h in sources.items():
        text = h.text.replace("</source>", "</ source>")  # a chunk can't close the tag early
        blocks.append(
            f'<source id="{sid}" company="{_attr(h.ticker)}" form="{_attr(h.form)}" '
            f'period="{_attr(h.period_end)}" '
            f'section="{_attr(h.section_key)} {_attr(h.section_title)}">\n{text}\n</source>'
        )
    return "\n\n".join(blocks), sources


# ---- checking the model's work --------------------------------------------------------


def _norm(s: str) -> str:
    """Lowercase, straighten curly quotes, and collapse whitespace for fair comparison."""
    s = s.lower().replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip(" .\"'…")


def quote_in(quote: str, text: str) -> bool:
    q = _norm(quote)
    return bool(q) and q in _norm(text)


@dataclass
class CheckedClaim:
    claim: Claim
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def verify(answer: CitedAnswer, sources: dict[str, StoredHit]) -> list[CheckedClaim]:
    checked = []
    for claim in answer.claims:
        problems = []
        if not claim.source_ids:
            problems.append("no citation")
        unknown = [sid for sid in claim.source_ids if sid not in sources]
        if unknown:
            problems.append(f"unknown source {', '.join(unknown)}")
        cited = [sources[sid] for sid in claim.source_ids if sid in sources]
        if cited and not any(quote_in(claim.quote, h.text) for h in cited):
            problems.append("quote not found in the cited source")
        checked.append(CheckedClaim(claim=claim, problems=problems))
    return checked


# ---- putting it together -------------------------------------------------------------


@dataclass
class AnswerResult:
    question: str
    answer: CitedAnswer
    sources: dict[str, StoredHit]
    checked: list[CheckedClaim]

    @property
    def cited_ids(self) -> list[str]:
        """Source IDs actually used by the claims, in order of first use."""
        seen: list[str] = []
        for c in self.answer.claims:
            seen += [sid for sid in c.source_ids if sid in self.sources and sid not in seen]
        return seen


def answer_question(
    gateway: LLMGateway, *, model: str, question: str, hits: list[StoredHit]
) -> AnswerResult:
    if not hits:  # nothing retrieved: don't spend an AI call
        empty = CitedAnswer(answerable=False, note="No indexed passages matched the question.")
        return AnswerResult(question, empty, {}, [])

    prompt = load_prompt(PROMPT_NAME)
    sources_text, sources = format_sources(hits)
    answer, _ = gateway.structured(
        model=model,
        system=prompt.system,
        prompt=prompt.render(question=question, sources=sources_text),
        output=CitedAnswer,
        purpose="answer",
        prompt_version=prompt.version,
    )
    return AnswerResult(question, answer, sources, verify(answer, sources))

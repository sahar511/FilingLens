"""Cut sections into overlapping chunks small enough to embed and retrieve."""

from __future__ import annotations

from dataclasses import dataclass

from .sections import Section

CHARS_PER_TOKEN = 4  # rough average for English text


@dataclass
class Chunk:
    section_key: str
    section_title: str
    index: int  # position within the section
    text: str


def chunk_section(section: Section, max_tokens: int = 600, overlap_tokens: int = 80) -> list[Chunk]:
    max_chars = max_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    # Paragraphs (one per line after html_to_text); hard-split any that are too long alone.
    paras: list[str] = []
    for line in section.text.split("\n"):
        line = line.strip()
        while len(line) > max_chars:
            paras.append(line[:max_chars])
            line = line[max_chars:]
        if line:
            paras.append(line)

    texts: list[str] = []
    current = ""
    for p in paras:
        if current and len(current) + 1 + len(p) > max_chars:
            texts.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = tail[tail.find(" ") + 1 :] if " " in tail else tail  # start on a word
        current = f"{current}\n{p}" if current else p
    if current:
        texts.append(current)

    return [Chunk(section.key, section.title, i, t) for i, t in enumerate(texts)]


def chunk_filing(sections: list[Section], **kw) -> list[Chunk]:
    return [c for s in sections for c in chunk_section(s, **kw)]

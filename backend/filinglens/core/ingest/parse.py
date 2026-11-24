"""Turn filing HTML into clean text and pick the part worth sending to the model.

Phase 1 uses a simple heuristic: start at the income statement and take a window of text.
Phase 2 replaces this with proper section parsing (Item 1, Item 2 MD&A, Item 1A Risk Factors).
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

_INCOME_HEADINGS = [
    r"statements? of operations",
    r"statements? of income",
    r"statements? of earnings",
    r"income statements?",
]


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Hidden inline-XBRL header blocks repeat raw tags; drop them.
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    # One line per table row, cells separated by " | ", so labels stay next to their numbers.
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c and c != "$"]
        tr.replace_with(soup.new_string("\n" + " | ".join(cells) + "\n"))
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def select_financials(text: str, max_chars: int) -> str:
    """Return a window starting at the income statement heading (skipping the table of
    contents), or the start of the document if none is found."""
    lower = text.lower()
    hits: list[int] = []
    for pattern in _INCOME_HEADINGS:
        hits += [m.start() for m in re.finditer(pattern, lower)]
    hits.sort()
    # The first hit is usually the table of contents; prefer the 2nd if there is one.
    start = hits[1] if len(hits) > 1 else (hits[0] if hits else 0)
    start = max(start - 500, 0)  # include the company/period header above the statement
    return text[start : start + max_chars]

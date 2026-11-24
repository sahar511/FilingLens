"""Split a filing's text into its official sections, e.g. "Item 1A. Risk Factors"."""

from __future__ import annotations

import re
from dataclasses import dataclass

_FLAGS = re.IGNORECASE | re.MULTILINE

# A line starting "Item 1A." / "ITEM 7 -". Group 1 = item number, group 2 = rest of line.
_ITEM = re.compile(r"^[ \t]*item[ \t]+(\d{1,2}[a-c]?)\b[ \t.:—–-]*(.*)$", _FLAGS)
# A line that starts with "PART I" / "Part II".
_PART = re.compile(r"^[ \t]*part[ \t]+(iv|i{1,3})\b", _FLAGS)


@dataclass
class Section:
    key: str  # e.g. "Part II - Item 1A"
    title: str  # e.g. "Risk Factors"
    text: str


def split_sections(text: str, min_chars: int = 300) -> list[Section]:
    parts = [(m.start(), m.group(1).upper()) for m in _PART.finditer(text)]
    heads = list(_ITEM.finditer(text))
    sections: list[Section] = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[m.end() : end].strip()
        if len(body) < min_chars:
            continue  # table-of-contents lines and short cross-references
        part = next((p for pos, p in reversed(parts) if pos < m.start()), None)
        item = m.group(1).upper()
        key = f"Part {part} - Item {item}" if part else f"Item {item}"
        title = m.group(2).strip(" .|") or body.split("\n", 1)[0].strip(" .|")
        sections.append(Section(key=key, title=title[:120], text=body))
    return sections

from itertools import pairwise

from filinglens.core.ingest.chunking import chunk_filing, chunk_section
from filinglens.core.ingest.sections import Section, split_sections

FILLER = "The company sells products worldwide and faces competition. " * 20

TEXT = f"""ACME CORP FORM 10-Q
Table of Contents
PART I
Item 1. | Financial Statements | 3
Item 2. | Management's Discussion and Analysis | 20
PART II
Item 1A. | Risk Factors | 40
PART I — FINANCIAL INFORMATION
Item 1. Financial Statements
Total net sales | 1,000
{FILLER}
Item 2. Management's Discussion and Analysis of Financial Condition
{FILLER}
PART II — OTHER INFORMATION
Item 1. Legal Proceedings
{FILLER}
Item 1A. Risk Factors
{FILLER}
"""


def test_split_skips_toc_and_tracks_parts():
    secs = split_sections(TEXT)
    assert [s.key for s in secs] == [
        "Part I - Item 1",
        "Part I - Item 2",
        "Part II - Item 1",
        "Part II - Item 1A",
    ]
    assert secs[0].title == "Financial Statements"
    assert secs[3].title == "Risk Factors"
    assert "Total net sales" in secs[0].text


def test_chunks_respect_size_and_overlap():
    words = [f"word{i}" for i in range(3000)]
    lines = [" ".join(words[i : i + 30]) for i in range(0, 3000, 30)]
    sec = Section("Item 7", "MD&A", "\n".join(lines))
    chunks = chunk_section(sec, max_tokens=200, overlap_tokens=20)
    assert len(chunks) > 5
    assert all(len(c.text) <= 200 * 4 + 20 * 4 for c in chunks)
    # the start of each chunk repeats the end of the previous one
    for a, b in pairwise(chunks):
        assert b.text.split()[0] in a.text
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_long_single_line_is_split():
    sec = Section("Item 1", "x", "a" * 10_000)
    chunks = chunk_section(sec, max_tokens=500, overlap_tokens=0)
    assert all(len(c.text) <= 2000 for c in chunks) and len(chunks) == 5


def test_chunk_filing_keeps_section_labels():
    chunks = chunk_filing(split_sections(TEXT), max_tokens=100, overlap_tokens=10)
    assert {c.section_key for c in chunks} == {
        "Part I - Item 1",
        "Part I - Item 2",
        "Part II - Item 1",
        "Part II - Item 1A",
    }

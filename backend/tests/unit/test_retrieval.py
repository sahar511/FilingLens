import re
import zlib

import numpy as np

from filinglens.core.ingest.chunking import Chunk
from filinglens.core.retrieval.embeddings import normalize
from filinglens.core.retrieval.search import text_for_embedding, top_k


class FakeEmbedder:
    """Stand-in for the real model: each word bumps one of 64 slots. Similar words → similar
    vectors, which is enough to test the search logic without downloading a model."""

    name = "fake"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"[a-z]+", text.lower()):
            v[zlib.crc32(word.encode()) % self.dim] += 1
        return v

    def embed_passages(self, texts):
        return normalize(np.array([self._vec(t) for t in texts]))

    def embed_query(self, text):
        return normalize(self._vec(text))


CHUNKS = [
    Chunk("Part I - Item 1", "Financial Statements", 0, "Total net sales and net income"),
    Chunk("Part II - Item 1A", "Risk Factors", 0, "Supply chain disruption in China"),
    Chunk("Part II - Item 1", "Legal Proceedings", 0, "Antitrust lawsuits and litigation"),
]


def test_normalize_makes_unit_length():
    v = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(v[0]), 1.0)
    assert np.allclose(v[1], 0.0)  # an all-zero vector stays zero instead of dividing by 0


def test_top_k_ranks_most_similar_first():
    e = FakeEmbedder()
    vecs = e.embed_passages([text_for_embedding(c) for c in CHUNKS])
    hits = top_k(e.embed_query("supply chain risks in China"), vecs, CHUNKS, k=2)
    assert hits[0].chunk.section_title == "Risk Factors"
    assert len(hits) == 2 and hits[0].score >= hits[1].score


def test_text_for_embedding_includes_section_title():
    assert text_for_embedding(CHUNKS[1]).startswith("Risk Factors\n")

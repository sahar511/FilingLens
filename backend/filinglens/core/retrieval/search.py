"""Find the chunks closest in meaning to a question (in memory, before we add a database)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ingest.chunking import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity: 1 = same meaning, around 0 = unrelated


def text_for_embedding(chunk: Chunk) -> str:
    """Prefix the section title so a chunk keeps its context ("Risk Factors: ...")."""
    return f"{chunk.section_title}\n{chunk.text}"


def top_k(
    query_vec: np.ndarray, chunk_vecs: np.ndarray, chunks: list[Chunk], k: int = 5
) -> list[Hit]:
    scores = chunk_vecs @ query_vec  # one similarity score per chunk
    best = np.argsort(-scores)[:k]  # positions of the k highest scores
    return [Hit(chunk=chunks[i], score=float(scores[i])) for i in best]

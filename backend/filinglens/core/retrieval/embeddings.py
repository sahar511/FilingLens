"""Turn text into vectors (embeddings) whose closeness reflects closeness in meaning."""

from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # small (about 70 MB), 384 numbers per text, runs on CPU


class Embedder(Protocol):
    name: str

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """One row per text: shape (len(texts), dim)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """One vector: shape (dim,)."""
        ...


class FastEmbedEmbedder:
    """Runs an embedding model locally. The model downloads once into cache_dir."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str = ".cache/fastembed"):
        from fastembed import TextEmbedding  # imported here so tests never load the model

        self.name = model_name
        self._model = TextEmbedding(model_name, cache_dir=cache_dir)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self._model.passage_embed(texts)), dtype=np.float32)
        return normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.array(next(iter(self._model.query_embed(text))), dtype=np.float32)
        return normalize(vector)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each vector to length 1, so a dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)

"""Vector store interface and the default in-memory NumPy implementation.

Only ``InMemoryVectorStore`` is implemented in this repository. ``docs/extending.md`` describes how a
pgvector or OpenSearch k-NN backend would implement the same ``VectorStore`` protocol.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from rag_assistant.types import Chunk


def matches_filters(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Exact-match metadata filter. A list value means "any of"."""
    if not filters:
        return True
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, list | tuple | set):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def chunk_metadata(chunk: Chunk) -> dict[str, Any]:
    return {**chunk.metadata, "doc_id": chunk.doc_id, "section": chunk.section, "source": chunk.source}


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def search(
        self, query_vector: np.ndarray, k: int, filters: dict[str, Any] | None = None
    ) -> list[tuple[Chunk, float]]: ...

    def clear(self) -> None: ...

    def __len__(self) -> int: ...


class InMemoryVectorStore:
    """Exact cosine search over a NumPy matrix (vectors are assumed L2-normalised)."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        vectors = np.asarray(vectors, dtype=np.float32)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        self._chunks.extend(chunks)

    def search(
        self, query_vector: np.ndarray, k: int, filters: dict[str, Any] | None = None
    ) -> list[tuple[Chunk, float]]:
        if self._matrix is None or not self._chunks:
            return []
        scores = self._matrix @ np.asarray(query_vector, dtype=np.float32)
        order = np.argsort(-scores, kind="stable")
        results: list[tuple[Chunk, float]] = []
        for idx in order:
            chunk = self._chunks[idx]
            if matches_filters(chunk_metadata(chunk), filters):
                results.append((chunk, float(scores[idx])))
                if len(results) >= k:
                    break
        return results

    def clear(self) -> None:
        self._chunks, self._matrix = [], None

    def __len__(self) -> int:
        return len(self._chunks)

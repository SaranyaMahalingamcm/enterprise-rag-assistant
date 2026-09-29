"""Embedding provider interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Turns text into L2-normalised dense vectors (float32, shape ``(n, dim)``)."""

    name: str

    def fit(self, corpus: list[str]) -> None:
        """Optional corpus-level fitting (no-op for hosted models)."""

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms

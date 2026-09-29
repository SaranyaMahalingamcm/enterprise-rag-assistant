"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # fused (RRF) score used for ranking
    bm25_score: float = 0.0
    dense_score: float = 0.0
    relevance: float = 0.0  # calibrated 0..1 signal used for the refusal threshold


@dataclass
class Citation:
    index: int
    chunk_id: str
    doc_id: str
    section: str
    source: str
    score: float
    snippet: str


@dataclass
class GenerationResult:
    answer: str
    cited_indices: list[int]

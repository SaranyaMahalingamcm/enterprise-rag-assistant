"""Hybrid retrieval: BM25 + dense vectors, combined with reciprocal rank fusion (RRF)."""

from __future__ import annotations

from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from rag_assistant.embeddings.base import Embedder
from rag_assistant.text import tokenize
from rag_assistant.types import Chunk, ScoredChunk
from rag_assistant.vectorstore import InMemoryVectorStore, VectorStore, chunk_metadata, matches_filters


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Standard RRF: score(d) = sum over rankings of 1 / (k + rank(d)), ranks starting at 1."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return fused


def query_coverage(query: str, text: str) -> float:
    """Fraction of distinct query content terms that appear in ``text`` (0..1)."""
    q_terms = set(tokenize(query))
    if not q_terms:
        return 0.0
    return len(q_terms & set(tokenize(text))) / len(q_terms)


class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        corpus = [tokenize(c.text) or ["_empty_"] for c in self._chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int, filters: dict[str, Any] | None = None) -> list[tuple[Chunk, float]]:
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = np.argsort(-scores, kind="stable")
        results: list[tuple[Chunk, float]] = []
        for idx in order:
            if scores[idx] <= 0:
                break
            chunk = self._chunks[idx]
            if matches_filters(chunk_metadata(chunk), filters):
                results.append((chunk, float(scores[idx])))
                if len(results) >= k:
                    break
        return results


class HybridRetriever:
    """Runs BM25 and dense search independently, then fuses the two rankings with RRF.

    RRF scores are rank-based and not comparable across queries, so a separate ``relevance`` value
    (mean of dense cosine and query-term coverage, both 0..1) is attached for the refusal threshold.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore | None = None,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store if vector_store is not None else InMemoryVectorStore()
        self.bm25 = BM25Index()
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self._size = 0

    def index(self, chunks: list[Chunk]) -> None:
        """(Re)build both indexes from the full chunk list."""
        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.embed_documents(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        self.vector_store.clear()
        if chunks:
            self.vector_store.add(chunks, vectors)
        self.bm25.build(chunks)
        self._size = len(chunks)

    def __len__(self) -> int:
        return self._size

    def retrieve(self, query: str, k: int = 4, filters: dict[str, Any] | None = None) -> list[ScoredChunk]:
        if self._size == 0:
            return []
        lexical = self.bm25.search(query, self.candidate_k, filters)
        dense = self.vector_store.search(self.embedder.embed_query(query), self.candidate_k, filters)

        by_id: dict[str, ScoredChunk] = {}
        for chunk, score in lexical:
            by_id[chunk.chunk_id] = ScoredChunk(chunk=chunk, score=0.0, bm25_score=score)
        for chunk, score in dense:
            item = by_id.setdefault(chunk.chunk_id, ScoredChunk(chunk=chunk, score=0.0))
            item.dense_score = score

        fused = reciprocal_rank_fusion([[c.chunk_id for c, _ in lexical], [c.chunk_id for c, _ in dense]], k=self.rrf_k)
        for chunk_id, score in fused.items():
            item = by_id[chunk_id]
            item.score = score
            item.relevance = 0.5 * max(item.dense_score, 0.0) + 0.5 * query_coverage(query, item.chunk.text)
        ranked = sorted(by_id.values(), key=lambda s: (-s.score, s.chunk.chunk_id))
        return ranked[:k]

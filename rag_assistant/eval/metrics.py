"""Evaluation metrics for retrieval, citation quality, groundedness and latency."""

from __future__ import annotations

import re

import numpy as np

from rag_assistant.text import tokenize

_CITATION_MARK_RE = re.compile(r"\[\d+\]")


def first_relevant_rank(retrieved_doc_ids: list[str], expected: set[str]) -> int | None:
    """1-based rank of the first retrieved item whose doc_id is expected, else None."""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected:
            return rank
    return None


def hit_at_k(retrieved_doc_ids: list[str], expected: set[str], k: int) -> float:
    rank = first_relevant_rank(retrieved_doc_ids[:k], expected)
    return 1.0 if rank is not None else 0.0


def reciprocal_rank(retrieved_doc_ids: list[str], expected: set[str]) -> float:
    rank = first_relevant_rank(retrieved_doc_ids, expected)
    return 1.0 / rank if rank else 0.0


def citation_precision(cited_doc_ids: list[str], expected: set[str]) -> float:
    """Fraction of citations that point at an expected source document."""
    if not cited_doc_ids:
        return 0.0
    return sum(1 for d in cited_doc_ids if d in expected) / len(cited_doc_ids)


def groundedness(answer: str, cited_texts: list[str]) -> float:
    """Share of the answer's content tokens that also appear in the cited chunks (0..1).

    A cheap lexical proxy: it catches answers that drift away from their sources, but it cannot
    detect a faithful-sounding paraphrase that changes meaning. An LLM-as-judge would complement it.
    """
    answer_tokens = tokenize(_CITATION_MARK_RE.sub(" ", answer))
    if not answer_tokens:
        return 0.0
    source_vocab = set(tokenize(" ".join(cited_texts)))
    return sum(1 for t in answer_tokens if t in source_vocab) / len(answer_tokens)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0

"""Deterministic offline "LLM": composes an answer from the most relevant retrieved sentences.

It never invents text, so it is useful for tests, demos and as a baseline in evaluation. Its answers
are less fluent than a real model's, and it cannot synthesise across passages.
"""

from __future__ import annotations

from rag_assistant.text import split_sentences, tokenize
from rag_assistant.types import GenerationResult, ScoredChunk


class ExtractiveLLM:
    name = "extractive"

    def __init__(self, max_sentences: int = 3, min_overlap: int = 1, relative_cutoff: float = 0.6) -> None:
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap
        self.relative_cutoff = relative_cutoff

    def generate(self, question: str, chunks: list[ScoredChunk]) -> GenerationResult:
        q_terms = set(tokenize(question))
        candidates: list[tuple[float, int, int, str]] = []  # (score, chunk_idx, position, sentence)
        for ci, sc in enumerate(chunks):
            # skip the "Title - Section" header line that ingestion prepends to every chunk
            body = sc.chunk.text.split("\n", 1)[1] if "\n" in sc.chunk.text else sc.chunk.text
            section_terms = set(tokenize(sc.chunk.section))
            for pos, sentence in enumerate(split_sentences(body)):
                s_terms = set(tokenize(sentence))
                overlap = len(q_terms & s_terms)
                if overlap < self.min_overlap and not (q_terms & section_terms):
                    continue
                # term overlap, a boost when the section heading matches, and a mild rank prior
                score = overlap + 0.5 * len(q_terms & section_terms) + 1.0 / (ci + 2)
                candidates.append((score, ci, pos, sentence))

        if not candidates:
            return GenerationResult(answer="", cited_indices=[])

        ranked = sorted(candidates, key=lambda c: (-c[0], c[1], c[2]))
        # keep only sentences reasonably close to the best one, to avoid padding with weak matches
        cutoff = self.relative_cutoff * ranked[0][0]
        best = [c for c in ranked if c[0] >= cutoff][: self.max_sentences]
        best.sort(key=lambda c: (c[1], c[2]))  # present in document order for readability
        parts = []
        cited: list[int] = []
        for _, ci, _, sentence in best:
            sentence = sentence if sentence.endswith((".", "!", "?")) else sentence + "."
            parts.append(f"{sentence} [{ci + 1}]")
            if ci + 1 not in cited:
                cited.append(ci + 1)
        return GenerationResult(answer=" ".join(parts), cited_indices=cited)

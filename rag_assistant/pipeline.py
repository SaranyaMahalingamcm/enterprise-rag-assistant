"""End-to-end RAG pipeline: ingest -> retrieve -> guard -> generate -> guard -> cite."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_assistant.config import Settings, get_settings
from rag_assistant.embeddings.base import Embedder
from rag_assistant.guardrails import REFUSAL_MESSAGE, detect_prompt_injection, redact_pii
from rag_assistant.ingestion import chunk_documents, load_directory
from rag_assistant.llm.base import LLMProvider
from rag_assistant.providers import build_embedder, build_llm
from rag_assistant.retrieval import HybridRetriever
from rag_assistant.types import Chunk, Citation, Document, ScoredChunk
from rag_assistant.vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    question: str
    answer: str
    citations: list[Citation]
    refused: bool
    refusal_reason: str | None
    retrieved: list[ScoredChunk]
    guardrails: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        llm: LLMProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or build_llm(self.settings)
        self.retriever = HybridRetriever(
            embedder or build_embedder(self.settings),
            vector_store=vector_store,
            candidate_k=self.settings.candidate_k,
            rrf_k=self.settings.rrf_k,
        )
        self._documents: dict[str, Document] = {}
        self._chunks: list[Chunk] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ ingestion
    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def ingest_documents(self, docs: list[Document], replace: bool = False) -> int:
        """Add (or upsert by doc_id) documents and rebuild the indexes. Returns chunks created."""
        with self._lock:
            if replace:
                self._documents.clear()
            for doc in docs:
                self._documents[doc.doc_id] = doc
            self._chunks = chunk_documents(
                self._documents.values(), self.settings.chunk_size, self.settings.chunk_overlap
            )
            self.retriever.index(self._chunks)
        new_ids = {d.doc_id for d in docs}
        created = sum(1 for c in self._chunks if c.doc_id in new_ids)
        logger.info(
            "ingest_complete",
            extra={"documents": len(docs), "chunks_created": created, "total_chunks": len(self._chunks)},
        )
        return created

    def ingest_directory(self, directory: Path | None = None, replace: bool = True) -> int:
        return self.ingest_documents(load_directory(directory or self.settings.docs_dir), replace=replace)

    # ------------------------------------------------------------------ query
    def query(self, question: str, top_k: int | None = None, filters: dict[str, Any] | None = None) -> QueryResult:
        start = time.perf_counter()
        k = top_k or self.settings.top_k

        # 1. Input guardrail: redact PII before it reaches retrieval, the LLM or the logs.
        redacted_q = redact_pii(question)
        clean_q = redacted_q.text

        # 2. Retrieve extra candidates so dropping flagged chunks still leaves k.
        candidates = self.retriever.retrieve(clean_q, k=k * 2, filters=filters)

        # 3. Context guardrail: drop retrieved chunks that look like prompt-injection attempts.
        flagged: list[dict[str, Any]] = []
        safe: list[ScoredChunk] = []
        for sc in candidates:
            hits = detect_prompt_injection(sc.chunk.text)
            if hits:
                flagged.append({"chunk_id": sc.chunk.chunk_id, "doc_id": sc.chunk.doc_id, "rules": hits})
            else:
                safe.append(sc)
        context = safe[:k]
        if flagged:
            logger.warning("prompt_injection_filtered", extra={"flagged_chunks": [f["chunk_id"] for f in flagged]})

        guardrails: dict[str, Any] = {
            "input_pii_redactions": redacted_q.counts,
            "flagged_chunks": flagged,
            "output_pii_redactions": {},
        }
        best = max((sc.relevance for sc in context), default=0.0)
        guardrails["best_relevance"] = round(best, 4)

        # 4. Relevance guardrail: refuse rather than answer from weak evidence.
        if not context or best < self.settings.min_relevance_score:
            return self._refusal(clean_q, context, guardrails, start, "insufficient_context")

        # 5. Generate and enforce citations (an uncited answer is treated as ungrounded).
        result = self.llm.generate(clean_q, context)
        if not result.answer or not result.cited_indices or result.answer.strip() == REFUSAL_MESSAGE:
            return self._refusal(clean_q, context, guardrails, start, "no_grounded_answer")

        # 6. Output guardrail: redact PII in the answer and in the snippets we return.
        redacted_a = redact_pii(result.answer)
        guardrails["output_pii_redactions"] = redacted_a.counts
        citations = [self._citation(i, context[i - 1]) for i in result.cited_indices]

        latency = (time.perf_counter() - start) * 1000
        logger.info(
            "query_answered",
            extra={"citations": [c.chunk_id for c in citations], "latency_ms": round(latency, 2)},
        )
        return QueryResult(
            question=clean_q,
            answer=redacted_a.text,
            citations=citations,
            refused=False,
            refusal_reason=None,
            retrieved=context,
            guardrails=guardrails,
            latency_ms=latency,
        )

    @staticmethod
    def _citation(index: int, sc: ScoredChunk) -> Citation:
        body = sc.chunk.text.split("\n", 1)[-1]
        snippet = redact_pii(body[:240] + ("..." if len(body) > 240 else "")).text
        return Citation(
            index=index,
            chunk_id=sc.chunk.chunk_id,
            doc_id=sc.chunk.doc_id,
            section=sc.chunk.section,
            source=sc.chunk.source,
            score=round(sc.score, 5),
            snippet=snippet,
        )

    def _refusal(
        self, question: str, context: list[ScoredChunk], guardrails: dict[str, Any], start: float, reason: str
    ) -> QueryResult:
        latency = (time.perf_counter() - start) * 1000
        logger.info("query_refused", extra={"reason": reason, "latency_ms": round(latency, 2)})
        return QueryResult(
            question=question,
            answer=REFUSAL_MESSAGE,
            citations=[],
            refused=True,
            refusal_reason=reason,
            retrieved=context,
            guardrails=guardrails,
            latency_ms=latency,
        )

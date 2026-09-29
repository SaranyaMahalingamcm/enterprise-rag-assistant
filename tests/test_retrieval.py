from __future__ import annotations

import numpy as np
import pytest

from rag_assistant.embeddings import HashingTfidfEmbedder
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.retrieval import reciprocal_rank_fusion
from rag_assistant.types import Chunk
from rag_assistant.vectorstore import InMemoryVectorStore


def test_rrf_rewards_agreement_between_rankers() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["a"] == fused["b"]
    assert fused["a"] > fused["c"] > 0
    assert fused["c"] == fused["d"]


def test_embedder_is_deterministic_and_normalised() -> None:
    emb = HashingTfidfEmbedder()
    emb.fit(["calibrate the robot", "clean the lens"])
    v1, v2 = emb.embed_query("calibration"), emb.embed_query("calibration")
    assert np.allclose(v1, v2)
    assert np.linalg.norm(v1) == pytest.approx(1.0, abs=1e-5)
    docs = emb.embed_documents(["calibrate the robot", "clean the lens"])
    assert docs[0] @ v1 > docs[1] @ v1  # char n-grams match morphological variants


def test_vector_store_orders_by_cosine_and_filters() -> None:
    store = InMemoryVectorStore()
    chunks = [Chunk(f"d{i}#0", f"d{i}", "t", "s", "src", {"doc_type": "sop" if i % 2 else "manual"}) for i in range(3)]
    vecs = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    store.add(chunks, vecs)
    q = np.array([1.0, 0.0], dtype=np.float32)
    assert [c.doc_id for c, _ in store.search(q, k=3)] == ["d0", "d1", "d2"]
    assert [c.doc_id for c, _ in store.search(q, k=3, filters={"doc_type": "sop"})] == ["d1"]


def test_hybrid_finds_exact_alarm_code(pipeline: RAGPipeline) -> None:
    # Short codes are where lexical search shines; fusion must keep that result on top.
    results = pipeline.retriever.retrieve("What does alarm R17 mean?", k=3)
    assert results[0].chunk.doc_id == "alarm-reference"
    assert results[0].bm25_score > 0


def test_hybrid_retrieves_paraphrased_question(pipeline: RAGPipeline) -> None:
    results = pipeline.retriever.retrieve("chamber takes forever to pump down, what should I check?", k=3)
    assert {"vc90-manual", "ts-003-vacuum-leaks"} & {r.chunk.doc_id for r in results}


def test_metadata_filter_restricts_results(pipeline: RAGPipeline) -> None:
    results = pipeline.retriever.retrieve("lens cleaning isopropyl alcohol", k=5, filters={"doc_type": "sop"})
    assert results
    assert all(r.chunk.metadata["doc_type"] == "sop" for r in results)

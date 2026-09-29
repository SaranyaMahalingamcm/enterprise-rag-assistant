# Extending the assistant

This page describes the extension points. Only the in-memory vector store, the offline embedder and
the extractive generator are exercised by the test suite; the Bedrock and OpenAI adapters are tested
against fake SDK clients, not live services. The pgvector and OpenSearch sections below are design
notes: **no such backend is implemented in this repository.**

## Interfaces

| Concern | Protocol | Default implementation | Other implementations |
|---|---|---|---|
| Embeddings | `rag_assistant.embeddings.base.Embedder` | `HashingTfidfEmbedder` (offline) | `BedrockTitanEmbedder`, `OpenAIEmbedder` (OpenAI and Azure OpenAI) |
| Vector search | `rag_assistant.vectorstore.VectorStore` | `InMemoryVectorStore` (NumPy, exact cosine) | none (see below) |
| Generation | `rag_assistant.llm.base.LLMProvider` | `ExtractiveLLM` (offline, deterministic) | `BedrockConverseLLM`, `OpenAIChatLLM` |

Providers are chosen by `RAG_EMBEDDING_PROVIDER` / `RAG_LLM_PROVIDER` in `rag_assistant/providers.py`.
Any object that satisfies a protocol can also be injected directly:

```python
from rag_assistant.pipeline import RAGPipeline

pipeline = RAGPipeline(embedder=my_embedder, llm=my_llm, vector_store=my_store)
```

## Vector store contract

```python
class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, query_vector: np.ndarray, k: int, filters: dict | None = None) -> list[tuple[Chunk, float]]: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...
```

`search` must return chunks ordered by descending similarity, apply `filters` as exact-match metadata
constraints (a list value means "any of"), and return cosine similarity (vectors are L2-normalised).

### How a pgvector backend would plug in

1. Table: `chunks(chunk_id text primary key, doc_id text, section text, source text, text text,
   metadata jsonb, embedding vector(<dim>))`, with an HNSW index using `vector_cosine_ops` and a GIN
   index on `metadata`.
2. `add` performs a batched `INSERT ... ON CONFLICT (chunk_id) DO UPDATE`.
3. `search` runs `SELECT ..., 1 - (embedding <=> %s) AS score FROM chunks WHERE metadata @> %s
   ORDER BY embedding <=> %s LIMIT %s`, so filtering happens inside the database.
4. The BM25 side could stay in-process for small corpora, or move to Postgres full-text search
   (`tsvector` + `ts_rank_cd`) so both rankings come from one store before fusion.

### How an OpenSearch backend would plug in

1. Index mapping with a `knn_vector` field (HNSW, cosine) plus keyword fields for metadata.
2. `search` issues a `knn` query with a `filter` clause built from `filters`.
3. OpenSearch can also serve the lexical side (native BM25), and its hybrid search pipeline with a
   normalization processor is an alternative to fusing results in application code.

In both cases the local `HashingTfidfEmbedder` should be replaced with a hosted embedding model,
because it has to be re-fitted on the whole corpus whenever documents change.

## Adding an LLM provider

Implement `generate(question, chunks) -> GenerationResult`. Build the prompt with
`rag_assistant.llm.base.build_user_prompt` and `SYSTEM_PROMPT`, then extract citations with
`parse_citations`. The pipeline rejects answers without at least one valid citation, so a provider
cannot bypass the grounding requirement.

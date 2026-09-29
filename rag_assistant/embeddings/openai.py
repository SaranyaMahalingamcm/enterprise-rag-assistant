"""OpenAI / Azure OpenAI embeddings. Requires ``pip install openai``."""

from __future__ import annotations

from typing import Any

import numpy as np

from rag_assistant.embeddings.base import l2_normalize


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str, client: Any, batch_size: int = 64) -> None:
        self._client = client
        self.model = model  # for Azure this is the deployment name
        self.batch_size = batch_size

    def fit(self, corpus: list[str]) -> None:
        return None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            response = self._client.embeddings.create(model=self.model, input=texts[i : i + self.batch_size])
            vectors.extend(item.embedding for item in response.data)
        return l2_normalize(np.array(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

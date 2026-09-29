"""Amazon Bedrock embeddings (Titan Text Embeddings V2). Requires ``pip install boto3``.

Credentials are resolved by the standard AWS credential chain (env vars, profile, IAM role);
this project never reads or stores AWS keys itself.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from rag_assistant.embeddings.base import l2_normalize


class BedrockTitanEmbedder:
    name = "bedrock-titan"

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        dimensions: int = 512,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError("Install the 'bedrock' extra: pip install '.[bedrock]'") from exc
            client = boto3.client("bedrock-runtime", region_name=region)
        self._client = client
        self.model_id = model_id
        self.dimensions = dimensions

    def fit(self, corpus: list[str]) -> None:  # hosted model: nothing to fit
        return None

    def _embed_one(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text, "dimensions": self.dimensions, "normalize": True})
        response = self._client.invoke_model(
            modelId=self.model_id, body=body, contentType="application/json", accept="application/json"
        )
        payload = json.loads(response["body"].read())
        return payload["embedding"]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        # Titan V2 embeds one input per call; batch jobs would use Bedrock batch inference instead.
        return l2_normalize(np.array([self._embed_one(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(np.array(self._embed_one(text)))[0]

"""Amazon Bedrock generation via the Converse API. Requires ``pip install boto3``."""

from __future__ import annotations

from typing import Any

from rag_assistant.llm.base import SYSTEM_PROMPT, build_user_prompt, parse_citations
from rag_assistant.types import GenerationResult, ScoredChunk


class BedrockConverseLLM:
    name = "bedrock"

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        max_tokens: int = 512,
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
        self.max_tokens = max_tokens

    def generate(self, question: str, chunks: list[ScoredChunk]) -> GenerationResult:
        response = self._client.converse(
            modelId=self.model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": build_user_prompt(question, chunks)}]}],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.0},
        )
        blocks = response["output"]["message"]["content"]
        answer = "".join(b.get("text", "") for b in blocks).strip()
        return GenerationResult(answer=answer, cited_indices=parse_citations(answer, len(chunks)))

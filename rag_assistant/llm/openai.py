"""OpenAI / Azure OpenAI chat generation. Requires ``pip install openai``."""

from __future__ import annotations

from typing import Any

from rag_assistant.llm.base import SYSTEM_PROMPT, build_user_prompt, parse_citations
from rag_assistant.types import GenerationResult, ScoredChunk


class OpenAIChatLLM:
    name = "openai"

    def __init__(self, model: str, client: Any, max_tokens: int = 512) -> None:
        self._client = client
        self.model = model  # for Azure this is the deployment name
        self.max_tokens = max_tokens

    def generate(self, question: str, chunks: list[ScoredChunk]) -> GenerationResult:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, chunks)},
            ],
        )
        answer = (response.choices[0].message.content or "").strip()
        return GenerationResult(answer=answer, cited_indices=parse_citations(answer, len(chunks)))

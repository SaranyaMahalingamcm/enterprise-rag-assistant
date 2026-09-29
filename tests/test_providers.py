"""Hosted-provider adapters tested offline against fake SDK clients."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any

import pytest

from rag_assistant.config import Settings
from rag_assistant.embeddings.bedrock import BedrockTitanEmbedder
from rag_assistant.embeddings.openai import OpenAIEmbedder
from rag_assistant.llm.base import parse_citations
from rag_assistant.llm.bedrock import BedrockConverseLLM
from rag_assistant.llm.openai import OpenAIChatLLM
from rag_assistant.providers import ProviderConfigError, build_embedder, build_llm
from rag_assistant.types import Chunk, ScoredChunk

CHUNKS = [ScoredChunk(Chunk("d#0", "d", "Doc - A\nThe limit is 5 mm.", "A", "d.md"), score=0.1)]


class FakeBedrock:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"output": {"message": {"content": [{"text": "The limit is 5 mm [1]. See also [7]."}]}}}

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"body": io.BytesIO(json.dumps({"embedding": [3.0, 4.0]}).encode())}


def test_bedrock_converse_request_and_citation_parsing() -> None:
    fake = FakeBedrock()
    result = BedrockConverseLLM("some-model", client=fake).generate("What is the limit?", CHUNKS)
    call = fake.calls[0]
    assert call["modelId"] == "some-model"
    assert call["inferenceConfig"]["temperature"] == 0.0
    assert "[1] (doc: d, section: A)" in call["messages"][0]["content"][0]["text"]
    assert "untrusted" in call["system"][0]["text"]
    assert result.cited_indices == [1]  # [7] is out of range and ignored


def test_bedrock_titan_embedder_normalises() -> None:
    vec = BedrockTitanEmbedder(client=FakeBedrock()).embed_query("hello")
    assert vec.tolist() == pytest.approx([0.6, 0.8])


def test_openai_adapters_with_fake_client() -> None:
    fake = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="5 mm [1]"))]
                )
            )
        ),
        embeddings=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in kw["input"]])
        ),
    )
    assert OpenAIChatLLM("m", fake).generate("q", CHUNKS).cited_indices == [1]
    assert OpenAIEmbedder("e", fake).embed_documents(["a", "b"]).shape == (2, 2)


def test_parse_citations_dedupes_and_orders() -> None:
    assert parse_citations("x [2] y [1] z [2] [0] [9]", n_chunks=3) == [2, 1]


def test_default_providers_are_offline_and_hosted_need_config() -> None:
    s = Settings(_env_file=None)
    assert build_llm(s).name == "extractive"
    assert build_embedder(s).name == "local-hashing-tfidf"
    with pytest.raises(ProviderConfigError):
        build_llm(Settings(_env_file=None, llm_provider="openai"))  # no API key configured
    with pytest.raises(ProviderConfigError):
        build_embedder(Settings(_env_file=None, embedding_provider="azure_openai"))

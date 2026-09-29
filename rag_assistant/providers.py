"""Factories that build embedding and LLM providers from settings.

Hosted providers are imported lazily, so the default offline configuration never needs boto3 or
the openai SDK installed.
"""

from __future__ import annotations

from typing import Any

from rag_assistant.config import Settings
from rag_assistant.embeddings.base import Embedder
from rag_assistant.embeddings.local import HashingTfidfEmbedder
from rag_assistant.llm.base import LLMProvider
from rag_assistant.llm.extractive import ExtractiveLLM


class ProviderConfigError(RuntimeError):
    """Raised when a hosted provider is selected but not configured."""


def _openai_client(settings: Settings, azure: bool) -> Any:
    if settings.openai_api_key is None:
        raise ProviderConfigError("RAG_OPENAI_API_KEY must be set for OpenAI/Azure OpenAI providers")
    if azure and not settings.azure_openai_endpoint:
        raise ProviderConfigError("RAG_AZURE_OPENAI_ENDPOINT must be set for Azure OpenAI")
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ProviderConfigError("Install the 'openai' extra: pip install '.[openai]'") from exc
    key = settings.openai_api_key.get_secret_value()
    if azure:
        return openai.AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=key,
            api_version=settings.azure_openai_api_version,
        )
    return openai.OpenAI(api_key=key)


def build_embedder(settings: Settings) -> Embedder:
    name = settings.embedding_provider
    if name == "local":
        return HashingTfidfEmbedder()
    if name == "bedrock":
        from rag_assistant.embeddings.bedrock import BedrockTitanEmbedder

        return BedrockTitanEmbedder(settings.bedrock_embedding_model_id, settings.aws_region)
    from rag_assistant.embeddings.openai import OpenAIEmbedder

    return OpenAIEmbedder(settings.openai_embedding_model, _openai_client(settings, name == "azure_openai"))


def build_llm(settings: Settings) -> LLMProvider:
    name = settings.llm_provider
    if name == "extractive":
        return ExtractiveLLM(max_sentences=settings.max_answer_sentences)
    if name == "bedrock":
        from rag_assistant.llm.bedrock import BedrockConverseLLM

        return BedrockConverseLLM(settings.bedrock_llm_model_id, settings.aws_region)
    from rag_assistant.llm.openai import OpenAIChatLLM

    return OpenAIChatLLM(settings.openai_model, _openai_client(settings, name == "azure_openai"))

"""Application configuration loaded from environment variables (prefix ``RAG_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EmbeddingProviderName = Literal["local", "bedrock", "openai", "azure_openai"]
LLMProviderName = Literal["extractive", "bedrock", "openai", "azure_openai"]


class Settings(BaseSettings):
    """All runtime settings. Defaults run fully offline with no credentials."""

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- General ---------------------------------------------------------------
    app_name: str = "enterprise-rag-assistant"
    log_level: str = "INFO"
    docs_dir: Path = PROJECT_ROOT / "data" / "sample_docs"
    auto_ingest: bool = Field(True, description="Ingest docs_dir when the API starts.")
    api_key: SecretStr | None = Field(
        None, description="If set, /ingest and /query require a matching X-API-Key header."
    )

    # --- Chunking --------------------------------------------------------------
    chunk_size: int = Field(900, ge=100, description="Max characters per chunk.")
    chunk_overlap: int = Field(150, ge=0, description="Character overlap between split chunks.")

    # --- Retrieval -------------------------------------------------------------
    top_k: int = Field(4, ge=1, le=20)
    candidate_k: int = Field(20, ge=1, description="Candidates taken from each retriever before fusion.")
    rrf_k: int = Field(60, ge=1, description="Reciprocal rank fusion constant.")
    min_relevance_score: float = Field(
        0.25, ge=0.0, le=1.0, description="Refuse to answer if the best chunk's relevance is below this."
    )

    # --- Providers -------------------------------------------------------------
    embedding_provider: EmbeddingProviderName = "local"
    llm_provider: LLMProviderName = "extractive"
    max_answer_sentences: int = 3

    # Amazon Bedrock (credentials come from the standard AWS credential chain, never from here)
    aws_region: str = "us-east-1"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    bedrock_llm_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"

    # OpenAI / Azure OpenAI
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-06-01"


@lru_cache
def get_settings() -> Settings:
    return Settings()

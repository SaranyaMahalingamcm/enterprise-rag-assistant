"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    version: str
    documents: int
    chunks: int
    embedding_provider: str
    llm_provider: str


class DocumentIn(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    title: str = Field(..., min_length=1, max_length=256)
    content: str = Field(..., min_length=1, max_length=200_000, description="Markdown or plain text.")
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(..., min_length=1, max_length=100)
    replace: bool = Field(False, description="Replace the whole corpus instead of upserting.")


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_created: int
    total_documents: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)
    filters: dict[str, str | list[str]] | None = Field(
        None, description='Exact-match metadata filters, e.g. {"doc_type": "sop"}.'
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return v


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    section: str
    source: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[CitationOut]
    refused: bool
    refusal_reason: str | None = None
    guardrails: dict[str, Any]
    latency_ms: float


class ErrorResponse(BaseModel):
    error: str
    detail: Any = None
    request_id: str

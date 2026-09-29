"""FastAPI application: /health, /ingest, /query."""

from __future__ import annotations

import logging
import re
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_assistant import __version__
from rag_assistant.api.schemas import (
    CitationOut,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from rag_assistant.config import Settings, get_settings
from rag_assistant.logging_config import configure_logging, request_id_var
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.types import Document

logger = logging.getLogger("rag_assistant.api")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def create_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = pipeline or RAGPipeline(settings)
        if settings.auto_ingest and app.state.pipeline.chunk_count == 0:
            app.state.pipeline.ingest_directory(settings.docs_dir)
        yield

    app = FastAPI(
        title="Enterprise RAG Assistant",
        version=__version__,
        description="Technical knowledge assistant over synthetic maintenance manuals and SOPs.",
        lifespan=lifespan,
    )

    # ---------------------------------------------------------------- middleware
    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception as exc:
                # Log the full traceback server-side; never leak internals to the client.
                logger.exception("unhandled_error", extra={"error_type": type(exc).__name__})
                response = _error(500, "internal_error", "An unexpected error occurred.")
            duration = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "http_request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": duration,
                },
            )
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_var.reset(token)

    # ---------------------------------------------------------------- errors
    def _error(status: int, error: str, detail: object = None) -> JSONResponse:
        return JSONResponse(
            status_code=status, content={"error": error, "detail": detail, "request_id": request_id_var.get()}
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, "http_error", exc.detail)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return _error(422, "validation_error", detail)

    # ---------------------------------------------------------------- auth
    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key is None:
            return
        expected = settings.api_key.get_secret_value()
        if not x_api_key or not secrets.compare_digest(x_api_key, expected):
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")

    def get_pipeline(request: Request) -> RAGPipeline:
        return request.app.state.pipeline

    # ---------------------------------------------------------------- routes
    @app.get("/health", response_model=HealthResponse)
    def health(pipeline: RAGPipeline = Depends(get_pipeline)) -> HealthResponse:
        return HealthResponse(
            status="ok" if pipeline.chunk_count > 0 else "empty_index",
            version=__version__,
            documents=pipeline.document_count,
            chunks=pipeline.chunk_count,
            embedding_provider=pipeline.retriever.embedder.name,
            llm_provider=pipeline.llm.name,
        )

    @app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
    def ingest(body: IngestRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> IngestResponse:
        docs = [
            Document(doc_id=d.doc_id, title=d.title, text=d.content, source=f"api:{d.doc_id}", metadata=d.metadata)
            for d in body.documents
        ]
        created = pipeline.ingest_documents(docs, replace=body.replace)
        return IngestResponse(
            documents_ingested=len(docs),
            chunks_created=created,
            total_documents=pipeline.document_count,
            total_chunks=pipeline.chunk_count,
        )

    @app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
    def query(body: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> QueryResponse:
        if pipeline.chunk_count == 0:
            raise HTTPException(status_code=409, detail="Index is empty; call /ingest first")
        result = pipeline.query(body.question, top_k=body.top_k, filters=body.filters)
        return QueryResponse(
            request_id=request_id_var.get(),
            answer=result.answer,
            citations=[CitationOut(**asdict(c)) for c in result.citations],
            refused=result.refused,
            refusal_reason=result.refusal_reason,
            guardrails=result.guardrails,
            latency_ms=round(result.latency_ms, 2),
        )

    return app


app = create_app()

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api.app import create_app
from rag_assistant.config import Settings
from rag_assistant.pipeline import RAGPipeline


@pytest.fixture(scope="session")
def settings() -> Settings:
    # _env_file=None: tests must not depend on a developer's local .env
    return Settings(_env_file=None, log_level="WARNING")


@pytest.fixture(scope="session")
def pipeline(settings: Settings) -> RAGPipeline:
    p = RAGPipeline(settings)
    p.ingest_directory()
    return p


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as c:
        yield c

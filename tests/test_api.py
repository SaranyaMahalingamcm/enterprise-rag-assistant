from __future__ import annotations

from fastapi.testclient import TestClient

from rag_assistant.api.app import create_app
from rag_assistant.config import Settings


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] >= 10 and body["chunks"] > body["documents"]
    assert body["llm_provider"] == "extractive"


def test_query_returns_answer_citations_and_request_id(client: TestClient) -> None:
    resp = client.post(
        "/query", json={"question": "Who is allowed to remove a lockout lock?"}, headers={"X-Request-ID": "abc-123"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert resp.headers["X-Request-ID"] == "abc-123" == body["request_id"]
    assert not body["refused"]
    assert body["citations"][0]["doc_id"] == "sop-001-loto"
    assert {"input_pii_redactions", "flagged_chunks", "output_pii_redactions"} <= body["guardrails"].keys()


def test_ingest_then_query_new_document(client: TestClient) -> None:
    doc = {
        "doc_id": "sop-099-test",
        "title": "SOP-099 Compressor Drain",
        "content": "## Drain Schedule\n\nDrain the air compressor condensate valve every Friday afternoon.",
        "metadata": {"doc_type": "sop"},
    }
    resp = client.post("/ingest", json={"documents": [doc]})
    assert resp.status_code == 200 and resp.json()["chunks_created"] == 1
    body = client.post(
        "/query", json={"question": "When should the compressor condensate be drained?", "filters": {"doc_type": "sop"}}
    ).json()
    assert body["citations"][0]["doc_id"] == "sop-099-test"
    assert "Friday" in body["answer"]


def test_validation_error_is_structured(client: TestClient) -> None:
    resp = client.post("/query", json={"question": "  ", "top_k": 99})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "validation_error" and body["request_id"]
    assert client.get("/nope").json()["error"] == "http_error"


def test_api_key_enforced_when_configured() -> None:
    settings = Settings(_env_file=None, api_key="unit-test-key", log_level="WARNING")  # pragma: allowlist secret
    with TestClient(create_app(settings)) as c:
        assert c.post("/query", json={"question": "lens cleaning"}).status_code == 401
        ok = c.post("/query", json={"question": "lens cleaning"}, headers={"X-API-Key": "unit-test-key"})
        assert ok.status_code == 200
        assert c.get("/health").status_code == 200  # health stays open for probes

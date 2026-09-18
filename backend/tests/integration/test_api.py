"""The app through its HTTP surface, with fake embeddings and a stub model."""
from __future__ import annotations

import io


def _upload(client, name: str, body: str, tenant: str | None = None):
    headers = {"X-Tenant-Id": tenant} if tenant else {}
    return client.post(
        "/api/upload",
        files={"file": (name, io.BytesIO(body.encode()), "text/plain")},
        headers=headers,
    )


def test_health_reports_the_running_configuration(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    for field in ("hybrid_retrieval", "rerank", "chunking", "llm_status"):
        assert field in body


def test_upload_then_query_cites_the_document(client):
    response = _upload(
        client,
        "policy.txt",
        "The reimbursement limit for co-working desks is 250 dollars per month.",
        tenant="alice",
    )
    assert response.status_code == 200
    assert response.json()["documents_ingested"] == 1

    answer = client.post(
        "/api/query",
        json={"question": "What is the co-working reimbursement limit?"},
        headers={"X-Tenant-Id": "alice"},
    )
    assert answer.status_code == 200
    body = answer.json()
    assert body["sources"], "an indexed document should produce sources"
    assert body["sources"][0]["source"] == "policy.txt"


def test_query_without_documents_refuses(client):
    body = client.post("/api/query", json={"question": "anything at all"}).json()
    assert body["sources"] == []
    assert any(g.startswith("grounding") for g in body["guardrails"])


def test_empty_question_is_rejected(client):
    assert client.post("/api/query", json={"question": "   "}).status_code == 400


def test_unsupported_file_type_is_rejected(client):
    response = client.post(
        "/api/upload",
        files={"file": ("notes.xyz", io.BytesIO(b"data"), "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_empty_file_is_rejected(client):
    response = client.post(
        "/api/upload", files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    )
    assert response.status_code == 400


def test_reupload_replaces_rather_than_duplicating(client):
    _upload(client, "report.txt", "First revision of the report.", tenant="alice")
    _upload(client, "report.txt", "Second revision of the report.", tenant="alice")

    listed = client.get("/api/documents", headers={"X-Tenant-Id": "alice"}).json()
    names = [d["filename"] for d in listed["documents"]]
    assert names.count("report.txt") == 1

"""Spec 004: a visitor sees their own documents and the public samples only."""
from __future__ import annotations

import io

from app.config import get_settings
from app.rag.ingest import ingest_file


def _upload(client, name, body, tenant):
    return client.post(
        "/api/upload",
        files={"file": (name, io.BytesIO(body.encode()), "text/plain")},
        headers={"X-Tenant-Id": tenant},
    )


def test_documents_are_not_visible_to_another_tenant(client):
    _upload(client, "alice-secret.txt", "Alice offer letter: salary 200000.", "alice")

    bob = client.get("/api/documents", headers={"X-Tenant-Id": "bob"}).json()
    assert "alice-secret.txt" not in [d["filename"] for d in bob["documents"]]

    alice = client.get("/api/documents", headers={"X-Tenant-Id": "alice"}).json()
    assert "alice-secret.txt" in [d["filename"] for d in alice["documents"]]


def test_another_tenant_cannot_retrieve_the_content(client):
    _upload(client, "alice-secret.txt", "Alice offer letter: salary 200000.", "alice")

    body = client.post(
        "/api/query",
        json={"question": "What is the salary in the offer letter?"},
        headers={"X-Tenant-Id": "bob"},
    ).json()
    assert all(s["source"] != "alice-secret.txt" for s in body["sources"])


def test_public_samples_are_readable_by_everyone(client, tmp_path):
    sample = tmp_path / "handbook.txt"
    sample.write_text("Core hours run from 10:00 to 15:00 local time.")
    ingest_file(str(sample), "handbook.txt", get_settings().public_tenant)

    for tenant in ("alice", "bob"):
        listed = client.get("/api/documents", headers={"X-Tenant-Id": tenant}).json()
        row = next(d for d in listed["documents"] if d["filename"] == "handbook.txt")
        assert row["shared"] is True


def test_a_caller_cannot_overwrite_the_public_samples(client, tmp_path):
    sample = tmp_path / "handbook.txt"
    sample.write_text("Core hours run from 10:00 to 15:00 local time.")
    ingest_file(str(sample), "handbook.txt", get_settings().public_tenant)

    # Claiming the reserved tenant must not replace the shared copy.
    _upload(client, "handbook.txt", "Core hours are whatever I say.", "public")

    listed = client.get("/api/documents", headers={"X-Tenant-Id": "public"}).json()
    rows = [d for d in listed["documents"] if d["filename"] == "handbook.txt"]
    assert any(r["shared"] for r in rows), "the public sample must survive"


def test_missing_tenant_header_still_sees_samples(client, tmp_path):
    sample = tmp_path / "handbook.txt"
    sample.write_text("Core hours run from 10:00 to 15:00 local time.")
    ingest_file(str(sample), "handbook.txt", get_settings().public_tenant)

    listed = client.get("/api/documents").json()
    assert "handbook.txt" in [d["filename"] for d in listed["documents"]]

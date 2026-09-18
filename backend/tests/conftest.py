"""Test fixtures.

Everything here runs offline: the Prompt Guard classifier is disabled so the
guardrails fall back to their regex heuristics, and each test that touches the
vector store gets its own directories.
"""
from __future__ import annotations

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def offline_settings(monkeypatch, tmp_path):
    """Isolate persistence and keep every test off the network."""
    monkeypatch.setenv("GUARD_ENABLED", "false")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("DOCSTORE_DIR", str(tmp_path / "docstore"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    get_settings.cache_clear()

    from app.rag import guardrails

    guardrails._guard_client.cache_clear()
    yield
    get_settings.cache_clear()
    guardrails._guard_client.cache_clear()


@pytest.fixture
def make_pdf(tmp_path):
    """Build a small PDF on the fly so tests carry no binary fixtures."""
    import pymupdf

    def _make(pages: list[str], name: str = "sample.pdf") -> str:
        doc = pymupdf.open()
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_text((72, 100), text, fontsize=11)
        path = tmp_path / name
        doc.save(str(path))
        doc.close()
        return str(path)

    return _make

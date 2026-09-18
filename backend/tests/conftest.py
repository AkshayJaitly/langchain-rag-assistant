"""Test fixtures.

Everything here runs offline: the Prompt Guard classifier is disabled so the
guardrails fall back to their regex heuristics, and each test that touches the
vector store gets its own directories.
"""
from __future__ import annotations

import math
import re

import pytest
from fastapi import FastAPI
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

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

    from app.rag import guardrails, retrieval
    from app.rag import vectorstore as vectorstore_module

    guardrails._guard_client.cache_clear()
    vectorstore_module.get_retriever.cache_clear()
    retrieval.invalidate_caches()
    yield
    get_settings.cache_clear()
    guardrails._guard_client.cache_clear()
    vectorstore_module.get_retriever.cache_clear()
    retrieval.invalidate_caches()


class FakeEmbeddings(Embeddings):
    """Deterministic hash-based embeddings (spec 007).

    A real model makes integration tests too slow to run on every change, and
    these tests are about wiring, not embedding quality. Same text always maps
    to the same vector, and shared words pull vectors together, which is enough
    for retrieval to behave sensibly.
    """

    dimensions = 64

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[hash(token) % self.dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


class FakeLLM:
    """Stub chat model: echoes the context so grounding checks pass."""

    def __init__(self, reply: str | None = None):
        self.reply = reply
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.reply is not None:
            return AIMessage(content=self.reply)
        human = messages[-1].content
        first = ""
        for line in human.splitlines():
            if line.startswith("[1]"):
                continue
            if line.strip() and not line.startswith(("Context:", "Question:")):
                first = line.strip()
                break
        return AIMessage(content=f"{first[:180]} [1]")


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Swap the embedding model out everywhere it is looked up."""
    from app.rag import embeddings as embeddings_module
    from app.rag import vectorstore as vectorstore_module

    fake = FakeEmbeddings()
    embeddings_module.get_embeddings.cache_clear()
    monkeypatch.setattr(embeddings_module, "get_embeddings", lambda: fake)
    monkeypatch.setattr(vectorstore_module, "get_embeddings", lambda: fake)
    # No cache_clear on teardown: monkeypatch has not been undone yet, so the
    # attribute is still the lambda and has no cache.
    yield fake


@pytest.fixture
def fake_llm(monkeypatch):
    from app.rag import graph as graph_module

    llm = FakeLLM()
    graph_module._get_llm.cache_clear()
    monkeypatch.setattr(graph_module, "_get_llm", lambda: llm)
    yield llm


@pytest.fixture
def client(fake_embeddings, fake_llm):
    """FastAPI TestClient with a fresh store and no network access."""
    from fastapi.testclient import TestClient

    from app.rag import vectorstore as vectorstore_module

    vectorstore_module.get_retriever.cache_clear()
    from app.api.routes import router

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as test_client:
        yield test_client
    vectorstore_module.get_retriever.cache_clear()


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

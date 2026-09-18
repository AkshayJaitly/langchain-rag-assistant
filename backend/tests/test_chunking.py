"""Spec 005: chunking strategy selection."""
from __future__ import annotations

import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.rag.vectorstore import _child_splitter


def test_recursive_is_the_default():
    assert get_settings().chunking_strategy == "recursive"
    assert isinstance(_child_splitter(get_settings()), RecursiveCharacterTextSplitter)


def test_semantic_strategy_selects_the_semantic_chunker(monkeypatch, fake_embeddings):
    monkeypatch.setenv("CHUNKING", "semantic")
    get_settings.cache_clear()
    splitter = _child_splitter(get_settings())
    assert type(splitter).__name__ == "SemanticChunker"


def test_unknown_strategy_fails_fast(monkeypatch):
    """AC-5: a typo must not silently fall back to the default."""
    monkeypatch.setenv("CHUNKING", "sematnic")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="Unknown CHUNKING"):
        get_settings().chunking_strategy


def test_gemini_provider_is_selectable(monkeypatch):
    """Spec: the provider abstraction must cover Gemini without code changes."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    get_settings.cache_clear()

    from app.rag.graph import _get_llm

    _get_llm.cache_clear()
    try:
        assert type(_get_llm()).__name__ == "ChatGoogleGenerativeAI"
    finally:
        _get_llm.cache_clear()


def test_unknown_provider_names_the_valid_options(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    get_settings.cache_clear()

    from app.rag.graph import _get_llm

    _get_llm.cache_clear()
    try:
        with pytest.raises(ValueError, match="gemini"):
            _get_llm()
    finally:
        _get_llm.cache_clear()

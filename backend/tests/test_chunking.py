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

"""Spec 002: reranking reorders, and fails open."""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.config import get_settings
from app.rag.rerank import _parse_order, rerank


class StubLLM:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error

    def invoke(self, messages):
        if self.error:
            raise self.error
        from langchain_core.messages import AIMessage

        return AIMessage(content=self.reply)


@pytest.fixture
def docs():
    return [Document(page_content=c) for c in ("alpha", "beta", "gamma")]


@pytest.fixture
def rerank_on(monkeypatch):
    monkeypatch.setenv("RERANK", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_disabled_by_default_leaves_order_untouched(docs):
    """AC-3: enabling reranking is a deliberate trade, not a silent default."""
    assert rerank("q", docs, StubLLM(reply="3,2,1")) == docs


def test_reorders_by_the_models_ranking(rerank_on, docs):
    result = rerank("q", docs, StubLLM(reply="3, 1, 2"))
    assert [d.page_content for d in result] == ["gamma", "alpha", "beta"]


def test_omitted_candidates_are_appended_not_dropped(rerank_on, docs):
    """AC-5: a partial ranking must not silently lose candidates."""
    result = rerank("q", docs, StubLLM(reply="2"))
    assert [d.page_content for d in result] == ["beta", "alpha", "gamma"]


def test_unparseable_output_keeps_the_fused_order(rerank_on, docs):
    assert rerank("q", docs, StubLLM(reply="no idea")) == docs


def test_model_failure_keeps_the_fused_order(rerank_on, docs):
    """AC-4: a query never fails because reranking did."""
    assert rerank("q", docs, StubLLM(error=RuntimeError("boom"))) == docs


def test_out_of_range_indices_are_ignored():
    assert _parse_order("9, 2", 3) == [1]

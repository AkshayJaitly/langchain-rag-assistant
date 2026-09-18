"""Spec 003: follow-up questions are condensed before retrieval."""
from __future__ import annotations

from langchain_core.messages import AIMessage

from app.rag.graph import condense_node


class StubLLM:
    def __init__(self, reply="rewritten question", error=None):
        self.reply = reply
        self.error = error
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        self.last = messages
        return AIMessage(content=self.reply)


def test_first_turn_is_not_condensed(monkeypatch):
    """AC-4: nothing to resolve, so no model call is spent."""
    from app.rag import graph as graph_module

    llm = StubLLM()
    monkeypatch.setattr(graph_module, "_get_llm", lambda: llm)

    result = condense_node({"question": "What is the uptime commitment?", "history": []})
    assert result["standalone_question"] == "What is the uptime commitment?"
    assert llm.calls == 0


def test_follow_up_is_rewritten_using_history(monkeypatch):
    from app.rag import graph as graph_module

    llm = StubLLM(reply="What is the service credit below 95% uptime?")
    monkeypatch.setattr(graph_module, "_get_llm", lambda: llm)

    result = condense_node(
        {
            "question": "and below that?",
            "history": [
                {"role": "user", "content": "What is the service credit at 97%?"},
                {"role": "assistant", "content": "25% of the monthly fee."},
            ],
        }
    )
    assert result["standalone_question"] == "What is the service credit below 95% uptime?"
    assert llm.calls == 1


def test_history_is_capped(monkeypatch):
    """AC-5: a long thread must not grow the prompt without bound."""
    from app.rag import graph as graph_module

    llm = StubLLM()
    monkeypatch.setattr(graph_module, "_get_llm", lambda: llm)

    history = [{"role": "user", "content": f"turn {i}"} for i in range(50)]
    condense_node({"question": "and then?", "history": history})

    transcript = llm.last[-1].content
    assert "turn 49" in transcript
    assert "turn 0" not in transcript


def test_condense_failure_falls_back_to_the_original(monkeypatch):
    """AC-6: a broken rewrite must not break the question."""
    from app.rag import graph as graph_module

    monkeypatch.setattr(
        graph_module, "_get_llm", lambda: StubLLM(error=RuntimeError("boom"))
    )
    result = condense_node(
        {"question": "and below that?", "history": [{"role": "user", "content": "x"}]}
    )
    assert result["standalone_question"] == "and below that?"


def test_empty_rewrite_falls_back_to_the_original(monkeypatch):
    from app.rag import graph as graph_module

    monkeypatch.setattr(graph_module, "_get_llm", lambda: StubLLM(reply="   "))
    result = condense_node(
        {"question": "and below that?", "history": [{"role": "user", "content": "x"}]}
    )
    assert result["standalone_question"] == "and below that?"

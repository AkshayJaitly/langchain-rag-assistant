"""LangGraph RAG pipeline with guardrails.

Flow:

    START
      -> input_guardrail  --(blocked)--> END
      -> retrieve         --(no docs)--> no_context -> END
      -> generate
      -> output_guardrail -> END

State is threaded through each node as a TypedDict.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.rag import guardrails
from app.rag.vectorstore import get_retriever

SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer the user's question using "
    "ONLY the information in the provided context. Follow these rules:\n"
    "- If the context does not contain the answer, say you don't know. Never "
    "invent facts or rely on outside knowledge.\n"
    "- Cite the sources you used with their bracketed numbers, e.g. [1], [2].\n"
    "- Be concise and factual."
)


class RAGState(TypedDict, total=False):
    question: str
    documents: list[Document]
    answer: str
    sources: list[dict[str, Any]]
    blocked: bool
    guardrails: list[str]


@lru_cache
def _get_llm() -> BaseChatModel:
    """Build the chat model for the configured provider.

    LLM_PROVIDER=anthropic -> Claude via the paid API (needs ANTHROPIC_API_KEY).
    LLM_PROVIDER=ollama     -> a local model served by Ollama (free, offline).
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        # Imported lazily so the Anthropic path doesn't require Ollama installed.
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            num_predict=settings.llm_max_tokens,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # No temperature/top_p: recent Claude models reject non-default sampling.
        return ChatAnthropic(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
        "Use 'anthropic' or 'ollama'."
    )


def _format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {d.metadata.get('source', 'unknown')})\n{d.page_content}"
        for i, d in enumerate(documents)
    )


def _sources_payload(documents: list[Document]) -> list[dict[str, Any]]:
    payload = []
    for i, d in enumerate(documents):
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        payload.append(
            {
                "index": i + 1,
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page"),
                "snippet": snippet,
            }
        )
    return payload


# --- Nodes ---------------------------------------------------------------


def input_guardrail_node(state: RAGState) -> RAGState:
    allowed, reason = guardrails.check_input(state["question"])
    triggered = list(state.get("guardrails", []))
    if not allowed:
        triggered.append(f"input:{reason}")
        return {
            "blocked": True,
            "answer": f"Your request was blocked by an input guardrail: {reason}",
            "documents": [],
            "sources": [],
            "guardrails": triggered,
        }
    return {"blocked": False, "guardrails": triggered}


def retrieve_node(state: RAGState) -> RAGState:
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    return {"documents": docs}


def no_context_node(state: RAGState) -> RAGState:
    triggered = list(state.get("guardrails", []))
    triggered.append("grounding:no_documents_retrieved")
    return {
        "answer": guardrails.REFUSAL_MESSAGE,
        "sources": [],
        "guardrails": triggered,
    }


def generate_node(state: RAGState) -> RAGState:
    documents = state["documents"]
    context = _format_context(documents)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion: {state['question']}"
        ),
    ]
    response = _get_llm().invoke(messages)
    answer = response.content if isinstance(response.content, str) else str(response.content)
    return {"answer": answer, "sources": _sources_payload(documents)}


def output_guardrail_node(state: RAGState) -> RAGState:
    triggered = list(state.get("guardrails", []))
    answer = state.get("answer", "")

    answer, redacted = guardrails.redact_secrets(answer)
    if redacted:
        triggered.append("output:redacted_secret")

    if not guardrails.is_grounded(answer, state.get("documents", [])):
        triggered.append("output:low_grounding")
        answer += (
            "\n\n⚠️ Note: this answer may not be fully supported by the "
            "retrieved context — please verify against the cited sources."
        )

    return {"answer": answer, "guardrails": triggered}


# --- Edges ---------------------------------------------------------------


def _after_input(state: RAGState) -> str:
    return "blocked" if state.get("blocked") else "retrieve"


def _after_retrieve(state: RAGState) -> str:
    return "generate" if state.get("documents") else "no_context"


@lru_cache
def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("no_context", no_context_node)
    graph.add_node("generate", generate_node)
    graph.add_node("output_guardrail", output_guardrail_node)

    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail", _after_input, {"blocked": END, "retrieve": "retrieve"}
    )
    graph.add_conditional_edges(
        "retrieve",
        _after_retrieve,
        {"generate": "generate", "no_context": "no_context"},
    )
    graph.add_edge("no_context", END)
    graph.add_edge("generate", "output_guardrail")
    graph.add_edge("output_guardrail", END)

    return graph.compile()


def answer_question(question: str) -> dict[str, Any]:
    """Run the full RAG graph and return a serializable result."""
    app = build_graph()
    final: RAGState = app.invoke(
        {"question": question, "guardrails": [], "documents": [], "sources": []}
    )
    return {
        "answer": final.get("answer", ""),
        "sources": final.get("sources", []),
        "guardrails": final.get("guardrails", []),
        "blocked": final.get("blocked", False),
    }

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

import re
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
    critique: str
    refine_count: int


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

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            max_tokens=settings.llm_max_tokens,
            temperature=0,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            max_tokens=settings.llm_max_tokens,
            temperature=0,
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
        "Use 'anthropic', 'openai', 'groq', or 'ollama'."
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
    human = f"Context:\n{context}\n\nQuestion: {state['question']}"
    critique = state.get("critique")
    if critique:
        human += (
            f"\n\nA reviewer flagged your previous draft: {critique}\n"
            "Revise so every claim is supported by the context above."
        )
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human)]
    response = _get_llm().invoke(messages)
    answer = response.content if isinstance(response.content, str) else str(response.content)
    return {"answer": answer, "sources": _sources_payload(documents)}


# --- Multi-agent nodes (grader + verifier) -------------------------------

GRADER_PROMPT = (
    "You are a relevance-grading agent. Given a question and a numbered list of "
    "documents, return a comma-separated list of the document numbers that "
    "contain information useful for answering the question. Return exactly "
    "'NONE' if none are relevant. Output only the numbers or NONE — no prose."
)

VERIFIER_PROMPT = (
    "You are a verification agent. Given the context and a drafted answer, "
    "check whether every claim in the answer is supported by the context. "
    "Reply with exactly 'GROUNDED' if fully supported. Otherwise reply "
    "'UNSUPPORTED: <one sentence on what is missing or unsupported>'."
)


def grade_documents_node(state: RAGState) -> RAGState:
    documents = state["documents"]
    triggered = list(state.get("guardrails", []))
    listing = "\n\n".join(
        f"[{i + 1}] {d.page_content[:1200]}" for i, d in enumerate(documents)
    )
    resp = _get_llm().invoke(
        [
            SystemMessage(content=GRADER_PROMPT),
            HumanMessage(content=f"Question: {state['question']}\n\n{listing}"),
        ]
    )
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    if "none" in text.lower():
        kept: list[Document] = []
    else:
        idxs = {int(n) for n in re.findall(r"\d+", text)}
        kept = [d for i, d in enumerate(documents) if (i + 1) in idxs]
        if not kept:  # grader returned nothing parseable -> keep all (fail open)
            kept = documents
    triggered.append(f"agent:graded {len(documents)}->{len(kept)} docs")
    return {"documents": kept, "guardrails": triggered}


def verify_node(state: RAGState) -> RAGState:
    triggered = list(state.get("guardrails", []))
    context = _format_context(state["documents"])
    resp = _get_llm().invoke(
        [
            SystemMessage(content=VERIFIER_PROMPT),
            HumanMessage(
                content=f"Context:\n{context}\n\nDrafted answer:\n{state.get('answer', '')}"
            ),
        ]
    )
    verdict = (resp.content if isinstance(resp.content, str) else str(resp.content)).strip()
    if verdict.upper().startswith("GROUNDED"):
        triggered.append("agent:verified grounded")
        return {"guardrails": triggered, "critique": ""}
    triggered.append("agent:verifier requested revision")
    return {
        "guardrails": triggered,
        "critique": verdict,
        "refine_count": state.get("refine_count", 0) + 1,
    }


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


def _after_grade(state: RAGState) -> str:
    return "generate" if state.get("documents") else "no_context"


def _after_verify(state: RAGState) -> str:
    # Allow at most one corrective revision.
    if state.get("critique") and state.get("refine_count", 0) < 2:
        return "generate"
    return "output_guardrail"


@lru_cache
def build_multi_agent_graph():
    """Corrective multi-agent RAG: grader -> generator -> verifier (-> refine)."""
    graph = StateGraph(RAGState)
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("no_context", no_context_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("output_guardrail", output_guardrail_node)

    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail", _after_input, {"blocked": END, "retrieve": "retrieve"}
    )
    graph.add_conditional_edges(
        "retrieve",
        _after_retrieve,
        {"generate": "grade_documents", "no_context": "no_context"},
    )
    graph.add_conditional_edges(
        "grade_documents",
        _after_grade,
        {"generate": "generate", "no_context": "no_context"},
    )
    graph.add_edge("no_context", END)
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        _after_verify,
        {"generate": "generate", "output_guardrail": "output_guardrail"},
    )
    graph.add_edge("output_guardrail", END)

    return graph.compile()


def answer_question(question: str) -> dict[str, Any]:
    """Run the configured RAG graph and return a serializable result."""
    settings = get_settings()
    app = (
        build_multi_agent_graph()
        if settings.pipeline.lower() == "multi_agent"
        else build_graph()
    )
    final: RAGState = app.invoke(
        {
            "question": question,
            "guardrails": [],
            "documents": [],
            "sources": [],
            "refine_count": 0,
        }
    )
    return {
        "answer": final.get("answer", ""),
        "sources": final.get("sources", []),
        "guardrails": final.get("guardrails", []),
        "blocked": final.get("blocked", False),
    }

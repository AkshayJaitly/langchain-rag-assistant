"""LLM reranking of fused candidates (spec 002).

The standard answer here is a cross-encoder, and it is the right answer when
there is memory for it. There is not: sentence-transformers pulls in torch, and
this deployment already OOM-killed itself on ingestion alone inside 512 MB. So
the reranker is the chat model that is already configured, asked for an ordering
in one call rather than one call per candidate, and it is off by default.

Everything here fails open. A reranker that breaks must cost ordering quality,
never an answer.
"""
from __future__ import annotations

import logging
import re

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings

logger = logging.getLogger("rag")

RERANK_PROMPT = (
    "You are a relevance ranker. Given a question and numbered passages, return "
    "the passage numbers ordered from most to least relevant to the question. "
    "Output only the numbers separated by commas, most relevant first. Include "
    "every number exactly once. No prose."
)

_SNIPPET_CHARS = 600


def _parse_order(text: str, count: int) -> list[int]:
    """Parse '3, 1, 2' into zero-based indices, ignoring anything out of range."""
    seen: set[int] = set()
    order: list[int] = []
    for token in re.findall(r"\d+", text):
        index = int(token) - 1
        if 0 <= index < count and index not in seen:
            seen.add(index)
            order.append(index)
    return order


def rerank(question: str, documents: list[Document], llm) -> list[Document]:
    """Reorder documents by relevance. Returns the input order on any failure."""
    settings = get_settings()
    if not settings.rerank or len(documents) < 2:
        return documents

    candidates = documents[: settings.rerank_candidates]
    remainder = documents[settings.rerank_candidates :]

    listing = "\n\n".join(
        f"[{i + 1}] {d.page_content[:_SNIPPET_CHARS]}"
        for i, d in enumerate(candidates)
    )
    try:
        response = llm.invoke(
            [
                SystemMessage(content=RERANK_PROMPT),
                HumanMessage(content=f"Question: {question}\n\n{listing}"),
            ]
        )
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        order = _parse_order(content, len(candidates))
    except Exception:  # noqa: BLE001 - spec 002 AC-4
        logger.exception("Rerank failed; keeping fused order")
        return documents

    if not order:
        logger.warning("Rerank returned no usable ordering; keeping fused order")
        return documents

    # AC-5: anything the model left out keeps its relative order, after the
    # passages it did rank. Dropping them would lose candidates silently.
    ranked = [candidates[i] for i in order]
    omitted = [d for i, d in enumerate(candidates) if i not in set(order)]
    return ranked + omitted + remainder

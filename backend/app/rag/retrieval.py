"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion (spec 001).

Dense embeddings are good at paraphrase and bad at rare literal tokens -- a
query for "ONEK2J" or "99.95%" is exactly the case where cosine similarity has
nothing to work with. BM25 is the mirror image. Running both and fusing the
rankings costs one in-process index and no extra model call.

Fusion is over *child* chunks, because that is what is embedded and what BM25
indexes; the parents of the winning children are what the model finally sees.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

from langchain_core.documents import Document

from app.config import get_settings

logger = logging.getLogger("rag")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int
) -> list[tuple[str, float]]:
    """Fuse ranked id lists. score(d) = sum over rankings of 1/(k + rank).

    Rank is 1-based. An item present in several rankings accumulates score, so
    agreement between retrievers outranks a strong showing in just one.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass
class _Corpus:
    """The child chunks BM25 indexes, and their parent keys."""

    ids: list[str]
    texts: list[str]
    parent_ids: list[str]
    tenants: list[str]
    index: object  # BM25Okapi


class BM25Index:
    """Lazily built, cached in memory, invalidated whenever the store changes."""

    def __init__(self) -> None:
        self._corpus: _Corpus | None = None
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        with self._lock:
            self._corpus = None

    def _build(self, vectorstore, id_key: str) -> _Corpus | None:
        from rank_bm25 import BM25Okapi

        record = vectorstore.get(include=["documents", "metadatas"])
        texts = record.get("documents") or []
        metadatas = record.get("metadatas") or []
        ids = record.get("ids") or []
        if not texts:
            return None

        settings = get_settings()
        return _Corpus(
            ids=ids,
            texts=texts,
            parent_ids=[(m or {}).get(id_key, "") for m in metadatas],
            tenants=[
                (m or {}).get("tenant_id", settings.public_tenant) for m in metadatas
            ],
            index=BM25Okapi([tokenize(t) for t in texts]),
        )

    def rank(
        self, query: str, vectorstore, id_key: str, tenants: set[str], limit: int
    ) -> list[str]:
        """Parent ids ranked by BM25, best first. Empty list if unavailable."""
        try:
            with self._lock:
                if self._corpus is None:
                    self._corpus = self._build(vectorstore, id_key)
                corpus = self._corpus
            if corpus is None:
                return []

            scores = corpus.index.get_scores(tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])

            ranked: list[str] = []
            seen: set[str] = set()
            for i in order:
                if scores[i] <= 0:
                    break
                if corpus.tenants[i] not in tenants:
                    continue
                parent = corpus.parent_ids[i]
                if not parent or parent in seen:
                    continue
                seen.add(parent)
                ranked.append(parent)
                if len(ranked) >= limit:
                    break
            return ranked
        except Exception:  # noqa: BLE001 - spec 001 AC-6: degrade, never fail
            logger.exception("BM25 ranking unavailable; using dense results only")
            return []


_bm25 = BM25Index()


def invalidate_caches() -> None:
    """Called whenever documents are added or removed (spec 001 AC-4)."""
    _bm25.invalidate()


def dense_parent_ranking(
    vectorstore, id_key: str, query: str, tenants: set[str], limit: int
) -> list[str]:
    """Parent ids ranked by vector similarity over their child chunks."""
    where = {"tenant_id": {"$in": sorted(tenants)}}
    children = vectorstore.similarity_search(query, k=limit * 4, filter=where)

    ranked: list[str] = []
    seen: set[str] = set()
    for child in children:
        parent = child.metadata.get(id_key)
        if not parent or parent in seen:
            continue
        seen.add(parent)
        ranked.append(parent)
        if len(ranked) >= limit:
            break
    return ranked


def retrieve(retriever, query: str, tenants: set[str]) -> list[Document]:
    """Fused retrieval. Returns parent documents, best first, deduplicated."""
    settings = get_settings()
    vectorstore = retriever.vectorstore
    id_key = retriever.id_key
    limit = max(settings.fusion_candidates, settings.retrieval_k)

    dense = dense_parent_ranking(vectorstore, id_key, query, tenants, limit)

    if settings.hybrid_retrieval:
        lexical = _bm25.rank(query, vectorstore, id_key, tenants, limit)
        fused = [pid for pid, _ in reciprocal_rank_fusion([dense, lexical], settings.rrf_k)]
    else:
        fused = dense

    if not fused:
        return []

    parents = retriever.docstore.mget(fused)
    return [p for p in parents if p is not None]

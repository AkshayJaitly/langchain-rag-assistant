"""Embeddings — local and offline.

Two backends (set EMBEDDING_BACKEND):
  * "huggingface" — sentence-transformers via torch. Best quality, but torch
    is memory-heavy (won't fit a 512 MB host).
  * "fastembed"   — ONNX runtime, no torch. ~10x lighter and faster to cold
    start; used for the hosted deploy (see Dockerfile).

Both run locally with no API cost.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> Embeddings:
    settings = get_settings()
    backend = settings.embedding_backend.lower()

    if backend == "fastembed":
        # Imported lazily so the torch path never pulls in fastembed and vice versa.
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

        options = {"model_name": settings.fastembed_model}
        if settings.fastembed_cache_dir:
            options["cache_dir"] = settings.fastembed_cache_dir
        return FastEmbedEmbeddings(**options)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

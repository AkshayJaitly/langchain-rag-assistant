"""Local embeddings via sentence-transformers (no API cost, runs offline)."""
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    # normalize_embeddings=True pairs well with cosine similarity in Chroma.
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

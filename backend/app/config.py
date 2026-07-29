"""Application settings, loaded from environment / .env file."""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM provider: "anthropic" (Claude), "openai" (GPT), "groq" (free/fast
    # hosted Llama), or "ollama" (local/free)
    llm_provider: str = "anthropic"
    llm_max_tokens: int = 2048

    # Pipeline: "simple" (retrieve -> generate) or "multi_agent" (grader,
    # generator, and verifier agents with corrective re-retrieval)
    pipeline: str = "simple"

    # Anthropic (used when llm_provider == "anthropic")
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5"

    # OpenAI (used when llm_provider == "openai")
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Groq (used when llm_provider == "groq"; free tier at console.groq.com)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Ollama (used when llm_provider == "ollama")
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"

    # Embeddings
    # backend: "huggingface" (torch/sentence-transformers, best quality, heavy)
    #          or "fastembed" (ONNX, low-memory — fits small hosts like Render free)
    embedding_backend: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"

    # Persistence
    chroma_dir: str = "./data/chroma"
    docstore_dir: str = "./data/docstore"
    upload_dir: str = "./data/uploads"

    # Parent-child chunking
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50

    # Retrieval
    retrieval_k: int = 4

    # CORS
    cors_origins: str = "http://localhost:5173"

    # LangSmith tracing / observability (optional)
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-assistant"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def configure_langsmith(settings: "Settings") -> bool:
    """Export LangSmith env vars so LangChain/LangGraph auto-trace runs.

    LangChain reads these from os.environ, so we bridge them from settings.
    Returns True if tracing was enabled.
    """
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    return True


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Application settings, loaded from environment / .env file."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM provider: "anthropic" (Claude, paid API) or "ollama" (local, free)
    llm_provider: str = "anthropic"
    llm_max_tokens: int = 2048

    # Anthropic (used when llm_provider == "anthropic")
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5"

    # Ollama (used when llm_provider == "ollama")
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

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

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

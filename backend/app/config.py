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
    # Groq retired llama-3.3-70b-versatile; gpt-oss-120b is the current free
    # general-purpose chat model there (131k context, sub-second first token).
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    # gpt-oss models think before answering. "low" keeps the reasoning budget
    # small, which matters because it is billed against max_tokens.
    groq_reasoning_effort: str = "low"

    # Ollama (used when llm_provider == "ollama")
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"

    # Embeddings
    # backend: "huggingface" (torch/sentence-transformers, best quality, heavy)
    #          or "fastembed" (ONNX, low-memory — fits small hosts like Render free)
    embedding_backend: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    fastembed_cache_dir: str = ""
    # The hosted instance has 512 MB. Embedding every child chunk of a document
    # in one call is what pushed it over, so cap the ONNX working set: one
    # inference batch at a time, in one thread.
    fastembed_batch_size: int = 8
    fastembed_threads: int = 1

    # Persistence
    chroma_dir: str = "./data/chroma"
    docstore_dir: str = "./data/docstore"
    upload_dir: str = "./data/uploads"

    # Parent-child chunking
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50

    # Prompt-injection classifier (Meta's Prompt Guard 2, served free by Groq).
    # It returns P(injection); anything above the threshold is treated as an
    # attack. Needs GROQ_API_KEY -- without one the regex heuristics are used.
    guard_model: str = "meta-llama/llama-prompt-guard-2-86m"
    guard_threshold: float = 0.5
    guard_enabled: bool = True

    # Number of source pages embedded per add_documents call. Ingestion peaks at
    # roughly one batch of child chunks, not the whole document.
    ingest_batch_size: int = 1

    # Retrieval (spec 001). Dense embeddings miss rare literal tokens that BM25
    # finds, and vice versa; RRF fuses the two rankings.
    retrieval_k: int = 4
    hybrid_retrieval: bool = True
    rrf_k: int = 60
    # Candidates pulled from each retriever before fusion.
    fusion_candidates: int = 20

    # Reranking (spec 002). A cross-encoder needs torch, which does not fit in
    # 512 MB next to the embedding model, so this reranks with the configured
    # LLM and stays off by default.
    rerank: bool = False
    rerank_candidates: int = 20

    # Chunking strategy (spec 005): "recursive" or "semantic".
    chunking: str = "recursive"
    semantic_breakpoint_percentile: int = 90

    # Conversation memory (spec 003). History is supplied by the caller, so it
    # is untrusted input and gets validated and bounded server-side.
    history_turns: int = 6
    history_max_chars: int = 4000

    # Tenancy (spec 004). Isolation between visitors, not authentication.
    public_tenant: str = "public"
    anonymous_tenant: str = "anonymous"

    # CORS
    cors_origins: str = "http://localhost:5173"

    # LangSmith tracing / observability (optional)
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-assistant"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_workspace_id: str = ""
    langsmith_environment: str = "development"
    langsmith_hide_inputs: bool = True
    langsmith_hide_outputs: bool = True

    @property
    def chunking_strategy(self) -> str:
        """Validated chunking strategy (spec 005 AC-5: fail fast, never guess)."""
        value = self.chunking.strip().lower()
        if value not in {"recursive", "semantic"}:
            raise ValueError(
                f"Unknown CHUNKING '{self.chunking}'. Use 'recursive' or 'semantic'."
            )
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def langsmith_enabled(settings: "Settings") -> bool:
    """Return whether tracing has both been requested and authenticated."""
    return settings.langsmith_tracing and bool(settings.langsmith_api_key.strip())


def configure_langsmith(settings: "Settings") -> bool:
    """Export LangSmith env vars so LangChain/LangGraph auto-trace runs.

    LangChain reads these from os.environ, so we bridge them from settings.
    Returns True if tracing was enabled.
    """
    if not langsmith_enabled(settings):
        return False

    # Current LangSmith environment variable names.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_HIDE_INPUTS"] = str(
        settings.langsmith_hide_inputs
    ).lower()
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = str(
        settings.langsmith_hide_outputs
    ).lower()
    if settings.langsmith_workspace_id:
        os.environ["LANGSMITH_WORKSPACE_ID"] = settings.langsmith_workspace_id

    # Legacy names remain set for compatibility with the project's pinned
    # LangChain/LangGraph versions.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    return True


@lru_cache
def get_settings() -> Settings:
    return Settings()

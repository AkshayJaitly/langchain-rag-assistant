"""FastAPI entrypoint for the RAG service."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import configure_langsmith, get_settings

settings = get_settings()

# Enable LangSmith tracing for all LangChain/LangGraph runs if configured.
TRACING_ON = configure_langsmith(settings)

app = FastAPI(title="RAG Vector DB API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "rag-vector-db", "docs": "/docs"}

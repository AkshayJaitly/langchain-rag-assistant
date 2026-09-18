"""FastAPI entrypoint for the RAG service."""
from __future__ import annotations

from contextlib import asynccontextmanager

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import router
from app.config import configure_langsmith, get_settings
from app.api.routes import set_llm_status
from app.rag.embeddings import get_embeddings
from app.rag.graph import probe_llm

settings = get_settings()

# Enable LangSmith tracing for all LangChain/LangGraph runs if configured.
TRACING_ON = configure_langsmith(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Render uses FastEmbed. Load its cached ONNX model before health checks pass
    # so the first user upload does not pay model initialization time.
    if settings.embedding_backend.lower() == "fastembed":
        get_embeddings()

    status = probe_llm()
    set_llm_status(status)
    if status != "ok":
        logging.getLogger("rag").error(
            "LLM probe failed for provider=%s model=%s: %s",
            settings.llm_provider,
            settings.llm_model,
            status,
        )
    yield


app = FastAPI(title="RAG Vector DB API", version="1.0.0", lifespan=lifespan)

logger = logging.getLogger("rag")


async def json_errors(request: Request, call_next):
    """Turn an unhandled error into JSON that the browser can actually read.

    Starlette generates its default 500 above the CORS middleware, so the
    response carries no Access-Control-Allow-Origin header and fetch() can only
    report "Failed to fetch" -- which is how a retired upstream model showed up
    in the UI. Registering this *before* the CORS middleware puts it inside that
    layer, so the error response gets CORS headers like any other.
    """
    try:
        return await call_next(request)
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": f"{type(exc).__name__}: {exc}"},
        )


# Added first, so it sits *inside* the CORS middleware added below.
app.add_middleware(BaseHTTPMiddleware, dispatch=json_errors)

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

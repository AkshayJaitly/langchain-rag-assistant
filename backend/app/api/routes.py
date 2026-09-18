"""HTTP API: upload documents, query the RAG pipeline, list ingested docs."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.config import get_settings, langsmith_enabled
from app.rag.graph import answer_question
from app.rag.ingest import SUPPORTED_EXTENSIONS, ingest_file, save_upload
from app.rag.vectorstore import read_manifest

router = APIRouter(prefix="/api")

# Result of the startup LLM probe (see app.main lifespan). "unknown" until the
# probe runs; anything other than "ok" is the provider error verbatim.
_llm_status = "unknown"


def set_llm_status(status: str) -> None:
    global _llm_status
    _llm_status = status


class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    index: int
    source: str
    page: int | None = None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    guardrails: list[str]
    blocked: bool
    trace_id: str | None = None


class UploadResponse(BaseModel):
    filename: str
    documents_ingested: int
    pages_without_text: int = 0
    suspect_injection_pages: int = 0


@router.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    active_model = {
        "ollama": settings.ollama_model,
        "openai": settings.openai_model,
        "groq": settings.groq_model,
    }.get(settings.llm_provider.lower(), settings.llm_model)
    active_embedding_model = (
        settings.fastembed_model
        if settings.embedding_backend.lower() == "fastembed"
        else settings.embedding_model
    )
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_model": active_model,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": active_embedding_model,
        "pipeline": settings.pipeline,
        "tracing": "on" if langsmith_enabled(settings) else "off",
        # "ok" once the configured model has answered a probe at startup.
        "llm_status": _llm_status,
        # Ingestion tuning, echoed so a running instance can be checked against
        # the code that is supposed to be deployed.
        "ingest_batch_size": str(settings.ingest_batch_size),
        "embed_batch_size": str(settings.fastembed_batch_size),
        "embed_threads": str(settings.fastembed_threads),
    }


@router.get("/documents")
def documents() -> dict[str, list]:
    return {"documents": read_manifest()}


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    settings = get_settings()
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file.")

    path = save_upload(contents, filename, settings.upload_dir)
    try:
        count, skipped, suspicious = await run_in_threadpool(
            ingest_file, path, filename
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadResponse(
        filename=filename,
        documents_ingested=count,
        pages_without_text=skipped,
        suspect_injection_pages=suspicious,
    )


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    result = answer_question(req.question)
    return QueryResponse(**result)

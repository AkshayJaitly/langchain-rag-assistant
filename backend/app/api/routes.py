"""HTTP API: upload documents, query the RAG pipeline, list ingested docs."""
from __future__ import annotations

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator
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


# Spec 003 AC-8: history arrives from the caller, so it is untrusted input.
# Anything outside this set is a client bug or an attempt to shape the prompt.
VALID_ROLES = {"user", "assistant"}


class Turn(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def known_role(cls, value: str) -> str:
        if value not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}")
        return value

    @field_validator("content")
    @classmethod
    def bounded_content(cls, value: str) -> str:
        limit = get_settings().history_max_chars
        if len(value) > limit:
            # AC-9: reject rather than silently truncating, so the caller is
            # never told a turn was used when it was quietly cut in half.
            raise ValueError(f"turn content exceeds {limit} characters")
        return value


class QueryRequest(BaseModel):
    question: str
    # Prior turns for follow-up resolution (spec 003). Sent by the client, which
    # already keeps them; the server holds no cross-request state.
    history: list[Turn] = Field(default_factory=list)


class Source(BaseModel):
    index: int
    source: str
    page: int | None = None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    # Set only when a follow-up was rewritten, so the UI can show what was
    # actually retrieved on.
    standalone_question: str | None = None
    sources: list[Source]
    guardrails: list[str]
    blocked: bool
    trace_id: str | None = None


def _tenants(header: str | None) -> tuple[str, set[str]]:
    """(write tenant, readable tenants) for a request (spec 004).

    A visitor reads their own documents plus the public samples. No header means
    an anonymous tenant, which still sees the samples.
    """
    settings = get_settings()
    tenant = (header or "").strip()[:64] or settings.anonymous_tenant
    if tenant == settings.public_tenant:
        # The public tenant is reserved for the bundled samples; a caller
        # claiming it writes as anonymous instead of overwriting them.
        tenant = settings.anonymous_tenant
    return tenant, {tenant, settings.public_tenant}


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
        "hybrid_retrieval": str(settings.hybrid_retrieval).lower(),
        "rerank": str(settings.rerank).lower(),
        "chunking": settings.chunking_strategy,
        "ingest_batch_size": str(settings.ingest_batch_size),
        "embed_batch_size": str(settings.fastembed_batch_size),
        "embed_threads": str(settings.fastembed_threads),
    }


@router.get("/documents")
def documents(x_tenant_id: str | None = Header(default=None)) -> dict[str, list]:
    settings = get_settings()
    _, readable = _tenants(x_tenant_id)
    rows = []
    for row in read_manifest(readable):
        shared = row.get("tenant_id", settings.public_tenant) == settings.public_tenant
        rows.append({**row, "shared": shared})
    return {"documents": rows}


@router.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    x_tenant_id: str | None = Header(default=None),
) -> UploadResponse:
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
        tenant, _ = _tenants(x_tenant_id)
        count, skipped, suspicious = await run_in_threadpool(
            ingest_file, path, filename, tenant
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
def query(
    req: QueryRequest, x_tenant_id: str | None = Header(default=None)
) -> QueryResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    _, readable = _tenants(x_tenant_id)
    result = answer_question(
        req.question,
        history=[t.model_dump() for t in req.history],
        tenants=readable,
    )
    return QueryResponse(**result)

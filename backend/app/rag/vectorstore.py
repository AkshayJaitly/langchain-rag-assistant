"""Vector store + parent-child retriever wiring.

Uses LangChain's ParentDocumentRetriever (the "parent-child" / "small-to-big"
algorithm):

  * Documents are split into large *parent* chunks and small *child* chunks.
  * Only the small child chunks are embedded and stored in Chroma. Small chunks
    embed more precisely, so retrieval is sharper.
  * At query time we search the child chunks, then return their *parent* chunks
    to the LLM — giving it the surrounding context a tiny chunk would lack.

The parent documents live in a persistent file-backed key/value docstore; the
child vectors live in a persistent Chroma collection.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.rag.embeddings import get_embeddings
from app.rag.retrieval import invalidate_caches

_COLLECTION = "rag_children"


def _manifest_path() -> str:
    settings = get_settings()
    return os.path.join(settings.docstore_dir, "_manifest.json")


def read_manifest(tenants: set[str] | None = None) -> list[dict]:
    """Ingested documents, optionally limited to a set of tenants (spec 004)."""
    path = _manifest_path()
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    if tenants is None:
        return rows
    default = get_settings().public_tenant
    return [r for r in rows if r.get("tenant_id", default) in tenants]


def record_ingested(
    filename: str, chunks: int, suspicious: int = 0, tenant_id: str | None = None
) -> None:
    """Record an ingested document, replacing this tenant's earlier entry only.

    Spec 004 AC-5: a re-upload must not touch another tenant's row of the same
    name, so identity here is (tenant, filename) rather than filename alone.
    """
    tenant = tenant_id or get_settings().public_tenant
    manifest = [
        d
        for d in read_manifest()
        if not (
            d.get("filename") == filename
            and d.get("tenant_id", get_settings().public_tenant) == tenant
        )
    ]
    entry = {"filename": filename, "chunks": chunks, "tenant_id": tenant}
    if suspicious:
        entry["suspect_injection_pages"] = suspicious
    manifest.append(entry)
    os.makedirs(os.path.dirname(_manifest_path()), exist_ok=True)
    with open(_manifest_path(), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def remove_document(filename: str, tenant_id: str | None = None) -> int:
    """Drop a document's existing chunks. Returns the number of children removed.

    Re-uploading a file used to embed a second copy of every chunk, so the same
    passage came back twice in retrieval and the sidebar listed the document
    twice. Clearing first makes re-upload a replace rather than an append.
    """
    retriever = get_retriever()
    tenant = tenant_id or get_settings().public_tenant
    collection = retriever.vectorstore.get(
        where={"$and": [{"source": filename}, {"tenant_id": tenant}]}
    )
    ids = collection.get("ids") or []
    if not ids:
        return 0

    # Children carry the key of the parent chunk they were split from; drop
    # those parents too so the docstore does not accumulate orphans.
    parent_ids = {
        meta.get(retriever.id_key)
        for meta in (collection.get("metadatas") or [])
        if meta and meta.get(retriever.id_key)
    }
    retriever.vectorstore.delete(ids=ids)
    if parent_ids:
        retriever.docstore.mdelete(list(parent_ids))
    invalidate_caches()
    return len(ids)


def _child_splitter(settings):
    """Child splitter for the configured strategy (spec 005).

    Semantic chunking embeds every sentence to find topic breakpoints, which is
    real work on a 512 MB host, so "recursive" stays the default and the
    strategy is reported by /api/health.
    """
    if settings.chunking_strategy == "semantic":
        from langchain_experimental.text_splitter import SemanticChunker

        return SemanticChunker(
            embeddings=get_embeddings(),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=settings.semantic_breakpoint_percentile,
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
    )


@lru_cache
def get_retriever() -> ParentDocumentRetriever:
    settings = get_settings()

    os.makedirs(settings.chroma_dir, exist_ok=True)
    os.makedirs(settings.docstore_dir, exist_ok=True)

    vectorstore = Chroma(
        collection_name=_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )

    # File-backed docstore for parent documents (survives restarts).
    fs = LocalFileStore(settings.docstore_dir)
    docstore = create_kv_docstore(fs)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_chunk_overlap,
    )
    child_splitter = _child_splitter(settings)

    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        # Pull extra children so parent de-duplication still yields ~k parents.
        search_kwargs={"k": max(settings.retrieval_k * 4, 8)},
    )

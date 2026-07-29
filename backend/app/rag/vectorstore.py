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

_COLLECTION = "rag_children"


def _manifest_path() -> str:
    settings = get_settings()
    return os.path.join(settings.docstore_dir, "_manifest.json")


def read_manifest() -> list[dict]:
    """Return the list of ingested documents ({filename, chunks})."""
    path = _manifest_path()
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def record_ingested(filename: str, chunks: int) -> None:
    manifest = read_manifest()
    manifest.append({"filename": filename, "chunks": chunks})
    os.makedirs(os.path.dirname(_manifest_path()), exist_ok=True)
    with open(_manifest_path(), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


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
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
    )

    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        # Pull extra children so parent de-duplication still yields ~k parents.
        search_kwargs={"k": max(settings.retrieval_k * 4, 8)},
    )

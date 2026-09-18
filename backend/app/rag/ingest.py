"""Document loading, parsing, and ingestion into the parent-child retriever."""
from __future__ import annotations

import gc
import os

from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_core.documents import Document

from app.config import get_settings
from app.rag import guardrails
from app.rag.pdf import extract_pages
from app.rag.vectorstore import get_retriever, record_ingested, remove_document

SUPPORTED_EXTENSIONS = {"pdf", "docx", "doc", "txt", "md"}


def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


SCANNED_PDF_MESSAGE = (
    "No text could be extracted from this PDF -- it looks like a scan or images "
    "of pages. Upload a text-based PDF, or run OCR on it first."
)


def load_document(path: str, filename: str) -> list[Document]:
    """Parse a PDF / Word / text file into LangChain Documents.

    PDFs go through app.rag.pdf, which keeps reading order and table structure
    that the plain PyPDF path used to lose. Other formats are simple enough that
    the stock loaders are fine.
    """
    ext = _extension(filename)
    if ext == "pdf":
        pages, empty_pages = extract_pages(path)
        if not pages:
            raise ValueError(SCANNED_PDF_MESSAGE)
        return [
            Document(
                page_content=page.text,
                metadata={
                    "source": filename,
                    "page": page.number,
                    # Surfaced so a partly-scanned PDF is visible, not silent.
                    "pages_without_text": empty_pages,
                },
            )
            for page in pages
        ]

    if ext in {"docx", "doc"}:
        loader = Docx2txtLoader(path)
    elif ext in {"txt", "md"}:
        loader = TextLoader(path, encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type '.{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    docs = loader.load()
    # Tag every document with a stable source name for citations.
    for doc in docs:
        doc.metadata["source"] = filename
    return docs


def ingest_file(path: str, filename: str) -> tuple[int, int, int]:
    """Load a file, split it into parent/child chunks, embed and store it.

    Returns (documents ingested, pages with no text, pages flagged as injection).
    """
    ext = _extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '.{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    docs = load_document(path, filename)
    if not docs or not any(d.page_content.strip() for d in docs):
        raise ValueError("No extractable text found in the document.")

    # Screen the document itself. Guarding only the question leaves indirect
    # prompt injection wide open: text inside an uploaded PDF reaches the model
    # through the retrieved context without ever passing an input guardrail.
    # Scoring here costs one classifier call per page at upload instead of one
    # per question, and the verdict rides along in the chunk's metadata.
    suspicious = 0
    for doc in docs:
        flagged, score = guardrails.looks_like_injection(doc.page_content)
        doc.metadata["injection_score"] = round(score, 4) if score is not None else None
        doc.metadata["suspect_injection"] = flagged
        suspicious += int(flagged)

    # Re-uploading replaces the previous copy instead of adding a second one.
    remove_document(filename)

    retriever = get_retriever()
    # ParentDocumentRetriever handles parent+child splitting and embedding, but
    # it embeds every child of everything it is handed in a single call. On a
    # 512 MB host that is what runs the container out of memory on a document of
    # any size, so feed it a few pages at a time and let each batch's arrays go.
    batch_size = max(1, get_settings().ingest_batch_size)
    for start in range(0, len(docs), batch_size):
        retriever.add_documents(docs[start : start + batch_size])
        gc.collect()

    skipped = docs[0].metadata.get("pages_without_text", 0)
    record_ingested(filename, len(docs), suspicious)
    return len(docs), skipped, suspicious


def save_upload(tmp_bytes: bytes, filename: str, upload_dir: str) -> str:
    """Persist an uploaded file to disk and return its path."""
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = os.path.basename(filename)
    dest = os.path.join(upload_dir, safe_name)
    with open(dest, "wb") as fh:
        fh.write(tmp_bytes)
    return dest

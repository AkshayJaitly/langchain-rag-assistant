"""Document loading, parsing, and ingestion into the parent-child retriever."""
from __future__ import annotations

import os

from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_core.documents import Document

from app.rag.pdf import extract_pages
from app.rag.vectorstore import get_retriever, record_ingested

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


def ingest_file(path: str, filename: str) -> tuple[int, int]:
    """Load a file, split it into parent/child chunks, embed and store it.

    Returns (documents ingested, pages that yielded no text).
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

    retriever = get_retriever()
    # ParentDocumentRetriever handles parent+child splitting and embedding.
    retriever.add_documents(docs)

    skipped = docs[0].metadata.get("pages_without_text", 0)
    record_ingested(filename, len(docs))
    return len(docs), skipped


def save_upload(tmp_bytes: bytes, filename: str, upload_dir: str) -> str:
    """Persist an uploaded file to disk and return its path."""
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = os.path.basename(filename)
    dest = os.path.join(upload_dir, safe_name)
    with open(dest, "wb") as fh:
        fh.write(tmp_bytes)
    return dest

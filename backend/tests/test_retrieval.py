"""Retrieval-side behaviour: dedupe, citation shape, and the manifest."""
from __future__ import annotations

from langchain_core.documents import Document

from app.rag.graph import _dedupe, _normalize_citations
from app.rag.vectorstore import read_manifest, record_ingested


def test_dedupe_drops_repeat_parents_keeping_order():
    """Several children resolving to one parent cited the same passage twice."""
    a = Document(page_content="alpha", metadata={"source": "a.pdf"})
    b = Document(page_content="beta", metadata={"source": "a.pdf"})
    duplicate = Document(page_content="alpha", metadata={"source": "a.pdf"})

    assert [d.page_content for d in _dedupe([a, b, duplicate])] == ["alpha", "beta"]


def test_same_text_from_different_sources_is_kept():
    a = Document(page_content="shared", metadata={"source": "a.pdf"})
    b = Document(page_content="shared", metadata={"source": "b.pdf"})
    assert len(_dedupe([a, b])) == 2


def test_normalizes_full_width_citations():
    """gpt-oss emits 【1】 and 【3†a】; the UI renders pills from ASCII [n]."""
    assert _normalize_citations("see 【1】") == "see [1]"
    assert _normalize_citations("see 【3†a】") == "see [3]"
    assert _normalize_citations("plain [2] stays") == "plain [2] stays"


def test_narrow_no_break_spaces_become_plain_spaces():
    assert _normalize_citations("UA 881") == "UA 881"


def test_reupload_replaces_the_manifest_row():
    """Re-uploading used to list the same document twice in the sidebar."""
    record_ingested("report.pdf", 4)
    record_ingested("other.pdf", 2)
    record_ingested("report.pdf", 7)

    manifest = read_manifest()
    names = [row["filename"] for row in manifest]
    assert names.count("report.pdf") == 1
    assert next(r for r in manifest if r["filename"] == "report.pdf")["chunks"] == 7
    assert "other.pdf" in names


def test_manifest_records_flagged_pages():
    record_ingested("poisoned.pdf", 3, suspicious=2)
    row = next(r for r in read_manifest() if r["filename"] == "poisoned.pdf")
    assert row["suspect_injection_pages"] == 2

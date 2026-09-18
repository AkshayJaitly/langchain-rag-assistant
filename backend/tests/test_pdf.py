"""PDF extraction: the hybrid pass, cleanup, and the scanned-PDF failure."""
from __future__ import annotations

import pymupdf

from app.rag import ingest
from app.rag.pdf import _drop_repeated_lines, _normalize, extract_pages


def test_extracts_each_page(make_pdf):
    path = make_pdf(["First page body text", "Second page body text"])
    pages, empty = extract_pages(path)

    assert [p.number for p in pages] == [0, 1]
    assert "First page" in pages[0].text
    assert "Second page" in pages[1].text
    assert empty == 0


def test_counts_pages_with_no_text(make_pdf):
    path = make_pdf(["Real content here", "", ""])
    pages, empty = extract_pages(path)

    assert len(pages) == 1
    assert empty == 2


def test_scanned_pdf_fails_loudly(make_pdf, tmp_path):
    """An image-only PDF must not index silently as an empty document."""
    doc = pymupdf.open()
    doc.new_page()
    path = tmp_path / "scanned.pdf"
    doc.save(str(path))
    doc.close()

    pages, empty = extract_pages(str(path))
    assert pages == []
    assert empty == 1

    try:
        ingest.load_document(str(path), "scanned.pdf")
    except ValueError as exc:
        assert "scan" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError for a scanned PDF")


def test_normalize_folds_ligatures_and_odd_spaces():
    """Ligatures broke keyword matching: 'conﬁrmation' never matched 'confirm'."""
    assert "confirmation" in _normalize("conﬁrmation")
    assert " " not in _normalize("a b")
    assert _normalize("a  \t b") == "a b"


def test_drops_running_heads_but_keeps_body_text():
    pages = [
        "ACME CONFIDENTIAL\nUnique body one\nPage footer",
        "ACME CONFIDENTIAL\nUnique body two\nPage footer",
        "ACME CONFIDENTIAL\nUnique body three\nPage footer",
    ]
    cleaned = _drop_repeated_lines(pages)

    assert all("ACME CONFIDENTIAL" not in page for page in cleaned)
    assert "Unique body two" in cleaned[1]


def test_keeps_repeated_lines_when_there_are_too_few_pages():
    """Two pages is not enough evidence that a repeated line is furniture."""
    pages = ["HEADER\nbody one", "HEADER\nbody two"]
    assert _drop_repeated_lines(pages) == pages

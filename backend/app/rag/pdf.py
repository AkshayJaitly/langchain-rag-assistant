"""PDF text extraction tuned for retrieval.

PyPDF's plain extraction was losing real content on these documents: text laid
out in columns came back interleaved, table rows arrived as run-on words
("Time2h"), and ligatures survived as single codepoints ("con<fi>rmation"),
which keyword-level retrieval never matches.

Each page is therefore extracted twice with PyMuPDF and the better result wins:

  * ``pymupdf4llm`` returns Markdown, so headings and tables stay structured for
    both the splitter and the LLM.
  * ``page.get_text(sort=True)`` returns plain text in reading order. It is the
    safety net: pages whose text is drawn inside graphics come back nearly empty
    from the Markdown pass (the flight itinerary loses its whole "DEPART /
    ARRIVE" block that way).

Repeated running heads and feet are dropped, and pages with no extractable text
are counted so a scanned PDF fails loudly instead of indexing an empty document.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pymupdf
import pymupdf4llm

# A Markdown page shorter than this fraction of the plain-text page is treated
# as a failed extraction and replaced by the plain text.
_MARKDOWN_COVERAGE_FLOOR = 0.8

# A running head/foot is short, sits in the top/bottom few lines of a page, and
# repeats on nearly every page. Only those lines are eligible for removal --
# matching anywhere on the page deletes real content from form-like documents.
_REPEAT_LINE_MAX_CHARS = 80
_REPEAT_LINE_MIN_PAGES = 3
_REPEAT_LINE_SHARE = 0.8
_REPEAT_LINE_EDGE = 2


@dataclass
class Page:
    number: int  # 0-based, matching the metadata PyPDFLoader used to emit
    text: str


def _normalize(text: str) -> str:
    """Fold ligatures and odd whitespace so retrieval can match plain words."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _drop_repeated_lines(pages: list[str]) -> list[str]:
    """Remove running heads/feet that repeat at the edges of most pages."""
    if len(pages) < _REPEAT_LINE_MIN_PAGES:
        return pages

    split = [[l for l in page.splitlines()] for page in pages]

    def edges(lines: list[str]) -> set[str]:
        stripped = [l.strip() for l in lines if l.strip()]
        edge = stripped[:_REPEAT_LINE_EDGE] + stripped[-_REPEAT_LINE_EDGE:]
        return {l for l in edge if len(l) <= _REPEAT_LINE_MAX_CHARS}

    counts: dict[str, int] = {}
    for lines in split:
        for line in edges(lines):
            counts[line] = counts.get(line, 0) + 1

    threshold = max(_REPEAT_LINE_MIN_PAGES, int(len(pages) * _REPEAT_LINE_SHARE))
    repeated = {line for line, n in counts.items() if n >= threshold}
    if not repeated:
        return pages

    kept = []
    for lines in split:
        page_edges = edges(lines)
        drop = repeated & page_edges
        kept.append("\n".join(l for l in lines if l.strip() not in drop))
    return kept


def extract_pages(path: str) -> tuple[list[Page], int]:
    """Return (pages with text, number of pages that yielded no text)."""
    with pymupdf.open(path) as doc:
        plain = [page.get_text("text", sort=True) or "" for page in doc]

    try:
        markdown = [
            chunk["text"] or ""
            for chunk in pymupdf4llm.to_markdown(
                path, page_chunks=True, show_progress=False
            )
        ]
    except Exception:
        # Never let the Markdown pass take the upload down; plain text is enough.
        markdown = [""] * len(plain)

    merged = [
        md if len(md) >= _MARKDOWN_COVERAGE_FLOOR * len(txt) else txt
        for md, txt in zip(markdown, plain)
    ]
    merged = [_normalize(text) for text in _drop_repeated_lines(merged)]

    pages = [Page(number=i, text=t) for i, t in enumerate(merged) if t]
    return pages, len(merged) - len(pages)

# 005 — Chunking strategies

## Goal

Let chunk boundaries follow meaning rather than a character count, and make the
choice measurable instead of assumed.

## Constraints

- Inherits C-1, C-2.
- Semantic chunking embeds every sentence at ingest. That is more work on a
  512 MB host, so it is **opt-in** and the recursive splitter stays the default.

## Behaviour

- **AC-1** `CHUNKING=recursive` (default) keeps the existing parent/child
  character splitter.
- **AC-2** `CHUNKING=semantic` splits children at embedding-similarity
  breakpoints instead of fixed character counts.
- **AC-3** The strategy is reported by `/api/health` so a running instance can
  be identified.
- **AC-4** Switching strategies does not migrate existing chunks; the README
  says to reindex.
- **AC-5** An unknown value fails fast at startup rather than silently falling
  back.

## Non-goals

Late chunking, layout-aware chunking.

## Verification

`tests/test_chunking.py`; `eval/` compares recall@k across strategies.

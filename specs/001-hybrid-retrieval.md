# 001 — Hybrid retrieval

## Goal

Retrieve with both lexical and dense signals instead of dense alone, so exact
terms (`ONEK2J`, `99.95%`, `Severity 1`) are findable. Embeddings are good at
paraphrase and bad at rare literal tokens; BM25 is the reverse.

## Constraints

- Inherits C-1, C-2.
- No second service. BM25 runs in-process over the child chunks already in
  Chroma (`rank_bm25` is pure Python, no native build, no model).
- Must not increase per-query model calls.

## Behaviour

- **AC-1** A query runs a dense search and a BM25 search over the same child
  chunks, and the two rankings are fused with Reciprocal Rank Fusion:
  `score(d) = Σ 1/(k + rank_i(d))` with `k = RRF_K` (default 60).
- **AC-2** A document ranked well by either retriever appears in the fused
  result; a document ranked well by both outranks one ranked well by a single
  retriever.
- **AC-3** Fusion happens over child chunks; the parents of the winning
  children are returned, deduplicated, capped at `RETRIEVAL_K`.
- **AC-4** The BM25 index is built lazily from the vector store, cached, and
  invalidated whenever documents are added or removed.
- **AC-5** With `HYBRID_RETRIEVAL=false` the dense-only path is used, so the
  two can be compared by the evaluation harness.
- **AC-6** If the BM25 index cannot be built the query still answers from the
  dense results, and the degradation is logged rather than raised.

## Non-goals

Query expansion, multi-query retrieval, HyDE.

## Verification

`tests/test_retrieval.py` (RRF maths, fusion ordering, cache invalidation),
`tests/integration/`, and `eval/` recall@k and MRR for both settings.

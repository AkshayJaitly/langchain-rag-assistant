# 002 — Reranking

## Goal

Reorder fused candidates by relevance to the question before they reach the
prompt, so the `RETRIEVAL_K` passages the model sees are the best ones rather
than the best-ranked by a fusion heuristic.

## Constraints

- Inherits C-1, C-2.
- **A cross-encoder does not fit.** The usual answer (`ms-marco-MiniLM` via
  sentence-transformers) needs torch, which does not fit alongside the
  embedding model in 512 MB — the host already OOM-killed on ingestion alone.
- Reranking therefore uses the LLM that is already configured, via Groq's free
  tier, and is **off by default on the hosted deployment**.

## Behaviour

- **AC-1** With `RERANK=true`, candidates are scored for relevance to the
  question and reordered before truncation to `RETRIEVAL_K`.
- **AC-2** Reranking sees at most `RERANK_CANDIDATES` (default 20) candidates,
  in one model call, not one call per candidate.
- **AC-3** The default is `RERANK=false`. Enabling it is a documented
  latency/quality trade, not a silent default.
- **AC-4** A rerank failure (timeout, bad output, unparseable ranking) leaves
  the fused order untouched and is logged. A query never fails because of it.
- **AC-5** Any candidate the reranker omits keeps its relative fused order
  after the ranked ones, so nothing is silently dropped.

## Non-goals

Cross-encoder or ColBERT-style late interaction. Revisit if the deployment ever
has more than 512 MB.

## Verification

`tests/test_rerank.py` with a stub model (ordering, malformed output, failure
passthrough); `eval/` compares nDCG with and without.

# 007 — Test strategy

## Goal

Four layers, each catching what the layer below cannot, all running offline.

## Constraints

- Inherits C-1, C-2. No test calls Groq, LangSmith or any network service.
- Real embedding models are too slow for unit and integration tests, so those
  layers use a deterministic fake embedding. Extraction tests use real PDFs
  built at runtime, because that is the thing under test.

## Behaviour

- **AC-1 Unit.** Pure logic: extraction, RRF maths, guardrail patterns,
  citation normalisation, chunking selection. No I/O.
- **AC-2 Integration.** The FastAPI app through `TestClient`, with fake
  embeddings and a stub model: upload → index → query → cite, error paths, and
  tenant isolation across the real HTTP surface.
- **AC-3 BDD.** `pytest-bdd` feature files in business language for the
  behaviours a reviewer would ask about: grounded answers, refusal, injection
  handling, multi-turn, isolation.
- **AC-4 Evaluation.** Spec 006, quality rather than correctness.
- **AC-5** Every test is hermetic: its own temp directories, no shared state,
  any order.
- **AC-6** CI runs unit + integration + BDD + the retrieval evaluation on every
  change under `backend/`.

## Verification

`python -m pytest` from `backend/`; the `Backend tests` workflow.

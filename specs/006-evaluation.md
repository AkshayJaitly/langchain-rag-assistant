# 006 — Evaluation harness

## Goal

Replace anecdote with numbers. Every claim about retrieval quality, pipeline
choice or guardrail accuracy should come from a dataset, not a demo.

## Constraints

- Inherits C-1, C-2.
- Retrieval and guardrail metrics must be computable **without a model call**,
  so they can run in CI. Answer quality needs a judge, so it is a manual run.
- The golden dataset is written against the bundled sample documents, so it
  works on any checkout with no private data.

## Behaviour

- **AC-1** A golden dataset holds, per case: question, the document that should
  be retrieved, substrings the answer must contain, and whether it is
  answerable.
- **AC-2** `python -m eval.run --retrieval` reports **recall@k**, **MRR** and
  **hit rate**, using embeddings only — no LLM call.
- **AC-3** `python -m eval.run --guardrails` reports block rate over an attack
  set and false-positive rate over a benign set.
- **AC-4** `python -m eval.run --answers` additionally reports answer match and
  refusal accuracy, and requires a configured provider.
- **AC-5** `--compare` runs a metric across configurations (hybrid on/off,
  rerank on/off, pipeline simple/multi_agent) and prints them side by side, so
  the README's open question is answerable.
- **AC-6** Results print as a table and are written as JSON for tracking.
- **AC-7** The retrieval suite runs in CI on every backend change and fails the
  build if recall@k drops below a floor.

## Non-goals

RAGAS (its defaults assume OpenAI), a hosted eval dashboard.

## Verification

`eval/` is itself covered by `tests/test_eval.py` (metric maths on fixed input).

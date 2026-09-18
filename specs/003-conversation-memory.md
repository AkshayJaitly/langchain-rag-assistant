# 003 — Multi-turn conversations

## Goal

Make follow-up questions work. "What about the second one?" currently retrieves
on those words alone, which matches nothing.

## Constraints

- Inherits C-1, C-2.
- No database. History lives in a LangGraph checkpointer in process, and is
  lost on restart — acceptable on a host that sleeps, and documented as such.

## Behaviour

- **AC-1** A query may carry a `thread_id`. Turns with the same `thread_id`
  share history; turns without one are independent.
- **AC-2** When history exists, the question is condensed into a standalone
  question before retrieval, using the history for referents.
- **AC-3** The condensed question is what gets retrieved on and what is
  reported in the response as `standalone_question`; the user's original
  wording is what the model answers.
- **AC-4** The first turn of a thread is never condensed — there is nothing to
  resolve — and costs no extra model call.
- **AC-5** History is capped at `HISTORY_TURNS` (default 6) most recent turns.
- **AC-6** A condense failure falls back to the original question.
- **AC-7** Guardrails apply to the original question, before condensing, so
  history cannot be used to smuggle an injection past the classifier.

## Non-goals

Cross-device or persistent history, summarisation of long threads.

## Verification

`tests/test_memory.py`, `features/conversation.feature`.

# 003 — Multi-turn conversations

## Goal

Make follow-up questions work. "What about the second one?" retrieved on those
words alone matches nothing.

## Constraints

- Inherits C-1, C-2.

### Revision: why history is client-supplied, not a server checkpointer

The first version of this spec required a `thread_id` and a LangGraph
checkpointer holding history in process. That was implemented as client-supplied
history instead, and on review the constraint analysis behind the original was
wrong:

- **The host sleeps after ~15 idle minutes and the disk is ephemeral.** An
  in-process checkpointer is empty almost every time a real visitor comes back,
  so the server would forget a conversation the browser still has on screen.
- **A checkpointer is per-process.** It stops being correct the moment there is
  more than one worker, so it would have to be replaced before it ever scaled —
  it buys nothing now and is not the thing to keep later.

A stateless API where the caller carries the conversation is a normal production
pattern for exactly these reasons. The cost is that history becomes untrusted
input, which the behaviour below has to account for — and that is a requirement
the original spec missed entirely.

## Behaviour

- **AC-1** A query may carry `history`: prior turns as `{role, content}`. Turns
  are supplied per request; the server holds no cross-request conversation
  state.
- **AC-2** When history is present, the question is condensed into a standalone
  question before retrieval, using the history to resolve referents.
- **AC-3** The condensed question is what retrieval runs on and what the
  response reports as `standalone_question`; the model answers the user's own
  wording.
- **AC-4** The first turn is never condensed — there is nothing to resolve —
  and costs no extra model call.
- **AC-5** History is capped at `HISTORY_TURNS` (default 6) most recent turns.
- **AC-6** A condense failure falls back to the original question.
- **AC-7** Guardrails apply to the original question, before condensing, so
  history cannot be used to smuggle an injection past the classifier.

### History is untrusted input

- **AC-8** History is validated server-side: turns with an unrecognised role are
  rejected, and content is truncated to `HISTORY_MAX_CHARS` per turn. A caller
  cannot grow the prompt without bound by sending a large history.
- **AC-9** Oversized or malformed history is rejected with a 422 rather than
  silently truncated into something the user did not ask for.
- **AC-10** History only ever influences *retrieval* (through condensing) and
  the conversational framing of the answer. It is never treated as retrieved
  context, and never as instructions.

## Non-goals

Cross-device or persistent history, thread summarisation, server-side threads.
Those need authenticated storage, which needs accounts — see spec 004.

## Verification

`tests/test_memory.py`, `tests/integration/test_history.py`,
`features/conversation.feature`.

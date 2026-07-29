"""Lightweight input/output guardrails for the RAG pipeline.

These run inside the LangGraph graph:

  * input guardrails  — reject empty/oversized queries and obvious prompt-
    injection attempts before we spend tokens.
  * grounding guardrail — if retrieval returned nothing, refuse to answer from
    the model's parametric memory (prevents hallucination).
  * output guardrails — redact anything that looks like a secret and flag when
    the answer isn't grounded in the retrieved context.
"""
from __future__ import annotations

import re

MAX_QUESTION_CHARS = 4000

# Heuristic prompt-injection / jailbreak patterns.
_INJECTION_PATTERNS = [
    r"ignore (all |the |your )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |the |your )?(previous|prior|above)",
    r"you are now",
    r"system prompt",
    r"reveal your (system )?prompt",
    r"pretend to be",
    r"developer mode",
    r"jailbreak",
]

# Patterns that look like leaked secrets in model output.
_SECRET_PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9\-_]{8,}"), "[REDACTED_ANTHROPIC_KEY]"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
]

_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

REFUSAL_MESSAGE = (
    "I couldn't find anything in the uploaded documents that answers that "
    "question, so I can't answer it reliably. Try rephrasing, or upload a "
    "document that covers this topic."
)


def check_input(question: str) -> tuple[bool, str]:
    """Return (allowed, reason). reason is non-empty only when blocked."""
    q = (question or "").strip()
    if not q:
        return False, "Empty question."
    if len(q) > MAX_QUESTION_CHARS:
        return False, f"Question exceeds {MAX_QUESTION_CHARS} characters."
    if _INJECTION_RE.search(q):
        return False, "Question looks like a prompt-injection attempt."
    return True, ""


def redact_secrets(text: str) -> tuple[str, bool]:
    """Redact secret-like tokens from output. Returns (clean_text, changed)."""
    changed = False
    for pattern, replacement in _SECRET_PATTERNS:
        text, n = pattern.subn(replacement, text)
        changed = changed or n > 0
    return text, changed


def is_grounded(answer: str, documents) -> bool:
    """Cheap grounding check: does the answer share vocabulary with context?

    Not a proof of faithfulness, but catches answers that clearly ignore the
    retrieved context. If the model already refused ("I don't know"), treat it
    as grounded.
    """
    lowered = answer.lower()
    if "don't know" in lowered or "cannot find" in lowered or "couldn't find" in lowered:
        return True
    context = " ".join(d.page_content for d in documents).lower()
    context_words = {w for w in re.findall(r"[a-z]{4,}", context)}
    answer_words = {w for w in re.findall(r"[a-z]{4,}", lowered)}
    if not answer_words:
        return True
    overlap = len(answer_words & context_words) / len(answer_words)
    return overlap >= 0.15

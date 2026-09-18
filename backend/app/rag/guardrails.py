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

import logging
import re
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger("rag")

MAX_QUESTION_CHARS = 4000

# Prompt Guard sees a limited window; classify the leading slice of long text.
GUARD_MAX_CHARS = 2000

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

# Patterns that look like leaked secrets or personal data in model output.
# Order matters: the most specific key formats come before the generic ones.
_SECRET_PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9\-_]{8,}"), "[REDACTED_ANTHROPIC_KEY]"),
    (re.compile(r"gsk_[A-Za-z0-9]{20,}"), "[REDACTED_GROQ_KEY]"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"AIza[A-Za-z0-9\-_]{30,}"), "[REDACTED_GOOGLE_KEY]"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "[REDACTED_SLACK_TOKEN]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED_API_KEY]"),
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+"), "[REDACTED_JWT]"),
    # The demo corpus is travel and HR documents, so personal identifiers are
    # the realistic leak, not cloud keys.
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[REDACTED_CARD_NUMBER]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
]

_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

REFUSAL_MESSAGE = (
    "I couldn't find anything in the uploaded documents that answers that "
    "question, so I can't answer it reliably. Try rephrasing, or upload a "
    "document that covers this topic."
)


@lru_cache
def _guard_client():
    """Groq client for the Prompt Guard classifier, or None if unavailable."""
    settings = get_settings()
    if not settings.guard_enabled or not settings.groq_api_key:
        return None
    try:
        from groq import Groq

        return Groq(api_key=settings.groq_api_key)
    except Exception:  # noqa: BLE001 - fall back to the regex heuristics
        logger.exception("Prompt Guard client unavailable; using regex fallback")
        return None


def injection_score(text: str) -> float | None:
    """P(prompt injection) from Prompt Guard 2, or None if it cannot be reached.

    The regex list this replaces failed in both directions: it missed anything
    but one literal phrasing ("ignore all prior directives", leetspeak, other
    languages) while blocking legitimate questions about documents that merely
    discuss prompts or jailbreaks. The classifier separates those cleanly.
    """
    client = _guard_client()
    if client is None or not text.strip():
        return None
    try:
        settings = get_settings()
        completion = client.chat.completions.create(
            model=settings.guard_model,
            messages=[{"role": "user", "content": text[:GUARD_MAX_CHARS]}],
        )
        return float(completion.choices[0].message.content)
    except Exception:  # noqa: BLE001 - never fail a request on the classifier
        logger.exception("Prompt Guard call failed; using regex fallback")
        return None


def looks_like_injection(text: str) -> tuple[bool, float | None]:
    """(is_injection, score). Falls back to the regex list when the model is out."""
    score = injection_score(text)
    if score is None:
        return bool(_INJECTION_RE.search(text)), None
    return score >= get_settings().guard_threshold, score


def check_input(question: str) -> tuple[bool, str]:
    """Return (allowed, reason). reason is non-empty only when blocked."""
    q = (question or "").strip()
    if not q:
        return False, "Empty question."
    if len(q) > MAX_QUESTION_CHARS:
        return False, f"Question exceeds {MAX_QUESTION_CHARS} characters."
    flagged, score = looks_like_injection(q)
    if flagged:
        detail = f" (score {score:.2f})" if score is not None else ""
        return False, f"Question looks like a prompt-injection attempt{detail}."
    return True, ""


def redact_secrets(text: str) -> tuple[str, bool]:
    """Redact secret-like tokens from output. Returns (clean_text, changed)."""
    changed = False
    for pattern, replacement in _SECRET_PATTERNS:
        text, n = pattern.subn(replacement, text)
        changed = changed or n > 0
    return text, changed


# A refusal is grounded by definition, but only when refusing is all it does.
# Treating any answer *containing* "I don't know" as grounded let a refusal
# prefix carry unsupported claims through ("I don't know. Also, the payout is
# $4,000,000."), so the answer counts as a refusal only when nothing survives
# removing its refusal sentences.
_REFUSAL_MARKERS = ("don't know", "do not know", "cannot find", "couldn't find")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]*")


def is_refusal(answer: str) -> bool:
    """True when the answer does nothing but refuse."""
    sentences = _SENTENCE_RE.findall(answer or "")
    if not sentences:
        return False
    remainder = [
        s
        for s in sentences
        if not any(m in s.lower() for m in _REFUSAL_MARKERS) and s.strip()
    ]
    return not remainder and any(
        m in (answer or "").lower() for m in _REFUSAL_MARKERS
    )


def is_grounded(answer: str, documents) -> bool:
    """Cheap grounding check: does the answer share vocabulary with context?

    Not a proof of faithfulness -- a fluent hallucination that reuses the
    document's vocabulary still passes -- but it catches answers that clearly
    ignore the retrieved context. Use the multi_agent pipeline's verifier when
    faithfulness actually matters.
    """
    lowered = answer.lower()
    if is_refusal(answer):
        return True
    context = " ".join(d.page_content for d in documents).lower()
    context_words = {w for w in re.findall(r"[a-z]{4,}", context)}
    answer_words = {w for w in re.findall(r"[a-z]{4,}", lowered)}
    if not answer_words:
        return True
    overlap = len(answer_words & context_words) / len(answer_words)
    return overlap >= 0.15

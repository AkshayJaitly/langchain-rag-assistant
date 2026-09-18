"""Spec 001: RRF maths and fusion ordering."""
from __future__ import annotations

from app.rag.retrieval import reciprocal_rank_fusion as rrf
from app.rag.retrieval import tokenize


def test_rrf_scores_follow_the_formula():
    scored = dict(rrf([["a", "b"]], k=60))
    assert scored["a"] == 1 / 61
    assert scored["b"] == 1 / 62


def test_agreement_between_retrievers_outranks_a_single_strong_hit():
    """AC-2: b is second in both lists, a is first in one and absent from the other."""
    fused = [doc for doc, _ in rrf([["a", "b"], ["c", "b"]], k=60)]
    assert fused[0] == "b"


def test_a_document_found_by_only_one_retriever_survives():
    fused = [doc for doc, _ in rrf([["a"], ["b"]], k=60)]
    assert set(fused) == {"a", "b"}


def test_ordering_is_deterministic_for_ties():
    """Equal scores must not reorder run to run, or evaluation drifts."""
    first = rrf([["x", "y"], ["y", "x"]], k=60)
    second = rrf([["x", "y"], ["y", "x"]], k=60)
    assert first == second


def test_tokenizer_keeps_alphanumeric_identifiers():
    """The whole point of BM25 here is literal tokens dense search misses."""
    assert "onek2j" in tokenize("Confirmation ONEK2J")
    assert "99" in tokenize("99.95% uptime")

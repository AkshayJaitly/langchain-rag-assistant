"""Spec 006: the metric maths, on fixed input."""
from __future__ import annotations

from eval.metrics import (
    hit_at_k,
    mean_reciprocal_rank,
    rate,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_at_k_respects_the_cutoff():
    assert hit_at_k(["a", "b", "c"], "c", 3) is True
    assert hit_at_k(["a", "b", "c"], "c", 2) is False


def test_reciprocal_rank_is_one_over_position():
    assert reciprocal_rank(["a", "b"], "a") == 1.0
    assert reciprocal_rank(["a", "b"], "b") == 0.5
    assert reciprocal_rank(["a", "b"], "z") == 0.0


def test_recall_and_mrr_over_a_small_set():
    results = [(["a", "b"], "a"), (["x", "y"], "y"), (["p"], "q")]
    assert recall_at_k(results, 2) == 2 / 3
    assert mean_reciprocal_rank(results) == (1.0 + 0.5 + 0.0) / 3


def test_empty_input_does_not_divide_by_zero():
    assert recall_at_k([], 3) == 0.0
    assert mean_reciprocal_rank([]) == 0.0
    assert rate([]) == 0.0

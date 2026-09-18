"""Metric maths, kept free of I/O so it can be tested directly (spec 006)."""
from __future__ import annotations


def hit_at_k(retrieved: list[str], expected: str, k: int) -> bool:
    return expected in retrieved[:k]


def recall_at_k(results: list[tuple[list[str], str]], k: int) -> float:
    """Share of cases whose expected document appears in the top k."""
    if not results:
        return 0.0
    return sum(hit_at_k(r, e, k) for r, e in results) / len(results)


def reciprocal_rank(retrieved: list[str], expected: str) -> float:
    for position, source in enumerate(retrieved, start=1):
        if source == expected:
            return 1.0 / position
    return 0.0


def mean_reciprocal_rank(results: list[tuple[list[str], str]]) -> float:
    if not results:
        return 0.0
    return sum(reciprocal_rank(r, e) for r, e in results) / len(results)


def rate(flags: list[bool]) -> float:
    return (sum(flags) / len(flags)) if flags else 0.0

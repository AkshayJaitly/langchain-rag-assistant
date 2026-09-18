"""Evaluation harness (spec 006).

    python -m eval.run --retrieval              # no model call; runs in CI
    python -m eval.run --guardrails             # needs GROQ_API_KEY
    python -m eval.run --answers                # needs a generation provider
    python -m eval.run --compare hybrid         # hybrid on vs off
    python -m eval.run --compare rerank
    python -m eval.run --compare pipeline

Retrieval and guardrail metrics are deterministic and need no judge, which is
what lets the retrieval suite gate CI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from app.config import get_settings
from app.rag import guardrails
from app.rag.retrieval import invalidate_caches
from app.rag.vectorstore import get_retriever
from eval.dataset import ATTACKS, BENIGN, GOLDEN
from eval.metrics import mean_reciprocal_rank, rate, recall_at_k


def _reload_settings() -> None:
    get_settings.cache_clear()
    get_retriever.cache_clear()
    invalidate_caches()


def _retrieved(question: str) -> tuple[list[str], list[str]]:
    """(ranked documents, ranked "file#page" chunks) for a question.

    No model call, so this is what CI can afford to run on every change.
    """
    from app.rag.retrieval import retrieve

    settings = get_settings()
    retriever = get_retriever()
    docs = retrieve(retriever, question, {settings.public_tenant})

    sources: list[str] = []
    chunks: list[str] = []
    for doc in docs:
        name = doc.metadata.get("source", "")
        if not name:
            continue
        if name not in sources:
            sources.append(name)
        key = f"{name}#{doc.metadata.get('page', 0)}"
        if key not in chunks:
            chunks.append(key)
    return sources, chunks


def _first_match_rank(retrieved: list[str], targets: set[str]) -> float:
    """Reciprocal rank of the first acceptable target, 0 when absent."""
    for position, item in enumerate(retrieved, start=1):
        if item in targets:
            return 1.0 / position
    return 0.0


def evaluate_retrieval(k: int = 4) -> dict:
    answerable = [c for c in GOLDEN if c.answerable and c.expects_source]
    doc_results, latencies = [], []
    page_hits, page_ranks, page_misses = [], [], []

    for case in answerable:
        started = time.perf_counter()
        sources, chunks = _retrieved(case.question)
        latencies.append(time.perf_counter() - started)

        doc_results.append((sources, case.expects_source))
        targets = case.targets
        if targets:
            hit = any(c in targets for c in chunks[:k])
            page_hits.append(hit)
            page_ranks.append(_first_match_rank(chunks, targets))
            if not hit:
                page_misses.append(case.question)

    return {
        "cases": len(answerable),
        f"doc_recall@{k}": round(recall_at_k(doc_results, k), 4),
        "doc_mrr": round(mean_reciprocal_rank(doc_results), 4),
        f"page_recall@{k}": round(rate(page_hits), 4),
        "page_mrr": round(sum(page_ranks) / len(page_ranks), 4) if page_ranks else 0.0,
        "median_latency_s": round(sorted(latencies)[len(latencies) // 2], 4),
        "page_misses": page_misses,
        "doc_misses": [
            expected for retrieved, expected in doc_results if expected not in retrieved[:k]
        ],
    }


def evaluate_guardrails() -> dict:
    blocked_attacks = [not guardrails.check_input(a)[0] for a in ATTACKS]
    blocked_benign = [not guardrails.check_input(b)[0] for b in BENIGN]
    return {
        "attacks": len(ATTACKS),
        "benign": len(BENIGN),
        "block_rate": round(rate(blocked_attacks), 4),
        "false_positive_rate": round(rate(blocked_benign), 4),
        "missed_attacks": [a for a, b in zip(ATTACKS, blocked_attacks) if not b],
        "wrongly_blocked": [b for b, f in zip(BENIGN, blocked_benign) if f],
    }


def evaluate_answers() -> dict:
    from app.rag.graph import answer_question

    settings = get_settings()
    matched, refusals_correct, latencies = [], [], []
    failures = []
    for case in GOLDEN:
        started = time.perf_counter()
        result = answer_question(case.question, tenants={settings.public_tenant})
        latencies.append(time.perf_counter() - started)
        answer = (result.get("answer") or "").lower()

        if case.answerable:
            hit = any(e.lower() in answer for e in case.expects) if case.expects else True
            matched.append(hit)
            if not hit:
                failures.append({"question": case.question, "answer": answer[:160]})
        else:
            refused = guardrails.is_refusal(result.get("answer") or "") or not result.get(
                "sources"
            )
            refusals_correct.append(refused)
            if not refused:
                failures.append({"question": case.question, "answer": answer[:160]})

    return {
        "answer_match": round(rate(matched), 4),
        "refusal_accuracy": round(rate(refusals_correct), 4),
        "median_latency_s": round(sorted(latencies)[len(latencies) // 2], 4),
        "failures": failures,
    }


CONFIGURATIONS = {
    "hybrid": [("hybrid on", {"HYBRID_RETRIEVAL": "true"}),
               ("dense only", {"HYBRID_RETRIEVAL": "false"})],
    "rerank": [("rerank off", {"RERANK": "false"}),
               ("rerank on", {"RERANK": "true"})],
    "pipeline": [("simple", {"PIPELINE": "simple"}),
                 ("multi_agent", {"PIPELINE": "multi_agent"})],
}


def compare(dimension: str, with_answers: bool) -> dict:
    """Run a metric across configurations (AC-5)."""
    out = {}
    for label, env in CONFIGURATIONS[dimension]:
        previous = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        _reload_settings()
        try:
            out[label] = (
                evaluate_answers() if with_answers else evaluate_retrieval()
            )
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            _reload_settings()
    return out


def _print(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    for key, value in payload.items():
        if isinstance(value, list) and not value:
            continue
        print(f"  {key:22} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG evaluation harness")
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--guardrails", action="store_true")
    parser.add_argument("--answers", action="store_true")
    parser.add_argument("--compare", choices=sorted(CONFIGURATIONS))
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="Exit non-zero if recall@k falls below this (used by CI).",
    )
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()

    if not any([args.retrieval, args.guardrails, args.answers, args.compare]):
        args.retrieval = True

    # The samples are the dataset, so make sure they are indexed.
    from app.rag.seed import seed_samples

    seed_samples()

    report: dict = {}
    if args.retrieval:
        report["retrieval"] = evaluate_retrieval(args.k)
        _print("retrieval", report["retrieval"])
    if args.guardrails:
        report["guardrails"] = evaluate_guardrails()
        _print("guardrails", report["guardrails"])
    if args.answers:
        report["answers"] = evaluate_answers()
        _print("answers", report["answers"])
    if args.compare:
        report["compare"] = compare(args.compare, with_answers=args.answers)
        for label, payload in report["compare"].items():
            _print(f"{args.compare}: {label}", payload)

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json_path}")

    if args.min_recall is not None:
        achieved = report.get("retrieval", {}).get(f"page_recall@{args.k}", 0.0)
        if achieved < args.min_recall:
            print(
                f"\nFAIL: page_recall@{args.k} {achieved} is below the floor "
                f"{args.min_recall}"
            )
            return 1
        print(f"\nOK: page_recall@{args.k} {achieved} >= {args.min_recall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

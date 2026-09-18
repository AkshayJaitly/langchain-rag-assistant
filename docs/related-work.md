# Related work

## Why this document exists

Every significant design decision in this project has prior art. This document
maps each one to the literature it sits next to, so the design can be discussed
in terms of what is known rather than what seemed reasonable at the time.

**It is positioning, not a novelty claim.** Nothing here is a research
contribution. Where this project does something that looks distinctive — ingest
-time injection screening, value-shadowed distractors in the evaluation set —
the literature got there first, and the relevant papers are named below.

**How this was compiled, and its limits.** These references were gathered from
literature search and read as abstracts and summaries, not in full. They are
accurate enough to orient a design discussion and *not* sufficient to support a
publication claim. Anyone extending this work should read the primary sources;
summaries are exactly where novelty claims go wrong.

---

## 1. Hybrid retrieval and rank fusion

**What this project does.** BM25 and dense retrieval run over the same child
chunks, and their rankings are combined with Reciprocal Rank Fusion,
`score(d) = Σ 1/(k + rank_i(d))` with `k = 60`
([spec 001](../specs/001-hybrid-retrieval.md)).

**Prior art.** RRF is Cormack, Clarke and Buettcher, *Reciprocal Rank Fusion
Outperforms Condorcet and Individual Rank Learning Methods* (SIGIR 2009), and
`k = 60` is their constant, carried forward largely unexamined by everyone since
— including here. Sparse/dense hybrid retrieval is standard practice rather than
a technique this project selected.

**Positioning.** Textbook application. The only locally interesting part is the
measurement in §2.

---

## 2. Evaluation: corpus saturation and distractor construction

**What this project does.** The evaluation corpus adds three *shadow* documents
to the three demo documents — a competing service agreement with its own
uptime-credit table, the previous quarter's metrics, a second policy. Same
template and vocabulary, different values. Scoring moved from document level to
page level because document level saturated.

The effect on the ability to tell two retrieval configurations apart:

| Corpus | Hybrid page MRR | Dense-only page MRR | Separable? |
| --- | --- | --- | --- |
| 3 documents, no distractors | 0.964 | 0.929 | no — within noise |
| 6 documents, value-shadowed | 0.879 | 0.731 | yes |

**Prior art.**
- [SeedRG](https://arxiv.org/html/2605.08838v1) generates leakage-free
  benchmarks and reports that they "restore discriminative power, revealing
  performance differences across RAG systems that are invisible on existing
  benchmarks." That is the same phenomenon, established properly.
- [EnterpriseRAG-Bench](https://arxiv.org/pdf/2605.05253) constructs
  "near-duplicates with conflicting facts... with specific facts changed" —
  the shadow-document construction used here, done first and at scale.
- [The Power of Noise](https://arxiv.org/pdf/2401.14887) measures accuracy
  degradation as distracting documents accumulate.
- [StratRAG](https://arxiv.org/abs/2604.22757) pairs gold documents with
  topically related distractors in a fixed pool.

**Positioning.** An independent, small-scale reproduction of a known result.
Useful as evidence that this project's own numbers are not self-flattering; not
a contribution.

---

## 3. Indirect prompt injection and ingest-time screening

**What this project does.** Uploaded documents are classified page by page at
ingest with Prompt Guard 2; flagged passages are excluded from retrieval and the
exclusion is surfaced in the response. Screening the question alone leaves the
document as an unguarded path into the prompt
([spec 004 context](../README.md#guardrails)).

**Prior art.**
- [Indirect Prompt Injection in the Wild for LLM Systems](https://arxiv.org/pdf/2601.07072)
  describes ingestion-time defenses directly, including sanitisation and
  attribution-gated answers.
- [A Layered Security Framework Against Prompt Injection in RAG-Based Chatbots](https://arxiv.org/pdf/2606.19660)
  proposes intercepting both direct and indirect injection across the pipeline.
- [Defending against Indirect Prompt Injection by Instruction Detection](https://arxiv.org/abs/2505.06311)
  detects injection from the model's own behavioural states rather than the text.
- [Defending Retrieval-Augmented Intrusion Detection Against Knowledge Poisoning and Prompt Injection](https://arxiv.org/abs/2608.08100)
  combines trust scoring, consistency checking and sanitisation at the retrieval
  boundary.

**Positioning.** A single-layer instance of a defense pattern the literature
treats as one layer among several. This project has no detection at generation
time, no output attribution gating, and no trust scoring.

**One local measurement worth keeping.** False-positive rate on ordinary
documents is very low: across the sample corpus and a set of real documents
(a signed NDA, flight itineraries, a conference slide deck), page-level injection
scores ranged from 0.0005 to 0.009 against a 0.5 threshold. That matters
practically — a screen that silently excluded real content would be worse than no
screen — but it is an observation on ~30 pages, not a study.

---

## 4. Abstention and grounded refusal

**What this project does.** The pipeline refuses when retrieval returns nothing,
the prompt forbids answering outside the context, and refusal accuracy is part of
the evaluation suite. Measured on the golden set: 1.00 refusal accuracy on
unanswerable questions, 0/8 refusals on answerable ones.

**Prior art.**
- [AbstentionBench](https://arxiv.org/abs/2506.09038) evaluates abstention over
  20 datasets and ~35k unanswerable questions, and reports that abstention is
  unsolved and does not improve with scale.
- [RefusalBench](https://arxiv.org/html/2510.10390) generates selective-refusal
  evaluations for grounded models.
- [Prompt-Based Abstention Fails Under Misleading Context](https://arxiv.org/html/2608.22228)
  finds models still answer 41.6% of misleading questions under explicit
  abstention prompting, with 63% of those echoing a planted wrong entity.
- [GRACE](https://arxiv.org/pdf/2601.04525) trains grounding and abstention with
  reinforcement learning rather than prompting.

**Positioning.** This project's refusal is prompt-based, which is precisely the
approach the fourth paper shows to be fragile under misleading context. Its
1.00 refusal accuracy is measured on questions that are *absent* from the corpus,
which is the easy case. See §7.

---

## 5. Chunking

**What this project does.** Parent/child ("small-to-big") splitting by default,
with semantic chunking available behind `CHUNKING=semantic` and deliberately not
the default, because embedding every sentence is expensive on a 512 MB host
([spec 005](../specs/005-chunking.md)).

**Prior art.**
- [Chunking Methods on RAG — Effectiveness Against Computational Cost](https://arxiv.org/abs/2606.00881)
  benchmarks 36 segmentation methods and finds sentence chunking the most
  cost-effective, matching semantic chunking up to roughly 5k tokens.
- [Evaluating Chunking Strategies for RAG on Academic Texts](https://arxiv.org/abs/2607.01852)
  finds semantic chunking's gains are not always proportional to its cost.
- [A Systematic Analysis of Chunking Strategies for Reliable Question Answering](https://arxiv.org/abs/2601.14123).

**Positioning.** The literature supports the default chosen here — for a corpus
this size, the cheaper splitter is the right one and semantic chunking is not
obviously worth its cost. The decision was made on deployment constraints and
happens to agree.

---

## 6. Running under a hard resource budget

**What this project does.** Everything fits a 512 MB instance: FastEmbed ONNX
instead of torch, page-at-a-time ingestion after the container was OOM-killed,
single-threaded ONNX, and an LLM reranker instead of a cross-encoder — because
the cross-encoder does not fit.

**Prior art.**
- [EdgeRAG](https://arxiv.org/html/2412.21023) — online indexing and
  memory-aware embedding caching for tight memory budgets.
- [Robust Implementation of RAG on Edge-based Computing-in-Memory Architectures](https://ieeexplore.ieee.org/document/11126301/)
  (IEEE/ACM ICCAD) — noise-aware training for CiM-backed retrieval.
- [Budget-Constrained Online RAG: Chunk-as-a-Service](https://arxiv.org/html/2604.26981)
  — cost-aware retrieval under a fixed budget.

**Positioning.** The academic work targets on-device and hardware-level
constraints; this project's constraint is the economics of a free cloud tier.
Related in spirit, different in kind, and not a gap worth a paper.

---

## 7. What this project has not measured

Stated explicitly because it is the most interesting open question in the repo,
and it was avoided rather than answered.

The golden questions were **deliberately disambiguated** — "*Under the Acme
agreement*, what is the service credit if uptime drops to 97%?" — so that the
shadow document could not be a legitimate answer. The unmeasured question is the
one that was dodged: when a question is *ambiguous* across two near-identical
contracts, how often does the system answer confidently from the wrong one, and
does the page-level citation let a reader catch it?

That failure mode is named in the literature —
[A-RAG](https://arxiv.org/pdf/2602.03442) reports entity confusion as the largest
secondary failure mode (40% on MuSiQue, 71% on 2Wiki) — and it is the realistic
enterprise case, where a document repository holds hundreds of agreements from
one template. The experiment is cheap to run here and has not been run.

---

## What this project claims

Only that it is a working, measured, documented system: the design decisions are
defensible, the quality claims come from a dataset rather than a demo, and the
constraints that shaped it are written down. Where it resembles published
research, it is applying that research, not extending it.

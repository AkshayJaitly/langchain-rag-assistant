# Specs

Each spec states what a capability must do, the constraints it works under, and
numbered acceptance criteria (`AC-n`). Tests reference those IDs, so a spec, its
implementation and its tests can be checked against each other.

Every spec inherits two project-wide constraints:

- **C-1 Zero cost.** Everything runs on free tiers: Render free (512 MB, sleeps
  when idle, ephemeral disk), GitHub Pages, GitHub Actions, and Groq's free API.
  No paid model, no managed vector service, no GPU.
- **C-2 Offline tests.** The automated suites never call a network service.
  Anything that needs a provider is a manual or opt-in run.

| Spec | Capability | Status |
| --- | --- | --- |
| [001](001-hybrid-retrieval.md) | Hybrid retrieval (BM25 + dense, fused with RRF) | implemented |
| [002](002-reranking.md) | Reranking | implemented (opt-in) |
| [003](003-conversation-memory.md) | Multi-turn conversations | implemented (revised — see the spec) |
| [004](004-tenant-isolation.md) | Per-visitor document isolation | implemented |
| [005](005-chunking.md) | Chunking strategies | implemented (semantic opt-in) |
| [006](006-evaluation.md) | Evaluation harness | implemented |
| [007](007-test-strategy.md) | Test strategy | implemented |

# Proposal: Optimizing A-MEM's Memory-Management Prompts with GEPA

## Research question

A-MEM ("Agentic Memory for LLM Agents", NeurIPS 2025) manages an agent's long-term
memory through two LLM-driven steps: **note construction** (turning a raw
interaction into a structured note with context/keywords/tags) and **memory
evolution** (deciding how a new note updates or links to existing notes). Both
steps run on hand-written prompts from the original paper.

GEPA ("Reflective Prompt Evolution Can Outperform Reinforcement Learning") is a
prompt/text optimizer that evolves candidates using LLM reflection over full
execution traces, keeping a Pareto front of candidates rather than collapsing
to a single scalar-reward search.

**Question:** does replacing A-MEM's hand-written note-construction and
memory-evolution prompts with GEPA-optimized versions improve downstream
memory-QA performance, and by how much relative to the cost of running the
optimization?

## Hypothesis

GEPA-optimized prompts will improve QA accuracy on held-out LoCoMo
conversations relative to A-MEM's original prompts, with the largest gains on
categories the original prompts handle poorly (expected: multi-hop, temporal,
adversarial) rather than uniformly across all categories.

## Success criteria (fixed before running experiments)

1. Statistically meaningful improvement (bootstrap CI on the test-set QA pairs
   does not overlap the baseline's CI) on the aggregate LoCoMo metric.
2. No category regresses beyond noise relative to baseline — a net win that
   trades adversarial-refusal accuracy for single-hop accuracy, for example,
   does not count as a clean success.
3. The $/token cost of the GEPA optimization run is reported alongside the
   accuracy delta, so "was it worth it" is answerable, not just "did it help."

## Non-goals

- Not attempting to beat other memory systems (Mem0, MemGPT, etc.) — the
  comparison is A-MEM-with-original-prompts vs. A-MEM-with-GEPA-prompts only.
- Not modifying A-MEM's retrieval/vector-store mechanics, only the two LLM
  prompts described above (see [decisions/0002](decisions/0002-optimization-targets.md)).
- Not fine-tuning any model weights — this is a prompt-space study only.

## Scope for v1

- Benchmark: LoCoMo only (see [decisions/0001](decisions/0001-base-repo-and-benchmark.md)).
  LongMemEval is an explicit follow-up once the pipeline is validated.
- Baseline: A-MEM's original paper prompts, unmodified
  (see [decisions/0002](decisions/0002-optimization-targets.md)).

## Milestones

1. **Dataset** — fetch LoCoMo, implement `datasets/locomo.py`, generate and
   commit the split manifest (see [decisions/0004](decisions/0004-train-val-test-split.md)).
2. **Baseline + reproduction sanity check** — implement `metrics.py`
   (the paper reports ROUGE-L on LoCoMo, not plain accuracy, so match that),
   reproduce the paper's numbers through our own harness on `Llama 3.2:1b`
   (see [decisions/0005](decisions/0005-reproduction-model.md)) as a
   correctness check before trusting the harness for anything else, then run
   the actual A-MEM-original-prompts baseline on the test split.
3. **Decide the GEPA trainset composition** — how `AMemGEPAAdapter.evaluate()`
   samples from the 5 train conversations per rollout batch. Deferred; not
   yet an ADR, revisit before starting milestone 4.
4. **GEPA adapter + optimization run** — fill in `gepa_adapter.py`, run a
   confirmed-budget GEPA optimization (see CLAUDE.md's "Cost awareness").
5. **Comparative analysis** — baseline vs. GEPA-optimized on the held-out
   test split, per-category deltas, cost accounting, checked against the
   success criteria above.
6. **Stretch** — LongMemEval generalization check; rotating-split robustness
   check (see [decisions/0004](decisions/0004-train-val-test-split.md)).

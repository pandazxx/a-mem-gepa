# 0005: Model for the paper-reproduction sanity check

## Status

Accepted (2026-09-06)

## Context

Milestone 2 includes reproducing the original A-MEM paper's LoCoMo results
through our own harness, as a sanity check that the harness itself is
correct before it's trusted for the GEPA comparison. The paper reports
numbers (F1 and BLEU on LoCoMo per category, not ROUGE-L -- corrected in
docs/decisions/0010 after an earlier secondary summary got this wrong) for
six backbones: `gpt-4o-mini`, `gpt-4o`, `Qwen2.5:3b`, `Qwen2.5:1.5b`,
`Llama 3.2:3b`, `Llama 3.2:1b`. This is a sanity check, not the headline
result, so the goal is minimum effort/cost to confirm the harness
reproduces the right shape of result — not picking the backbone that will
eventually carry the GEPA comparison.

## Decision

Use **Llama 3.2:1b** (Ollama tag `llama3.2:1b`) for the reproduction sanity
check:

- Smallest model in the paper's own benchmarked set — cheapest and fastest
  to run locally, no API cost.
- The paper reports a number for exactly this model, so there's a concrete
  target to compare against rather than eyeballing plausibility.

**Known risk:** at 1B params, the structured-JSON outputs both A-MEM prompts
require (note construction, evolution) can be unreliable. If `evaluate()`
shows a high rate of JSON-parse failures rather than genuine QA misses,
that's a signal to fall back to `Qwen2.5:1.5b` (next smallest in the paper's
set) for this sanity check — not evidence the harness itself is broken.

## Consequences

- This model choice is scoped to the milestone-2 sanity check only. It does
  not determine which model(s) `task_lm`/`AMEM_LLM_MODEL` use for the actual
  GEPA optimization runs — that's a separate decision (docs/decisions/0003
  covers routing, not backbone choice for the optimization itself).
- If the fallback to Qwen2.5:1.5b is needed, update this ADR's status rather
  than silently switching models in config.

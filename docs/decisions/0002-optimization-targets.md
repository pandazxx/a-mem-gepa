# 0002: Which prompts GEPA optimizes

## Status

Partially superseded (2026-09-19) by
[0014](0014-gepa-targets-repro-pipeline.md). Originally accepted 2026-09-06.

- **No longer valid:** the two-prompt candidate (note construction + one
  evolution prompt) against the `agiresearch/A-mem` library. GEPA's candidate
  is the repro pipeline's four prompts (note construction + three evolution
  steps), see 0014.
- **Still in force:** the baseline is the paper's verbatim prompts with no
  hand-tuning pass; all prompts in a candidate are optimized jointly;
  retrieval/QA prompts are frozen.

## Context

A-MEM has (at least) two LLM-driven prompts: note construction and memory
evolution/linking. We need to decide whether GEPA optimizes one or both, and
what the fixed baseline is.

## Decision

- Optimize **both** the note-construction prompt and the memory-evolution/
  linking prompt. Run them as a joint optimization target (both prompts are
  part of the same GEPA "candidate" being evolved) rather than optimizing one
  with the other frozen, since the two steps interact — a note-construction
  prompt that changes what attributes get extracted changes what the
  evolution prompt has to work with.
- Baseline for comparison is **A-MEM's original paper prompts, verbatim,
  unmodified**. No hand-tuning pass on top of the original prompts before
  comparing — GEPA's gain is measured against zero human prompt-engineering
  effort, not against an already-improved starting point.
- Retrieval/query prompts (if any exist in A-MEM beyond vector similarity
  search) and the vector-store/embedding mechanics are out of scope for v1.

## Consequences

- A single GEPA "candidate" is a pair of prompt strings, not one. The
  `GEPAAdapter` implementation must mutate/score both together.
- Because we're not hand-tuning a middle baseline, we can't distinguish "GEPA
  found real improvements" from "any prompt-engineering effort would have
  found the same improvements" — noted as a limitation in the final report,
  not solved in v1.

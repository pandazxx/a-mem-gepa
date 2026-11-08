# 0001: Base A-MEM repo and benchmark choice

## Status

Partially superseded (2026-11-08) by
[0013](0013-paper-reproduction-pipeline.md) and
[0014](0014-gepa-targets-repro-pipeline.md). Originally accepted 2026-09-06.

- **No longer valid:** the premise that `WujiangXu/A-mem-sys` is the paper's
  reproduction code (it is the same library code as `agiresearch/A-mem`, see
  [0006](0006-qa-answer-prompt.md)) and that our own harness around
  `AgenticMemorySystem` would be the system under test. The paper's real
  reproduction code is `WujiangXu/AgenticMemory`, vendored at
  `external/agentic-memory-repro/` (0013); it is what `just reproduce`
  measures and what GEPA optimizes (0014).
- **Still in force:** LoCoMo as the v1 benchmark; `external/a-mem` stays
  vendored as a read-only submodule for the legacy `just demo`/`just baseline`
  development loop.

## Context

A-MEM has two public repos: `agiresearch/A-mem` (clean library, importable
`AgenticMemorySystem`) and `WujiangXu/A-mem-sys` (paper reproduction, includes
the original eval scripts against LoCoMo). We need one to vendor as the
system under test, and one benchmark to start with.

## Decision

- Vendor `agiresearch/A-mem` as a git submodule at `external/a-mem`, since we
  are building our own adapter/eval harness around `AgenticMemorySystem`
  rather than reusing a paper-reproduction script wholesale.
- Pull specific pieces from `WujiangXu/A-mem-sys` (e.g. LoCoMo loading/scoring
  logic) into `src/amem_gepa/datasets/` only if reimplementing them from
  scratch turns out to be wasted effort — evaluated case by case, not
  vendored wholesale.
- Benchmark: LoCoMo first. LongMemEval is an explicit v2 follow-up (see
  [../01-related-work.md](../01-related-work.md)) once the LoCoMo pipeline is
  validated.

## Consequences

- We own the adapter code that calls into `AgenticMemorySystem`, so upstream
  A-MEM changes need a submodule bump + adapter review, not a silent break.
- LoCoMo-only scope for v1 means all v1 conclusions are conditional on LoCoMo
  generalizing — flagged explicitly in the final report, not just implied.

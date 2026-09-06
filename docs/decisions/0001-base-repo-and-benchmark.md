# 0001: Base A-MEM repo and benchmark choice

## Status

Accepted (2026-09-06)

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

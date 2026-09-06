# 0006: A fixed, non-optimized QA-answering prompt is needed for evaluation

## Status

Accepted (2026-09-06)

## Context

Neither `agiresearch/A-mem` nor `WujiangXu/A-mem-sys` (checked directly --
they turned out to be the same library code, not a separate eval harness,
contrary to what [decisions/0001](0001-base-repo-and-benchmark.md) assumed)
ships a question-answering step. `AgenticMemorySystem.search`/`search_agentic`
only retrieve memory notes (pure ChromaDB/embedding similarity, no LLM call)
-- there's no "given these retrieved notes, answer the question" prompt
anywhere in either repo. To report QA accuracy/ROUGE-L at all, our own eval
loop needs to add that step.

## Decision

- Add a third prompt, `src/amem_gepa/prompts/qa_answer.txt`, used only by
  the evaluation loop (`src/amem_gepa/evaluate.py`): given the top-k
  retrieved memories for a question (via `search_agentic`) plus the question
  itself, produce an answer.
- This prompt is **not a GEPA optimization target** (docs/decisions/0002
  still stands: only note-construction and evolution are optimized). It's
  held fixed and identical across the baseline and every GEPA-optimized
  candidate, so a comparison isolates the effect of memory-management prompt
  quality, not QA-prompt quality. If the QA prompt were also mutable, an
  improvement could come from better answer phrasing rather than better
  memory notes, which would defeat the point of the study.
- Since this prompt is ours, not the paper's, it doesn't go under
  `prompts/baseline/` (which is reserved for A-MEM's verbatim original
  prompts, docs/decisions/0002) -- it lives directly under `prompts/`.

## Consequences

- The milestone-2 "reproduction" is a reproduction of A-MEM's *memory
  management*, not a bit-exact reproduction of the paper's full eval
  pipeline -- the paper's own QA-answering prompt (if it differs from ours)
  isn't publicly available in either repo to match against. Sanity-check
  expectations accordingly: matching the paper's reported *shape* of result
  (A-MEM's memory helps vs. not) matters more here than matching its exact
  number.
- Retrieval itself (`search_agentic`) costs no extra LLM calls -- only the
  QA-answer step and A-MEM's own note-construction/evolution steps do.

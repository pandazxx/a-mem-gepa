# 0011: Verbose tracing for `just demo`, `just baseline` stays clean

## Status

Accepted (2026-09-09)

## Context

Walking through `evaluate.py`/`amem_adapter.py` line by line (written up in
GitHub issue #4) explains the code, but doesn't show the actual note
construction/evolution decisions, retrieved memories, or QA-answer text a
real run produces -- those only exist as opaque LLM calls today. The user
asked for detailed tracing specifically for `just demo`, with `just
baseline` staying clean -- a real reproduction run processes ~600 questions
over many hours (docs/decisions/0005, 0007); per-call tracing at that scale
would be unreadable noise, not a debugging aid.

## Decision

- `PromptInjectableMemorySystem` (`amem_adapter.py`) gets a `trace: bool =
  False` constructor param. When true:
  - `analyze_content()` prints the content being analyzed and the resulting
    keywords/context/tags (it already parses these for its own use, so this
    is a direct print, no extra call).
  - `add_note()` temporarily swaps `self.llm_controller.llm` for
    `_EvolutionTraceBackend` for the duration of `super().add_note()` --
    the only call upstream's `process_memory()` can make -- printing the
    parsed evolution decision (`should_evolve`, `actions`,
    `suggested_connections`, `tags_to_update`). Skips (with a one-line
    print) when there are no existing memories yet, since upstream skips
    the evolution call itself in that case. Tracing parse failures never
    propagate -- upstream already handles unparseable evolution responses
    by recovering, tracing shouldn't be the reason a call fails.
- `evaluate.py` gets a matching `trace: bool = False` threaded through
  `evaluate_candidate` -> `replay_conversation` (prints a turn header) and
  `answer_question` (prints the retrieved memories and the final QA-answer
  text). `_default_memory_system_factory` passes `trace` through to
  `PromptInjectableMemorySystem`; a caller-supplied `memory_system_factory`
  (as tests use) doesn't get this automatically, which is fine -- trace is
  a real-system debugging aid, not a general contract.
- **Trace mode bypasses the checkpoint/prediction cache reads** (still
  writes). Without this, a demo rerun -- the common case, since `just demo`
  is meant to be rerun cheaply -- would print nothing, because everything
  is already cached. Tracing is a request to watch the process happen live;
  honoring the cache would defeat that.
- Only `scripts/run_demo.py` passes `trace=True` (default `True` there,
  overridable with `--no-trace`). `run_baseline.py`/`run_eval.py` never
  pass it, so `just baseline`/`just eval` are unaffected regardless of any
  future changes to the demo script.

## Consequences

- A traced `just demo` run always re-calls the LLM for every turn/question,
  even on a rerun -- small (~20-45 calls total at default settings), so
  this is a deliberate cost/visibility tradeoff, not a performance
  regression worth worrying about at this scale.
- `PredictionStore`'s JSONL can accumulate duplicate lines for the same
  (conversation, question) key after repeated traced reruns (last line
  wins on reload) -- harmless at demo scale, not something this adds
  deduplication for.

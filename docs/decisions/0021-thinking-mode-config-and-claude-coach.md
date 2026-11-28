# 0021: Configurable thinking mode, and Claude as the coach through the official SDK

## Status

Accepted (2026-11-22). Refines [0016](0016-disable-ollama-thinking-mode.md)
(the task model's `think=False` becomes a config default instead of a
constant) and the reflection-model half of [0003](0003-model-routing.md)
(Ollama and Anthropic coaches get native clients; LiteLLM stays the
fallback for everything else).

## Context

Stage 1 of the sequential training ([experiments/005](../experiments/005-gepa-stage1-note-construction-qwen3.5-9b.md))
ran both the task model and the coach on the local `qwen3.5:9b`. Two
things about that setup were implicit rather than chosen:

1. **Thinking mode was hard-wired.** 0016 forces `think=False` on every
   Ollama call the *task* model makes, for good reason (a ~100x per-call
   slowdown with no measured benefit for keyword extraction). But it is a
   constant in `_patch_ollama_chat`, not a setting: there is no way to run
   the comparison the other way and find out whether a thinking task model
   builds a better memory. Meanwhile the *coach* went through LiteLLM with
   no `think` parameter at all, so it ran with the model's default, which
   for qwen3.5 is thinking **on**. 004's doc had to note this from
   inference ("thinking on, not covered by the patch").
2. **The coach was the same 9B model as the task.** 004 listed "a stronger
   reflection model" as the cheapest change that targets proposal quality
   rather than budget: the coach makes about one call per GEPA iteration
   (14 calls in all of stage 1), so a hosted frontier model costs cents to a
   few dollars per stage. `.env.example` already named an Anthropic model
   as the intended coach, but nothing in the repo configured it, metered
   it, or handled the current Claude API's request shape (adaptive
   thinking and `effort`; `temperature` and `budget_tokens` are rejected on
   the current generation).

## Decision

- **One knob for the task model's thinking mode: `paper_repro.thinking`**
  (default `false`, i.e. 0016 unchanged). `_patch_ollama_chat(think=...)`
  reads it; `just reproduce`, `just k-sweep`, GEPA builds/QA and
  `just gepa-test-eval` all pass the same value, so baseline and candidate
  arms can never run under different generation settings by accident.
  The adapter keys the build cache on it (`<candidate-hash>-think/`): the
  same prompts build a different memory with thinking on, and a cached
  thinking-off build must never be served to a thinking-on run.
- **The coach is built by `reflection_lm.build_reflection_lm`** from
  `GEPA_REFLECTION_LM` plus a new `gepa.reflection` config block
  (`thinking`, `effort`, `max_tokens`, `max_cost_usd`, `fallbacks`),
  dispatching on the provider prefix:
  - `ollama/…`: the native `ollama` client, the same package the task
    model uses, so `think` is an explicit request parameter. `null` keeps
    the model's default (what 004/005 ran); `true`/`false` pins it.
  - `anthropic/…`: the official `anthropic` SDK, not LiteLLM. Requests use
    adaptive thinking with `output_config.effort` (default `high`), no
    `temperature`, no token budget; `thinking: false` maps to effort
    `low` with a warning because the current Claude generation cannot
    switch thinking off. The server-side refusal fallback is on by default
    per the Claude API guidance and can be disabled; a fallback is logged
    with the model that actually answered so the experiment doc can
    record it. A refusal with no fallback raises; gepa logs it and moves
    on (`raise_on_exception=False`). Default model when the env var points
    at Anthropic: `claude-opus-5-5`.
  - anything else: gepa's LiteLLM-backed `LM`, as before.
- **Coach cost is metered and capped.** Both native backends count
  tokens; the Anthropic one prices them from a small table
  (`ANTHROPIC_PRICING_USD_PER_MTOK`, first-party rates). gepa's
  `max_reflection_cost` is wired to `gepa.reflection.max_cost_usd`
  (default $25, a safety net well above a stage's expected $1–3), and
  `summary.json` gains `reflection_calls`, `reflection_tokens_in/out`,
  `reflection_cost_usd` plus the full reflection settings and
  `task_thinking`, so CLAUDE.md's "Cost" section can be filled from one
  file. The run plan prints the coach line and the task-thinking line
  before asking for `--yes`.
- **Every coach call is logged** (`[coach] call N -> model ... / done in Ns:
  tokens`), and the Ollama coach reports after its first call where Ollama
  placed the model (`ollama ps`: size vs. VRAM share, a warning when it is
  split to CPU). `gepa.reflection.request_timeout_s` (Ollama only; default
  none, the client's own default) bounds a single call so a CPU-spilled
  27B that would take an hour fails instead of hanging. Added after the
  first 007 attempt stalled silently at the first coach call.
- **Credentials stay in the environment** (`ANTHROPIC_API_KEY`, or an
  `ant auth login` profile the SDK picks up); never in configs or docs.

## Consequences

- **Comparability.** Any run with `paper_repro.thinking: true` is a
  different system under test from every reported run so far; its
  experiment doc must say so, and `gepa-test-eval` must be run with the
  same setting for both arms (it is, by construction). Switching the
  coach changes the search, not the system under test, so stage results
  with different coaches remain comparable on the held-out split; the
  experiment doc names the coach either way.
- **The Ollama coach path changed client** (LiteLLM → native) even at
  default settings. The request is the same chat call with the model's
  default sampling; 004/005 are not invalidated, but the first Ollama-coach
  run after this should confirm iteration time is unchanged.
- **Cost of a Claude coach**, from stage 1's shape (14 iterations, one
  reflection prompt of ~4–8K tokens in and ~1–2K out each): roughly $0.5–1.5
  per stage on `claude-opus-5-5` ($4/$20 per MTok). The `max_cost_usd` cap
  turns a surprise into a stopped run rather than a bill.
- **Thinking task model cost**, if anyone turns it on: 003 measured ~100x
  the eval tokens per call on qwen3.5:9b; at stage 1's 1,700 calls per
  build, expect builds in the many-hours range and plan the budget
  accordingly. The setting exists so that experiment can be run
  deliberately, not as a recommendation.
- `litellm` is no longer needed for an Ollama or Anthropic coach; it is
  still a dependency for the legacy pipeline and the LiteLLM fallback.

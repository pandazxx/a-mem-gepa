# 0016: Disable Ollama's thinking mode for the paper-reproduction pipeline

## Status

Accepted (2026-11-06)

## Context

A local `qwen3.5:9b` k-sweep (`k=40,50`) that should have been a few hours
took more than 15 days on an 8GB RTX 3070, then still 3+ days after
upgrading to a 32GB Tesla V100. Hardware was systematically ruled out
across several real diagnostic steps (all against the live machine, not
guessed):

- `ollama ps` → `100% GPU` — not a VRAM/offload problem.
- `nvidia-smi` during an active call → 77% GPU utilization — not idle,
  not CPU-bound.
- `ollama run --verbose` → 72.92 tok/s eval rate — a healthy, normal
  generation speed for a 9B quantized model, ruling out "old GPU
  architecture lacks efficient quantized-matmul kernels."
- Added per-call Ollama timing + content-preview logging (this PR,
  `_patch_ollama_chat`, originally `_patch_ollama_chat_for_timing`) to
  `src/amem_gepa/paper_repro.py`: real runs showed `unaccounted=0.0s` and
  a steady ~73 tok/s throughout — confirming the GPU was doing exactly
  what it should, just generating far more tokens than the task needed
  (1,200–15,000+ eval tokens for what should be a short structured
  extraction or a direct QA answer).

The root cause: `ollama show qwen3.5:9b` reports

```
thinking
    levels     false, true
    default    true
```

`qwen3.5:9b` generates an extended reasoning block before every answer
unless told not to — confirmed visually too (`ollama run qwen3.5:9b "What
is 2+2?"` prints a multi-paragraph "Thinking..." block before "Four").
The installed `ollama` Python package's `chat()` has an explicit
`think: bool | str | None` parameter for this; the vendored
`RobustOllamaController.get_completion`
(`external/agentic-memory-repro/memory_layer_robust.py`, read-only) never
sets it, only `temperature`.

A live side-by-side comparison on the same prompt, same machine:

| | eval_count | eval_duration | output |
|---|---|---|---|
| default (thinking on) | 1,115 | 15.3s | `Audrey, dog, fetch, park, playing` |
| `think=False` | 11 | 0.1s | `Audrey, dog, fetch, park, play` |

~100x fewer tokens, ~150x faster, functionally identical output for this
task.

## Decision

Extend the existing `ollama.chat()` monkeypatch
(`src/amem_gepa/paper_repro.py:_patch_ollama_chat`, `backend="ollama"`
only) to also inject `kwargs.setdefault("think", False)` before calling
the real `ollama.chat`. `setdefault`, not an unconditional override: the
vendored code never passes `think` today, so this is the only thing that
changes in practice, but a future/different caller that explicitly sets
it keeps that choice.

None of the paper's six originally-benchmarked models (GPT-4o-mini,
GPT-4o, Qwen2.5-1.5b, Qwen2.5-3b, Llama-3.2-1b, Llama-3.2-3b) are
reasoning models. Disabling thinking for `qwen3.5:9b` — which isn't one
of those six anyway (docs/decisions/0013) — keeps this project's
comparisons consistent with the paper's own fast, direct
instruction-following methodology rather than introducing a new kind of
model behavior the paper never accounted for.

## Consequences

- Scoped to `backend="ollama"` only, since `think` is an Ollama-specific
  API concept. `openai`/OpenRouter runs (GPT-4o-mini, experiment 002) are
  entirely unaffected — `_patch_ollama_chat` is never installed for that
  backend.
- Harmless for models without thinking capability: `llama3.2:1b`
  (experiment 001, M2's original reproduction) has no thinking mode at
  all, so `think=False` is simply ignored by Ollama for that model —
  001's result stands as-is, not invalidated by this change.
- **Any already-cached `qwen3.5:9b` progress predates this fix and was
  generated with thinking *on*.** Memory notes built, and QA answers
  computed, under `results/paper_repro/cached_memories_ollama_qwen3.5:9b/`
  and `qa_progress_ollama_qwen3.5:9b_k*.jsonl` (plus any
  `k_sweep/results_*qwen3.5:9b*.json` already written) are not
  representative of what a clean `think=False` run would produce.
  Reusing them would silently mix thinking-mode memory construction with
  non-thinking QA-answering (or vice versa, depending on what's cached at
  which layer). Recommended: clear those paths for `qwen3.5:9b`
  specifically (not the `gpt-4o-mini`/`llama3.2:1b` caches, which are
  unaffected and still valid) and rerun from scratch now that it should
  be dramatically cheaper.
- Not yet measured: the actual end-to-end speedup for a full `qwen3.5:9b`
  run with this fix. The isolated comparison above is for one short
  keyword-extraction prompt; real prompts (especially QA-answering at
  `k=50`, with 20,000+ prompt tokens) will have a different, but still
  expected to be large, ratio. A fresh run with the logging from this PR
  will show the real number.

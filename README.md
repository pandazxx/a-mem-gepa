# a-mem-gepa

Studying whether [GEPA](https://github.com/gepa-ai/gepa)'s reflective prompt
evolution improves [A-MEM](https://github.com/agiresearch/A-mem)'s
memory-management prompts (note construction, memory evolution/linking),
measured on the LoCoMo long-term-conversation benchmark against A-MEM's
original, unmodified paper prompts.

See `docs/00-proposal.md` for the research question, hypothesis, and success
criteria, and `docs/decisions/` for why the base repos, benchmark, prompt
scope, model routing, dataset split, and evaluation methodology were chosen
the way they were — especially `docs/decisions/0013` if you're wondering why
there are two vendored A-MEM repos and two evaluation pipelines below. See
`CLAUDE.md` for how the repo is organized and the working conventions.

## Status

Milestone 1 (dataset) and milestone 2 (baseline reproduction) are in
progress on `topic/m2-baseline-reproduction`. The GEPA adapter itself
(`gepa_adapter.py`'s `evaluate()`/`make_reflective_dataset()`) is still
stubbed — milestone 3.

## Setup

```
direnv allow          # or: nix develop
just setup            # uv sync
cp .env.example .env  # fill in model routing + API keys
just --list           # see available commands
```

This repo has **two** git submodules — clone with
`git submodule update --init --recursive` if either is empty:

- `external/a-mem` (`agiresearch/A-mem`) — the general-purpose memory
  *library*. `just demo`/`just baseline`/`just eval` are built on this,
  via `src/amem_gepa/amem_adapter.py`; these are what GEPA will optimize
  prompts against in milestone 3+.
- `external/agentic-memory-repro` (`WujiangXu/AgenticMemory`) — the
  paper's *actual reproduction code* (its own README says to use it, not
  the library, to reproduce reported numbers). `just reproduce` is built
  on this, via `src/amem_gepa/paper_repro.py`. See `docs/decisions/0013`
  for why these two are meaningfully different pipelines, not redundant.

## Running something

- `just demo` — fast (~minutes) end-to-end sanity check: one truncated
  conversation, a few questions, verbose tracing of note construction/
  evolution/retrieval/answer by default. Start here.
- `just baseline` — A-MEM-library pipeline, original unmodified prompts,
  on the committed test split (`configs/locomo_split.json`). Resumable;
  can take a long time against a local model.
- `just reproduce` — faithful, full-10-conversation reproduction of the
  paper's own benchmark table, via `external/agentic-memory-repro`. This
  is the number to compare against the paper, not `just baseline`'s.
- `just rescore results/<run>/<split>.json` — recompute an existing run's
  summary against the current `metrics.py`, no LLM calls.

## Layout

- `external/a-mem/` — vendored A-MEM library (read-only)
- `external/agentic-memory-repro/` — vendored paper-reproduction code (read-only)
- `src/amem_gepa/` — adapters for both, prompts, datasets, metrics
- `configs/` — run configs + the committed LoCoMo split manifest
- `scripts/` — CLI entry points, run via `just`
- `docs/` — proposal, related work, decisions, per-experiment write-ups

# a-mem-gepa

Studying whether [GEPA](https://github.com/gepa-ai/gepa)'s reflective prompt
evolution improves [A-MEM](https://github.com/agiresearch/A-mem)'s
memory-management prompts (note construction, memory evolution/linking),
measured on the LoCoMo long-term-conversation benchmark against A-MEM's
original, unmodified paper prompts.

See `docs/00-proposal.md` for the research question, hypothesis, and success
criteria, and `docs/decisions/` for why the base repo, benchmark, prompt
scope, model routing, and dataset split were chosen the way they were. See
`CLAUDE.md` for how the repo is organized and the working conventions.

## Status

Scaffolding only — the adapter code type-checks against A-MEM's real
interface (see `src/amem_gepa/amem_adapter.py`) and the baseline prompts are
verified to format correctly (`tests/`), but the dataset loader, metrics, and
the GEPA adapter's `evaluate()`/`make_reflective_dataset()` are not yet
implemented. Milestones tracked in `docs/00-proposal.md` and the current
project discussion.

## Setup

```
direnv allow          # or: nix develop
just setup            # uv sync
cp .env.example .env  # fill in model routing + API keys
just --list           # see available commands
```

`external/a-mem` is a git submodule (`agiresearch/A-mem`) — clone with
`git submodule update --init` if it's empty.

## Layout

- `external/a-mem/` — vendored upstream A-MEM (read-only)
- `src/amem_gepa/` — adapter, prompts, datasets, metrics
- `configs/` — run configs + the committed LoCoMo split manifest
- `scripts/` — CLI entry points, run via `just`
- `docs/` — proposal, related work, decisions, per-experiment write-ups

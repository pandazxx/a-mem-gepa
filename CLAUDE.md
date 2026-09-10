# a-mem-gepa

Research project: optimize A-MEM's memory-management prompts (note
construction, memory evolution/linking) with GEPA, and measure the effect on
LoCoMo QA performance vs. the paper's original prompts. Read
`docs/00-proposal.md` and `docs/decisions/` before making scope calls —
several open questions (base repo, benchmark, split strategy, model routing)
are already decided there; don't re-litigate them without a new ADR.

## Project layout

- `external/a-mem/` — `agiresearch/A-mem`, vendored as a git submodule. Treat
  as read-only upstream code; project logic goes in `src/amem_gepa/`, not here.
  This is the general-purpose *library*, not the paper's reproduction code —
  see `docs/decisions/0013` before assuming a number from it is comparable
  to the paper's benchmark table.
- `external/agentic-memory-repro/` — `WujiangXu/AgenticMemory`, vendored
  read-only the same way. This *is* the paper's actual reproduction code
  (see `docs/decisions/0013`); `just reproduce` is built on it.
- `src/amem_gepa/` — adapter code: `amem_adapter.py` (wraps
  `AgenticMemorySystem`), `llm/litellm_controller.py` (routes A-MEM's own LLM
  calls through LiteLLM), `gepa_adapter.py` (`GEPAAdapter` implementation),
  `metrics.py`, `datasets/` (LoCoMo loading + the train/val/test split),
  `paper_repro.py` (wraps `external/agentic-memory-repro/`, see 0013).
- `src/amem_gepa/prompts/` — prompt candidates as plain text files, one per
  optimization target. The original A-MEM baseline prompts live here too,
  under a `baseline/` subfolder, verbatim — never edit those in place.
- `configs/` — YAML run configs (models, budget, split) + `locomo_split.json`
  (the committed conversation-ID split manifest, see
  `docs/decisions/0004-train-val-test-split.md`).
- `scripts/` — CLI entry points, wrapped by `justfile` recipes.
- `results/` — gitignored raw run output. Never the source of truth for a
  result — `docs/experiments/NNN-*.md` is.
- `docs/experiments/` — one committed markdown file per run, written
  immediately after the run per `docs/experiments/README.md`'s template.

## Environment

- Nix flake devshell (`flake.nix`) provides non-Python deps; `direnv allow`
  (or `nix develop`) to enter it.
- Python deps via `pyproject.toml` — use `uv sync` inside the devshell.
- Secrets/model config via `.env` (gitignored), loaded from `.env.example`.
  Never commit a populated `.env` or paste API keys into code, configs, or
  experiment docs.
- All commands run through `just` recipes (see `justfile`) — prefer
  `just <recipe>` over calling scripts directly so cost/logging conventions
  stay consistent.

## Prompt versioning

Prompt candidates are plain text files under `src/amem_gepa/prompts/`, not
inline strings in Python. Every GEPA run's winning candidate gets saved as a
new file (`prompts/note_construction.v<N>.txt`, etc.) rather than overwriting
the previous version in place — we need the diff history for the experiment
docs.

## Cost awareness

GEPA optimization runs and any NIM/Anthropic-backed eval sweep cost real API
money and wall-clock time. Before starting one:

- Confirm the `max_metric_calls` budget and which models (`task_lm`/
  `reflection_lm`) are configured — state them back to the user and get an
  explicit go-ahead before kicking off a run, don't just launch one because a
  script exists.
- Prefer running long GEPA/eval jobs in the background (with a way to check
  progress) over blocking the conversation on them.
- Log token/cost totals for every run; they belong in that run's
  `docs/experiments/NNN-*.md` under "Cost," not just in raw logs.

## Git workflow

- One branch per experiment or per meaningful pipeline change; land the
  corresponding `docs/experiments/NNN-*.md` (or ADR, for a scope change) in
  the same PR as the code that produced it.
- The `external/a-mem`/`external/agentic-memory-repro` submodule pointers
  only move on a deliberate bump — call it out explicitly in the commit
  message, since it changes the system under test.
- Don't commit anything under `results/` or a populated `.env`.

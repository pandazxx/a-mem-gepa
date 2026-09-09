set dotenv-load := true

default:
    @just --list

# Install/sync Python dependencies
setup:
    uv sync

# Fast end-to-end sanity demo: one truncated conversation, a few questions
# (docs/decisions/0008) -- run this before `just baseline`, not instead of it
demo config="configs/base.yaml" max_turns="15" max_questions="5":
    uv run python scripts/run_demo.py --config {{config}} --max-turns {{max_turns}} --max-questions {{max_questions}}

# Evaluate A-MEM with the original, unmodified paper prompts
baseline config="configs/base.yaml":
    uv run python scripts/run_baseline.py --config {{config}}

# Run a GEPA optimization job (costs real API calls — confirm budget first)
gepa-optimize config="configs/base.yaml":
    uv run python scripts/run_gepa_optimize.py --config {{config}}

# Evaluate a specific prompt candidate on the test set
eval candidate config="configs/base.yaml":
    uv run python scripts/run_eval.py --candidate {{candidate}} --config {{config}}

# Regenerate the LoCoMo train/val/test split manifest (see docs/decisions/0004)
split:
    uv run python scripts/make_locomo_split.py --out configs/locomo_split.json

# Recompute a results/<run_label>/<split>.json's summaries with the current
# metrics.py, no LLM calls -- for when metrics.py changes after a run already
# finished (e.g. a scoring bug fix) and redoing the run isn't practical
rescore results_path:
    uv run python scripts/rescore.py {{results_path}}

test:
    uv run pytest

lint:
    uv run ruff check src scripts tests

fmt:
    uv run ruff format src scripts tests

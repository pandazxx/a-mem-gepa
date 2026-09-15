# 002: M2 baseline reproduction — GPT-4o-mini (via OpenRouter), full LoCoMo

- Date: 2026-09-13
- Commands: `uv run python scripts/run_paper_reproduction.py --ratio 0.1`
  (cost calibration only), then `uv run python scripts/run_k_sweep.py
  --k-values 40` and `--k-values 50` (run separately; commit: `d71f798`).
- Pipeline: `src/amem_gepa/paper_repro.py`, via `external/agentic-memory-repro`
  (`WujiangXu/AgenticMemory`), per
  [decisions/0013](../decisions/0013-paper-reproduction-pipeline.md).
- `PAPER_REPRO_BACKEND=openai`, `PAPER_REPRO_MODEL=openai/gpt-4o-mini`,
  routed through OpenRouter (`OPENAI_BASE_URL=https://openrouter.ai/api/v1`)
  rather than the real OpenAI API — see decisions/0013's OpenRouter
  addendum. Full 10-conversation dataset, n=1986 questions.

## What changed vs. the previous experiment

First reproduction run against a paid API backend (every prior run,
[001](001-m2-llama3.2-1b-full-reproduction.md), used local $0 Ollama).
Two real issues hit and fixed getting here, both in this project's own
wrapper code, not the vendored submodule or the paper's own code:

1. A `run_k_sweep` invocation got killed by exhausted OpenRouter credits
   mid QA-answering. Found `run_full_reproduction`'s QA-answering loop had
   zero checkpointing (unlike memory-building) — fixed with per-question
   resumability to `results_dir/qa_progress_*.jsonl` (commit `cad825e`).
2. That fix itself had a bug: `model="openai/gpt-4o-mini"` contains a `/`,
   which makes the progress file's path land in a nested directory
   (pathlib splits on `/`) that nothing had created yet — bare
   `FileNotFoundError` on the first answered question. Fixed with an
   explicit `mkdir(parents=True)` before every append (commit `d71f798`).

Also, per [001](001-m2-llama3.2-1b-full-reproduction.md)'s 2026-09-12
correction: read the paper directly (arxiv.org/pdf/2502.12110, Appendix
A.5, Table 8) and found GPT-4o-mini's own headline number does **not**
use a uniform `k` — it uses `k=40` for Multi Hop/Temporal/Adversarial and
`k=50` for Open Domain/Single Hop. `run_full_reproduction`/`run_k_sweep`
only support one uniform `k` per call, so getting the paper-matching
number means running both `k=40` and `k=50` and *splicing* the right
category from each — formalized as `splice_per_category_results`/
`format_spliced_summary` in `paper_repro.py` (this run), used by
`scripts/run_k_sweep.py` automatically when a model has a known
per-category `k` (`PAPER_CATEGORY_K`).

## Results

Raw sweep output (uniform `k` across the whole dataset, n=1986):

| k  | overall F1 | overall BLEU-1 |
|----|-----------|----------------|
| 40 | 0.4268    | 0.3685         |
| 50 | 0.4368    | 0.3745         |

Spliced per-category (paper's actual `k` per category, via
`format_spliced_summary`):

| category    | k  | F1     | BLEU-1 | n    |
|-------------|----|--------|--------|------|
| multi_hop   | 40 | 0.2707 | 0.2000 | 282  |
| temporal    | 40 | 0.4367 | 0.3672 | 321  |
| open_domain | 50 | 0.1510 | 0.1211 | 96   |
| single_hop  | 50 | 0.4358 | 0.3525 | 841  |
| adversarial | 40 | 0.5631 | 0.5512 | 446  |

## Comparison to the paper's reported GPT-4o-mini row

| category    | ours F1 | paper F1 | Δ F1   | ours BLEU | paper BLEU | Δ BLEU |
|-------------|---------|----------|--------|-----------|------------|--------|
| multi_hop   | 27.07   | 27.02    | +0.05  | 20.00     | 20.09      | -0.09  |
| temporal    | 43.67   | 45.85    | -2.18  | 36.72     | 36.67      | +0.05  |
| open_domain | 15.10   | 12.14    | +2.96  | 12.11     | 12.00      | +0.11  |
| single_hop  | 43.58   | 44.65    | -1.07  | 35.25     | 37.06      | -1.81  |
| adversarial | 56.31   | 50.03    | +6.28  | 55.12     | 49.47      | +5.65  |

**This is a materially closer reproduction than
[001](001-m2-llama3.2-1b-full-reproduction.md)'s Llama-3.2:1b run.**
Multi_hop is an essentially exact match (within noise). Temporal and
single_hop sit 1-2 points below the paper. Open_domain and adversarial are
actually *above* the paper's numbers by a non-trivial margin (+3 and +6
F1 respectively) — adversarial's gap is the largest of any category here,
same as in 001, but this time in the opposite direction (there it was
-23.6; here it's +6.3), which fits with the adversarial-scoring-sign
ambiguity flagged in
[decisions/0013](../decisions/0013-paper-reproduction-pipeline.md#context)
being genuinely a wash rather than a one-directional bias — not resolved
either way by this run, just further evidence it's noisy/model-dependent.

This result is a meaningful data point for 001's still-open question
("what actually explains the Llama-3.2:1b gap, if not `k`"): since the
*same* pipeline, on a *different* model, using the paper's own correct
`k` configuration, reproduces the paper closely — the pipeline itself
looks faithful. That makes it more likely the Llama-3.2:1b gap is
something specific to that model/run (quantization, single-run variance,
Ollama-specific behavior) rather than a systematic bug in this project's
reproduction code.

## Cost

Real API spend, OpenRouter/GPT-4o-mini. Calibration run (`--ratio 0.1`,
1 conversation): 55 min, $0.34 — used to derive the estimates below (this
project's own cost/time predictions, not logged automatically by the
pipeline). Full memory-building (10 conversations, one-time, shared
across every k): ~9.3h / ~$3.46 (paid once, prior to this run's own
sessions). `k=40` and `k=50` QA-answering (this run, after the crash and
resume): ~2.5h / ~$0.93 each, ~5h / ~$1.87 combined — real total spend
not independently confirmed against the OpenRouter dashboard for this
writeup.

## Conclusion

Meets milestone 2's bar more convincingly than 001 did: full-dataset,
paper-code-faithful, and — using the paper's own per-category `k` — closely
matches the paper's reported GPT-4o-mini numbers (one near-exact category,
two slightly under, two meaningfully over). Two real orchestration bugs
were found and fixed along the way (QA-answering had no checkpointing at
all; the fix for that had its own path bug with slash-containing model
names), both now covered by regression tests.

Still open: the adversarial-scoring-sign ambiguity (still unresolved, now
showing up in both directions across two different models); the
Llama-3.2:1b gap's actual cause (weakened as a "pipeline bug" explanation
by this run's fidelity); the two milestone-3 questions from
decisions/0013 (2 vs. 4 prompts for GEPA to optimize; whether to expand
the paper-comparison to GPT-4o and the two Qwen/Llama-3b models before
starting milestone 3).

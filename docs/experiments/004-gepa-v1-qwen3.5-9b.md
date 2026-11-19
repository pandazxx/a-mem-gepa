# 004: First GEPA run — qwen3.5:9b task + reflection, held-out test comparison

- Date: training 2026-11-10 → 2026-11-16; test comparison 2026-11-19
- Config: `configs/base.yaml` (branch `feat/gepa-local-model-progress-resume`,
  PR #10; training started at commit `68cf999`, test eval at `bc513c9`)
- GEPA budget: `max_metric_calls=2000` (actual 2,216 — the final full-val
  pass overshoots), `reflection_minibatch_size=2`, task model
  `ollama/qwen3.5:9b` (thinking off, ADR 0016/0017), reflection model
  `ollama/qwen3.5:9b` via LiteLLM (thinking on — not covered by the patch),
  `retrieve_k=10`, `adversarial_scoring=refusal`.
- Commands: `just gepa-optimize configs/base.yaml --run-label gepa-qwen35-9b-v1 --yes`
  (resumed at least once with the same label), then
  `just gepa-test-eval results/gepa-qwen35-9b-v1/best_candidate --yes`
  (ADR 0018).

## What changed vs. the previous experiment

[003](003-qwen3.5-9b-exploration.md) ran the paper's prompts on all 10
conversations with a uniform k-sweep. This is the first run of the GEPA
training method (ADRs 0014/0015) end to end, and the first held-out
baseline-vs-candidate comparison (ADR 0018). Pipeline changes that this run
forced, all on PR #10: `think=False` applied to GEPA rollouts, progress/ETA
logging, mid-build checkpoints and resumable run accounting (ADR 0017),
NLTK `punkt_tab` pre-flight in the adapter, the `propose_new_texts = None`
protocol hook (without which the first smoke run never proposed a single
candidate), and `just gepa-test-eval` itself.

## Results

Aggregate (test set: conv-26, conv-44, conv-48; n=596 questions; paired
bootstrap over questions, 10,000 resamples; `retrieve_k=10` for both arms;
paper metric suite):

| | baseline F1 | candidate F1 | delta F1 [95% CI] |
|---|---|---|---|
| overall | 0.3623 | 0.3679 | +0.0056 [-0.0241, +0.0354] |

Per-category breakdown:

| category | n | baseline F1 | candidate F1 | delta F1 [95% CI] | baseline BLEU-1 | candidate BLEU-1 |
|---|---|---|---|---|---|---|
| multi_hop | 83 | 0.2881 | 0.2804 | -0.0077 [-0.0602, +0.0453] | 0.2071 | 0.2160 |
| temporal | 103 | 0.3788 | 0.3403 | -0.0385 [-0.0972, +0.0227] | 0.2758 | 0.2631 |
| open_domain | 30 | 0.1178 | 0.1373 | +0.0195 [-0.0314, +0.0632] | 0.1148 | 0.1092 |
| single_hop | 250 | 0.4090 | 0.4101 | +0.0011 [-0.0433, +0.0443] | 0.3667 | 0.3527 |
| adversarial | 130 | 0.3634 | 0.4179 | +0.0545 [-0.0354, +0.1459] | 0.3464 | 0.4067 |

Adversarial, refusal reading (picked "Not mentioned in the conversation"):
baseline 0.6846, candidate 0.6077, delta -0.0769 [-0.1692, +0.0154].

Reading the two adversarial rows together: the candidate matches the *trap*
text more (F1 row up) and refuses less (refusal row down). Both point the
same way — the trained memory makes the model fall for the trap more
often on held-out conversations — even though training rewarded the
refusal direction on the validation set (ADR 0015). Neither interval
excludes zero, but this is the largest and most consistent signal in the
table and it is in the wrong direction.

Validation (training-time, refusal direction, stratified ~150-question
subset of conv-42/conv-50 — not comparable to the test numbers above):
best candidate (index 13 of 14) aggregate **0.3784**. Seed aggregate:
**TODO** — read from the iteration-0 line of
`results/gepa-qwen35-9b-v1.console.log` (`grep -n "Iteration 0"`).
Without it, the size of the validation-to-test gap (i.e. how much of the
validation gain was overfitting to 2 conversations) cannot be stated.

Not comparable to [003](003-qwen3.5-9b-exploration.md)'s 0.46 overall:
that was all 10 conversations at k=40; this is 3 held-out conversations at
k=10 (ADR 0018's comparability note).

## Pareto front summary

14 candidates after 39 iterations (13 accepted proposals). Final best by
validation aggregate: candidate 13, which was also the "linear Pareto front
program index" at the last logged iteration. **TODO:** per-candidate
validation aggregates and which component each accepted proposal changed —
available in `results/gepa-qwen35-9b-v1/run_log.json` /
`candidates.json`; not pasted into this doc yet.

## Before/after prompt diff

**TODO.** The four files under `results/gepa-qwen35-9b-v1/best_candidate/`
have not been committed. Per CLAUDE.md they should land as
`src/amem_gepa/prompts/<component>.v1.txt` in this PR, and the diff against
`prompts/baseline_repro/` goes here, even though the candidate is not
adopted (section "Conclusion") — the history is the point.

## Cost

Local Ollama only (`qwen3.5:9b` on the user's box), $0 API spend for the
task model and the reflection model alike.

Training (`summary.json` / `adapter_stats.json`, whole run across resumes):

| | |
|---|---|
| wall-clock | 545,221 s = **6.3 days** |
| of which memory building | 522,832 s (96%) — 98 fresh builds, **~89 min/build** |
| of which QA | 18,591 s — 3,059 questions, ~6.1 s/question |
| build cache hits | 53 (of 151 build requests) |
| mid-build resumes | 0 |
| LLM calls | 188,865 (≈1,900 per build after subtracting ~2 per question — high; suggests frequent parser-fallback retries, not yet inspected) |
| metric calls | 2,216 / 2,000 budget; 88 evaluations |

Test comparison: 6 fresh builds (3 test conversations × 2 arms) + 1,192
answered questions. **TODO:** actual wall-clock from
`results/test_eval_gepa-qwen35-9b-v1_test/test_eval.json` → `adapter_stats`
(estimate from training rates: ~9 h builds + ~2 h QA).

The 89 min/build figure is 4x the 6–25 min extrapolated from the smoke
run's 3.5 s/turn (ADR 0017's consequences section): full conversations
reach dense evolution with large neighbor prompts that a 40-turn smoke
never sees.

## Conclusion

Against the success criteria in `docs/00-proposal.md`:

1. *Statistically meaningful aggregate improvement* — **no.** +0.0056
   overall, CI [-0.024, +0.035], straddling zero.
2. *No category regresses beyond noise* — **not shown either way**, but
   the adversarial refusal rate dropped 7.7 points (CI just touching zero)
   and temporal F1 dropped 3.9 points (CI including zero). Criterion 2
   would fail on adversarial if that signal held up with more data.
3. *Cost reported next to the delta* — yes: 6.3 days of local compute for
   a delta indistinguishable from zero.

What this run establishes: the full method works end to end on a local 9B
model (build → cache → QA → reflection → acceptance → Pareto → held-out
comparison), and one data point for what a 2,000-metric-call budget buys
with a 9B reflector: nothing measurable.

Candidate 13 is **not adopted** as a new baseline. Next steps, in order of
expected information per hour:

- Fill the three TODOs above (seed validation score, prompt diff, Pareto
  table) — all from files already on disk, no compute.
- Inspect one cached build trace for the ~1,900 calls/build figure: if
  the plain-text parsers are falling back to retries on most turns, the
  reflection dossiers are being built from degraded outputs.
- A stronger reflection model (ADR 0003's recommendation; the run used the
  same 9B model, with thinking on through LiteLLM) is the cheapest change
  that targets proposal quality rather than budget. Reflection is ~1 call
  per iteration, so a hosted model costs cents, not dollars.
- Validation is 2 conversations. If the seed's validation score turns out
  to be well below 0.378, the gain was real on val and vanished on test —
  i.e. overfitting to two conversations — which argues for 0004's deferred
  rotating-split check before spending another week.
- Build cost dominates; ADR 0015's deferred truncated-conversation train
  instances would cut iteration cost several-fold and are now worth a
  calibration run.

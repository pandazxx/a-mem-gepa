# 006: Stage 2 of sequential training — evolution decision only, qwen3.5:9b

- Date: training 2026-11-24 → 2026-11-25 (35.0 h wall-clock); held-out
  comparison **pending**
- Config: `configs/base.yaml` with the `gepa_single_component` overrides
  (branch `topic/single-prompt-training-5ed55fc`, PR #11, at or after
  `56ba02f`; the thinking/coach settings of PR #13 were **not** in this run —
  `summary.json` has no `reflection_*` fields)
- GEPA budget: `max_metric_calls=800` (actual 872), `reflection_minibatch_size=2`,
  task model `ollama/qwen3.5:9b` (thinking off, ADR 0016/0017), reflection
  model `ollama/qwen3.5:9b` via LiteLLM (thinking on by model default),
  `retrieve_k=10`, `adversarial_scoring=refusal`
- Component optimized: **`evolution_decision`**; frozen byte-for-byte at the
  seed: `note_construction`, `evolution_strengthen`,
  `evolution_update_neighbors` (`frozen_intact: true`)
- Seed: `baseline_repro/` with `note_construction` replaced by
  `src/amem_gepa/prompts/note_construction.v1.txt` (stage 1's winner,
  [005](005-gepa-stage1-note-construction-qwen3.5-9b.md)), i.e. the seed
  differs from the paper's prompts on `note_construction` only
- Command: `just gepa-optimize-component evolution_decision --seed-file note_construction=src/amem_gepa/prompts/note_construction.v1.txt --yes`
  (run label `gepa_evolution_decision_20261124_002304`, not resumed)

## What changed vs. the previous experiment

Second stage of [ADR 0019](../decisions/0019-sequential-single-prompt-training.md)'s
order. Same pipeline, models, budget and validation subset as 005; the
seed carries stage 1's note-construction prompt, which was kept after its
held-out check (005: single-hop +6.3 with the interval clear of zero,
nothing significantly worse). The prompt under optimization is the one
that decides, per new note, whether to do nothing, link, rewrite
neighbours, or both.

## Results

**Validation** (training objective; the fixed 138-question subset of
conv-42/conv-50):

| | val aggregate |
|---|---|
| paper's prompts (seed of stage 1; 004/005) | 0.3221 |
| stage 1 winner = **this stage's seed**, as scored in stage 1 | 0.3798 |
| this stage's seed, as re-scored at the start of this run | **TODO** — the "Base program full valset score" line in `results/gepa_evolution_decision_20261124_002304/run_log.txt` |
| **this stage's winner** (`evolution_decision` changed) | **0.3668** |

The winner's 0.3668 is *below* the 0.3798 the identical seed scored in
stage 1. Since the winner differs from the seed and GEPA's best-by-
validation includes the seed, the seed must have re-scored below 0.3668
in this run: the same memory (its validation builds were cache hits),
re-answered, moved by more than 1.3 points on 138 questions. That is the
same QA sampling noise 005 flagged on the test set (baseline arm drifted
1.3 points overall between 004 and 005). Two consequences:

- Validation gains of a few points are within one noise width on this
  subset; the held-out table, not the validation number, is the only
  evidence that counts (ADR 0019's gate).
- Stage-over-stage comparisons of *validation* numbers across runs are
  not meaningful; only within-run comparisons (candidate vs. its own
  seed, same run) are, and the seed line is needed to make one here.

**Held-out test comparison: TODO.** Two runs are useful, both resumable:

```
# stage gate: stage 2 winner vs stage 1 winner (the seed this stage started from)
just gepa-test-eval results/gepa_evolution_decision_20261124_002304/best_candidate \
    --baseline-dir results/gepa_note_construction_20261120_042519/best_candidate --yes
# for the running table: stage 2 winner vs the paper's prompts
just gepa-test-eval results/gepa_evolution_decision_20261124_002304/best_candidate --yes
```

The `--baseline-dir` form is new in this PR: `gepa-test-eval` previously
compared only against the paper's prompts, which answers "is the whole
stack better than the paper" but not "did stage 2 add anything to stage
1" — and the latter is what decides whether this winner seeds stage 3.
Cost: stage 1's winner already has its 3 test builds cached (005's test
eval), the paper's prompts too, so each run is ~3 fresh builds (stage 2
winner) + 1,192 questions; the second run reuses the first's 3 builds and
is QA only.

## Pareto front summary

6 candidates after 11 iterations (5 accepted proposals, 45%; stage 1: 5 of
14). Every candidate differs from the seed in `evolution_decision` only.
**TODO:** per-candidate validation aggregates from `run_log.txt`.

## Before/after prompt diff

**TODO.** `best_candidate/evolution_decision.txt` to be committed as
`src/amem_gepa/prompts/evolution_decision.v1.txt`; diff against
`baseline_repro/evolution_decision.txt` goes here (committed whether or not
adopted).

## Cost

Local Ollama only (`qwen3.5:9b`), $0 API spend. From `summary.json`:

| | |
|---|---|
| wall-clock | 126,158 s = **35.0 h** |
| of which memory building | 119,122 s (94%) — 36 fresh builds, **~55 min/build** |
| of which QA | 5,878 s — 1,180 questions, ~5.0 s/question |
| build cache hits | 14 (of 50 build requests) |
| mid-build resumes | 0 |
| LLM calls | 65,432 (≈1,750 per build after subtracting ~2 per question; 005: ~1,700) |
| metric calls | 872 / 800 budget; 28 evaluations (6 full-val + 22 minibatch = 11 iterations) |

Build time: 55 min here vs 42 min in 005 vs 89 min in 004. Plausible
driver: a decision prompt that says STRENGTHEN/UPDATE more often triggers
more downstream evolution calls per turn (calls/build 1,750 vs 1,700 is
consistent with slightly more, and longer, prompts). Not investigated.

Test comparison: **TODO** wall-clock from the test-eval `adapter_stats`.

## Conclusion

**Not yet assessable** — the held-out comparison has not run. What the
training numbers establish:

- The stage machinery held: frozen components intact, only
  `evolution_decision` changed, 5 accepted rewrites in 35 hours.
- The validation subset's run-to-run noise is now visible directly (the
  identical seed scored 0.3798 in one run and below 0.3668 in the next).
  Any future reading of a validation delta should be against the seed line
  *of the same run*.

Next steps:

1. Run the stage-gate test eval (`--baseline-dir` stage 1 winner) and,
   if cheap enough, the paper-baseline one; paste both `summary.md`.
2. Decide keep/discard for stage 3 per ADR 0019; record here.
3. Commit `evolution_decision.v1.txt`; fill the diff; fill the Pareto table
   and the seed validation line from `run_log.txt`.
4. Stage 3 (`evolution_strengthen`), seeded from v1 note construction plus
   (if kept) v1 evolution decision:
   `just gepa-optimize-component evolution_strengthen --seed-file note_construction=src/amem_gepa/prompts/note_construction.v1.txt --seed-file evolution_decision=src/amem_gepa/prompts/evolution_decision.v1.txt --yes`

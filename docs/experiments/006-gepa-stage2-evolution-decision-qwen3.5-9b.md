# 006: Stage 2 of sequential training — evolution decision only, qwen3.5:9b

- Date: training 2026-11-24 → 2026-11-25 (35.0 h wall-clock); held-out
  stage gate 2026-11-25 → 2026-11-26 (5.2 h)
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

**Held-out stage gate: stage 2 winner vs stage 1 winner** (the seed this
stage started from), via the `--baseline-dir` form added in this PR
(`gepa-test-eval` previously compared only against the paper's prompts,
which answers "is the whole stack better than the paper" but not "did
stage 2 add anything to stage 1"). Test set conv-26/44/48, n=596, paired
bootstrap over questions, 10,000 resamples, `retrieve_k=10` both arms,
paper metric suite. "baseline" below = stage 1's winner (`note_construction.v1`
+ paper's three evolution prompts); "candidate" = that plus this stage's
`evolution_decision` rewrite.
`results/test_eval_gepa_evolution_decision_20261124_002304_vs_gepa_note_construction_20261120_042519_test/summary.md`:

| category | n | baseline F1 | candidate F1 | delta F1 [95% CI] | baseline BLEU-1 | candidate BLEU-1 |
|---|---|---|---|---|---|---|
| multi_hop | 83 | 0.2704 | 0.2748 | +0.0044 [-0.0628, +0.0726] | 0.2048 | 0.2205 |
| temporal | 103 | 0.3284 | 0.3683 | +0.0399 [-0.0105, +0.0921] | 0.2389 | 0.2766 |
| open_domain | 30 | 0.1663 | 0.1516 | -0.0147 [-0.0962, +0.0808] | 0.1505 | 0.1418 |
| single_hop | 250 | 0.4358 | 0.4670 | +0.0312 [-0.0159, +0.0765] | 0.3757 | 0.4091 |
| adversarial | 130 | 0.2990 | 0.3743 | +0.0752 [-0.0059, +0.1575] | 0.2801 | 0.3519 |
| overall | 596 | 0.3508 | **0.3871** | **+0.0363 [+0.0070, +0.0662]** | 0.2961 | 0.3340 |

Adversarial, refusal reading (picked "Not mentioned in the conversation"):
baseline 0.7692, candidate 0.6846, **delta -0.0846 [-0.1692, +0.0000]**.

Reading it — two true statements that point in opposite directions:

- **Under the paper's metric suite this is the first held-out *overall*
  improvement whose interval excludes zero**: +3.6 points, [+0.7, +6.6].
  Every factual category moved up (single-hop +3.1, temporal +4.0,
  multi-hop +0.4; open-domain -1.5 on n=30 is noise), none significantly on
  its own. Weighted over the 466 non-adversarial questions the gain is
  **+2.5 points**.
- **Under the project's own adversarial direction it is a regression.**
  The candidate matches the trap text *more* (F1 row +7.5, interval
  nearly clear of zero) and refuses *less* (refusal rate 0.769 → 0.685,
  -8.5 points, interval [-16.9, +0.0], upper edge exactly at zero). ADR
  0015 chose refusal as the rewarded direction because scoring against the
  trap text rewards falling for the trap; by that reading about a third of
  the overall +3.6 is the model being fooled more often. **Under the
  training objective** (refusal rate in place of adversarial F1, weighted
  by n): baseline **0.453**, candidate **0.455**, **+0.1 points — a wash.**
  The factual gains and the refusal loss cancel almost exactly.

This is, almost word for word, the case `docs/00-proposal.md`'s success
criterion 2 pre-committed against: "a net win that trades adversarial-
refusal accuracy for single-hop accuracy does not count as a clean
success."

Why a decision prompt would do this is mechanically plausible: a prompt
that says STRENGTHEN/UPDATE more often produces a more densely linked
memory, so retrieval drags more context into every answer. For factual
questions that means the evidence is more often present; for trap
questions it means there is more material from which to assemble a
confident wrong answer instead of refusing. The prompt diff (TODO below)
should show whether the rewrite does push toward more evolution; the
build time (55 min vs 42) and calls per build (1,750 vs 1,700) already
hint that way.

**Noise-floor note, again.** The baseline arm here is byte-identical in
memory to 005's candidate arm (same cached builds), only re-answered. Its
scores moved: overall 0.3685 → 0.3508 (-1.8), temporal -3.5, single-hop
-2.8, open-domain +6.3 (n=30), adversarial refusal 0.7154 → 0.7692
(+5.4, i.e. 7 of 130 questions flipped). The paired interval within a run
is the valid statistic; cross-run *levels* are not comparable at the
precision these deltas need. Any running "vs. the paper" table must come
from a single run per comparison, not by chaining tables.

The paper-baseline run (`just gepa-test-eval <stage 2 winner> --yes`,
QA only now that the builds exist) was not run; given the above it would
add a level, not a decision.

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

Stage gate: 3 fresh builds (candidate arm; the stage 1 winner's 3 test
builds were cache hits from 005's test eval) = 11,789 s, **~65 min/build**;
1,192 questions re-answered (both arms, ~5.9 s/question); **5.2 h**
wall-clock in total.

## Conclusion

Against `docs/00-proposal.md`'s criteria, for this stage alone (stage 2
winner vs stage 1 winner):

1. *Statistically meaningful aggregate improvement* — **yes under the
   paper's metric** (+3.6, [+0.7, +6.6]); **no under the training
   objective** (+0.1).
2. *No category regresses beyond noise* — **fails, or sits exactly on the
   line**: adversarial refusal -8.5 points with the interval's upper edge
   at 0.000. The proposal names this precise trade as not a clean success.
3. *Cost next to the delta* — 35.0 h training + 5.2 h gate.

What this stage establishes:

- The sequential method can produce a held-out, paired, overall-significant
  gain on the paper's metrics in one stage — the first in the project.
- GEPA optimized a *blended* objective and found a prompt that moves its
  components in opposite directions; the blend hid the trade during
  training (the stage's own validation gain was within noise anyway).
  Whatever is decided below, future stages should report the refusal row
  next to the overall as a matter of course.

**Stage verdict (ADR 0019 gate): recommended *discard* as the seed of
stage 3 — the user's call.** The case for discarding: criterion 2 was
pre-registered to rule out exactly this trade, the training-objective
delta is zero, and stage 3 (`evolution_strengthen`, which also governs how
densely memory is linked) would then start from a memory already tilted
toward over-linking. The case for keeping: the paper-metric gain is real
and the first of its kind, and the refusal interval does touch zero. A
middle path worth considering: keep the candidate *file* committed
(`evolution_decision.v1.txt`, history is the point) but seed stage 3 from
stage 1's winner, and revisit the decision prompt after the two remaining
evolution stages, when the adversarial behaviour of the whole evolution
stack can be judged together. **Decision recorded here once made:** TODO.

Next steps:

1. Record the keep/discard decision above.
2. Commit `best_candidate/evolution_decision.txt` as
   `src/amem_gepa/prompts/evolution_decision.v1.txt`; fill the diff; fill
   the Pareto table and the seed validation line from `run_log.txt`.
3. Stage 3 (`evolution_strengthen`). If discarded:
   `just gepa-optimize-component evolution_strengthen --seed-file note_construction=src/amem_gepa/prompts/note_construction.v1.txt --yes`
   If kept, add `--seed-file evolution_decision=src/amem_gepa/prompts/evolution_decision.v1.txt`.

# 005: Stage 1 of sequential training — note construction only, qwen3.5:9b

- Date: training 2026-11-20 → 2026-11-21 (26.3 h wall-clock); held-out
  comparison 2026-11-21 → 2026-11-22
- Config: `configs/base.yaml` with the `gepa_single_component` overrides
  (branch `topic/single-prompt-training-5ed55fc`, PR #11; code at commit
  `70b0f8f`, docs-only commits followed)
- GEPA budget: `max_metric_calls=800` (actual 884 — the final full-val pass
  overshoots, as in 004), `reflection_minibatch_size=2`, task model
  `ollama/qwen3.5:9b` (thinking off, ADR 0016/0017), reflection model
  `ollama/qwen3.5:9b` via LiteLLM, `retrieve_k=10`,
  `adversarial_scoring=refusal`
- Component optimized: **`note_construction`**; frozen byte-for-byte at the
  paper's prompts: `evolution_decision`, `evolution_strengthen`,
  `evolution_update_neighbors` (`frozen_intact: true` in `summary.json`)
- Seed: `src/amem_gepa/prompts/baseline_repro/` (the paper's prompts, no
  overrides — stage 1 of ADR 0019's order)
- Command: `just gepa-optimize-component note_construction --yes`
  (run label `gepa_note_construction_20261120_042519`, not resumed)

## What changed vs. the previous experiment

[004](004-gepa-v1-qwen3.5-9b.md) let GEPA's round-robin selector rewrite all
four prompts in one run; its winner differed from the seed on all four and
the +5.6 validation gain became -2.3 on test, unattributable to any one
prompt. This is the first run under [ADR 0019](../decisions/0019-sequential-single-prompt-training.md):
one prompt per run, the other three frozen, the held-out check between
stages. Same task model, same reflection model, same validation subset and
training objective as 004, so the validation numbers are directly
comparable; budget 800 instead of 2000.

## Results

**Validation** (training-time objective: paper F1 for categories 1–4,
refusal direction for category 5; the fixed stratified subset of
conv-42/conv-50 = 138 questions):

| | val aggregate | vs. seed |
|---|---|---|
| seed (paper's prompts) | 0.3221 (004's seed line; same candidate, same cached val builds — **TODO** confirm from this run's `run_log.txt`) | — |
| 004 winner (all 4 prompts changed) | 0.3784 | +5.6 |
| **this stage's winner (note_construction only)** | **0.3798** | **+5.8** |

Changing one prompt reproduced the whole of 004's validation gain. On its
own that says nothing about the test split — 004's gain was the same size
and reversed there — but it does say the other three rewrites in 004 were
not where the validation gain came from.

**Held-out test comparison** (`just gepa-test-eval
results/gepa_note_construction_20261120_042519/best_candidate --yes`; test
set conv-26, conv-44, conv-48; n=596 questions; paired bootstrap over
questions, 10,000 resamples; `retrieve_k=10` both arms; paper metric
suite; `results/test_eval_gepa_note_construction_20261120_042519_test/summary.md`):

| category | n | baseline F1 | candidate F1 | delta F1 [95% CI] | baseline BLEU-1 | candidate BLEU-1 |
|---|---|---|---|---|---|---|
| multi_hop | 83 | 0.2519 | 0.2710 | +0.0192 [-0.0393, +0.0784] | 0.1817 | 0.2159 |
| temporal | 103 | 0.3553 | 0.3633 | +0.0081 [-0.0469, +0.0652] | 0.2660 | 0.2717 |
| open_domain | 30 | 0.1105 | 0.1033 | -0.0072 [-0.0927, +0.0685] | 0.1098 | 0.0896 |
| single_hop | 250 | 0.4008 | **0.4635** | **+0.0627 [+0.0171, +0.1089]** | 0.3492 | 0.4116 |
| adversarial | 130 | 0.3635 | 0.3132 | -0.0503 [-0.1413, +0.0371] | 0.3489 | 0.2996 |
| overall | 596 | 0.3494 | 0.3685 | +0.0190 [-0.0129, +0.0504] | 0.2994 | 0.3195 |

Adversarial, refusal reading (picked "Not mentioned in the conversation"):
baseline 0.6846, candidate 0.7154, delta +0.0308 [-0.0615, +0.1231].

Reading it:

- **Single-hop improved and the interval excludes zero** (+6.3 points,
  n=250, the largest category). This is the first held-out effect in the
  project whose interval does not straddle zero. It is also the category
  one would predict for a better note-construction prompt: a single fact
  on a single card, found or not found.
- **Overall +1.9 points, interval [-1.3, +5.0]** — straddles zero, so
  proposal criterion 1 (aggregate CI clear of baseline) is **not met**.
  Point estimate is 3.4x 004's (+0.56), and the lower bound is much closer
  to zero.
- **Adversarial moved in the right direction this time, both readings
  agree, neither is significant.** The candidate matches the trap text
  less (F1 row -5.0) and refuses more (refusal row +3.1). In 004 both
  readings pointed the wrong way. Note that the paper's F1 reading going
  *down* is the good outcome here (ADR 0013/0015's sign ambiguity).
- **No category regresses beyond noise.** open_domain -0.7 on n=30 is
  well inside its interval; adversarial F1 is the wanted direction.
- **Under the training objective** (refusal rate in place of adversarial
  F1, weighted by n): baseline **0.419**, candidate **0.456**, **+3.7
  points on test**, against +5.8 on validation. 004 was +5.6 on
  validation and -2.3 on test. So about two-thirds of the validation gain
  transferred, where 004's reversed.
- **Noise floor caveat.** The baseline arm's memory was identical in 004
  and here (cache hit on the same candidate hash), yet its re-answered
  scores differ: overall 0.3623 → 0.3494, multi_hop 0.2881 → 0.2519,
  temporal 0.3788 → 0.3553, single_hop 0.4090 → 0.4008 (adversarial
  refusal identical, 0.6846). Re-answering the same memory moves overall
  F1 by ~1.3 points and multi-hop by ~3.6, i.e. the QA/retrieval sampling
  noise is of the same order as the overall delta measured here. The
  paired bootstrap accounts for per-question variance *within* this run,
  but this cross-run drift is the right mental yardstick for the
  non-significant rows. A baseline-vs-baseline re-answer (zero builds,
  ~1.5 h of QA) would quantify it properly; not done.

## Pareto front summary

6 candidates after 14 iterations (5 accepted proposals, 36% acceptance; 004
was 13 of 39, 33%). Every candidate differs from the seed in
`note_construction` only. **TODO:** per-candidate validation aggregates
from `results/gepa_note_construction_20261120_042519/run_log.txt` /
`candidates.json`.

## Before/after prompt diff

**TODO.** `best_candidate/note_construction.txt` to be committed as
`src/amem_gepa/prompts/note_construction.v1.txt` (the other three files in
`best_candidate/` are identical to `baseline_repro/` and are not
re-committed). Diff against `baseline_repro/note_construction.txt` goes
here whether or not the candidate is adopted — the history is the point.

## Cost

Local Ollama only (`qwen3.5:9b`), $0 API spend. From `summary.json`
(whole run, not resumed):

| | |
|---|---|
| wall-clock | 94,634 s = **26.3 h** |
| of which memory building | 87,532 s (92%) — 35 fresh builds, **~42 min/build** |
| of which QA | 5,556 s — 1,199 questions, ~4.6 s/question |
| build cache hits | 22 (of 57 build requests) — includes the seed's 2 val builds from 004 |
| mid-build resumes | 0 |
| LLM calls | 61,915 (≈1,700 per build after subtracting ~2 per question; 004: ~1,900) |
| metric calls | 884 / 800 budget; 34 evaluations (6 full-val + 28 minibatch) |

Test comparison: 3 fresh builds (candidate arm; the baseline arm's 3 test
builds were cache hits from 004's test eval) + 1,192 answered questions.
**TODO:** wall-clock from
`results/test_eval_gepa_note_construction_20261120_042519_test/test_eval.json`
→ `adapter_stats` (estimate at this run's rates: ~2 h builds + ~1.5 h QA).

Notable and unexplained: **42 min/build vs. 89 min/build in 004**, same
model, same hardware as far as known. Candidates for the difference, to be
checked against the per-build progress lines in `amem_gepa.log`: shorter
note-construction outputs shrinking every downstream evolution prompt;
different Ollama/driver state; or 004's figure being inflated by its
resumed segments. Stage 2's number will say whether 42 min is the new
normal.

## Conclusion

Against `docs/00-proposal.md`'s criteria, for this stage alone:

1. *Statistically meaningful aggregate improvement* — **no.** +1.9
   overall, CI [-1.3, +5.0].
2. *No category regresses beyond noise* — **yes**, as far as 596 questions
   can tell. The one significant movement (single-hop, +6.3) is an
   improvement; adversarial moved in the wanted direction on both readings.
3. *Cost reported next to the delta* — yes: 26.3 h training + the test
   eval (cost TODO below) for +1.9 overall / +6.3 single-hop.

What this stage establishes:

- A note-construction-only rewrite reproduced 004's whole validation gain
  (+5.8 vs +5.6) **and, unlike 004, most of it transferred to the held-out
  split** (+3.7 under the training objective, vs 004's -2.3). The three
  evolution rewrites in 004 were therefore not where its validation gain
  came from, and they are the most plausible source of its test reversal
  (adversarial went the wrong way there and the right way here).
- Single-hop retrievability is where a better note prompt helps, which is
  what ADR 0019 predicted when putting this component first.

**Stage verdict (ADR 0019 gate): recommended *keep* — the user's call.**
The strict reading of the gate ("overall range above zero") is not met.
The case for carrying the winner into stage 2 anyway: one category is
significantly better, none is significantly worse, both adversarial
readings agree on the direction, and the training objective's test delta
is positive. The case against: the overall interval includes zero, and
the cross-run noise floor above is about as large as the overall delta.
Discarding would mean stage 2 searches on top of a note prompt we have
evidence is worse on single-hop. **Decision recorded here once made:**
TODO.

Next steps, in order:

1. Record the keep/discard decision above.
2. Commit `best_candidate/note_construction.txt` as
   `src/amem_gepa/prompts/note_construction.v1.txt`; fill the diff
   section. Fill the test-eval cost line from
   `results/test_eval_gepa_note_construction_20261120_042519_test/test_eval.json`
   → `adapter_stats`.
3. Fill the Pareto table from `run_log.txt`.
4. Stage 2 (`evolution_decision`):
   `just gepa-optimize-component evolution_decision --seed-file note_construction=src/amem_gepa/prompts/note_construction.v1.txt --yes`
   (drop `--seed-file` if the verdict is discard).
5. Optional, zero builds: a baseline-vs-baseline re-answer of the test
   set to put a number on the QA noise floor, so later stages' overall
   deltas can be read against it.

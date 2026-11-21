# 005: Stage 1 of sequential training — note construction only, qwen3.5:9b

- Date: training 2026-11-20 → 2026-11-21 (26.3 h wall-clock); held-out
  comparison **pending**
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

**Held-out test comparison: TODO.** Run

```
just gepa-test-eval results/gepa_note_construction_20261120_042519/best_candidate --yes
```

and paste `summary.md` here. The baseline arm's three test builds should
come from the cache (004's test eval built them under the same candidate
hash), so the cost is ~3 fresh builds + 1,192 questions. Per ADR 0019 the
verdict (carry forward to stage 2 or not) is decided from that table, not
from the validation number above.

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

Notable and unexplained: **42 min/build vs. 89 min/build in 004**, same
model, same hardware as far as known. Candidates for the difference, to be
checked against the per-build progress lines in `amem_gepa.log`: shorter
note-construction outputs shrinking every downstream evolution prompt;
different Ollama/driver state; or 004's figure being inflated by its
resumed segments. Stage 2's number will say whether 42 min is the new
normal.

## Conclusion

Against `docs/00-proposal.md`'s criteria: **not yet assessable** — the
held-out comparison has not run. What this stage establishes so far:

- The single-component machinery works end to end on the real model:
  frozen components verified intact, only `note_construction` changed, 5
  accepted rewrites in 26 hours.
- A note-construction-only rewrite matches 004's full validation gain
  (+5.8 vs +5.6), which is consistent with note construction being the
  prompt that matters most for retrievability (ADR 0019's reason for
  putting it first) — or with both runs fitting the same two validation
  conversations the same way. The test table decides which.

Next steps, in order:

1. `just gepa-test-eval` on the winner (above); paste the table; decide
   keep/discard per ADR 0019.
2. Commit `note_construction.v1.txt` and fill the diff section.
3. Fill the Pareto table from `run_log.txt`.
4. Stage 2 (`evolution_decision`), seeded from v1 if kept, from
   `baseline_repro` otherwise:
   `just gepa-optimize-component evolution_decision --seed-file note_construction=src/amem_gepa/prompts/note_construction.v1.txt --yes`

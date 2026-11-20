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

Validation (training-time objective: paper F1 for categories 1-4, refusal
direction for category 5; the fixed stratified subset of conv-42/conv-50 =
138 questions): seed **0.3221**, best candidate (index 13 of 14) **0.3784**,
i.e. **+5.6 points on validation**. (The seed number comes from the "Base
program full valset score ... over 138 / 138 examples" line GEPA re-logs on
resume; the iteration-0 lines themselves were lost because the console log
was re-created with `tee` instead of `tee -a` on the resumed run.)

Re-scoring the test table under that same training objective (swap the
adversarial F1 row for the refusal rate, weight by n) gives baseline
**0.432** vs candidate **0.409**: **-2.3 points on test**. So the +5.6 seen
during training did not transfer at all; it reversed. With only 2
validation conversations, the simplest reading is that GEPA found prompts
that suit conv-42/conv-50 specifically, and most of the validation gain was
fit to those two conversations rather than to the task.

Not comparable to [003](003-qwen3.5-9b-exploration.md)'s 0.46 overall:
that was all 10 conversations at k=40; this is 3 held-out conversations at
k=10 (ADR 0018's comparability note).

## Pareto front summary

From `results/gepa-qwen35-9b-v1/run_log.json` + `candidates.json`
(39 iterations, 14 candidates, 13 accepted proposals; minibatch scores
below are the sum over the 2 groups of an iteration, so 2.0 is perfect):

| iter | parent (val) | minibatch before → after | accepted as | component rewritten | child full-val aggregate |
|---|---|---|---|---|---|
| 2 | 0 (0.3221) | 0.93 → 0.97 | 1 | evolution_strengthen | **0.2871** (-0.0350 vs parent) |
| 3 | 1 (0.2871) | 0.88 → 0.92 | 2 | evolution_update_neighbors | **0.3642** (+0.0771 vs parent) |
| 5 | 1 (0.2871) | 1.05 → 1.16 | 3 | evolution_decision | **0.3250** (+0.0379 vs parent) |
| 10 | 0 (0.3221) | 0.76 → 0.92 | 4 | evolution_decision | **0.3321** (+0.0100 vs parent) |
| 11 | 4 (0.3321) | 0.71 → 0.89 | 5 | evolution_strengthen | **0.3387** (+0.0066 vs parent) |
| 13 | 1 (0.2871) | 1.03 → 1.22 | 6 | evolution_update_neighbors | **0.3206** (+0.0335 vs parent) |
| 21 | 4 (0.3321) | 0.48 → 0.93 | 7 | evolution_update_neighbors | **0.3216** (-0.0105 vs parent) |
| 24 | 2 (0.3642) | 0.53 → 0.70 | 8 | note_construction | **0.3721** (+0.0079 vs parent) |
| 25 | 3 (0.3250) | 1.04 → 1.07 | 9 | evolution_update_neighbors | **0.3541** (+0.0291 vs parent) |
| 27 | 5 (0.3387) | 0.63 → 0.69 | 10 | note_construction | **0.3502** (+0.0115 vs parent) |
| 33 | 1 (0.2871) | 0.95 → 1.20 | 11 | evolution_strengthen | **0.3741** (+0.0870 vs parent) |
| 34 | 11 (0.3741) | 0.69 → 0.85 | 12 | evolution_update_neighbors | **0.3301** (-0.0440 vs parent) |
| 38 | 11 (0.3741) | 0.69 → 0.78 | 13 | evolution_decision | **0.3784** (+0.0043 vs parent) |

Lineage of the final best: **13 ← 11 ← 1 ← 0** (seed → strengthen rewrite →
strengthen rewrite → decision rewrite). Candidate 13 therefore differs from
the seed in exactly two of the four prompts; note construction and
update-neighbors are byte-identical to the paper's.

What the log shows about the search itself:

- **Parent selection**: the seed was chosen as parent in 11 of 39
  iterations, candidate 1 in 8, candidate 3 in 7 -- the Pareto lottery kept
  going back to early, broadly-good candidates rather than chasing the
  latest accepted one.
- **Component choice**: accepted rewrites were update-neighbors ×5,
  strengthen ×3, decision ×3, note construction ×2.
- **8 of 39 children scored exactly 0** on their minibatch (iters 4, 6,
  12, 14, 23, 26, 35, 36). A build cannot legitimately score 0 across ~30
  questions, so these are almost certainly template-validation rejections
  (0014: a rewrite that drops/invents a `{placeholder}` scores 0 without
  running) -- i.e. ~20% of reflection calls produced an unusable template.
  Each still cost the parent's minibatch rollout.
- **6 iterations (15-20) logged no child scores at all** -- these coincide
  with the interruption/resume boundary (the resumed log starts at
  iteration 23); treat them as lost iterations.
- **Minibatch gains were small and noisy**: typical accepted improvement
  was +0.05 to +0.2 on a 0-2 scale over ~30 questions, i.e. a few
  questions flipping. With 2 groups per minibatch, acceptance is close to
  a coin flip on noise -- consistent with the validation gain not
  transferring to test.

Seed full-validation aggregate: **0.3221** (138 questions). Per-candidate
aggregates above come from `gepa_state.bin` (`prog_candidate_val_subscores`;
the snippet below prints them).

What the full-validation column adds:

- **The final gain came from one rewrite.** Along the winning lineage
  0 → 1 → 11 → 13 the aggregates are 0.3221 → **0.2871** → 0.3741 → 0.3784:
  candidate 1 was 3.5 points *worse* than the seed, candidate 11's
  strengthen rewrite then jumped +8.7 over its parent, and candidate 13's
  decision rewrite added +0.4. Essentially the whole +5.6 over the seed is
  one iteration-33 rewrite of the strengthen prompt.
- **Minibatch acceptance let a regression through.** Candidate 1 won its
  2-group minibatch (0.93 → 0.97) yet lost 3.5 points on the full 138
  questions. Three of the 13 accepted children (1, 6, 7) were below the
  seed on full validation; a fourth (3) was within noise of it. A 2-group
  minibatch is too small a gate.
- **The Pareto lottery kept picking the worst candidate.** Candidate 1 was
  parent in 8 of 39 iterations despite the lowest aggregate of all 14,
  because it was the per-question best on enough individual validation
  questions. That is Pareto selection working as designed -- it protects
  specialists -- but on a 138-question set from 2 conversations it mostly
  protected noise.
- A second lineage, 8 ← 2 ← 1 (0.3721, note-construction rewrite on top of
  an update-neighbors rewrite), reached nearly the same validation score
  through entirely different prompts. Two unrelated candidates at ~0.375
  on validation, and the one that was tested landing at -2.3 on test under
  the same objective, is further evidence the validation set, not the
  prompts, was being fit.

```
uv run python -c "import pickle,statistics; d=pickle.load(open('results/gepa-qwen35-9b-v1/gepa_state.bin','rb')); print([(i, len(s), round(statistics.mean(s.values()),4)) for i,s in enumerate(d['prog_candidate_val_subscores'])])"
```

## Before/after prompt diff

Candidate 13 is committed as `src/amem_gepa/prompts/{note_construction,
evolution_decision,evolution_strengthen,evolution_update_neighbors}.v1.txt`
(the two unchanged components are committed too, so v1 is a complete
candidate). `diff src/amem_gepa/prompts/baseline_repro/X.txt src/amem_gepa/prompts/X.v1.txt`
for the full text; the substance:

**evolution_decision** (788 → 1,406 chars). The four-option menu is kept
verbatim, but the neutral "decide:" preamble became three numbered
directives that *bias against NO_EVOLUTION*: "RETRIEVAL PRIORITY ...
Avoid NO_EVOLUTION if a connection to a neighbor cluster could improve the
embedding space's density", "LINKING OVER ISOLATION ... prefer STRENGTHEN
to link them into a cohesive cluster rather than leaving them fragmented",
"CONTEXT SYNCHRONIZATION ... consider UPDATE_NEIGHBOR or
STRENGTHEN_AND_UPDATE". The reflection LM has clearly absorbed the
dossier's framing ("links drive neighborhood expansion at retrieval time",
"tag/context updates change retrieval embeds" are the adapter's own
feedback sentences, 0015) and turned it into "evolve more".

**evolution_strengthen** (363 → 2,810 chars, ~8x). From a two-line
instruction to a role statement ("optimizing a memory knowledge graph for
future retrieval performance ... multi-hop questions"), a three-step task
list (extract entities/dates/objects/counts; link neighbors sharing
speaker, complementary facts or temporal proximity, "avoid ... broad theme
similarity"; emit specific lowercase tags), and a long output-constraints
block. Two details worth flagging:

- "generally aim for at least one connection" -- together with the
  decision prompt's anti-NO_EVOLUTION bias, the trained candidate links
  more aggressively than the seed. More links means more neighborhood
  expansion at answer time: more text in the QA context, including
  plausible-but-irrelevant text. That is a mechanism by which the
  adversarial regression (model picks the trap more often, section
  "Results") could arise -- a hypothesis, not checked against traces.
- The example indices changed from the seed's `CONNECTIONS: 0, 2, 3` to
  `CONNECTIONS: 177, 309` -- copied from a reflective-dataset example. In
  this pipeline the neighbors are labelled with their *global* position
  ("memory index: 177"), and `parse_strengthen_details` does no range
  check, so large indices are legal and the seed's `0, 2, 3` example was
  the misleading one. No bug, but a reminder that the reflection LM
  copies whatever concrete examples it is shown.

**note_construction, evolution_update_neighbors**: unchanged from the seed
in the final candidate (both were rewritten in other lineages -- 8, 10 and
2, 6, 7, 9, 12 -- that did not win).

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

Test comparison (`test_eval.json` → `adapter_stats`): 6 fresh builds
(3 test conversations × 2 arms), **12,783 s = 3.55 h of building, ~35.5
min/build** -- well under the training average of 89 min/build, so the
test conversations are shorter and/or trigger less evolution than the
training ones (the pre-run estimate of ~9 h for builds was therefore ~2.5x
pessimistic). 9,804 LLM calls during builds (~1,634/build). QA time and
question count were **not recorded** by this first version of the tool
(it bypassed the adapter's own accounting; fixed in the same PR after this
run -- `questions_answered=0 / qa_seconds=0` in the stats file is an
instrumentation gap, not a fact about the run). Estimate from training's
6 s/question: 1,192 questions ≈ 2 h.

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

- Inspect one cached build trace for the ~1,900 calls/build figure: if
  the plain-text parsers are falling back to retries on most turns, the
  reflection dossiers are being built from degraded outputs.
- A stronger reflection model (ADR 0003's recommendation; the run used the
  same 9B model, with thinking on through LiteLLM) is the cheapest change
  that targets proposal quality rather than budget. Reflection is ~1 call
  per iteration, so a hosted model costs cents, not dollars.
- Validation is 2 conversations, and the +5.6 validation gain became -2.3
  on test under the same objective: overfitting to two conversations is
  now the leading explanation, not a hypothesis. Before spending another
  week, either widen validation (0004's deferred rotating-split check, or
  a 4/3/3 split at the cost of training conversations) or at minimum make
  acceptance require a gain on *both* validation conversations rather than
  on their pooled average.
- Build cost dominates; ADR 0015's deferred truncated-conversation train
  instances would cut iteration cost several-fold and are now worth a
  calibration run.

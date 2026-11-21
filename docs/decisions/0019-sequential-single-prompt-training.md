# 0019: Train one prompt at a time, the other three frozen — note construction first

## Status

Accepted (2026-11-20). Refines [0014](0014-gepa-targets-repro-pipeline.md)'s
"GEPA handles multi-component candidates natively (round-robin component
selection)" and [0002](0002-optimization-targets.md)'s joint-optimization
decision: the four components are still the optimization targets, but a
run now optimizes one of them, and the four runs are chained.

## Context

The first full GEPA run ([experiments/004](../experiments/004-gepa-v1-qwen3.5-9b.md),
`qwen3.5:9b`, 6.3 days) used GEPA's default `round_robin` component
selector: iteration *i* rewrites component *i mod 4* of whichever parent
the Pareto selector picked. 39 iterations, 13 accepted proposals, 14
candidates, +5.6 points on validation that became -2.3 on the held-out
test split under the same objective. Several things about that setup
make the result hard to learn from:

- **Diluted attention.** Each prompt got roughly ten iterations of
  reflection, with ~3 accepted rewrites, spread over the whole run.
- **Confounded credit.** The winning candidate differed from the seed on
  all four prompts. Nothing in the run tells us which of the four changes
  carried the validation gain, which were neutral, and which caused the
  adversarial regression on test. The experiment doc's "which component
  each accepted proposal changed" TODO is exactly that question, and the
  round-robin run cannot answer it without further compute.
- **Moving targets for reflection.** The reflection dossier for, say,
  `evolution_strengthen` is assembled from rollouts whose notes were
  written by whatever `note_construction` the parent happened to carry.
  A strengthen prompt tuned against one note format is then inherited by
  children whose note format changes two iterations later.
- **Cost of a joint search.** Every accepted candidate costs a full
  validation pass (2 builds, ~3 hours on the local model). A joint
  search over four prompts needs many more accepted candidates to cover
  the space than a search over one.

The user's direction after reading 004: instead of training all four
prompts at the same time, train them one by one while freezing the
others, starting with the memory-construction prompt.

## Decision

- **One component per GEPA run.** `scripts/run_gepa_optimize.py
  --component <name>` (`just gepa-optimize-component <name>`) passes a
  `FixedComponentSelector` (`src/amem_gepa/gepa_stages.py`) as gepa's
  `module_selector`. gepa builds a child as `parent.copy()` with only the
  selected components' texts replaced, and merge proposals are off, so
  the other three components can never change within the run. The
  adapter enforces it anyway: `AMemGEPAAdapter(frozen_components=...)`
  scores any candidate that differs from the seed on a frozen component
  0 without spending a build, the same path a template that fails
  placeholder validation takes ([0014](0014-gepa-targets-repro-pipeline.md)).
  The run plan prints the optimized component and the frozen three;
  `summary.json` records `component`, `frozen_components`,
  `frozen_intact` and `best_changed_components`.
- **Stage order: `note_construction` → `evolution_decision` →
  `evolution_strengthen` → `evolution_update_neighbors`.** Note
  construction first because it is upstream of everything else: its
  keywords/context/tags are what retrieval embeds and what all three
  evolution prompts read as input; it fires on every turn, so a stage has
  the most data to reflect on; and its failure attribution is the most
  direct one in `make_reflective_dataset` (an evidence turn that was
  never retrieved points at the note written for it,
  [0015](0015-gepa-training-method.md)). The evolution prompts follow in
  pipeline order: decide → link → rewrite. The order is a default, not a
  constraint; `COMPONENT_ORDER` is the single place it lives.
- **Stages are separate GEPA runs chained through the seed.** A stage
  starts from the previous stage's committed winner: `--seed-file
  <component>=src/amem_gepa/prompts/<component>.v<N>.txt` layers one
  versioned file over the default seed directory (`--seed-candidate-dir`
  swaps the whole directory, e.g. a `results/<label>/best_candidate/`
  for a quick continuation). Each stage has its own Pareto front, its own
  `run_dir`, and its own resume state. A single long-lived GEPA state
  across stages was rejected: the Pareto selector could then pick a
  parent from an earlier stage whose "frozen" components differ from the
  current stage's seed, which is exactly the confound this ADR removes.
  Every stage writes the seed it started from to
  `results/<label>/seed_candidate/` next to `best_candidate/`.
- **Whether a stage's winner is carried forward is a human decision**,
  made after `just gepa-test-eval` ([0018](0018-held-out-test-comparison.md))
  on that stage's winner, not automatically. 004 showed a validation
  gain that reversed on test; the stage boundary is where that is caught
  before it seeds the next three stages. A stage whose winner does not
  hold up on test is recorded in its experiment doc and the next stage
  seeds from the previous accepted candidate (or the paper's prompts).
- **Per-stage budget `gepa_single_component.max_metric_calls: 800`**
  (`configs/base.yaml`), applied on top of `gepa:` when `--component` is
  given, overridable per run with `--max-metric-calls`. Arithmetic: seed
  validation (138) + 138 per accepted candidate + 2 per minibatch
  iteration, so 800 ≈ seed + 4 accepted + ~55 iterations on one prompt,
  versus ~10 iterations per prompt in 004. Four stages ≈ 3,200 calls ≈
  1.6× the joint run. CLAUDE.md's go-ahead rule applies to every stage
  individually; the run plan (no `--yes`) states models and budget.

## Consequences

- **What is given up: joint optima.** Coordinate ascent finds the best
  note-construction prompt *for the paper's evolution prompts*, then the
  best decision prompt *for that note prompt*, and so on. A pair of
  changes that only helps together will not be found. Accepted: 004
  suggests the budget is nowhere near the regime where that matters, and
  attributable single-prompt results are worth more now than a possibly
  better but uninterpretable joint one.
- **Cache reuse.** Builds are keyed by the full candidate hash
  ([0015](0015-gepa-training-method.md)). Stage 1's seed is the paper's
  prompts, whose two validation builds and the train builds already in
  `results/gepa_cache/` from 004 are reused for free. Later stages start
  from a new seed and pay its validation builds once.
- **The validation-overfitting concern from 004 is unchanged.** Two
  validation conversations are still two; a stage can still fit them.
  The stage-boundary test eval makes it visible per prompt rather than
  once at the end. Widening validation remains a separate decision.
- **Experiment docs per stage.** Each stage is one
  `docs/experiments/NNN-*.md` with the stage's component named in the
  title, its seed stated (which earlier stage's winner, or the paper's
  prompts), its before/after diff for the *one* changed file, and its
  test-eval table. The winner lands as
  `src/amem_gepa/prompts/<component>.v<N>.txt` per CLAUDE.md.
- **Stage-over-stage comparison is possible offline but not built.**
  `gepa-test-eval` compares a candidate to the paper's baseline. Because
  every arm answers every test question, two stages' `test_eval.json`
  files can be paired per question to get stage-2-minus-stage-1
  intervals; nothing does that yet.
- The default round-robin mode (`just gepa-optimize`) is untouched and
  remains available for a joint run if a later ADR wants one.

# 0018: The held-out test comparison runs on the repro pipeline, paired, with the paper's metrics

## Status

Accepted (2026-11-19).

## Context

The first real GEPA run ([0017](0017-gepa-local-model-progress-and-resume.md),
`qwen3.5:9b`, 6.3 days) produced a best candidate with a validation
aggregate of 0.378. Nothing in the repo could turn that into the headline
number the proposal asks for: `just eval` drives the legacy
`agiresearch/A-mem` library pipeline (2-prompt candidates, JSON evolution,
our own QA prompt, `metrics.py`) and cannot run a 4-prompt repro-pipeline
candidate at all -- the gap is behavioral, not just file-format, see
[docs/02-training-explained.md](../02-training-explained.md) section 8.
`just reproduce` runs the paper's prompts on the full dataset with no
candidate injection and no split. GEPA's own validation scores use the
refusal direction for adversarial questions and a stratified question
subset, so they are not comparable to experiments 002/003 either.

## Decision

`just gepa-test-eval <candidate dir>` (`scripts/run_gepa_test_eval.py`,
library `src/amem_gepa/gepa_test_eval.py`):

- **Same pipeline, same adapter.** Both arms -- `prompts/baseline_repro/`
  (the paper's prompts as the vendored robust code expresses them) and the
  candidate directory -- are built with `AMemGEPAAdapter.build_for_conversation`,
  i.e. exactly the injection, build cache, mid-build checkpoints,
  `think=False` and tracing that training used. A difference between arms is
  then attributable to the four prompt files and nothing else.
- **Every question of the test conversations**, not the training-time
  stratified subset. The test split (conv-26, 44, 48; [0004](0004-train-val-test-split.md))
  was never seen by GEPA.
- **Same `retrieve_k` for both arms**, defaulting to `gepa.retrieve_k` (10,
  what training optimized under), overridable with `--retrieve-k`. The
  paper's per-model k-sweep ([0013](0013-paper-reproduction-pipeline.md))
  is about matching the paper's table, not about comparing two prompt sets;
  mixing k across arms would confound the comparison.
- **The paper's metric suite** (`utils.calculate_metrics`: token F1, BLEU,
  ROUGE, BERTScore, METEOR, SBERT) per question, exactly as `just reproduce`
  scores. Category 5 is reported **both ways**: the paper's reading (metrics
  against the trap text, as their aggregation does) and the refusal reading
  (picked "Not mentioned in the conversation", the direction training
  rewarded, [0015](0015-gepa-training-method.md)).
- **Paired bootstrap confidence interval.** Both arms answer the identical
  question set, so the headline is the per-question *difference*
  (candidate minus baseline), resampled over questions 10,000 times
  (`evaluation.bootstrap_resamples`), overall and per category. 0004 asked
  for a CI rather than a bare point estimate; pairing is what makes a
  ~600-question test set informative about a few-point difference.
- **Resumable per question** (`qa_progress_<arm>.jsonl`, same mechanism as
  `paper_repro`'s QA progress log), builds cached and checkpointed by the
  adapter. Output: `results/<label>/test_eval.json` (everything) and
  `summary.md` (the table for the experiment doc).
- The legacy `just eval` recipe is marked legacy-only in the justfile and
  left in place for the `just demo`/`just baseline` loop.

## Consequences

- Cost per comparison: one build per (arm, test conversation) -- 6 builds
  for a candidate, roughly 9 hours at the first run's ~89 min/build on a
  local 9B model -- plus every test question answered twice. The baseline
  arm's three builds are reusable across candidates via the cache.
- Multi-hop has few test questions ([0004](0004-train-val-test-split.md)
  estimated ~29); its interval will be wide, and the doc convention is to
  say so rather than read the point estimate.
- The number this produces is comparable to experiments 002/003 only in
  *method* (same pipeline, same metrics), not in *data*: those ran all 10
  conversations with the paper's per-model k; this runs 3 held-out
  conversations at the training k. The experiment doc must state both.

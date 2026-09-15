# 0010: Match the paper's exact F1 metric; fix a category-label bug it exposed

## Status

Accepted (2026-09-09)

## Context

The user's stated goal for milestone 2 is to replicate and match the
paper's benchmark, not just sanity-check the pipeline with an approximate
metric. Two things followed from actually trying to do that:

1. The paper's real metric is **F1 and BLEU**, not ROUGE-L. Earlier docs
   (00-proposal.md, ADRs 0005/0006/0009) said ROUGE-L based on a secondary
   web summary that turned out to be wrong -- confirmed once the user
   shared the actual paper benchmark table (F1/BLEU columns per category,
   per model).
2. Implementing the paper's real F1 (ported from the released eval code,
   `snap-research/locomo`'s `task_eval/evaluation.py`) required knowing
   exactly which `category` integer means which reasoning type per
   question. Checking that against the paper's own numbers (its "QA
   Benchmark Statistics" appendix reports exact counts: single-hop
   retrieval=841, multi-hop retrieval=282, temporal reasoning=321,
   open-domain knowledge=96, adversarial=446) against what
   `category_counts_by_conversation` actually finds in the released data
   ({1: 282, 2: 321, 3: 96, 4: 841, 5: 446}) showed **`CATEGORY_LABELS` had
   category 1 and 4 swapped, and category 3 mislabeled**: the correct
   mapping is 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop,
   5=adversarial -- not the 1=single-hop/3=multi-hop/4=open-domain this
   project used since milestone 1. Every per-category number reported
   before this fix (docs/experiments discussion, ADR 0009) was tied to the
   right data but the wrong category *name*.

## Decision

- Fix `CATEGORY_LABELS` (`datasets/locomo.py`) to the verified mapping.
- Port the paper's exact scoring (`metrics.py`), not a generic F1/ROUGE
  implementation:
  - `normalize_answer`: strip commas, lowercase, strip punctuation, strip
    articles (`a`/`an`/`the`/`and` -- yes, "and" too, matching their regex
    exactly), collapse whitespace.
  - `f1_score`: Porter-stemmed (`nltk.stem.PorterStemmer`), multiset
    (`Counter`) token-overlap F1. New dependency: `nltk` (just the
    stemmer, no corpus/data download needed).
  - `f1_multi_answer`: category 1 (multi-hop) only -- splits both
    prediction and gold on commas, takes the best-matching predicted
    sub-phrase per gold sub-answer, averages.
  - category 3 (open-domain) gold answers get truncated at the first `;`
    before scoring, matching a quirk in the paper's eval code.
  - category 5 (adversarial): **not** text-overlap based at all --
    verbatim from their script, 1.0 if the prediction contains the literal
    substring `"no information available"` or `"not mentioned"`, 0.0
    otherwise. This is brittle by construction (very literal phrase
    matching) -- that's the paper's actual method, not something to
    "improve" while the goal is fidelity.
  - `rouge_l_f1` (0009's fix) is kept in `metrics.py` as an available
    secondary signal but is no longer what `score_answer`/`score_batch`
    use.

## Consequences

- **The adversarial QA prompt likely needs revisiting separately.** Our
  `prompts/qa_answer.txt` (docs/decisions/0006) asks the model to "say so
  explicitly" when it can't answer, but doesn't ask for either of the two
  exact phrases the scorer now requires. A well-phrased refusal like "I
  don't have enough context to answer that" now scores 0, including the
  `just demo` example from earlier in this project's history that looked
  like a correct refusal. Whether to adjust the prompt to elicit matching
  phrasing is a separate decision, not made here -- flagged for the user.
- Every previously reported per-category number (this project's own
  results, ADR 0009's before/after table) used the wrong category *name*
  for the same underlying data. The `docs/experiments/` writeup for this
  milestone must use the corrected labels, and should note this history
  rather than silently presenting only the final numbers.
- `scripts/rescore.py` (added in 0009) already supports recomputing an
  existing run's summary against whatever `metrics.py` says today, with no
  new LLM calls -- this fix is "free" to apply to results already on disk.

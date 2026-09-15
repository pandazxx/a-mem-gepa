# 0009: ROUGE-L tokenization bug and cache-freshness fix

## Status

Accepted (2026-09-09)

## Context

The first full `just baseline` run against Llama 3.2:1b produced an
aggregate of 0.282, but with `single_hop`/`temporal`/`multi_hop` all in the
0.03-0.05 range -- near the noise floor. Inspecting real predictions
(`results/baseline/test.json`) turned up cases like predicting "...Caroline's
identity is transgender." against gold "Transgender woman" and scoring
*0.0* -- despite containing the correct word. `rouge_l_f1` (metrics.py)
tokenized with plain `.lower().split()`, so "transgender." (with the
trailing period) was a different token from gold's "transgender", and any
answer phrased as a full sentence lost credit for exactly the word that
mattered.

## Decision

- Tokenize by extracting alphanumeric runs (`re.findall(r"[a-z0-9]+", ...)`)
  instead of splitting on whitespace, so trailing/surrounding punctuation no
  longer breaks a match.
- Rescoring the real baseline run's existing predictions (no new LLM calls)
  with the fix: `single_hop` 0.059->0.123, `temporal` 0.031->0.152,
  `multi_hop` 0.034->0.052, `open_domain` 0.116->0.231, `adversarial`
  1.000->0.915 (some predictions that "avoided" the trap under the old
  tokenization turn out to have actually restated it once matching is
  accurate), `aggregate` 0.282->0.342.
- `evaluate_candidate`'s per-question cache (docs/decisions/0007) now always
  rescores a cached prediction fresh instead of trusting its stored `score`
  -- a metrics.py fix shouldn't require invalidating every on-disk
  prediction cache to take effect.
- Added `scripts/rescore.py` (`just rescore <results.json>`) to recompute an
  already-written results JSON's summaries against the current metrics.py,
  with zero LLM calls -- for exactly this situation, where a scoring bug is
  found after a run that took days.

## Consequences

- Every category's real number moved substantially just from this fix --
  a reminder that a "sanity check" run's job is partly to shake out bugs
  like this one, not just to produce a number to report. The corrected
  numbers are still low enough that real capability/retrieval-quality
  questions remain open (see the ongoing discussion in the PR), not
  resolved by this fix alone.
- `rouge_l_f1` is still not bit-identical to whatever toolkit the paper
  used (docs/decisions/0006 already flagged this) -- this fix makes it a
  more reasonable approximation, not an exact match.

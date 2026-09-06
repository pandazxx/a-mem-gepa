# 0004: LoCoMo train/val/test split

## Status

Accepted (2026-09-06)

## Context

LoCoMo has only 10 conversations (~750 QA pairs each on average, across 5
categories: single-hop, multi-hop, temporal, open-domain, adversarial). GEPA
needs a trainset (for reflective rollouts) and a valset (for Pareto candidate
selection during search); we separately need a held-out test set that GEPA
never sees, for the final reported comparison against baseline. The scarce
resource here is conversation diversity (10), not QA-pair count (~7,500).

## Decision

- **Split unit is the conversation, not the QA pair.** A-MEM builds memory
  incrementally over an entire conversation before questions are asked
  against it; splitting at the QA-pair level would leak memory context
  across train/val/test.
- Default split: **5 train / 2 val / 3 test** conversations, chosen (not
  purely randomly) to keep a comparable mix of the 5 QA categories in each
  split.
- The split is a **fixed, checked-in manifest**
  (`configs/locomo_split.json` — conversation IDs per split), generated once
  a script assigns them, not re-randomized per run.
- Report **per-category metrics**, not just an aggregate, so GEPA overfitting
  the easy categories while regressing hard ones (adversarial, temporal) is
  visible rather than averaged away. Prefer scoring multiple category metrics
  per instance and letting GEPA's Pareto front reflect that, over collapsing
  to one blended scalar, if the adapter implementation supports it cleanly.
- Report a **bootstrap confidence interval** over the test conversations' QA
  pairs (~2,000+) for the headline number, not a bare point estimate.
- **Stretch goal, deferred**: rotate which conversations are held out across
  2-3 different split assignments and report mean ± std, to check the result
  isn't an artifact of one particular 5/2/3 assignment. Only worth the extra
  GEPA runs once a first pass shows a signal worth double-checking.

## Consequences

- Conversation-level variance is real even with many QA pairs per
  conversation — headline numbers should always be read with the CI, not as
  a bare percentage.
- The actual conversation-ID assignment happens once the dataset is
  downloaded (a milestone-1 task), not at scaffold time — this doc fixes the
  *method*, not the concrete IDs yet.

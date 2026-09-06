# 0004: LoCoMo train/val/test split

## Status

Accepted (2026-09-06)

## Context

LoCoMo has only 10 conversations. Verified against the actual released
`locomo10.json` (see [../01-related-work.md](../01-related-work.md)):
**1,986 QA pairs total** (105-260 per conversation), across 5 categories with
an *uneven* distribution — single-hop 282, temporal 321, multi-hop only 96,
open-domain 841, adversarial 446. GEPA needs a trainset (for reflective
rollouts) and a valset (for Pareto candidate selection during search); we
separately need a held-out test set that GEPA never sees, for the final
reported comparison against baseline. The scarce resource here is
conversation diversity (10), not QA-pair count — but multi-hop specifically
is scarce on *both* axes (only ~9.6 pairs/conversation on average).

## Decision

- **Split unit is the conversation, not the QA pair.** A-MEM builds memory
  incrementally over an entire conversation before questions are asked
  against it; splitting at the QA-pair level would leak memory context
  across train/val/test.
- Default split: **5 train / 2 val / 3 test** conversations.
- Assignment method (`datasets/locomo.py:make_split`): **exhaustive search**,
  not a heuristic or random draw. With only 10 conversations, all
  C(10,5)×C(5,2) = 2,520 ways to assign train/val/test are cheap to score
  directly. For each candidate assignment, compute each split's per-category
  QA-pair fraction and compare it to the whole-dataset's per-category
  fraction; pick the assignment minimizing total squared deviation across all
  three splits and all five categories. This is deterministic and
  reproducible — no seed needed, and none is used.
- The split is a **fixed, checked-in manifest**
  (`configs/locomo_split.json` — conversation IDs per split), generated once
  by `scripts/make_locomo_split.py`, not re-randomized per run.
- Report **per-category metrics**, not just an aggregate, so GEPA overfitting
  the easy categories while regressing hard ones (adversarial, temporal) is
  visible rather than averaged away. Prefer scoring multiple category metrics
  per instance and letting GEPA's Pareto front reflect that, over collapsing
  to one blended scalar, if the adapter implementation supports it cleanly.
- Report a **bootstrap confidence interval** over the test conversations' QA
  pairs for the headline number, not a bare point estimate. With a 3-way
  30%-ish split, expect on the order of ~600 total test QA pairs, but as few
  as ~29 for multi-hop specifically — call out multi-hop's per-category CI as
  low-confidence/wide rather than treating it at face value against the
  other four categories.
- **Stretch goal, deferred**: rotate which conversations are held out across
  2-3 different split assignments and report mean ± std, to check the result
  isn't an artifact of one particular 5/2/3 assignment. Only worth the extra
  GEPA runs once a first pass shows a signal worth double-checking.

## Consequences

- Conversation-level variance is real, and for multi-hop specifically the
  per-category sample size is genuinely thin — headline numbers should
  always be read with the CI, not as a bare percentage, and multi-hop
  conclusions should be held to a lower confidence bar than the other four
  categories.
- The category-balanced split found by `make_split` is committed as data
  (`configs/locomo_split.json`), not re-derived at runtime — if the dataset
  file ever changes upstream, regenerate and diff it deliberately rather than
  silently picking up a new assignment.

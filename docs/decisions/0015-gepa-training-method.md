# 0015: GEPA training method — grouped minibatches, fixed val subset, build-count budgeting

## Status

Accepted (2026-09-19). Closes milestone 3's deferred question ("decide the
GEPA trainset composition", docs/00-proposal.md) and the adversarial reward
direction left open in [0013](0013-paper-reproduction-pipeline.md).

## Context

A GEPA rollout here is: fresh memory system with the candidate's prompts →
replay one whole conversation (memory build) → answer questions against the
final memory → score. [experiments/002](../experiments/002-m2-gpt4o-mini-reproduction.md)'s
calibration puts one build at ~45 min / ~$0.35 (GPT-4o-mini via OpenRouter)
and one answered question at ~$0.0005 — a build costs as much as ~700
questions. Every new candidate invalidates all memory state (the mutated
prompts are the ones that build it), and GEPA's engine samples train
minibatches itself — individual (conversation, question) instances scattered
across conversations would cost one build *per question*.

## Decision

- **The budget currency is the memory build**, not the metric call or the
  QA pair. Everything below minimizes distinct (candidate, conversation)
  builds; wall-clock (~45 min/build, sequential engine) is the binding
  constraint, dollars second.
- **Train instances are `LoCoMoQuestionGroup`s**: one conversation + up to
  `per_category` (default 3) of its questions from each of the 5
  categories, questions consumed exactly once across a conversation's
  groups (`datasets/locomo.py:build_question_groups`). One minibatch
  element = one build amortized over ~15 questions. Fixed per-category
  quotas, not proportional draws: multi-hop/open-domain are scarce (~10
  pairs/conversation), and a proportional minibatch would show the
  reflection LM those categories too rarely to ever optimize for them.
  Group scores are the mean over their questions.
- **Val instances are individual questions**: a fixed, deterministic,
  stratified subset (default 15 per category per val conversation, ~150
  total, `build_val_subset`, index-strided — no randomness, per 0004's
  convention). Per-instance granularity is what makes GEPA's
  per-val-instance Pareto front preserve category diversity (0004's
  "per-category metrics, not a blended scalar" preference); the adapter
  groups them by conversation internally, so a full val eval still costs
  exactly 2 builds per candidate. The derived subset is committed as
  `configs/gepa_val_subset.json` (written by the run script on first use);
  a drift between derived and committed aborts the run rather than
  silently shifting the selection.
- **Builds are disk-cached** keyed by (candidate content hash,
  conversation id) — `results/gepa_cache/<hash>/<conv_id>/` holding the
  pickled memories, the saved retriever, and the build's LLM-call trace.
  Re-evaluating a candidate (parent side of a minibatch comparison, Pareto
  re-checks, resumed runs) costs zero builds. Traces are persisted at build
  time precisely so a later `capture_traces=True` evaluation that hits the
  cache still has construction/evolution traces to reflect over.
- **Per-question score**: the repro pipeline's own token-F1
  (`utils.calculate_metrics`) against the gold answer for categories 1–4.
  **Adversarial (category 5) trains on the refusal direction**: score 1.0
  iff the prediction picks "Not mentioned in the conversation" in their
  multiple-choice QA format. This is the paper's *stated intent* for the
  category ("assess models' ability to identify unanswerable queries");
  0013 documented that their own aggregation scores against the trap text
  with no sign flip and left the ambiguity open — for *training* we must
  pick a direction, because GEPA will push hard on whatever is rewarded,
  and rewarding trap-matching would train the memory to mislead. The
  final test-set comparison reports the paper's own metric suite (both
  interpretations shown for category 5), so comparability to
  experiments/002 is preserved regardless. `adversarial_scoring: paper`
  remains selectable for investigating the ambiguity.
- **Reflective dataset = attribution, not raw dumps**
  (`gepa_adapter.make_reflective_dataset`). Notes store the raw turn
  verbatim in this pipeline, so note construction controls
  *retrievability* (keywords/context/tags are what the retriever embeds)
  and evolution controls links (neighborhood expansion at retrieval) and
  neighbor context/tag rewrites. Using LoCoMo's per-question evidence
  dia_ids (now carried on `LoCoMoInstance`): a failed question whose
  evidence turn is absent from the retrieved context indicts the
  construction/evolution calls that touched that turn — those calls (from
  the persisted build trace) become the component's reflection examples,
  with the downstream failure spelled out. Questions that fail with
  evidence retrieved are flagged as likely QA-side and excluded from
  attribution. Every example ends with the component's placeholder/format
  constraints (0014), and a component with no attributable calls gets one
  aggregate per-category-score example rather than an empty dataset.
- **Failure threshold** for "counts as a failure in reflection": F1 < 0.5.
  Above that, the gap is usually phrasing owned by the frozen QA prompt,
  not memory management.
- **Scenario-coverage check, not scenario-datasets**: rollouts traverse
  blank→sparse→dense memory states and same-session→cross-session updates
  automatically by replaying whole conversations; there are no
  per-scenario labels, only downstream QA. What we owe the training loop
  is an audit that reflection examples span those regimes (early and late
  turns, sparse and dense neighborhoods) — to be checked on the first real
  run's traces, recorded in that run's experiment doc.
- **Deferred, explicitly**: truncated-conversation (session-prefix) train
  instances. They would cut build cost/latency several-fold but
  systematically over-represent sparse/blank memory states; only worth
  adopting after a calibration run shows the full-conversation loop is too
  slow in practice. Val/test always stay full-conversation either way.
- **Smoke mode (addendum, 2026-09-19)**: `--smoke` / `just gepa-smoke`
  runs the identical loop on conversations truncated to their first
  `max_turns` turns (questions filtered to evidence-complete ones, same
  logic as 0008's demo sampler), 2 train conversations, ≤10 val questions,
  tiny `max_metric_calls`. This is the sanctioned use of truncation:
  validating build → cache → QA → reflection → acceptance plumbing before
  committing days of wall-clock, **never** producing reported numbers.
  Smoke runs skip the val-subset manifest entirely, and the build cache
  keys include the turn count so a truncated build can never be mistaken
  for (or collide with) a full build of the same conversation.

## Consequences

- GEPA budget accounting: one metric call per batch element, so a full val
  eval costs ~150 calls and each accepted candidate ~150 more.
  `configs/base.yaml`'s `max_metric_calls: 2000` ≈ seed + ~11 accepted
  candidates + ~100 minibatch evals. Estimated builds for such a run:
  ~100–140 train + ~24 val ≈ $45–60 and multiple days of sequential
  wall-clock on GPT-4o-mini — the run plan (`just gepa-optimize`, no
  `--yes`) prints the arithmetic, and CLAUDE.md's explicit-go-ahead rule
  applies before any real run.
- The adversarial training direction (refusal) means the GEPA arm's
  category-5 val scores are *not* comparable to experiments/002's
  category-5 numbers during the run — only the final test-set report,
  which uses the paper's metrics, is.
- `results/gepa_cache/` grows with candidates × conversations (pickled
  memories are a few MB each); it's disposable cache under gitignored
  `results/`, safe to delete between runs at the cost of rebuilding.
- The tail groups of each conversation are category-lopsided (scarce
  categories run out first). Accepted: GEPA's shuffled sampler mixes
  groups across epochs, and the alternative (dropping leftovers) would
  waste exactly the scarce-category questions we quota'd to protect.

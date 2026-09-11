# 001: M2 baseline reproduction — Llama 3.2:1b, full LoCoMo

- Date: 2026-09-11
- Command: `just reproduce` (config: `configs/base.yaml`, commit: `ec4b73b`)
- Pipeline: `src/amem_gepa/paper_repro.py`, via `external/agentic-memory-repro`
  (`WujiangXu/AgenticMemory`) — the paper's own reproduction code, per
  [decisions/0013](../decisions/0013-paper-reproduction-pipeline.md). **Not**
  the `amem_adapter.py`/`evaluate.py` library pipeline (`just baseline`).
- `PAPER_REPRO_BACKEND=ollama`, `PAPER_REPRO_MODEL=llama3.2:1b`,
  `retrieve_k=10` (the vendored repo's own default), `ratio=1.0` (full
  10-conversation dataset, no train/val/test split applied).

## What changed vs. the previous experiment

First successful end-to-end `just reproduce` run — no prior experiment doc
exists for this pipeline. Three real environment issues were hit and fixed
getting here (all on this branch, `topic/m2-baseline-reproduction`):

1. `ensure_repro_repo_importable()` didn't detect an uninitialized git
   submodule (commit `599ce99`).
2. `LookupError` for NLTK's `punkt_tab` resource (commit `dd9bfe1`, kept).
3. A transient `ParseError` from NLTK's own package-index fetch — initially
   given a find-first/retry wrapper (`712b2cc`), then reverted (`ec4b73b`)
   once the user identified it as a local network issue, not something to
   paper over in this project's code.

## Results

Full 10-conversation LoCoMo dataset, n=1986 questions — matches the paper's
own reported total QA-pair count.

Per-category (F1 / BLEU-1 / n), via `format_summary()`:

| category    | F1     | BLEU-1 | n    |
|--------------|--------|--------|------|
| multi_hop    | 0.1666 | 0.1219 | 282  |
| temporal     | 0.0927 | 0.0581 | 321  |
| open_domain  | 0.0870 | 0.0738 | 96   |
| single_hop   | 0.2483 | 0.1882 | 841  |
| adversarial  | 0.3524 | 0.2848 | 446  |
| **overall**  | **0.2271** | **0.1739** | **1986** |

Adversarial's F1/BLEU are the highest of any category — consistent with the
scoring ambiguity flagged in
[decisions/0013](../decisions/0013-paper-reproduction-pipeline.md#context):
`calculate_metrics` scores directly against the trap answer with no
sign-flip found in the vendored code, so this may mean the model matched
the trap text more, not that it correctly abstained. Not resolved by this
run.

## Comparison to the paper's reported Llama 3.2:1b row

Paper's benchmark table (Llama 3.2, 1b, A-Mem row), converted from the
paper's 0–100 scale to match ours ×100 below. The paper reports "Average
Ranking" (F1/BLEU rank across LoCoMo/ReadAgent/MemoryBank/MemGPT/A-Mem for
that model, not a combined score), not an overall F1/BLEU — so there is no
paper "overall" cell to diff against; only the five per-category columns
are directly comparable.

| category    | ours F1 | paper F1 | Δ F1   | ours BLEU-1 | paper BLEU | Δ BLEU |
|-------------|---------|----------|--------|-------------|------------|--------|
| multi_hop   | 16.66   | 19.06    | -2.40  | 12.19       | 11.71      | +0.48  |
| temporal    | 9.27    | 17.80    | -8.53  | 5.81        | 10.28      | -4.47  |
| open_domain | 8.70    | 17.55    | -8.85  | 7.38        | 14.67      | -7.29  |
| single_hop  | 24.83   | 28.51    | -3.68  | 18.82       | 24.13      | -5.31  |
| adversarial | 35.24   | 58.81    | -23.57 | 28.48       | 54.28      | -25.80 |

**This run underperforms the paper's reported A-Mem/Llama-3.2:1b numbers in
every category** except multi_hop BLEU (+0.48, roughly a wash). The gap
grows from mild (multi_hop, single_hop: -2 to -4 F1) to large (open_domain,
temporal: -8 to -9 F1) to severe (adversarial: -23.6 F1 / -25.8 BLEU).

Leading suspect, not yet confirmed: **this run used the vendored code's
default `retrieve_k=10` for every category, but the paper's headline
numbers come from a per-model `k`-sweep (`k ∈ {10,...,50}`,
`run_k_sweep.sh`)** — deferred as a "cheap, do-later" follow-up in
[decisions/0013](../decisions/0013-paper-reproduction-pipeline.md), on the
assumption the gap would be small. This result suggests it isn't small,
at least for the smallest model — worth promoting the k-sweep from
deferred-follow-up to the next thing to actually run before treating this
number as GEPA's baseline, since a GEPA-optimized prompt beating a
baseline that's using a suboptimal `k` wouldn't isolate the prompt's
actual effect.

Other plausible (unranked, unconfirmed) contributors: single-run variance
(no repeated-run averaging here, vs. unknown methodology in the paper),
Ollama's specific `llama3.2:1b` quantization vs. whatever weights the
paper actually ran, and the still-open adversarial-scoring-sign ambiguity
(noted above) inflating or deflating that category's gap specifically.

## Cost

Local Ollama (`llama3.2:1b`), $0 API cost. Wall-clock and token counts not
captured by this run — `paper_repro.py` doesn't currently log either (the
vendored pipeline's own per-conversation caching means most of the cost
signal would need to come from Ollama's own logs, not ours).

## Conclusion

Meets milestone 2's *mechanical* bar from
[00-proposal.md](../00-proposal.md#milestones): a genuine, end-to-end,
full-dataset run using the paper's own code and scoring, on one model
(Llama 3.2:1b) to bound cost/time — no more environment errors, `just
reproduce` runs clean start to finish.

Does **not** yet meet the bar of "verified baseline to compare GEPA
against": this run is meaningfully below the paper's own reported
Llama-3.2:1b/A-Mem numbers in every category but one (see comparison
above), and the leading suspect (fixed `k=10` vs. the paper's per-model
`k`-sweep for its headline numbers) is unconfirmed. Using this number as
GEPA's baseline as-is risks attributing a `k`-selection gap to prompt
quality. Next step before milestone 3: run the `k`-sweep
(`run_k_sweep.sh`-equivalent) for `llama3.2:1b` and see how much of this
gap it closes.

Still open after that: the two milestone-3 questions flagged in
decisions/0013 (does GEPA optimize 2 prompts via the library pipeline, or
4 via this reproduction's note-construction + 3-step evolution flow), and
the adversarial-scoring-sign ambiguity.

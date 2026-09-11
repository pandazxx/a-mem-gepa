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

**Not filled in.** The paper's own benchmark table was shown earlier in this
conversation as a pasted screenshot, not committed anywhere in this repo,
and I don't have its exact per-category numbers memorized precisely enough
to cite without risking a wrong transcription — especially given this
project's explicit goal of a number that's directly comparable to the
paper. Need the paper's Llama 3.2:1b F1/BLEU-1 row (multi_hop / temporal /
open_domain / single_hop / adversarial / overall) re-shared to complete this
section accurately.

## Cost

Local Ollama (`llama3.2:1b`), $0 API cost. Wall-clock and token counts not
captured by this run — `paper_repro.py` doesn't currently log either (the
vendored pipeline's own per-conversation caching means most of the cost
signal would need to come from Ollama's own logs, not ours).

## Conclusion

Meets milestone 2's stated bar from
[00-proposal.md](../00-proposal.md#milestones): a genuine, end-to-end,
full-dataset reproduction using the paper's own code and scoring, on one
model (Llama 3.2:1b) to bound cost/time. This is now the verified baseline
GEPA (milestone 3) prompt candidates get compared against — not
`just baseline`'s number.

Still open: the paper-row comparison above, and the two milestone-3
questions flagged in decisions/0013 (does GEPA optimize 2 prompts via the
library pipeline, or 4 via this reproduction's note-construction + 3-step
evolution flow; and whether to run the paper's `k`-sweep for a headline
number vs. this run's fixed `k=10`).

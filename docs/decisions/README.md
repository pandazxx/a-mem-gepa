# Architecture decision records

One file per decision, numbered in the order they were made. ADR bodies are
never rewritten -- they record what was known and decided at the time. When a
later decision invalidates an earlier one, the earlier ADR's **Status**
section is updated to say so and point forward; the body stays as history.

"Measured path" below means the code behind every reported number:
`just reproduce` / `just k-sweep` (the paper-reproduction pipeline, 0013) and
GEPA optimization on that same pipeline (0014/0015). "Legacy pipeline" means
the `agiresearch/A-mem` library harness (`amem_adapter.py`, `evaluate.py`,
`metrics.py`) behind `just demo` / `just baseline`, kept as a development loop
but behind no reported result.

| ADR | Title | Status (2026-11-08) |
|-----|-------|---------------------|
| [0001](0001-base-repo-and-benchmark.md) | Base A-MEM repo and benchmark choice | Partially superseded by 0013/0014. LoCoMo still the benchmark; the system under test is now `WujiangXu/AgenticMemory`. |
| [0002](0002-optimization-targets.md) | Which prompts GEPA optimizes | Partially superseded by 0014. Verbatim baseline + joint optimization stand; the candidate is 4 repro-pipeline prompts, not 2 library prompts. |
| [0003](0003-model-routing.md) | Model routing via LiteLLM | Partially superseded by 0013/0014. LiteLLM routes `reflection_lm` and the legacy pipeline only; the measured path uses `PAPER_REPRO_*`. |
| [0004](0004-train-val-test-split.md) | LoCoMo train/val/test split | Accepted. Governs GEPA (0015), not `just reproduce` (full dataset). |
| [0005](0005-reproduction-model.md) | Model for the reproduction sanity check | Superseded by 0013 + experiments 001/002. Reproduction ran on the paper's code; the verified baseline is GPT-4o-mini. |
| [0006](0006-qa-answer-prompt.md) | Fixed, non-optimized QA-answering prompt | Superseded by 0013/0014. The paper's own QA prompts exist and are what is held fixed; `qa_answer.txt` is legacy-only. |
| [0007](0007-resumable-eval-runs.md) | Resumable, progress-reporting eval runs | Legacy pipeline only. Measured path resumes via per-conversation pickles + QA progress log (0013) and the build cache (0015). |
| [0008](0008-demo-before-reproduction.md) | A small demo run before the full reproduction | Superseded by 0015's smoke mode for the measured path; `just demo` is a legacy dev tool. |
| [0009](0009-rouge-tokenization-fix.md) | ROUGE-L tokenization bug and cache-freshness fix | Superseded by 0010/0013. ROUGE-L is not the reported metric; the ADR's numbers used the pre-0010 category labels. |
| [0010](0010-paper-exact-f1-and-category-fix.md) | Paper-exact F1; category-label fix | Partially superseded by 0013. Category-label fix stands; the scoring port (from `snap-research/locomo`) is legacy-only. |
| [0011](0011-demo-tracing.md) | Verbose tracing for `just demo` | Legacy pipeline only. Measured path traces via `paper_repro.py` logging (0016) and `gepa_adapter.py` trace recording. |
| [0012](0012-evolution-decision-guard.md) | Reject fabricated evolution decisions | Legacy pipeline only. The guarded bug is specific to `agiresearch/A-mem`; not present in the repro code (0013). |
| [0013](0013-paper-reproduction-pipeline.md) | A separate, faithful paper-reproduction pipeline | Accepted. Its open questions were closed by 0014 (prompt count) and 0015 (adversarial training direction). |
| [0014](0014-gepa-targets-repro-pipeline.md) | GEPA optimizes the repro pipeline's four prompts | Accepted. |
| [0015](0015-gepa-training-method.md) | GEPA training method | Accepted. |
| [0016](0016-disable-ollama-thinking-mode.md) | Disable Ollama thinking mode for the repro pipeline | Accepted. |

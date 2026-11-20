# 0014: GEPA optimizes the repro pipeline's four prompts, injected at runtime

## Status

Accepted (2026-09-19). Supersedes the "which pipeline / how many prompts"
part of [0002](0002-optimization-targets.md); 0002's baseline-is-verbatim
and joint-optimization decisions still stand.

## Context

[0013](0013-paper-reproduction-pipeline.md) left an explicit milestone-3
question open: the verified baseline (`just reproduce`,
[experiments/002](../experiments/002-m2-gpt4o-mini-reproduction.md)) runs
`WujiangXu/AgenticMemory`'s code with **4** memory-management prompts (note
construction + a 3-step evolution flow), while 0002 planned GEPA around the
`agiresearch/A-mem` library's **2** prompts. Optimizing the 2-prompt library
pipeline would mean the final headline comparison uses a baseline number
that is *not* the one milestone 2 spent two experiments verifying against
the paper.

## Decision

- **GEPA's system under test is the paper-repro pipeline** (`external/
  agentic-memory-repro/`, via `RobustAdvancedMemAgent`), not the
  `agiresearch/A-mem` library. The headline comparison is then
  paper-verified-baseline vs. GEPA, on the same pipeline end to end.
- **A candidate is 4 components**, named for what they do, mapped to the
  constants they replace (`repro_injection.COMPONENT_TO_CONSTANT`):
  `note_construction` (ANALYZE_CONTENT_PROMPT), `evolution_decision`
  (EVOLUTION_DECISION_PROMPT), `evolution_strengthen`
  (STRENGTHEN_DETAILS_PROMPT), `evolution_update_neighbors`
  (UPDATE_NEIGHBORS_PROMPT). The 3 evolution prompts are individually more
  optimizable than one mega-prompt (each is small, plain-text-parsed, with
  one job), and GEPA handles multi-component candidates natively
  (round-robin component selection).
- **Injection is runtime patching, not a submodule edit**: the four
  templates are module-level constants imported by name into
  `memory_layer_robust`'s namespace, where every call site resolves them --
  `repro_injection.injected_prompts()` swaps them on that module object for
  the duration of an evaluation and restores the originals after. Same
  category of wrapper as 0013's file-path redirection: nothing scientific
  in the vendored code changes.
- **Frozen, deliberately** (0006's isolation argument, extended):
  `FOCUSED_KEYWORDS_PROMPT` (an error-recovery fallback, not a
  memory-management decision), `generate_query_llm`'s retrieval-keyword
  prompt, the category-specific QA prompts (including the category-5
  multiple-choice mechanic), and all response parsers. If retrieval/QA
  prompts were also mutable, an improvement could come from better querying
  or answer phrasing rather than better memory -- which would change what
  the experiment measures.
- **Template safety is validated before spending**: the prompts are
  consumed via `str.format()`, so an evolved candidate that drops a
  required `{placeholder}`, invents a new one, or emits unbalanced braces
  would crash mid-build after real API spend.
  `repro_injection.validate_candidate()` checks each component against the
  exact field set its call site supplies; invalid candidates score 0
  without running, and the reflective dataset feeds the validation errors
  (plus the placeholder constraints) back to the reflection LM so the next
  mutation can repair the template.
- The seed candidate is the paper's verbatim prompts as files under
  `src/amem_gepa/prompts/baseline_repro/` -- **generated from
  `llm_text_parsers`' constants, not hand-copied** (this project has been
  bitten by transcription before, 0013), with a test asserting
  byte-equality against the vendored submodule so a submodule bump that
  changes a prompt fails loudly.

## Consequences

- `gepa_adapter.py` (previously a 2-prompt skeleton against
  `amem_adapter.py`) is rewritten against the repro pipeline.
  `amem_adapter.py`/`evaluate.py`/`run_baseline.py` remain for the
  `just demo`/`just baseline` development loop, but they are no longer on
  the milestone-4/5 critical path; the GEPA comparison runs entirely on the
  repro pipeline.
- The evolved prompts must keep working with the repro pipeline's fixed
  plain-text parsers (`parse_evolution_decision` expects `DECISION:` /
  `REASON:` lines, etc.). Reflection feedback states this constraint;
  a candidate that drifts to an unparseable output format will score
  poorly (parse failures degrade to no-evolution/heuristics in their code)
  and be selected against, rather than crashing.
- Because both LLM controllers in `RobustAdvancedMemAgent` are wrapped for
  trace recording, every LLM call in a rollout is counted and attributable
  -- the basis for 0015's reflective-dataset attribution and for per-run
  cost accounting.
- Model/backend for GEPA rollouts is configured with the repro pipeline's
  own `PAPER_REPRO_BACKEND`/`PAPER_REPRO_MODEL` variables (0013's scoped
  exception to 0003's LiteLLM routing), so the GEPA arm and the baseline
  are guaranteed to run the same stack. GEPA's `reflection_lm` stays a
  LiteLLM string (`GEPA_REFLECTION_LM`), per 0003.

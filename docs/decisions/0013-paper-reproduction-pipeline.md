# 0013: A separate, faithful paper-reproduction pipeline

## Status

Accepted (2026-09-10)

## Context

`agiresearch/A-mem`'s own README (`external/a-mem/README.md`) has a note we'd
missed: *"This repository provides a memory system to facilitate agent
construction. If you want to reproduce the results presented in our paper,
please refer to: `WujiangXu/AgenticMemory`."* Everything M2 had built so far
-- `amem_adapter.py`, `evaluate.py`, `metrics.py`'s paper-exact F1
(docs/decisions/0010), `prompts/qa_answer.txt` -- was built around
`agiresearch/A-mem`, the general-purpose library, not the paper's actual
reproduction code.

Reading `WujiangXu/AgenticMemory` (MIT licensed, vendored at
`external/agentic-memory-repro/`) directly confirmed it's the real thing --
`run_all_experiments.sh` runs exactly the six models in the benchmark table
shared earlier in this project, via `test_advanced_robust.py`, and prints
`overall['f1']['mean']`/`overall['bleu1']['mean']` as the reported
F1/BLEU columns. It differs from what we'd built in every layer:

- **Retrieval**: an LLM call turns the question into keywords first
  (`generate_query_llm`), *then* those keywords are embedded for
  similarity search -- not the raw question text.
- **QA-answering**: category-specific prompts, and critically, adversarial
  (category 5) questions are **multiple-choice** -- "Select the correct
  answer: {trap} or {'Not mentioned in the conversation'}" with the two
  options randomly ordered -- not open-ended generation scored on refusal
  phrases (which is what docs/decisions/0010 built).
- **Evolution**: 3 separate, conditional, plain-text-parsed LLM calls
  (decision -> strengthen-details -> update-neighbors), not one strict-
  JSON-schema call. Crucially, `suggested_connections`/`CONNECTIONS` are
  explicit **integer positions** matching the "memory index: N" labels
  shown in context, consumed consistently as positions throughout their
  code (`all_memories[i].links` is indexed directly). The ID/index
  mismatch bug docs/decisions/0012 guards against is specific to
  `agiresearch/A-mem`'s library packaging, not present here.
- **Scoring**: a uniform multi-metric suite (`exact_match`, token-set F1,
  ROUGE-1/2/L, BLEU-1-4, BERTScore, METEOR, SBERT cosine similarity) via
  `utils.calculate_metrics`, computed the same way for every category
  against `QA.final_answer` -- not the category-specific dispatch
  (comma-split multi-hop, semicolon-truncated open-domain,
  refusal-phrase-substring adversarial) docs/decisions/0010 ported from
  `snap-research/locomo`'s eval script, a different repo entirely.
- **Dataset scope**: the full 10-conversation LoCoMo dataset, no train/
  val/test split. That split (docs/decisions/0004) is this project's own
  invention for the later GEPA comparison phase -- it isn't part of the
  paper's own benchmark, so a number computed on 3 held-out conversations
  isn't comparable to the paper's reported number at all.

**Explicit user direction**: M2's target is a verified, faithful baseline
that GEPA optimization can be compared against later -- "pointless if we
diverge from the original paper." One model (cost/time), but genuinely
end-to-end, genuinely comparable to the paper's reported number.

**One thing checked and left unresolved, not swept under the rug**: for
category 5, `QA.final_answer` returns the trap text (`adversarial_answer`),
and `calculate_metrics(prediction, final_answer)` scores directly against
it -- no sign-flip found anywhere in their aggregation/reporting code. That
would mean a *higher* adversarial F1/BLEU correlates with the model
matching the trap *more*, not less. The paper text itself gives no
methodological detail on this beyond "adversarial questions assess models'
ability to identify unanswerable queries" (checked directly). Flagging this
as a live ambiguity in the paper's own reported methodology, not something
this project has resolved.

## Decision

- Vendor `WujiangXu/AgenticMemory` read-only at
  `external/agentic-memory-repro/`, same convention as `external/a-mem`.
  It's not a pip-installable package (flat sibling-import scripts, no
  `pyproject.toml`) -- rather than porting its logic by hand (real risk of
  transcription bugs, which this project has hit more than once), import
  its modules directly via `sys.path` insertion
  (`src/amem_gepa/paper_repro.py:ensure_repro_repo_importable`).
- `paper_repro.py:run_full_reproduction` ports `test_advanced_robust.py`'s
  `evaluate_dataset()` orchestration loop, redirecting only *where files
  go* (their version writes memory/retriever pickle caches and logs
  relative to their own module's `__file__`, which would write into our
  read-only vendored submodule directory -- this version writes under
  `results/paper_repro/`, gitignored per CLAUDE.md). Everything that
  affects the actual science -- prompts, retrieval, evolution, category-
  specific QA handling, scoring -- calls straight into their unmodified
  code.
- `scripts/run_paper_reproduction.py` (`just reproduce`) runs the full
  10-conversation dataset by default, `retrieve_k=10` (their own default).
  The paper's headline numbers used a per-model k-sweep
  (`k ∈ {10,...,50}`, `run_k_sweep.sh`) -- deferred as a follow-up (their
  sweep only re-runs the cheap QA-answering step against cached memories,
  not memory-building, so it's a cheap addition later, not a blocker now).
- Model/backend config is separate from the main pipeline's
  (`PAPER_REPRO_BACKEND`/`PAPER_REPRO_MODEL` env vars, `bare model name`,
  not LiteLLM-prefixed) since this reproduction uses their own
  `RobustLLMController` (native `ollama` client, `openai` SDK, or raw
  HTTP for vllm/sglang) rather than our LiteLLM routing
  (docs/decisions/0003) -- a scoped exception for this specific faithful-
  reproduction path, not a reversal of that decision for the rest of the
  project.
- `run_baseline.py`/`evaluate.py`/`metrics.py` (the `agiresearch/A-mem`-
  based pipeline) are **not deleted**. They remain the adapter GEPA will
  optimize prompts against in milestone 3+, since GEPA needs a library it
  can swap prompt strings into at each candidate, not a flat script
  collection. What changes is what we call a "verified baseline" -- that's
  now `just reproduce`'s number, not `just baseline`'s.
- **A second, easy-to-miss Ollama endpoint variable.** `RobustOllamaController`
  (`memory_layer_robust.py`) calls the native `ollama` package's `chat()`
  directly -- no `api_base`/host parameter is threaded through anywhere in
  `RobustAgenticMemorySystem`/`RobustLLMController`/`RobustAdvancedMemAgent`.
  **Correction, verified by re-reading `RobustLLMController.__init__` and
  every call site in this project's own `paper_repro.py`: `api_base` isn't
  threaded through for *any* of the four backends, not just `ollama` as
  originally written here** -- `RobustLLMController`'s factory accepts an
  `api_base` parameter but never actually passes it to any of the
  `openai`/`ollama`/`sglang`/`vllm` controller constructors it builds, and
  nothing on our side passes one either. The `ollama` package resolves its
  endpoint from the `OLLAMA_HOST` env var (default `http://127.0.0.1:11434`),
  read once when its internal client singleton is first constructed. This
  is a *different* variable from `OLLAMA_API_BASE` (used by the
  LiteLLM-routed `just demo`/`just baseline` pipeline, docs/decisions/0003)
  -- added both to `.env.example` with a note, since having two
  same-purpose, differently-named variables for two pipelines is exactly
  the kind of thing that causes a silent misconfiguration later.
- **Routing the `openai` backend through OpenRouter (or any other
  OpenAI-API-compatible endpoint).** `RobustOpenAIController` just does
  `OpenAI(api_key=api_key)` (`memory_layer_robust.py`) -- no `base_url`
  passed, consistent with the `api_base`-is-never-threaded-through
  correction above. The `openai` Python SDK itself, however, falls back to
  the `OPENAI_BASE_URL` env var when `base_url` isn't passed explicitly
  (verified directly against the installed SDK: `OpenAI(api_key=...)`
  with `OPENAI_BASE_URL` set in the environment produces a client whose
  `.base_url` is that value) -- so `PAPER_REPRO_BACKEND=openai` +
  `OPENAI_API_KEY=<your OpenRouter key>` + `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
  + `PAPER_REPRO_MODEL=openai/gpt-4o-mini` (OpenRouter's own model-slug
  naming) routes through OpenRouter with zero code changes, no submodule
  edit required. Added to `.env.example`; also added `openai` to
  `pyproject.toml` (not previously a direct dependency -- only reachable
  via `backend="ollama"` until now).

## Consequences

- Two prompt counts to track going forward: this reproduction uses 4
  (note construction + 3-step evolution), the GEPA-optimizable pipeline
  still uses 2 (docs/decisions/0002). Milestone 3 will need to decide
  whether GEPA's optimization targets expand to match this reproduction's
  architecture, or stay on the simpler 2-prompt one -- not decided here.
- New, heavier dependencies: `ollama`, `rouge-score`, `bert-score`,
  `pandas`, `tqdm` (their `requirements.txt`). `sentence-transformers`/
  `torch`/`transformers`/`scikit-learn`/`rank-bm25` were already pulled in
  transitively via `agentic-memory`.
- Resumability here is coarser than docs/decisions/0007's per-turn
  checkpointing -- their caching is per-conversation (pickle the whole
  `memory_system.memories` dict once a conversation finishes). An
  interruption mid-conversation loses that conversation's progress, not
  just the last turn. Accepted as-is for now, since it's their exact
  mechanism and changing it would be exactly the kind of "improvement that
  diverges from the paper" this ADR exists to avoid; revisit only if it
  becomes actual observed pain, the same way per-turn checkpointing
  (0007) was added after actually hitting a 12+-hour run, not
  speculatively.
- I could not execute this end-to-end myself -- this sandbox has no
  Ollama/GPU (same limitation as every other real run in this project).
  `run_full_reproduction`'s orchestration logic (caching paths, category
  filtering, resume behavior, ratio slicing) is covered by tests against
  stubbed versions of the four imported modules; the actual prompts/
  scoring/retrieval calls into `external/agentic-memory-repro/` are
  untested by this project directly -- they're the paper authors' own
  code, run as-is.

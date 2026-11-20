# 0003: Model routing via LiteLLM, configured by env var

## Status

Partially superseded (2026-11-08) by
[0013](0013-paper-reproduction-pipeline.md) and
[0014](0014-gepa-targets-repro-pipeline.md). Originally accepted 2026-09-06.

- **No longer valid:** "LiteLLM routes every LLM call in this project." The
  measured path -- `just reproduce`, `just k-sweep`, and GEPA's task rollouts
  -- runs the vendored repro code's own `RobustLLMController`, configured by
  `PAPER_REPRO_BACKEND`/`PAPER_REPRO_MODEL` (bare model names, not LiteLLM
  strings); see 0013 for the `OLLAMA_HOST`/`OPENAI_BASE_URL` details.
- **Still in force:** LiteLLM routes GEPA's `reflection_lm`
  (`GEPA_REFLECTION_LM`) and the legacy `just demo`/`just baseline` pipeline
  (`AMEM_LLM_MODEL`, `llm/litellm_controller.py`).

## Context

Need to support a lab-hosted Ollama instance and NVIDIA NIM's free tier, in
addition to Anthropic, for both (a) A-MEM's own LLM backend (the system under
test) and (b) GEPA's `task_lm`/`reflection_lm`. GEPA already accepts LiteLLM
model-ID strings (or a raw callable) for both LM roles by default.

## Decision

- Use LiteLLM as the single routing layer for every LLM call in this project.
- A-MEM's native LLM backend only supports OpenAI/Ollama; add a small
  LiteLLM-backed controller (`src/amem_gepa/llm/litellm_controller.py`) that
  plugs into `AgenticMemorySystem` so its calls go through the same routing
  as GEPA's.
- All model choices are env-var configured, no hardcoded model names:
  `AMEM_LLM_MODEL`, `AMEM_EMBEDDING_MODEL`, `GEPA_TASK_LM`, `GEPA_REFLECTION_LM`,
  plus provider credentials/endpoints (`OLLAMA_API_BASE`, `NVIDIA_NIM_API_KEY`,
  `ANTHROPIC_API_KEY`). See `.env.example`.
- Because NVIDIA NIM's free tier is rate-limited and GEPA runs many rollouts,
  default guidance (not enforced in code) is: use NIM or Ollama for
  `task_lm`/`AMEM_LLM_MODEL` (many cheap calls), reserve a stronger model
  (Anthropic or a larger Ollama model) for `reflection_lm` (few, higher-value
  calls).

## Consequences

- Adding a new provider later is a LiteLLM model-string change plus a
  credential env var, not a code change.
- The LiteLLM controller for A-MEM is new code we own and must keep in sync
  with A-MEM's controller interface across submodule bumps.
- Rate limits on free-tier NIM may throttle GEPA runs; retry/backoff needs to
  be handled in the controller, not assumed away.

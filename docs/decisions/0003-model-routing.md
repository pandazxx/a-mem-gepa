# 0003: Model routing via LiteLLM, configured by env var

## Status

Accepted (2026-09-06)

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

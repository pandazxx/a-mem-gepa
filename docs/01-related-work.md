# Related Work

## A-MEM: Agentic Memory for LLM Agents

- Paper: "A-Mem: Agentic Memory for LLM Agents" (NeurIPS 2025)
- Official memory system library: https://github.com/agiresearch/A-mem
  (vendored here at `external/a-mem` as a git submodule)
- Paper-reproduction code / eval harness: https://github.com/WujiangXu/A-mem-sys

A-MEM organizes an agent's memory as a Zettelkasten-style network of notes
rather than a flat log or fixed-schema store. Two LLM-driven steps matter for
this project:

1. **Note construction** — given a new interaction, an LLM prompt extracts
   structured attributes (context summary, keywords, tags) to turn raw text
   into a memory note.
2. **Memory evolution** — when a new note is added, an LLM prompt (using
   ChromaDB similarity search to find candidate neighbors) decides how
   existing notes' context/tags/links should be updated, letting the memory
   network reorganize itself over time.

Both steps are driven entirely by prompt text against a configurable LLM
backend (natively OpenAI or Ollama). These two prompts are this project's
optimization targets (see [decisions/0002](decisions/0002-optimization-targets.md)).

## GEPA: Reflective Prompt Evolution

- Paper: "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
- Library: https://github.com/gepa-ai/gepa
- DSPy integration: https://github.com/stanfordnlp/dspy (`dspy.GEPA`)

GEPA optimizes any textual artifact (a prompt, a piece of code, a config)
against a metric by having a "reflection" LM read full execution traces —
not just a scalar reward — to diagnose failures and propose targeted rewrites.
It maintains a Pareto front of candidates across generations rather than
converging on a single greedy best, which matters here because we care about
*not regressing* individual LoCoMo question categories, not just improving
the aggregate score.

Two ways to integrate a system with GEPA:

- `dspy.GEPA`, if the system's prompts are expressed as DSPy signatures/modules.
- The raw `GEPAAdapter` interface (`evaluate()` + `make_reflective_dataset()`),
  which we use here since A-MEM's prompts are plain strings, not DSPy modules.

GEPA's `task_lm` and `reflection_lm` accept LiteLLM-style model ID strings by
default (e.g. `"ollama/llama3.1"`, `"anthropic/claude-opus-4-8"`) or a raw
callable, which is what makes routing task/reflection calls through Ollama,
NVIDIA NIM, or Anthropic (see [decisions/0003](decisions/0003-model-routing.md))
straightforward.

## LoCoMo benchmark

- 10 curated multi-session conversations (avg. ~588 turns / ~35 sessions each),
  drawn from a larger 50-conversation pool.
- ~7,500 human-authored QA pairs across five categories: single-hop,
  multi-hop, temporal, open-domain, adversarial (unanswerable-question refusal).
- Used in the original A-MEM paper as an evaluation benchmark, which makes it
  the natural default for a like-for-like baseline comparison
  (see [decisions/0001](decisions/0001-base-repo-and-benchmark.md)).

## Follow-up benchmark: LongMemEval

Deferred to a v2 milestone once the LoCoMo pipeline is validated end-to-end —
tracked as a generalization check (does a prompt optimized on LoCoMo transfer,
or did GEPA just overfit LoCoMo's specific conversation style?).

"""Wraps agentic_memory.memory_system.AgenticMemorySystem so both of its LLM
prompts are plain injectable strings instead of hardcoded/fixed attributes.

Two things this works around, found by reading external/a-mem directly
rather than trusting the paper's description of the code:

1. AgenticMemorySystem.llm_controller only supports backend="openai" or
   "ollama" (see llm_controller.LLMController). To route through Ollama,
   NVIDIA NIM, or Anthropic via LiteLLM (docs/decisions/0003), we replace
   `.llm_controller` after construction with LiteLLMController instead of
   passing an unsupported backend string.

2. analyze_content() -- the note-construction prompt -- is defined on
   AgenticMemorySystem but is dead code in this snapshot: add_note() never
   calls it, so MemoryNote.keywords/context/tags default to [] / "General"
   unless the caller passes them explicitly. PromptInjectableMemorySystem
   below calls analyze_content() itself inside add_note() so that prompt is
   actually exercised end-to-end.
"""

from __future__ import annotations

from typing import Optional

from agentic_memory.memory_system import AgenticMemorySystem

from amem_gepa.llm.litellm_controller import LiteLLMController, parse_json_response

# Same response schema as upstream AgenticMemorySystem.analyze_content, kept
# in sync manually -- see docs/decisions/0001 on why we don't subclass instead.
_NOTE_ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}},
                "context": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


class PromptInjectableMemorySystem(AgenticMemorySystem):
    """AgenticMemorySystem with both LLM prompts swappable at construction time.

    `note_construction_prompt` must contain a `{content}` placeholder.
    `evolution_prompt` must contain the same placeholders as upstream's
    `_evolution_system_prompt` ({content}, {context}, {keywords},
    {nearest_neighbors_memories}, {neighbor_number}) -- GEPA mutates the
    wording around them, not the placeholders themselves.

    Both are plain `str.format()` templates (see prompts/baseline/*.txt), so
    any literal `{`/`}` in a JSON example within the prompt text must be
    doubled (`{{`/`}}`). If GEPA-proposed text drops that escaping, evaluate()
    will raise a KeyError/IndexError on .format() -- treat that as a failed
    rollout (score 0), not a crash to propagate.
    """

    def __init__(
        self,
        note_construction_prompt: str,
        evolution_prompt: str,
        llm_model: str,
        llm_api_base: Optional[str] = None,
        **kwargs,
    ):
        # AgenticMemorySystem.__init__ unconditionally builds a real
        # LLMController before we get a chance to override it below, and
        # defaults to backend="openai" -- which raises ValueError if
        # OPENAI_API_KEY isn't set, even though that controller is about to
        # be discarded. Feed it a throwaway key so that construction doesn't
        # crash; the OpenAI SDK client it builds just stores the string, it
        # doesn't validate it until an actual request is made, and we never
        # make one through it.
        kwargs.setdefault("api_key", "unused-discarded-immediately-below")
        super().__init__(**kwargs)
        self.llm_controller = LiteLLMController(model=llm_model, api_base=llm_api_base)
        self._note_construction_prompt = note_construction_prompt
        self._evolution_system_prompt = evolution_prompt

    def analyze_content(self, content: str) -> dict:
        prompt = self._note_construction_prompt.format(content=content)
        try:
            response = self.llm_controller.llm.get_completion(
                prompt, response_format=_NOTE_ANALYSIS_SCHEMA
            )
            return parse_json_response(response)
        except Exception as exc:  # matches upstream's own broad catch + fallback
            print(f"Error analyzing content: {exc}")
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        if "keywords" not in kwargs:
            analysis = self.analyze_content(content)
            kwargs.setdefault("keywords", analysis["keywords"])
            kwargs.setdefault("context", analysis["context"])
            kwargs.setdefault("tags", analysis["tags"])
        return super().add_note(content, time=time, **kwargs)

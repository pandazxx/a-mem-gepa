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

from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote

from amem_gepa.llm.litellm_controller import LiteLLMController, parse_json_response


class _EvolutionTraceBackend:
    """Wraps a LiteLLMBackend so the evolution-step LLM call prints its
    parsed decision (docs/decisions/0011) -- only installed for the
    duration of one super().add_note() call, and only when trace=True.
    Note-construction tracing doesn't need this: analyze_content() already
    parses its own response and can print directly."""

    def __init__(self, inner):
        self._inner = inner

    def get_completion(self, prompt, response_format=None, temperature=0.7):
        response = self._inner.get_completion(prompt, response_format=response_format, temperature=temperature)
        # Tracing must never be the reason a call fails -- upstream's own
        # process_memory will separately print "Error in memory evolution"
        # and recover if this response is unparseable; mirror that here,
        # don't propagate.
        try:
            decision = parse_json_response(response)
        except Exception:
            decision = None
        if isinstance(decision, dict) and "should_evolve" in decision:
            print(
                f"  [trace:evolution] should_evolve={decision.get('should_evolve')} "
                f"actions={decision.get('actions')} "
                f"suggested_connections={decision.get('suggested_connections')} "
                f"tags_to_update={decision.get('tags_to_update')}"
            )
        else:
            print(f"  [trace:evolution] unparseable response: {str(response)[:200]!r}")
        return response

# Same fields add_note() writes into ChromaDB metadata (memory_system.py) --
# kept in sync manually, same reasoning as _NOTE_ANALYSIS_SCHEMA below.
_NOTE_FIELDS = [
    "id", "content", "keywords", "links", "retrieval_count", "timestamp",
    "last_accessed", "context", "evolution_history", "category", "tags",
]

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
        trace: bool = False,
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
        # docs/decisions/0011 -- verbose per-call tracing, `just demo` only.
        # `just baseline`/`just eval` never pass trace=True, so this is a
        # no-op there.
        self._trace = trace

    def analyze_content(self, content: str) -> dict:
        prompt = self._note_construction_prompt.format(content=content)
        try:
            response = self.llm_controller.llm.get_completion(
                prompt, response_format=_NOTE_ANALYSIS_SCHEMA
            )
            parsed = parse_json_response(response)
        except Exception as exc:  # matches upstream's own broad catch + fallback
            print(f"Error analyzing content: {exc}")
            parsed = {}
        # Smaller/weaker models (docs/decisions/0005) sometimes return JSON
        # that parses fine but is missing a key entirely -- e.g. a truncated
        # response that happens to close its braces early. A successful
        # parse isn't the same as a complete one; fill in the same defaults
        # upstream uses on an outright parse failure, per key, not just here.
        result = {
            "keywords": parsed.get("keywords") or [],
            "context": parsed.get("context") or "General",
            "tags": parsed.get("tags") or [],
        }
        if self._trace:
            print(
                f"  [trace:note_construction] content={content!r}\n"
                f"  [trace:note_construction] -> keywords={result['keywords']} "
                f"context={result['context']!r} tags={result['tags']}"
            )
        return result

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        if "keywords" not in kwargs:
            analysis = self.analyze_content(content)
            kwargs.setdefault("keywords", analysis["keywords"])
            kwargs.setdefault("context", analysis["context"])
            kwargs.setdefault("tags", analysis["tags"])
        if not self._trace:
            return super().add_note(content, time=time, **kwargs)

        # Evolution (process_memory) is upstream code we don't call
        # ourselves -- add_note() triggers it internally. Swap in a tracing
        # backend just for this call so evolution's LLM response gets
        # printed too, then restore the real one immediately after.
        if not self.memories:
            print("  [trace:evolution] skipped (first memory, nothing to compare against)")
            return super().add_note(content, time=time, **kwargs)
        real_backend = self.llm_controller.llm
        self.llm_controller.llm = _EvolutionTraceBackend(real_backend)
        try:
            return super().add_note(content, time=time, **kwargs)
        finally:
            self.llm_controller.llm = real_backend

    def snapshot_notes(self) -> list[dict]:
        """JSON-serializable dump of every note built so far, for
        checkpointing (checkpoint.py) -- a single conversation replay can
        run for hours against a slow local model (docs/experiments), so
        resuming needs to reload this state without re-calling the LLM."""
        return [{field: getattr(note, field) for field in _NOTE_FIELDS} for note in self.memories.values()]

    def restore_notes(self, notes: list[dict]) -> None:
        """Inverse of snapshot_notes(): repopulates self.memories and the
        ChromaDB retriever directly, bypassing add_note()/analyze_content()/
        process_memory() entirely -- no LLM calls, since this data was
        already produced by them in a prior (interrupted) run. Embeddings
        are recomputed locally by ChromaDB on insert, which is cheap."""
        for fields in notes:
            note = MemoryNote(**fields)
            self.memories[note.id] = note
            metadata = {field: getattr(note, field) for field in _NOTE_FIELDS}
            self.retriever.add_document(note.content, metadata, note.id)

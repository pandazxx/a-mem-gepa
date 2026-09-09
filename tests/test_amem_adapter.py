"""Regression test for a real bug hit on a live run (M2, Llama 3.2:1b via
Ollama): AgenticMemorySystem.__init__ eagerly builds its own LLMController
with backend="openai" by default, which raises ValueError if OPENAI_API_KEY
isn't set -- even though PromptInjectableMemorySystem immediately discards
that controller in favor of LiteLLMController. Stubs out `agentic_memory`
(the real package needs ChromaDB/sentence-transformers, too heavy for this
suite) with a fake that reproduces just that crash behavior.
"""

import itertools
import sys
import types

import pytest

_id_counter = itertools.count()


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeLiteLLMResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeRetriever:
    def __init__(self):
        self.documents = {}

    def add_document(self, document, metadata, doc_id):
        self.documents[doc_id] = (document, metadata)


class _FakeMemoryNote:
    def __init__(
        self,
        content,
        id=None,
        keywords=None,
        links=None,
        retrieval_count=None,
        timestamp=None,
        last_accessed=None,
        context=None,
        evolution_history=None,
        category=None,
        tags=None,
    ):
        self.content = content
        self.id = id or f"generated-id-{next(_id_counter)}"
        self.keywords = keywords or []
        self.links = links or []
        self.retrieval_count = retrieval_count or 0
        self.timestamp = timestamp or "t"
        self.last_accessed = last_accessed or "t"
        self.context = context or "General"
        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []


@pytest.fixture
def fake_agentic_memory(monkeypatch):
    calls = {"llm_responses": ["{}"]}

    class FakeAgenticMemorySystem:
        def __init__(self, model_name="all-MiniLM-L6-v2", llm_backend="openai", llm_model="gpt-4o-mini", evo_threshold=100, api_key=None):
            calls["kwargs"] = {
                "model_name": model_name,
                "llm_backend": llm_backend,
                "llm_model": llm_model,
                "api_key": api_key,
            }
            if llm_backend == "openai" and api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.memories = {}
            self.retriever = _FakeRetriever()

        def add_note(self, content, time=None, **kwargs):
            if self.memories:
                # Mirrors upstream: process_memory() calls out to the LLM
                # for an evolution decision once there's something to
                # compare the new note against.
                self.llm_controller.llm.get_completion("evolution prompt", response_format={})
            note = _FakeMemoryNote(content=content, timestamp=time, **kwargs)
            self.memories[note.id] = note
            self.retriever.add_document(note.content, {}, note.id)
            return note.id

    fake_memory_system_module = types.ModuleType("agentic_memory.memory_system")
    fake_memory_system_module.AgenticMemorySystem = FakeAgenticMemorySystem
    fake_memory_system_module.MemoryNote = _FakeMemoryNote
    fake_package = types.ModuleType("agentic_memory")
    fake_package.memory_system = fake_memory_system_module

    calls["llm_call_count"] = 0

    def fake_completion(**kwargs):
        responses = calls["llm_responses"]
        idx = min(calls["llm_call_count"], len(responses) - 1)
        calls["llm_call_count"] += 1
        return _FakeLiteLLMResponse(responses[idx])

    fake_litellm_module = types.ModuleType("litellm")
    fake_litellm_module.completion = fake_completion

    monkeypatch.setitem(sys.modules, "agentic_memory", fake_package)
    monkeypatch.setitem(sys.modules, "agentic_memory.memory_system", fake_memory_system_module)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm_module)

    for mod_name in ["amem_gepa.amem_adapter", "amem_gepa.llm.litellm_controller"]:
        sys.modules.pop(mod_name, None)

    return calls


def test_construction_does_not_require_openai_api_key(fake_agentic_memory, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from amem_gepa.amem_adapter import PromptInjectableMemorySystem

    # Should not raise even though no api_key was passed and OPENAI_API_KEY
    # isn't set -- this is exactly the crash seen on a real Llama 3.2:1b run.
    PromptInjectableMemorySystem(
        note_construction_prompt="construct {content}",
        evolution_prompt="evolve",
        llm_model="ollama/llama3.2:1b",
    )

    assert fake_agentic_memory["kwargs"]["api_key"] is not None


def test_analyze_content_fills_in_missing_keys(fake_agentic_memory):
    """Real crash seen on a live Llama 3.2:1b run: the model returned JSON
    that parsed successfully but was missing "tags" entirely (likely
    truncated output that happened to close its braces early), and
    add_note()'s analysis["tags"] direct indexing raised KeyError."""
    from amem_gepa.amem_adapter import PromptInjectableMemorySystem

    fake_agentic_memory["llm_responses"][0] = '{"keywords": ["a", "b"]}'
    system = PromptInjectableMemorySystem(
        note_construction_prompt="construct {content}",
        evolution_prompt="evolve",
        llm_model="ollama/llama3.2:1b",
    )

    result = system.analyze_content("some memory content")

    assert result == {"keywords": ["a", "b"], "context": "General", "tags": []}


def test_analyze_content_falls_back_on_unparseable_json(fake_agentic_memory):
    from amem_gepa.amem_adapter import PromptInjectableMemorySystem

    fake_agentic_memory["llm_responses"][0] = "not json at all, just rambling"
    system = PromptInjectableMemorySystem(
        note_construction_prompt="construct {content}",
        evolution_prompt="evolve",
        llm_model="ollama/llama3.2:1b",
    )

    result = system.analyze_content("some memory content")

    assert result == {"keywords": [], "context": "General", "tags": []}


def test_snapshot_and_restore_roundtrip_without_llm_calls(fake_agentic_memory):
    """restore_notes() must fully reconstruct memories + retriever state
    from a snapshot, with zero calls to the (expensive, slow) LLM."""
    from amem_gepa.amem_adapter import PromptInjectableMemorySystem

    fake_agentic_memory["llm_responses"][0] = '{"keywords": ["hiking"], "context": "hobbies", "tags": ["x"]}'
    source = PromptInjectableMemorySystem(
        note_construction_prompt="construct {content}",
        evolution_prompt="evolve",
        llm_model="ollama/llama3.2:1b",
    )
    source.add_note("Alice: I love hiking.", time="t1")
    snapshot = source.snapshot_notes()
    assert len(snapshot) == 1

    # A fresh system, with the LLM response now poisoned so any real call
    # would produce different/wrong data -- restore must not touch it.
    fake_agentic_memory["llm_responses"][0] = '{"keywords": ["WRONG"], "context": "WRONG", "tags": ["WRONG"]}'
    restored = PromptInjectableMemorySystem(
        note_construction_prompt="construct {content}",
        evolution_prompt="evolve",
        llm_model="ollama/llama3.2:1b",
    )
    restored.restore_notes(snapshot)

    assert len(restored.memories) == 1
    restored_note = next(iter(restored.memories.values()))
    assert restored_note.keywords == ["hiking"]
    assert restored_note.context == "hobbies"
    assert len(restored.retriever.documents) == 1


def test_trace_true_prints_note_construction_details(fake_agentic_memory, capsys):
    from amem_gepa.amem_adapter import PromptInjectableMemorySystem

    fake_agentic_memory["llm_responses"][0] = '{"keywords": ["hiking"], "context": "hobbies", "tags": ["x"]}'
    system = PromptInjectableMemorySystem(
        note_construction_prompt="construct {content}",
        evolution_prompt="evolve",
        llm_model="ollama/llama3.2:1b",
        trace=True,
    )

    system.add_note("Alice: I love hiking.", time="t1")  # first note -- nothing to evolve against

    out = capsys.readouterr().out
    assert "[trace:note_construction]" in out
    assert "keywords=['hiking']" in out
    assert "[trace:evolution] skipped" in out  # correctly skipped, nothing to compare against yet
    assert "should_evolve" not in out  # no actual evolution decision was made


def test_trace_true_prints_evolution_details_on_second_note(fake_agentic_memory, capsys):
    from amem_gepa.amem_adapter import PromptInjectableMemorySystem

    fake_agentic_memory["llm_responses"] = [
        '{"keywords": ["hiking"], "context": "hobbies", "tags": ["x"]}',  # note 1 construction
        '{"keywords": ["camping"], "context": "hobbies", "tags": ["y"]}',  # note 2 construction
        '{"should_evolve": true, "actions": ["strengthen"], "suggested_connections": ["n1"], '
        '"tags_to_update": ["outdoors"], "new_context_neighborhood": [], "new_tags_neighborhood": []}',  # evolution
    ]
    system = PromptInjectableMemorySystem(
        note_construction_prompt="construct {content}",
        evolution_prompt="evolve",
        llm_model="ollama/llama3.2:1b",
        trace=True,
    )

    system.add_note("Alice: I love hiking.", time="t1")
    capsys.readouterr()  # discard first note's trace, only care about the second below
    system.add_note("Alice: I also love camping.", time="t2")

    out = capsys.readouterr().out
    assert "[trace:note_construction]" in out
    assert "keywords=['camping']" in out
    assert "[trace:evolution]" in out
    assert "should_evolve=True" in out
    assert "actions=['strengthen']" in out


def test_trace_false_by_default_produces_no_trace_output(fake_agentic_memory, capsys):
    """Regression guard: run_baseline.py/run_eval.py never pass trace=True,
    so this must stay silent by default."""
    from amem_gepa.amem_adapter import PromptInjectableMemorySystem

    fake_agentic_memory["llm_responses"] = [
        '{"keywords": ["hiking"], "context": "hobbies", "tags": ["x"]}',
        '{"keywords": ["camping"], "context": "hobbies", "tags": ["y"]}',
        '{"should_evolve": false, "actions": [], "suggested_connections": [], '
        '"tags_to_update": [], "new_context_neighborhood": [], "new_tags_neighborhood": []}',
    ]
    system = PromptInjectableMemorySystem(
        note_construction_prompt="construct {content}",
        evolution_prompt="evolve",
        llm_model="ollama/llama3.2:1b",
    )

    system.add_note("Alice: I love hiking.", time="t1")
    system.add_note("Alice: I also love camping.", time="t2")

    out = capsys.readouterr().out
    assert "[trace:" not in out

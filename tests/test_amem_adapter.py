"""Regression test for a real bug hit on a live run (M2, Llama 3.2:1b via
Ollama): AgenticMemorySystem.__init__ eagerly builds its own LLMController
with backend="openai" by default, which raises ValueError if OPENAI_API_KEY
isn't set -- even though PromptInjectableMemorySystem immediately discards
that controller in favor of LiteLLMController. Stubs out `agentic_memory`
(the real package needs ChromaDB/sentence-transformers, too heavy for this
suite) with a fake that reproduces just that crash behavior.
"""

import sys
import types

import pytest


@pytest.fixture
def fake_agentic_memory(monkeypatch):
    calls = {}

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

    fake_memory_system_module = types.ModuleType("agentic_memory.memory_system")
    fake_memory_system_module.AgenticMemorySystem = FakeAgenticMemorySystem
    fake_package = types.ModuleType("agentic_memory")
    fake_package.memory_system = fake_memory_system_module

    monkeypatch.setitem(sys.modules, "agentic_memory", fake_package)
    monkeypatch.setitem(sys.modules, "agentic_memory.memory_system", fake_memory_system_module)
    monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))

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

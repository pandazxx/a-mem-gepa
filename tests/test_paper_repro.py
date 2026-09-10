"""Tests for paper_repro.py's own orchestration logic (caching, path
redirection, category filtering, result shape) -- stubs out the four
sibling modules it imports from external/agentic-memory-repro/
(test_advanced_robust, load_dataset, utils, llm_text_parsers), since those
pull in torch/sentence-transformers/bert-score, too heavy for this suite
(same reasoning as test_amem_adapter.py's agentic_memory stub).
"""

import sys
import types

import pytest

from amem_gepa.paper_repro import ensure_repro_repo_importable, format_summary


def test_ensure_repro_repo_importable_adds_real_submodule_to_path():
    # external/agentic-memory-repro is a real submodule in this repo --
    # this doesn't stub anything, it's checking the actual vendored path.
    ensure_repro_repo_importable()
    import amem_gepa.paper_repro as pr

    assert str(pr.REPRO_DIR) in sys.path
    assert pr.REPRO_DIR.exists()


def test_format_summary_uses_correct_category_labels():
    final_results = {
        "model": "llama3.2:1b",
        "backend": "ollama",
        "retrieve_k": 10,
        "total_questions": 3,
        "category_distribution": {"1": 1, "5": 2},
        "aggregate_metrics": {
            "overall": {"f1": {"mean": 0.5}, "bleu1": {"mean": 0.4}},
            "category_1": {"f1": {"mean": 0.3, "count": 1}, "bleu1": {"mean": 0.2}},
            "category_5": {"f1": {"mean": 0.7, "count": 2}, "bleu1": {"mean": 0.6}},
        },
    }

    out = format_summary(final_results)

    assert "multi_hop" in out  # category 1, per docs/decisions/0010
    assert "adversarial" in out  # category 5
    assert "overall" in out


# ---------------------------------------------------------------------------
# Fakes for external/agentic-memory-repro/'s modules
# ---------------------------------------------------------------------------

class _FakeTurn:
    def __init__(self, speaker, text):
        self.speaker = speaker
        self.text = text


class _FakeSession:
    def __init__(self, date_time, turns):
        self.date_time = date_time
        self.turns = turns


class _FakeQA:
    def __init__(self, question, category, final_answer):
        self.question = question
        self.category = category
        self.final_answer = final_answer


class _FakeSample:
    def __init__(self, sample_id, sessions, qa):
        class _Conv:
            pass

        self.sample_id = sample_id
        self.conversation = _Conv()
        self.conversation.sessions = sessions
        self.qa = qa


class _FakeRetriever:
    def __init__(self):
        self.saved = False

    def save(self, *args, **kwargs):
        self.saved = True

    def load(self, *args, **kwargs):
        return self

    def load_from_local_memory(self, *args, **kwargs):
        return self


class _FakeMemorySystem:
    def __init__(self):
        self.memories = {}
        self.retriever = _FakeRetriever()


class _FakeAgent:
    instances = []

    def __init__(self, model, backend, retrieve_k, temperature_c5):
        self.model = model
        self.backend = backend
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.memory_system = _FakeMemorySystem()
        self.added = []
        _FakeAgent.instances.append(self)

    def add_memory(self, content, time=None):
        self.added.append(content)
        self.memory_system.memories[f"note-{len(self.added)}"] = content

    def answer_question(self, question, category, answer):
        return (f"answer to: {question}", "prompt", "context")


@pytest.fixture
def fake_repro_modules(monkeypatch, tmp_path):
    _FakeAgent.instances = []

    fake_test_advanced_robust = types.ModuleType("test_advanced_robust")
    fake_test_advanced_robust.RobustAdvancedMemAgent = _FakeAgent

    fake_load_dataset = types.ModuleType("load_dataset")

    def load_locomo_dataset(path):
        turns_a = [_FakeTurn("Alice", "hi"), _FakeTurn("Bob", "hey")]
        sessions_a = {1: _FakeSession("t1", turns_a)}
        qa_a = [
            _FakeQA("Q1?", 1, "gold1"),
            _FakeQA("Q2?", 5, "trap"),
        ]
        turns_b = [_FakeTurn("Carol", "yo")]
        sessions_b = {1: _FakeSession("t2", turns_b)}
        qa_b = [_FakeQA("Q3?", 4, "gold3")]
        return [
            _FakeSample("conv-a", sessions_a, qa_a),
            _FakeSample("conv-b", sessions_b, qa_b),
        ]

    fake_load_dataset.load_locomo_dataset = load_locomo_dataset

    fake_utils = types.ModuleType("utils")
    fake_utils.calculate_metrics = lambda prediction, reference: {"f1": 1.0, "bleu1": 1.0}
    fake_utils.aggregate_metrics = lambda all_metrics, all_categories: {
        "overall": {"f1": {"mean": 1.0}, "bleu1": {"mean": 1.0}},
    }

    fake_llm_text_parsers = types.ModuleType("llm_text_parsers")
    fake_llm_text_parsers.parse_plain_text_answer = lambda response: response

    monkeypatch.setitem(sys.modules, "test_advanced_robust", fake_test_advanced_robust)
    monkeypatch.setitem(sys.modules, "load_dataset", fake_load_dataset)
    monkeypatch.setitem(sys.modules, "utils", fake_utils)
    monkeypatch.setitem(sys.modules, "llm_text_parsers", fake_llm_text_parsers)

    return tmp_path


def test_run_full_reproduction_replays_all_conversations_and_questions(fake_repro_modules):
    from amem_gepa.paper_repro import run_full_reproduction

    results = run_full_reproduction(
        dataset_path=fake_repro_modules / "fake.json",
        backend="ollama",
        model="llama3.2:1b",
        retrieve_k=10,
        results_dir=fake_repro_modules / "results",
    )

    assert results["total_questions"] == 3
    assert len(_FakeAgent.instances) == 2, "one agent per conversation"
    assert _FakeAgent.instances[0].added == ["Speaker Alicesays : hi", "Speaker Bobsays : hey"]
    assert results["category_distribution"] == {"1": 1, "5": 1, "4": 1}


def test_run_full_reproduction_caches_under_results_dir_not_the_submodule(fake_repro_modules):
    from amem_gepa.paper_repro import REPRO_DIR, run_full_reproduction

    results_dir = fake_repro_modules / "results"
    run_full_reproduction(
        dataset_path=fake_repro_modules / "fake.json",
        backend="ollama",
        model="llama3.2:1b",
        results_dir=results_dir,
    )

    cache_dir = results_dir / "cached_memories_ollama_llama3.2:1b"
    assert cache_dir.exists()
    assert (cache_dir / "memory_cache_sample_0.pkl").exists()
    # Nothing should have been written into the read-only vendored submodule.
    assert not (REPRO_DIR / "cached_memories_ollama_llama3.2:1b").exists()


def test_run_full_reproduction_resumes_from_cache_without_replaying(fake_repro_modules):
    from amem_gepa.paper_repro import run_full_reproduction

    results_dir = fake_repro_modules / "results"
    common_kwargs = dict(
        dataset_path=fake_repro_modules / "fake.json",
        backend="ollama",
        model="llama3.2:1b",
        results_dir=results_dir,
    )

    run_full_reproduction(**common_kwargs)
    first_run_agent_count = len(_FakeAgent.instances)

    run_full_reproduction(**common_kwargs)

    # New agent objects are still created (one per conversation, per run),
    # but their memories should come from the cache, not add_memory().
    second_run_agents = _FakeAgent.instances[first_run_agent_count:]
    assert len(second_run_agents) == 2
    assert all(agent.added == [] for agent in second_run_agents), "cached conversations must not be replayed"


def test_run_full_reproduction_respects_ratio(fake_repro_modules):
    from amem_gepa.paper_repro import run_full_reproduction

    results = run_full_reproduction(
        dataset_path=fake_repro_modules / "fake.json",
        backend="ollama",
        model="llama3.2:1b",
        ratio=0.5,
        results_dir=fake_repro_modules / "results",
    )

    # ratio=0.5 of 2 conversations -> just conv-a's 2 questions
    assert results["total_questions"] == 2

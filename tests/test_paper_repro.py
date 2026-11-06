"""Tests for paper_repro.py's own orchestration logic (caching, path
redirection, category filtering, result shape) -- stubs out the four
sibling modules it imports from external/agentic-memory-repro/
(test_advanced_robust, load_dataset, utils, llm_text_parsers), since those
pull in torch/sentence-transformers/bert-score, too heavy for this suite
(same reasoning as test_amem_adapter.py's agentic_memory stub).
"""

import json
import sys
import types

import pytest

from amem_gepa.paper_repro import ensure_nltk_data, ensure_repro_repo_importable, format_summary


def test_ensure_repro_repo_importable_adds_real_submodule_to_path():
    # external/agentic-memory-repro is a real submodule in this repo --
    # this doesn't stub anything, it's checking the actual vendored path.
    ensure_repro_repo_importable()
    import amem_gepa.paper_repro as pr

    assert str(pr.REPRO_DIR) in sys.path
    assert pr.REPRO_DIR.exists()


def test_ensure_repro_repo_importable_gives_a_clear_error_when_uninitialized(tmp_path, monkeypatch):
    """Real bug hit on a live run: an uninitialized git submodule leaves an
    *empty* directory in place, not a missing one -- REPRO_DIR.exists()
    alone would pass, and the failure would only surface much later as a
    bare `ModuleNotFoundError: No module named 'test_advanced_robust'`
    from deep inside the lazy import in run_full_reproduction()."""
    import amem_gepa.paper_repro as pr

    empty_dir = tmp_path / "external" / "agentic-memory-repro"
    empty_dir.mkdir(parents=True)  # exists, but empty -- exactly an uninitialized submodule
    monkeypatch.setattr(pr, "REPRO_DIR", empty_dir)

    with pytest.raises(FileNotFoundError, match="git submodule update --init"):
        pr.ensure_repro_repo_importable()


def test_ensure_nltk_data_downloads_punkt_tab():
    """Real bug hit on a live run: external/agentic-memory-repro's own
    pre-flight check (test_advanced_robust.py, utils.py) only downloads
    'punkt'/'wordnet' -- correct when written, but NLTK 3.8.2+ split
    punkt's tokenizer data into a separate 'punkt_tab' resource that
    word_tokenize() needs at call time, so their check doesn't catch a
    missing one and it surfaces later as a bare LookupError deep inside
    nltk's tokenizer. This is real (not stubbed) nltk -- confirms the fix
    actually resolves the LookupError, not just that download() was called."""
    import nltk

    ensure_nltk_data()

    # Would raise LookupError before ensure_nltk_data() downloaded punkt_tab.
    assert nltk.word_tokenize("This is a test sentence.") == [
        "This", "is", "a", "test", "sentence", "."
    ]


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
        self.answered = []
        _FakeAgent.instances.append(self)

    def add_memory(self, content, time=None):
        self.added.append(content)
        self.memory_system.memories[f"note-{len(self.added)}"] = content

    def answer_question(self, question, category, answer):
        self.answered.append(question)
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


def test_run_k_sweep_reuses_cached_memories_across_k_values(fake_repro_modules):
    from amem_gepa.paper_repro import run_k_sweep

    results_dir = fake_repro_modules / "results"
    results_by_k = run_k_sweep(
        dataset_path=fake_repro_modules / "fake.json",
        backend="ollama",
        model="llama3.2:1b",
        k_values=(10, 20, 30),
        results_dir=results_dir,
    )

    assert set(results_by_k) == {10, 20, 30}
    for result in results_by_k.values():
        assert result["total_questions"] == 3

    # 2 conversations for k=10 (memory-building), then cache hits for the
    # other two k values -- no further add_memory() calls for any of them.
    assert len(_FakeAgent.instances) == 6, "one agent per conversation per k"
    building_agents, cached_agents = _FakeAgent.instances[:2], _FakeAgent.instances[2:]
    assert all(agent.added for agent in building_agents)
    assert all(agent.added == [] for agent in cached_agents), "later k values must reuse cached memories"


def test_run_k_sweep_writes_one_result_file_per_k(fake_repro_modules):
    from amem_gepa.paper_repro import run_k_sweep

    results_dir = fake_repro_modules / "results"
    run_k_sweep(
        dataset_path=fake_repro_modules / "fake.json",
        backend="ollama",
        model="llama3.2:1b",
        k_values=(10, 20),
        results_dir=results_dir,
    )

    assert (results_dir / "k_sweep" / "results_ollama_llama3.2:1b_k10.json").exists()
    assert (results_dir / "k_sweep" / "results_ollama_llama3.2:1b_k20.json").exists()


def test_run_k_sweep_resumes_from_an_existing_k_output_file(fake_repro_modules):
    from amem_gepa.paper_repro import run_k_sweep

    results_dir = fake_repro_modules / "results"
    sweep_dir = results_dir / "k_sweep"
    sweep_dir.mkdir(parents=True)
    sentinel = {"total_questions": 999, "aggregate_metrics": {"overall": {"f1": {"mean": 0.42}}}}
    (sweep_dir / "results_ollama_llama3.2:1b_k99.json").write_text(json.dumps(sentinel))

    results_by_k = run_k_sweep(
        dataset_path=fake_repro_modules / "fake.json",
        backend="ollama",
        model="llama3.2:1b",
        k_values=(99,),
        results_dir=results_dir,
    )

    assert results_by_k[99] == sentinel
    assert _FakeAgent.instances == [], "a resumed k must not re-run QA-answering at all"


def test_run_k_sweep_does_not_reuse_another_models_cached_k_result(fake_repro_modules):
    """Real bug hit on a live run: the per-k output filename used to be
    just f"results_k{k}.json" -- no backend/model in it. Sweeping a second
    model at the same k values silently loaded the *first* model's cached
    result file and reported it as the second model's, no error, no
    warning -- confirmed on a real run where two completely different
    models (GPT-4o-mini and a local Ollama model) produced byte-identical
    output."""
    from amem_gepa.paper_repro import run_k_sweep

    results_dir = fake_repro_modules / "results"
    run_k_sweep(
        dataset_path=fake_repro_modules / "fake.json",
        backend="openai",
        model="openai/gpt-4o-mini",
        k_values=(40,),
        results_dir=results_dir,
    )
    first_model_agent_count = len(_FakeAgent.instances)

    results_by_k = run_k_sweep(
        dataset_path=fake_repro_modules / "fake.json",
        backend="ollama",
        model="qwen3.5:9b",
        k_values=(40,),
        results_dir=results_dir,
    )

    second_model_agents = _FakeAgent.instances[first_model_agent_count:]
    assert second_model_agents, "a different model at the same k must actually run, not load the first model's cache"
    assert any(agent.answered for agent in second_model_agents)
    assert results_by_k[40] is not None


def test_format_k_sweep_summary_picks_best_overall_f1_and_breaks_down_by_category():
    from amem_gepa.paper_repro import format_k_sweep_summary

    results_by_k = {
        10: {
            "category_distribution": {"1": 1},
            "aggregate_metrics": {
                "overall": {"f1": {"mean": 0.20}, "bleu1": {"mean": 0.10}},
                "category_1": {"f1": {"mean": 0.20, "count": 1}, "bleu1": {"mean": 0.10}},
            },
        },
        20: {
            "category_distribution": {"1": 1},
            "aggregate_metrics": {
                "overall": {"f1": {"mean": 0.35}, "bleu1": {"mean": 0.30}},
                "category_1": {"f1": {"mean": 0.35, "count": 1}, "bleu1": {"mean": 0.30}},
            },
        },
    }

    out = format_k_sweep_summary(results_by_k)

    assert "best_k=" not in out  # not a literal token -- just guards against a leftover debug format
    assert "best overall F1" in out
    lines = out.splitlines()
    best_line = next(line for line in lines if "best overall F1" in line)
    assert best_line.strip().startswith("20"), "k=20 has the higher overall F1 and should be marked best"
    assert "multi_hop" in out


def test_run_full_reproduction_resumes_qa_answering_from_progress_file(fake_repro_modules):
    """Real bug hit on a live run: a GPT-4o-mini/OpenRouter run got killed
    mid QA-answering -- unlike memory-building (cached per-conversation),
    the QA-answering loop had zero checkpointing before this, so a rerun
    re-answered every question from scratch, wasting real API spend.
    Pre-seeding one question's result in the progress file simulates
    resuming after a kill partway through."""
    import json

    from amem_gepa.paper_repro import _qa_progress_path, run_full_reproduction

    results_dir = fake_repro_modules / "results"
    progress_path = _qa_progress_path(results_dir, "ollama", "llama3.2:1b", 10)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps({
        "sample_id": 0, "q_idx": 0, "question": "Q1?", "prediction": "cached answer",
        "reference": "gold1", "category": 1, "metrics": {"f1": 0.99, "bleu1": 0.99},
    }) + "\n")

    results = run_full_reproduction(
        dataset_path=fake_repro_modules / "fake.json",
        backend="ollama",
        model="llama3.2:1b",
        retrieve_k=10,
        results_dir=results_dir,
    )

    all_answered = [q for agent in _FakeAgent.instances for q in agent.answered]
    assert "Q1?" not in all_answered, "already-cached question must not be re-answered"
    assert "Q2?" in all_answered and "Q3?" in all_answered, "uncached questions still get answered"
    assert results["total_questions"] == 3
    cached_result = next(r for r in results["individual_results"] if r["question"] == "Q1?")
    assert cached_result["prediction"] == "cached answer"


def test_run_full_reproduction_writes_qa_progress_incrementally(fake_repro_modules):
    import json

    from amem_gepa.paper_repro import _qa_progress_path, run_full_reproduction

    results_dir = fake_repro_modules / "results"
    run_full_reproduction(
        dataset_path=fake_repro_modules / "fake.json",
        backend="ollama",
        model="llama3.2:1b",
        retrieve_k=10,
        results_dir=results_dir,
    )

    progress_path = _qa_progress_path(results_dir, "ollama", "llama3.2:1b", 10)
    lines = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 3
    assert {line["question"] for line in lines} == {"Q1?", "Q2?", "Q3?"}


def test_run_full_reproduction_qa_progress_is_isolated_per_retrieve_k(fake_repro_modules):
    """QA answers legitimately differ by k -- a progress file from one k
    must not short-circuit answering at a different k."""
    from amem_gepa.paper_repro import run_full_reproduction

    results_dir = fake_repro_modules / "results"
    run_full_reproduction(
        dataset_path=fake_repro_modules / "fake.json",
        backend="ollama",
        model="llama3.2:1b",
        retrieve_k=10,
        results_dir=results_dir,
    )
    first_run_agent_count = len(_FakeAgent.instances)

    run_full_reproduction(
        dataset_path=fake_repro_modules / "fake.json",
        backend="ollama",
        model="llama3.2:1b",
        retrieve_k=20,
        results_dir=results_dir,
    )

    second_run_agents = _FakeAgent.instances[first_run_agent_count:]
    all_answered = [q for agent in second_run_agents for q in agent.answered]
    assert all_answered.count("Q1?") == 1, "k=20 must answer every question fresh, not reuse k=10's progress"


def test_run_full_reproduction_qa_progress_survives_a_slash_in_model_name(fake_repro_modules):
    """Real bug hit on a live run: model="openai/gpt-4o-mini" (OpenRouter's
    naming) makes _qa_progress_path land in a nested directory (pathlib
    splits on the "/"), same as the existing cached_memories_{backend}_{model}
    dir already does -- but _append_qa_progress's open(path, "a") doesn't
    create parent directories, unlike the memory cache dir's mkdir(parents=True).
    Raised a bare FileNotFoundError on the very first answered question."""
    from amem_gepa.paper_repro import run_full_reproduction

    results_dir = fake_repro_modules / "results"

    results = run_full_reproduction(
        dataset_path=fake_repro_modules / "fake.json",
        backend="openai",
        model="openai/gpt-4o-mini",
        retrieve_k=40,
        results_dir=results_dir,
    )

    assert results["total_questions"] == 3
    progress_file = results_dir / "qa_progress_openai_openai" / "gpt-4o-mini_k40.jsonl"
    assert progress_file.exists()


def _fake_category_result(f1_by_cat, bleu_by_cat, n_by_cat):
    return {
        "aggregate_metrics": {
            f"category_{cat}": {"f1": {"mean": f1_by_cat[cat], "count": n_by_cat[cat]}, "bleu1": {"mean": bleu_by_cat[cat]}}
            for cat in f1_by_cat
        }
    }


def test_splice_per_category_results_picks_the_right_k_per_category():
    from amem_gepa.paper_repro import splice_per_category_results

    results_by_k = {
        40: _fake_category_result(
            {1: 0.20, 2: 0.30, 5: 0.40}, {1: 0.15, 2: 0.25, 5: 0.35}, {1: 10, 2: 10, 5: 10}
        ),
        50: _fake_category_result(
            {3: 0.50, 4: 0.60}, {3: 0.45, 4: 0.55}, {3: 10, 4: 10}
        ),
    }
    category_k = {1: 40, 2: 40, 3: 50, 4: 50, 5: 40}

    spliced = splice_per_category_results(results_by_k, category_k)

    assert spliced[1] == {"f1": 0.20, "bleu1": 0.15, "n": 10, "k": 40}
    assert spliced[3] == {"f1": 0.50, "bleu1": 0.45, "n": 10, "k": 50}
    # n-weighted mean across all 5 categories, all n=10 here so it's a plain mean
    assert spliced["overall"]["f1"] == pytest.approx((0.20 + 0.30 + 0.50 + 0.60 + 0.40) / 5)
    assert spliced["overall"]["n"] == 50


def test_splice_per_category_results_raises_on_missing_k():
    from amem_gepa.paper_repro import splice_per_category_results

    results_by_k = {40: _fake_category_result({1: 0.20}, {1: 0.15}, {1: 10})}
    category_k = {1: 40, 3: 50}  # k=50 never run

    with pytest.raises(KeyError, match="50"):
        splice_per_category_results(results_by_k, category_k)


def test_format_spliced_summary_shows_k_per_category():
    from amem_gepa.paper_repro import format_spliced_summary

    spliced = {
        1: {"f1": 0.27, "bleu1": 0.20, "n": 282, "k": 40},
        3: {"f1": 0.15, "bleu1": 0.12, "n": 96, "k": 50},
        "overall": {"f1": 0.30, "bleu1": 0.25, "n": 378},
    }

    out = format_spliced_summary(spliced)

    assert "multi_hop" in out and " 40 " in out
    assert "open_domain" in out and " 50 " in out
    assert "overall" in out


# ---------------------------------------------------------------------------
# Progress tracing / ETA / diagnosis (added after a real run -- even on a
# GPU upgrade that should comfortably fit the model -- took 3+ days on a
# single k and gave zero visibility into whether it was progressing,
# stalled, or just slow throughout).
# ---------------------------------------------------------------------------

import logging


def test_format_duration_formats_various_magnitudes():
    from amem_gepa.paper_repro import _format_duration

    assert _format_duration(5) == "5s"
    assert _format_duration(65) == "1m5s"
    assert _format_duration(3665) == "1h1m"
    assert _format_duration(90000) == "1d1h0m"


def test_progress_tracker_avg_latency_excludes_cached_entries():
    from amem_gepa.paper_repro import _ProgressTracker

    tracker = _ProgressTracker(logging.getLogger("test"), "label", total_units=10)
    tracker.record(0.0, cached=True)
    tracker.record(0.0, cached=True)
    tracker.record(10.0, cached=False)
    tracker.record(20.0, cached=False)

    assert tracker.cached_units == 2
    assert tracker.completed_units == 4
    assert tracker.avg_latency == pytest.approx(15.0), "cached (near-instant) entries must not drag the average down"


def test_progress_tracker_eta_uses_avg_latency_times_remaining_units():
    from amem_gepa.paper_repro import _ProgressTracker

    tracker = _ProgressTracker(logging.getLogger("test"), "label", total_units=10)
    for _ in range(4):
        tracker.record(10.0, cached=False)

    # 4/10 done at 10s/call avg -> 6 remaining -> ETA 60s
    assert tracker.eta_seconds == pytest.approx(60.0)


def test_progress_tracker_eta_is_none_before_any_real_call():
    from amem_gepa.paper_repro import _ProgressTracker

    tracker = _ProgressTracker(logging.getLogger("test"), "label", total_units=10)
    assert tracker.eta_seconds is None
    tracker.record(0.0, cached=True)  # a cache hit alone still shouldn't produce a fake ETA
    assert tracker.eta_seconds is None


def test_progress_tracker_outlier_detection_needs_warmup_then_flags_slow_calls():
    from amem_gepa.paper_repro import _ProgressTracker

    tracker = _ProgressTracker(logging.getLogger("test"), "label", total_units=100)
    # Not enough real calls yet -- a lone slow first call shouldn't trip
    # the detector against an average that's essentially just itself.
    assert tracker.is_outlier(100.0) is False
    tracker.record(100.0, cached=False)
    assert tracker.is_outlier(100.0) is False  # still only 1 real call, under the 5-call warmup

    # 4 more real calls at 10s each -> 5 total, avg = (100+10*4)/5 = 28s
    for _ in range(4):
        tracker.record(10.0, cached=False)
    assert tracker.avg_latency == pytest.approx(28.0)
    # a 50s call (< 3x the 28s avg) should not flag; a 100s call (> 3x) should.
    assert tracker.is_outlier(50.0) is False
    assert tracker.is_outlier(100.0) is True


def test_progress_tracker_progress_line_includes_expected_fields():
    from amem_gepa.paper_repro import _ProgressTracker

    tracker = _ProgressTracker(logging.getLogger("test"), "QA-answering k=50", total_units=10)
    tracker.record(5.0, cached=False)
    tracker.record(0.0, cached=True)

    line = tracker.progress_line(extra="sample=0 cat=5")

    assert "QA-answering k=50: 2/10" in line
    assert "1 cached" in line
    assert "avg=5.0s" in line
    assert "ETA=" in line
    assert "elapsed=" in line
    assert "sample=0 cat=5" in line


def test_progress_tracker_progress_line_shows_real_sample_count_for_eta_confidence():
    """Real UX finding: a live run's ETA swung from ~8h to >1d within a
    handful of log lines -- correct behavior (a running average over very
    few real samples is volatile), not a bug, but confusing without
    visibility into how many samples it was based on."""
    from amem_gepa.paper_repro import _ProgressTracker

    tracker = _ProgressTracker(logging.getLogger("test"), "label", total_units=100)
    tracker.record(10.0, cached=False)
    tracker.record(0.0, cached=True)  # cached entries must not count toward n
    tracker.record(20.0, cached=False)

    line = tracker.progress_line()

    assert "avg=15.0s(n=2)" in line


# ---------------------------------------------------------------------------
# Ollama per-call timing breakdown (added after a live diagnosis session:
# a verbose `ollama run` benchmark on the exact machine running a slow
# sweep showed ~73% of one request's total_duration wasn't accounted for
# by Ollama's own load+prompt_eval+eval breakdown, with GPU contention
# ruled out -- `_ProgressTracker`'s own `elapsed` can't say whether slow
# calls are prefill-bound, eval-bound, or something else, so this pulls
# Ollama's own per-call breakdown through the monkeypatch instead of
# editing the vendored RobustOllamaController, which discards it).
# ---------------------------------------------------------------------------


def test_ns_to_s_converts_nanoseconds_and_passes_through_none():
    from amem_gepa.paper_repro import _ns_to_s

    assert _ns_to_s(5_000_000_000) == pytest.approx(5.0)
    assert _ns_to_s(0) == pytest.approx(0.0)
    assert _ns_to_s(None) is None


def test_summarize_ollama_calls_returns_none_for_empty_list():
    from amem_gepa.paper_repro import _summarize_ollama_calls

    assert _summarize_ollama_calls([]) is None


def test_summarize_ollama_calls_aggregates_across_multiple_calls():
    from amem_gepa.paper_repro import _summarize_ollama_calls

    # Two ollama.chat() calls, e.g. keyword-generation then the answer
    # itself (answer_question() makes exactly this pair).
    calls = [
        {
            "wall_elapsed": 10.0, "total_duration": 8.0, "load_duration": 0.1,
            "prompt_eval_count": 20, "prompt_eval_duration": 1.0,
            "eval_count": 15, "eval_duration": 3.0,
        },
        {
            "wall_elapsed": 90.0, "total_duration": 60.0, "load_duration": 0.0,
            "prompt_eval_count": 1500, "prompt_eval_duration": 8.0,
            "eval_count": 200, "eval_duration": 5.0,
        },
    ]

    summary = _summarize_ollama_calls(calls)

    assert "2 call(s)" in summary
    assert "wall=100.0s" in summary  # 10.0 + 90.0
    assert "reported=68.0s" in summary  # 8.0 + 60.0
    assert "prompt=1520tok/9.0s" in summary  # (20+1500)tok, (1.0+8.0)s
    assert "eval=215tok/8.0s" in summary  # (15+200)tok, (3.0+5.0)s
    assert "unaccounted=32.0s" in summary  # wall(100) - reported(68)


def test_summarize_ollama_calls_handles_missing_fields_as_zero():
    from amem_gepa.paper_repro import _summarize_ollama_calls

    # A call where Ollama didn't report some fields (e.g. total_duration
    # None) must not crash the aggregation -- treat as 0, not fail.
    calls = [{
        "wall_elapsed": 5.0, "total_duration": None, "load_duration": None,
        "prompt_eval_count": None, "prompt_eval_duration": None,
        "eval_count": None, "eval_duration": None,
    }]

    summary = _summarize_ollama_calls(calls)

    assert "wall=5.0s" in summary
    assert "reported=0.0s" in summary
    assert "unaccounted=5.0s" in summary


def test_patch_ollama_chat_captures_fields_and_preserves_return_value(monkeypatch):
    fake_ollama = types.ModuleType("ollama")
    call_log = []

    class _FakeResponse(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    def fake_chat(*args, **kwargs):
        call_log.append(kwargs)
        return _FakeResponse({
            "message": {"content": "the answer"},
            "total_duration": 5_000_000_000,
            "load_duration": 100_000_000,
            "prompt_eval_count": 50,
            "prompt_eval_duration": 1_000_000_000,
            "eval_count": 20,
            "eval_duration": 2_000_000_000,
        })

    fake_ollama.chat = fake_chat
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    from amem_gepa.paper_repro import _ollama_call_stats, _patch_ollama_chat

    _ollama_call_stats.clear()
    _patch_ollama_chat()

    import ollama
    response = ollama.chat(model="qwen3.5:9b", messages=[{"role": "user", "content": "hi"}])

    # The wrapper must not change the caller-visible response at all --
    # RobustOllamaController.get_completion only ever reads this field.
    assert response["message"]["content"] == "the answer"
    # It DOES change what's actually sent to Ollama, deliberately: think=False
    # gets injected since the vendored code never sets it and qwen3.5:9b
    # defaults to thinking mode on (docs/decisions/0016).
    assert call_log == [{
        "model": "qwen3.5:9b", "messages": [{"role": "user", "content": "hi"}], "think": False,
    }]

    assert len(_ollama_call_stats) == 1
    stat = _ollama_call_stats[0]
    assert stat["total_duration"] == pytest.approx(5.0)
    assert stat["prompt_eval_count"] == 50
    assert stat["prompt_eval_duration"] == pytest.approx(1.0)
    assert stat["eval_count"] == 20
    assert stat["eval_duration"] == pytest.approx(2.0)
    assert stat["wall_elapsed"] >= 0.0
    assert stat["content_preview"] == "the answer"

    _ollama_call_stats.clear()


def test_patch_ollama_chat_forces_think_false_by_default(monkeypatch):
    """Root-cause fix (docs/decisions/0016): `ollama show qwen3.5:9b`
    reports thinking defaults to on, and a live side-by-side comparison
    showed think=False cut a short keyword-extraction call from 1,115
    eval tokens/15.3s to 11 eval tokens/0.1s with a functionally
    identical answer. The vendored RobustOllamaController.get_completion
    never sets `think` at all, so this must inject it."""
    fake_ollama = types.ModuleType("ollama")
    call_log = []

    class _FakeResponse(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    def fake_chat(*args, **kwargs):
        call_log.append(kwargs)
        return _FakeResponse({"message": {"content": "hi"}})

    fake_ollama.chat = fake_chat
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    from amem_gepa.paper_repro import _ollama_call_stats, _patch_ollama_chat

    _ollama_call_stats.clear()
    _patch_ollama_chat()

    import ollama
    ollama.chat(model="qwen3.5:9b", messages=[])

    assert call_log[0]["think"] is False

    _ollama_call_stats.clear()


def test_patch_ollama_chat_respects_an_explicit_think_value(monkeypatch):
    """setdefault, not an unconditional override -- a caller that
    explicitly asks for thinking (now or in some future code path) must
    not be silently fought."""
    fake_ollama = types.ModuleType("ollama")
    call_log = []

    class _FakeResponse(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    def fake_chat(*args, **kwargs):
        call_log.append(kwargs)
        return _FakeResponse({"message": {"content": "hi"}})

    fake_ollama.chat = fake_chat
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    from amem_gepa.paper_repro import _ollama_call_stats, _patch_ollama_chat

    _ollama_call_stats.clear()
    _patch_ollama_chat()

    import ollama
    ollama.chat(model="qwen3.5:9b", messages=[], think=True)

    assert call_log[0]["think"] is True

    _ollama_call_stats.clear()


def test_patch_ollama_chat_truncates_long_content_preview(monkeypatch):
    """Real diagnosis finding: token counts alone could only suggest a
    call generated far more than a short structured task should need --
    the actual text (e.g. a visible `<think>...` reasoning block) is what
    confirms why. Capped at 80 chars so a multi-thousand-token response
    doesn't blow up the log line; newlines flattened so the preview stays
    on one line."""
    fake_ollama = types.ModuleType("ollama")

    class _FakeResponse(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    long_content = "<think>\nOkay, let me reason through this step by step. " + "x" * 200

    def fake_chat(*args, **kwargs):
        return _FakeResponse({"message": {"content": long_content}})

    fake_ollama.chat = fake_chat
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    from amem_gepa.paper_repro import _ollama_call_stats, _patch_ollama_chat

    _ollama_call_stats.clear()
    _patch_ollama_chat()

    import ollama
    ollama.chat(model="x", messages=[])

    preview = _ollama_call_stats[0]["content_preview"]
    assert len(preview) <= 82, "truncated to 80 raw chars, plus at most one \\n -> \\\\n expansion"
    assert "\n" not in preview
    assert preview.startswith("<think>\\nOkay, let me reason")

    stat = _ollama_call_stats[0]
    assert stat["content_len"] == len(long_content)
    assert stat["content_tail"] == long_content[-80:]

    _ollama_call_stats.clear()


def test_patch_ollama_chat_omits_tail_for_short_content(monkeypatch):
    """A response short enough to fit entirely in the head preview (<=80
    chars) has nothing new to show in a tail -- must be empty, not a
    duplicate of the head."""
    fake_ollama = types.ModuleType("ollama")

    class _FakeResponse(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    def fake_chat(*args, **kwargs):
        return _FakeResponse({"message": {"content": "Keywords: Audrey, dogs, games, park"}})

    fake_ollama.chat = fake_chat
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    from amem_gepa.paper_repro import _ollama_call_stats, _patch_ollama_chat

    _ollama_call_stats.clear()
    _patch_ollama_chat()

    import ollama
    ollama.chat(model="x", messages=[])

    assert _ollama_call_stats[0]["content_tail"] == ""

    _ollama_call_stats.clear()


def test_summarize_ollama_calls_includes_content_previews():
    from amem_gepa.paper_repro import _summarize_ollama_calls

    calls = [
        {
            "wall_elapsed": 1.0, "total_duration": 1.0, "load_duration": 0.0,
            "prompt_eval_count": 10, "prompt_eval_duration": 0.1,
            "eval_count": 5, "eval_duration": 0.9,
            "content_len": 24, "content_preview": "keywords, context, tags", "content_tail": "",
        },
        {
            "wall_elapsed": 2.0, "total_duration": 2.0, "load_duration": 0.0,
            "prompt_eval_count": 20, "prompt_eval_duration": 0.2,
            "eval_count": 500, "eval_duration": 1.8,
            "content_len": 3000, "content_preview": "<think>\\nOkay so the user wants",
            "content_tail": "...and that concludes my reasoning, so the answer is 42.",
        },
    ]

    summary = _summarize_ollama_calls(calls)

    assert '[0] (len=24) head="keywords, context, tags"' in summary
    assert '[1] (len=3000) head="<think>\\nOkay so the user wants"' in summary
    assert 'tail="...and that concludes my reasoning, so the answer is 42."' in summary
    assert "tail=" not in summary.split("[1]")[0], (
        "call 0 had no tail (short content) -- must not show an empty tail=\"\" segment"
    )


def test_patch_ollama_chat_is_idempotent(monkeypatch):
    fake_ollama = types.ModuleType("ollama")
    call_count = 0

    class _FakeResponse(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    def fake_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _FakeResponse({"message": {"content": "hi"}})

    fake_ollama.chat = fake_chat
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    from amem_gepa.paper_repro import _ollama_call_stats, _patch_ollama_chat

    _ollama_call_stats.clear()
    _patch_ollama_chat()
    _patch_ollama_chat()  # second call must not double-wrap

    import ollama
    ollama.chat(model="x", messages=[])

    assert call_count == 1, "double-wrapping would call the underlying fake_chat more than once per call"
    assert len(_ollama_call_stats) == 1, "double-wrapping would record two entries for one chat() call"

    _ollama_call_stats.clear()

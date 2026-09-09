from types import SimpleNamespace

from amem_gepa.datasets.locomo import LoCoMoInstance, LoCoMoTurn
from amem_gepa.evaluate import evaluate_candidate


class FakeMemorySystem:
    """Stands in for PromptInjectableMemorySystem: same add_note/search_agentic/
    llm_controller.llm.get_completion shape, no real A-MEM/ChromaDB/LLM."""

    def __init__(self, note_construction_prompt, evolution_prompt, answers_by_question):
        self.note_construction_prompt = note_construction_prompt
        self.evolution_prompt = evolution_prompt
        self.notes = []
        self.answers_by_question = answers_by_question
        self.llm_controller = SimpleNamespace(llm=SimpleNamespace(get_completion=self._get_completion))

    def add_note(self, content, time=None):
        self.notes.append(content)

    def search_agentic(self, query, k=10):
        return [{"content": n, "timestamp": "t", "context": "c"} for n in self.notes[:k]]

    def snapshot_notes(self):
        return [{"content": n} for n in self.notes]

    def restore_notes(self, notes):
        self.notes.extend(n["content"] for n in notes)

    def _get_completion(self, prompt, response_format=None, temperature=0.7):
        for question, answer in self.answers_by_question.items():
            if question in prompt:
                return answer
        return "I don't know"


def _make_instances():
    turns = [
        LoCoMoTurn(session=1, dia_id="D1:1", speaker="Alice", text="I love hiking.", date_time="t1"),
        LoCoMoTurn(session=1, dia_id="D1:2", speaker="Bob", text="Nice, where?", date_time="t1"),
    ]
    return [
        LoCoMoInstance(
            conversation_id="conv-x",
            turns=turns,
            question="What does Alice love?",
            gold_answer="hiking",
            category=1,  # multi_hop
        ),
        LoCoMoInstance(
            conversation_id="conv-x",
            turns=turns,
            question="What did Alice realize about hiking?",
            gold_answer=None,
            category=5,  # adversarial
            adversarial_answer="hiking is dangerous",
        ),
    ]


def test_evaluate_candidate_replays_turns_once_per_conversation():
    instances = _make_instances()
    created_systems = []

    def factory(note_construction_prompt, evolution_prompt):
        system = FakeMemorySystem(
            note_construction_prompt,
            evolution_prompt,
            answers_by_question={
                "What does Alice love?": "hiking",
                "What did Alice realize about hiking?": "That is not mentioned in the retrieved memories.",
            },
        )
        created_systems.append(system)
        return system

    result = evaluate_candidate(
        instances,
        note_construction_prompt="construct: {content}",
        evolution_prompt="evolve",
        qa_prompt_template="Memories:\n{retrieved_memories}\n\nQ: {question}",
        llm_model="unused",
        memory_system_factory=factory,
        n_bootstrap_resamples=100,
    )

    assert len(created_systems) == 1, "one conversation should build exactly one memory system"
    assert len(created_systems[0].notes) == 2, "both turns should have been replayed"
    assert len(result.instance_results) == 2
    assert result.summaries["aggregate"].n == 2
    assert result.summaries["multi_hop"].mean == 1.0
    assert result.summaries["adversarial"].mean == 1.0


def test_evaluate_candidate_groups_by_conversation():
    turns_a = [LoCoMoTurn(session=1, dia_id="D1:1", speaker="A", text="hi", date_time="t")]
    turns_b = [LoCoMoTurn(session=1, dia_id="D1:1", speaker="B", text="hey", date_time="t")]
    instances = [
        LoCoMoInstance("conv-a", turns_a, "Q1?", "x", 1),
        LoCoMoInstance("conv-a", turns_a, "Q2?", "x", 1),
        LoCoMoInstance("conv-b", turns_b, "Q3?", "x", 1),
    ]
    created = []

    def factory(note_construction_prompt, evolution_prompt):
        s = FakeMemorySystem(note_construction_prompt, evolution_prompt, answers_by_question={})
        created.append(s)
        return s

    evaluate_candidate(
        instances,
        note_construction_prompt="p",
        evolution_prompt="e",
        qa_prompt_template="{retrieved_memories} {question}",
        llm_model="unused",
        memory_system_factory=factory,
    )

    assert len(created) == 2, "two distinct conversations should build two memory systems"


def test_trace_true_prints_turn_retrieval_and_answer_details(capsys):
    instances = _make_instances()

    def factory(note_construction_prompt, evolution_prompt):
        return FakeMemorySystem(
            note_construction_prompt,
            evolution_prompt,
            answers_by_question={
                "What does Alice love?": "hiking",
                "What did Alice realize about hiking?": "not mentioned",
            },
        )

    evaluate_candidate(
        instances,
        note_construction_prompt="construct: {content}",
        evolution_prompt="evolve",
        qa_prompt_template="Memories:\n{retrieved_memories}\n\nQ: {question}",
        llm_model="unused",
        memory_system_factory=factory,
        n_bootstrap_resamples=100,
        trace=True,
    )

    out = capsys.readouterr().out
    assert "[trace:turn 1/2]" in out
    assert "[trace:turn 2/2]" in out
    assert "[trace:retrieval] query='What does Alice love?'" in out
    assert "[trace:qa_answer] -> 'hiking'" in out


def test_trace_false_by_default_prints_no_trace_lines(capsys):
    """Regression guard: run_baseline.py/run_eval.py never pass trace=True."""
    instances = _make_instances()

    def factory(note_construction_prompt, evolution_prompt):
        return FakeMemorySystem(
            note_construction_prompt,
            evolution_prompt,
            answers_by_question={
                "What does Alice love?": "hiking",
                "What did Alice realize about hiking?": "not mentioned",
            },
        )

    evaluate_candidate(
        instances,
        note_construction_prompt="construct: {content}",
        evolution_prompt="evolve",
        qa_prompt_template="Memories:\n{retrieved_memories}\n\nQ: {question}",
        llm_model="unused",
        memory_system_factory=factory,
        n_bootstrap_resamples=100,
    )

    out = capsys.readouterr().out
    assert "[trace:" not in out


def test_trace_true_bypasses_prediction_cache(tmp_path):
    """trace is a request to watch it happen live -- a cached prediction
    from a prior non-traced run must not silently skip answer_question."""
    instances = _make_instances()
    call_count = {"n": 0}

    def factory(note_construction_prompt, evolution_prompt):
        system = FakeMemorySystem(
            note_construction_prompt,
            evolution_prompt,
            answers_by_question={
                "What does Alice love?": "hiking",
                "What did Alice realize about hiking?": "not mentioned",
            },
        )
        original = system._get_completion

        def counting_get_completion(prompt, response_format=None, temperature=0.7):
            call_count["n"] += 1
            return original(prompt, response_format, temperature)

        system.llm_controller.llm.get_completion = counting_get_completion
        return system

    common_kwargs = dict(
        note_construction_prompt="construct: {content}",
        evolution_prompt="evolve",
        qa_prompt_template="Memories:\n{retrieved_memories}\n\nQ: {question}",
        llm_model="unused",
        memory_system_factory=factory,
        run_label="test-trace",
        split="test",
        results_dir=tmp_path,
    )

    evaluate_candidate(instances, **common_kwargs)
    calls_after_first_run = call_count["n"]
    assert calls_after_first_run == 2  # both questions answered fresh

    evaluate_candidate(instances, **common_kwargs, trace=True)
    assert call_count["n"] == calls_after_first_run + 2, "trace run should re-answer both, not reuse the cache"

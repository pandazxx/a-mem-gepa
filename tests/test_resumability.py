"""Proves the fix for a real problem: a baseline run against a slow local
model ran 12+ hours without finishing, and any interruption before then
would have lost all of it. These tests simulate a mid-run crash (by raising
partway through) and assert a second run resumes rather than redoing
already-completed LLM work.
"""

from types import SimpleNamespace

import pytest

from amem_gepa.checkpoint import CheckpointStore, PredictionStore
from amem_gepa.datasets.locomo import LoCoMoInstance, LoCoMoTurn
from amem_gepa.evaluate import evaluate_candidate


class ResumableFakeMemorySystem:
    """Like tests/test_evaluate.py's FakeMemorySystem, but with
    snapshot_notes/restore_notes (so it can round-trip through a
    CheckpointStore, the same way the real PromptInjectableMemorySystem
    does) and an injectable "crash after N calls" hook."""

    def __init__(self, answers_by_question, add_note_call_log, crash_after_add_notes=None):
        self.notes = []
        self.answers_by_question = answers_by_question
        self.add_note_call_log = add_note_call_log
        self.crash_after_add_notes = crash_after_add_notes
        self.llm_controller = SimpleNamespace(llm=SimpleNamespace(get_completion=self._get_completion))

    def add_note(self, content, time=None):
        if self.crash_after_add_notes is not None and len(self.add_note_call_log) >= self.crash_after_add_notes:
            raise RuntimeError("simulated crash mid-replay")
        self.notes.append(content)
        self.add_note_call_log.append(content)

    def search_agentic(self, query, k=10):
        return [{"content": n, "timestamp": "t", "context": "c"} for n in self.notes[:k]]

    def _get_completion(self, prompt, response_format=None, temperature=0.7):
        for question, answer in self.answers_by_question.items():
            if question in prompt:
                return answer
        return "I don't know"

    def snapshot_notes(self):
        return [{"content": n} for n in self.notes]

    def restore_notes(self, notes):
        for n in notes:
            self.notes.append(n["content"])


def _turns(n):
    return [
        LoCoMoTurn(session=1, dia_id=f"D1:{i}", speaker="A", text=f"turn {i}", date_time="t")
        for i in range(n)
    ]


def test_resumed_run_does_not_replay_already_checkpointed_turns(tmp_path):
    turns = _turns(4)
    instances = [LoCoMoInstance("conv-x", turns, "Q1?", "gold", 1)]

    first_run_log = []

    def crashing_factory(note_construction_prompt, evolution_prompt):
        return ResumableFakeMemorySystem({}, first_run_log, crash_after_add_notes=2)

    with pytest.raises(RuntimeError, match="simulated crash"):
        evaluate_candidate(
            instances,
            note_construction_prompt="p",
            evolution_prompt="e",
            qa_prompt_template="{retrieved_memories} {question}",
            llm_model="unused",
            memory_system_factory=crashing_factory,
            run_label="test-run",
            split="test",
            results_dir=tmp_path,
        )

    assert len(first_run_log) == 2, "should have processed exactly 2 turns before the simulated crash"

    checkpoint = CheckpointStore("test-run", results_dir=tmp_path).load("conv-x")
    assert checkpoint is not None
    assert checkpoint.turns_processed == 2
    assert checkpoint.completed is False

    second_run_log = []

    def resuming_factory(note_construction_prompt, evolution_prompt):
        return ResumableFakeMemorySystem(
            {"Q1?": "an answer"}, second_run_log, crash_after_add_notes=None
        )

    result = evaluate_candidate(
        instances,
        note_construction_prompt="p",
        evolution_prompt="e",
        qa_prompt_template="{retrieved_memories} {question}",
        llm_model="unused",
        memory_system_factory=resuming_factory,
        run_label="test-run",
        split="test",
        results_dir=tmp_path,
    )

    assert len(second_run_log) == 2, "should only replay the 2 turns that weren't checkpointed yet"
    assert len(result.instance_results) == 1

    final_checkpoint = CheckpointStore("test-run", results_dir=tmp_path).load("conv-x")
    assert final_checkpoint.turns_processed == 4
    assert final_checkpoint.completed is True


def test_resumed_run_does_not_re_answer_already_cached_questions(tmp_path):
    turns = _turns(1)
    instances = [
        LoCoMoInstance("conv-x", turns, "Q1?", "gold1", 1),
        LoCoMoInstance("conv-x", turns, "Q2?", "gold2", 1),
        LoCoMoInstance("conv-x", turns, "Q3?", "gold3", 1),
    ]

    call_count = {"n": 0}

    class CrashAfterOneAnswer(ResumableFakeMemorySystem):
        def _get_completion(self, prompt, response_format=None, temperature=0.7):
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise RuntimeError("simulated crash mid-answering")
            return super()._get_completion(prompt, response_format, temperature)

    def crashing_factory(note_construction_prompt, evolution_prompt):
        return CrashAfterOneAnswer({"Q1?": "answer one"}, [])

    with pytest.raises(RuntimeError, match="simulated crash"):
        evaluate_candidate(
            instances,
            note_construction_prompt="p",
            evolution_prompt="e",
            qa_prompt_template="{retrieved_memories} {question}",
            llm_model="unused",
            memory_system_factory=crashing_factory,
            run_label="test-run-2",
            split="test",
            results_dir=tmp_path,
        )

    predictions = PredictionStore("test-run-2", "test", results_dir=tmp_path)
    assert len(predictions) == 1
    assert predictions.get("conv-x", "Q1?")["prediction"] == "answer one"

    second_run_calls = {"n": 0}

    def resuming_factory(note_construction_prompt, evolution_prompt):
        system = ResumableFakeMemorySystem(
            {"Q2?": "answer two", "Q3?": "answer three"}, []
        )
        original = system._get_completion

        def counting_get_completion(prompt, response_format=None, temperature=0.7):
            second_run_calls["n"] += 1
            return original(prompt, response_format, temperature)

        system.llm_controller.llm.get_completion = counting_get_completion
        return system

    result = evaluate_candidate(
        instances,
        note_construction_prompt="p",
        evolution_prompt="e",
        qa_prompt_template="{retrieved_memories} {question}",
        llm_model="unused",
        memory_system_factory=resuming_factory,
        run_label="test-run-2",
        split="test",
        results_dir=tmp_path,
    )

    assert second_run_calls["n"] == 2, "only the 2 unanswered questions should hit the LLM again"
    assert len(result.instance_results) == 3
    predictions_by_q = {r.question: r.prediction for r in result.instance_results}
    assert predictions_by_q["Q1?"] == "answer one"  # reused from cache, not regenerated
    assert predictions_by_q["Q2?"] == "answer two"
    assert predictions_by_q["Q3?"] == "answer three"

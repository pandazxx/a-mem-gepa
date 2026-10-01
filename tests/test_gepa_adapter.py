"""Tests for the GEPA adapter (docs/decisions/0014, 0015) -- stubs the
vendored repro modules (same reasoning as test_paper_repro.py: the real
ones pull torch/sentence-transformers), and exercises evaluation grouping,
the build cache, scoring (incl. the adversarial direction), candidate
validation, and reflective-dataset attribution."""

import sys
import types
from typing import ClassVar

import pytest

from amem_gepa.datasets.locomo import LoCoMoInstance, LoCoMoQuestionGroup, LoCoMoTurn
from amem_gepa.gepa_adapter import AMemGEPAAdapter

CANDIDATE = {
    "note_construction": "NOTECON analyze: {content}",
    "evolution_decision": "EVODEC ctx={context} c={content} k={keywords} n={nearest_neighbors_memories}",
    "evolution_strengthen": "STRENGTH c={content} k={keywords} n={nearest_neighbors_memories}",
    "evolution_update_neighbors": (
        "UPDNEI c={content} ctx={context} n={nearest_neighbors_memories} "
        "max={max_neighbor_idx} count={neighbor_count}"
    ),
}

TURNS = [
    LoCoMoTurn(session=1, dia_id="D1:1", speaker="Alice", text="I adopted a dog named Rex", date_time="t1"),
    LoCoMoTurn(session=1, dia_id="D1:2", speaker="Bob", text="Rex loves the beach in June", date_time="t1"),
    LoCoMoTurn(session=2, dia_id="D2:1", speaker="Alice", text="We moved to Lisbon last month", date_time="t2"),
]


def _instance(question, gold, category=4, conv="conv-a", evidence=None, adversarial=None):
    return LoCoMoInstance(
        conversation_id=conv,
        turns=TURNS,
        question=question,
        gold_answer=gold,
        category=category,
        adversarial_answer=adversarial,
        evidence=evidence,
    )


class _EchoLLM:
    def get_completion(self, prompt, temperature=0.7):
        return "DECISION: NO_EVOLUTION\nREASON: fake"


class _FakeRetriever:
    def save(self, path, embeddings_path):
        from pathlib import Path

        Path(path).write_text("retriever")
        Path(embeddings_path).write_text("embeddings")

    def load(self, path, embeddings_path):
        return self

    def load_from_local_memory(self, memories, model_name):
        return self


class _FakeMemorySystem:
    def __init__(self):
        self.memories = {}
        self.retriever = _FakeRetriever()
        self.llm_controller = types.SimpleNamespace(llm=_EchoLLM())


class _FakeAgent:
    """Mimics RobustAdvancedMemAgent's surface: reads the (stubbed)
    memory_layer_robust module's prompt constants at call time -- so these
    tests exercise real prompt injection end to end, not just the wrapper."""

    instances: ClassVar[list] = []
    answers: ClassVar[dict] = {}  # question -> canned prediction
    context_mode = "all"  # "all": every stored memory in raw_context; "empty": none
    fail_build = False

    def __init__(self, model, backend, retrieve_k, temperature_c5):
        self.model = model
        self.backend = backend
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.memory_system = _FakeMemorySystem()
        self.retriever_llm = types.SimpleNamespace(llm=_EchoLLM())
        self.added = []
        self.answer_args = []
        _FakeAgent.instances.append(self)

    def add_memory(self, content, time=None):
        if _FakeAgent.fail_build:
            raise RuntimeError("backend exploded")
        mlr = sys.modules["memory_layer_robust"]
        llm = self.memory_system.llm_controller.llm  # RecordingLLM after adapter wrap
        llm.get_completion(mlr.ANALYZE_CONTENT_PROMPT.format(content=content))
        if self.added:
            llm.get_completion(
                mlr.EVOLUTION_DECISION_PROMPT.format(
                    context="ctx",
                    content=content,
                    keywords="[]",
                    nearest_neighbors_memories="\n".join(self.added),
                )
            )
        self.added.append(content)
        self.memory_system.memories[f"note-{len(self.added)}"] = content

    def answer_question(self, question, category, answer):
        self.answer_args.append((question, category, answer))
        self.retriever_llm.llm.get_completion(
            f"Given the following question, generate several keywords separated by commas.\n\nQuestion: {question}\n\nKeywords:"
        )
        stored = list(self.memory_system.memories.values())
        raw_context = "" if _FakeAgent.context_mode == "empty" else "memory content: " + " ".join(stored)
        prediction = self.memory_system.llm_controller.llm.get_completion(
            f"Based on the context: {raw_context}, answer the following question. {question}"
        )
        return prediction, "user prompt", raw_context


class _AnswerFromTable(_EchoLLM):
    """The QA-answer call returns the canned answer for the question
    embedded in the prompt; everything else gets the echo default."""

    def get_completion(self, prompt, temperature=0.7):
        if prompt.startswith("Based on the context:"):
            for question, prediction in _FakeAgent.answers.items():
                if question in prompt:
                    return prediction
            return "dunno"
        return super().get_completion(prompt, temperature=temperature)


@pytest.fixture
def fake_repro(monkeypatch, tmp_path):
    _FakeAgent.instances = []
    _FakeAgent.answers = {}
    _FakeAgent.context_mode = "all"
    _FakeAgent.fail_build = False

    fake_tar = types.ModuleType("test_advanced_robust")
    fake_tar.RobustAdvancedMemAgent = _FakeAgent

    fake_mlr = types.ModuleType("memory_layer_robust")
    fake_mlr.ANALYZE_CONTENT_PROMPT = "ORIGINAL ANALYZE {content}"
    fake_mlr.EVOLUTION_DECISION_PROMPT = "ORIGINAL DECISION {context}{content}{keywords}{nearest_neighbors_memories}"
    fake_mlr.STRENGTHEN_DETAILS_PROMPT = "ORIGINAL STRENGTHEN {content}{keywords}{nearest_neighbors_memories}"
    fake_mlr.UPDATE_NEIGHBORS_PROMPT = (
        "ORIGINAL UPDATE {content}{context}{nearest_neighbors_memories}{max_neighbor_idx}{neighbor_count}"
    )
    fake_mlr.FOCUSED_KEYWORDS_PROMPT = "ORIGINAL FOCUSED {content}"

    fake_utils = types.ModuleType("utils")
    fake_utils.calculate_metrics = lambda prediction, reference: {
        "f1": 1.0 if prediction.strip() == reference.strip() else 0.0
    }

    fake_parsers = types.ModuleType("llm_text_parsers")
    fake_parsers.parse_plain_text_answer = lambda response: response

    monkeypatch.setitem(sys.modules, "test_advanced_robust", fake_tar)
    monkeypatch.setitem(sys.modules, "memory_layer_robust", fake_mlr)
    monkeypatch.setitem(sys.modules, "utils", fake_utils)
    monkeypatch.setitem(sys.modules, "llm_text_parsers", fake_parsers)

    def make_adapter(**kwargs):
        defaults = {"backend": "openai", "model": "gpt-4o-mini", "cache_dir": tmp_path / "gepa_cache"}
        defaults.update(kwargs)
        return AMemGEPAAdapter(**defaults)

    # QA-answer calls route through the memory system's controller; give
    # every fake agent the canned-answer LLM instead of the echo default.
    original_init = _FakeMemorySystem.__init__

    def init_with_answers(self):
        original_init(self)
        self.llm_controller = types.SimpleNamespace(llm=_AnswerFromTable())

    monkeypatch.setattr(_FakeMemorySystem, "__init__", init_with_answers)

    return make_adapter


def test_evaluate_builds_once_per_conversation_and_shapes_outputs(fake_repro):
    adapter = fake_repro()
    group = LoCoMoQuestionGroup(
        conversation_id="conv-a",
        turns=TURNS,
        questions=[_instance("Q1", "Rex"), _instance("Q2", "Lisbon")],
    )
    single = _instance("Q3", "June")
    _FakeAgent.answers = {"Q1": "Rex", "Q2": "wrong", "Q3": "June"}

    batch = adapter.evaluate([group, single], CANDIDATE, capture_traces=True)

    assert len(_FakeAgent.instances) == 1, "one memory build serves the whole conversation"
    assert _FakeAgent.instances[0].added == [
        "Speaker Alicesays : I adopted a dog named Rex",
        "Speaker Bobsays : Rex loves the beach in June",
        "Speaker Alicesays : We moved to Lisbon last month",
    ]
    assert batch.scores == [pytest.approx(0.5), pytest.approx(1.0)]  # group mean, single
    assert batch.outputs[0] == ["Rex", "wrong"]
    assert batch.outputs[1] == "June"
    assert batch.num_metric_calls == 2
    assert adapter.builds_performed == 1
    assert batch.trajectories[0].build_records, "construction/evolution calls must be traced"
    kinds = {r.kind for r in batch.trajectories[0].build_records}
    assert kinds == {"note_construction", "evolution_decision"}
    # Injection reached the (stubbed) vendored namespace: the recorded
    # prompts are the candidate's templates, not the originals.
    assert all(not r.prompt.startswith("ORIGINAL") for r in batch.trajectories[0].build_records)


def test_second_evaluation_hits_the_cache_and_keeps_traces(fake_repro, tmp_path):
    _FakeAgent.answers = {"Q1": "Rex"}
    item = _instance("Q1", "Rex")

    adapter1 = fake_repro()
    adapter1.evaluate([item], CANDIDATE)
    assert adapter1.builds_performed == 1

    # Fresh adapter (fresh process in real life), same cache dir.
    adapter2 = fake_repro()
    batch = adapter2.evaluate([item], CANDIDATE, capture_traces=True)

    assert adapter2.builds_performed == 0
    assert adapter2.build_cache_hits == 1
    second_agent = _FakeAgent.instances[-1]
    assert second_agent.added == [], "cache hit must not replay the conversation"
    assert batch.trajectories[0].build_cache_hit
    assert batch.trajectories[0].build_records, "persisted build trace must survive the cache hit"
    assert batch.scores == [pytest.approx(1.0)]


def test_different_candidate_forces_a_rebuild(fake_repro):
    _FakeAgent.answers = {"Q1": "Rex"}
    item = _instance("Q1", "Rex")
    adapter = fake_repro()

    adapter.evaluate([item], CANDIDATE)
    changed = dict(CANDIDATE, note_construction="NOTECON v2: {content}")
    adapter.evaluate([item], changed)

    assert adapter.builds_performed == 2, "a mutated candidate invalidates the memory cache"


def test_invalid_candidate_scores_zero_without_building(fake_repro):
    adapter = fake_repro()
    broken = dict(CANDIDATE, evolution_decision="EVODEC missing everything")
    item = _instance("Q1", "Rex")

    batch = adapter.evaluate([item], broken, capture_traces=True)

    assert batch.scores == [0.0]
    assert _FakeAgent.instances == [], "invalid template must not spend a build"
    assert batch.trajectories[0].validation_errors

    dataset = adapter.make_reflective_dataset(broken, batch, ["evolution_decision", "note_construction"])
    assert set(dataset) == {"evolution_decision", "note_construction"}
    feedback = dataset["evolution_decision"][0]["Feedback"]
    assert "invalid" in feedback and "CONSTRAINT" in feedback and "{nearest_neighbors_memories}" in feedback


def test_adversarial_scoring_refusal_direction(fake_repro):
    adapter = fake_repro()
    refused = _instance("Q-adv-1", None, category=5, adversarial="Rex hates water", evidence=None)
    trapped = _instance("Q-adv-2", None, category=5, adversarial="Rex hates water", evidence=None)
    _FakeAgent.answers = {"Q-adv-1": "Not mentioned in the conversation", "Q-adv-2": "Rex hates water"}

    batch = adapter.evaluate([refused, trapped], CANDIDATE)

    assert batch.scores == [pytest.approx(1.0), pytest.approx(0.0)]
    # The trap text is passed through as the multiple-choice option.
    assert _FakeAgent.instances[0].answer_args[0] == ("Q-adv-1", 5, "Rex hates water")


def test_adversarial_scoring_paper_direction_is_selectable(fake_repro):
    adapter = fake_repro(adversarial_scoring="paper")
    trapped = _instance("Q-adv", None, category=5, adversarial="Rex hates water")
    _FakeAgent.answers = {"Q-adv": "Rex hates water"}

    batch = adapter.evaluate([trapped], CANDIDATE)

    assert batch.scores == [pytest.approx(1.0)], "paper mode scores against the trap text"


def test_build_failure_degrades_to_zero_scores_not_an_exception(fake_repro):
    adapter = fake_repro()
    _FakeAgent.fail_build = True
    item = _instance("Q1", "Rex")

    batch = adapter.evaluate([item], CANDIDATE, capture_traces=True)

    assert batch.scores == [0.0]
    assert "backend exploded" in batch.trajectories[0].error
    assert "memory build failed" in batch.trajectories[0].questions[0].error


def test_reflective_dataset_attributes_unretrieved_evidence(fake_repro):
    adapter = fake_repro()
    _FakeAgent.context_mode = "empty"  # nothing retrieved -> evidence missing from context
    group = LoCoMoQuestionGroup(
        conversation_id="conv-a",
        turns=TURNS,
        questions=[_instance("Where did they move?", "Lisbon", evidence=["D2:1"])],
    )
    _FakeAgent.answers = {"Where did they move?": "no idea"}

    batch = adapter.evaluate([group], CANDIDATE, capture_traces=True)
    assert batch.scores == [pytest.approx(0.0)]
    qt = batch.trajectories[0].questions[0]
    assert qt.evidence_turn_texts == ["We moved to Lisbon last month"]
    assert qt.evidence_retrieved == [False]

    dataset = adapter.make_reflective_dataset(CANDIDATE, batch, ["note_construction", "evolution_decision"])

    nc = dataset["note_construction"]
    assert nc, "the failed question's evidence turn was note-constructed -- must yield an example"
    assert "We moved to Lisbon last month" in nc[0]["Inputs"]["turn_content (the {content} input)"]
    assert "NOT among the retrieved" in nc[0]["Feedback"]
    assert "CONSTRAINT" in nc[0]["Feedback"]

    evo = dataset["evolution_decision"]
    assert evo, "the evolution step that processed the evidence turn must yield an example"
    assert "did not make the memory reachable" in evo[0]["Feedback"]


def test_reflective_dataset_falls_back_to_aggregate_example(fake_repro):
    adapter = fake_repro()
    # No evidence annotations at all -> no per-call attribution possible.
    group = LoCoMoQuestionGroup(
        conversation_id="conv-a",
        turns=TURNS,
        questions=[_instance("Q1", "Rex"), _instance("Q2", "Lisbon")],
    )
    _FakeAgent.answers = {"Q1": "wrong", "Q2": "also wrong"}

    batch = adapter.evaluate([group], CANDIDATE, capture_traces=True)
    dataset = adapter.make_reflective_dataset(CANDIDATE, batch, ["evolution_strengthen"])

    examples = dataset["evolution_strengthen"]
    assert len(examples) == 1
    assert "single_hop" in examples[0]["Feedback"], "aggregate feedback must include per-category scores"
    assert "CONSTRAINT" in examples[0]["Feedback"]


def test_truncated_and_full_builds_do_not_share_cache(fake_repro):
    adapter = fake_repro()
    _FakeAgent.answers = {"Q1": "Rex"}
    full = _instance("Q1", "Rex")
    truncated = LoCoMoInstance(
        conversation_id="conv-a",
        turns=TURNS[:1],
        question="Q1",
        gold_answer="Rex",
        category=4,
    )

    adapter.evaluate([full], CANDIDATE)
    adapter.evaluate([truncated], CANDIDATE)

    assert adapter.builds_performed == 2, "different turn counts are different memory states"
    assert adapter.build_cache_hits == 0
    assert _FakeAgent.instances[1].added == ["Speaker Alicesays : I adopted a dog named Rex"]

    # Same truncation again: now it's a legitimate cache hit.
    adapter2 = fake_repro()
    adapter2.evaluate([truncated], CANDIDATE)
    assert adapter2.build_cache_hits == 1

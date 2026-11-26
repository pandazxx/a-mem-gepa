"""docs/decisions/0020: proxy evaluation for a note_construction stage --
seed-built memory with only the question-relevant notes re-constructed.

Own fakes (richer than test_gepa_adapter's): note objects with
keywords/context/tags, a retriever with a corpus/embedding matrix, and a
RobustMemoryNote.analyze_content that records which prompt it was given.
"""

import sys
import types
from typing import ClassVar

import numpy as np
import pytest

from amem_gepa.datasets.locomo import LoCoMoInstance, LoCoMoQuestionGroup, LoCoMoTurn
from amem_gepa.gepa_adapter import AMemGEPAAdapter
from amem_gepa.gepa_stages import frozen_components
from amem_gepa.gepa_test_eval import CANDIDATE_ARM, run_test_eval

SEED = {
    "note_construction": "ORIGINAL ANALYZE {content}",
    "evolution_decision": "ORIGINAL DECISION {context}{content}{keywords}{nearest_neighbors_memories}",
    "evolution_strengthen": "ORIGINAL STRENGTHEN {content}{keywords}{nearest_neighbors_memories}",
    "evolution_update_neighbors": (
        "ORIGINAL UPDATE {content}{context}{nearest_neighbors_memories}{max_neighbor_idx}{neighbor_count}"
    ),
}
CAND = dict(SEED, note_construction="NOTECON analyze: {content}")
FROZEN = frozen_components(SEED, "note_construction")

TURNS = [
    LoCoMoTurn(session=1, dia_id="D1:1", speaker="Alice", text="I adopted a dog named Rex", date_time="t1"),
    LoCoMoTurn(session=1, dia_id="D1:2", speaker="Bob", text="Rex loves the beach in June", date_time="t1"),
    LoCoMoTurn(session=2, dia_id="D2:1", speaker="Alice", text="We moved to Lisbon last month", date_time="t2"),
]


def _instance(question, gold, conv="conv-a", category=4, evidence=None):
    return LoCoMoInstance(
        conversation_id=conv, turns=TURNS, question=question, gold_answer=gold, category=category, evidence=evidence
    )


# ---------------------------------------------------------------- fakes


class _Note:
    def __init__(self, content, keywords, context, tags):
        self.content, self.keywords, self.context, self.tags, self.links = content, keywords, context, tags, []


def _vec(doc: str) -> np.ndarray:
    # Deterministic 2-d "embedding": length and a hash-ish of the text.
    return np.array([float(len(doc)), float(sum(map(ord, doc)) % 997)])


class _Model:
    def encode(self, docs):
        return np.stack([_vec(d) for d in docs])


class _Retriever:
    def __init__(self):
        self.model = _Model()
        self.corpus: list[str] = []
        self.embeddings = None
        self.document_ids: dict[str, int] = {}

    def add_documents(self, docs):
        start = len(self.corpus)
        self.corpus.extend(docs)
        new = self.model.encode(docs)
        self.embeddings = new if self.embeddings is None else np.vstack([self.embeddings, new])
        for i, d in enumerate(docs):
            self.document_ids[d] = start + i

    def search(self, query, k):
        return list(range(min(k, len(self.corpus))))

    def save(self, path, emb_path):
        import pickle

        with open(path, "wb") as f:
            pickle.dump({"corpus": self.corpus, "document_ids": self.document_ids}, f)
        np.save(emb_path, self.embeddings)

    def load(self, path, emb_path):
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)
        self.corpus, self.document_ids = state["corpus"], state["document_ids"]
        self.embeddings = np.load(emb_path)
        return self

    @classmethod
    def load_from_local_memory(cls, memories, model_name):
        r = cls()
        r.add_documents([m.content + " , " + m.context for m in memories.values()])
        return r


class _LLM:
    """Analysis responses encode which prompt family produced them; QA
    answers come from a table; everything else is a no-evolution decision."""

    answers: ClassVar[dict] = {}

    def get_completion(self, prompt, temperature=0.7):
        if prompt.startswith("NOTECON"):
            return "KEYWORDS: cand\nCONTEXT: cand ctx\nTAGS: cand"
        if prompt.startswith("ORIGINAL ANALYZE"):
            return "KEYWORDS: seed\nCONTEXT: seed ctx\nTAGS: seed"
        if prompt.startswith("Based on the context:"):
            for q, a in _LLM.answers.items():
                if q in prompt:
                    return a
            return "dunno"
        return "DECISION: NO_EVOLUTION\nREASON: fake"


class _FakeRobustMemoryNote:
    @staticmethod
    def analyze_content(content, llm_controller):
        mlr = sys.modules["memory_layer_robust"]
        resp = llm_controller.llm.get_completion(mlr.ANALYZE_CONTENT_PROMPT.format(content=content))
        fields = dict(line.split(": ", 1) for line in resp.splitlines())
        return {
            "keywords": [k.strip() for k in fields["KEYWORDS"].split(",")],
            "context": fields["CONTEXT"],
            "tags": [t.strip() for t in fields["TAGS"].split(",")],
        }


class _MemorySystem:
    def __init__(self):
        self.memories = {}
        self.retriever = _Retriever()
        self.llm_controller = types.SimpleNamespace(llm=_LLM())
        self.evo_cnt = 0


class _Agent:
    instances: ClassVar[list] = []

    def __init__(self, model, backend, retrieve_k, temperature_c5):
        self.memory_system = _MemorySystem()
        self.retriever_llm = types.SimpleNamespace(llm=_LLM())
        self.added = []
        _Agent.instances.append(self)

    def add_memory(self, content, time=None):
        a = _FakeRobustMemoryNote.analyze_content(content, self.memory_system.llm_controller)
        note = _Note(content, a["keywords"], a["context"], a["tags"])
        self.memory_system.memories[f"note-{len(self.memory_system.memories) + 1}"] = note
        self.memory_system.retriever.add_documents(
            ["content:" + content + " context:" + note.context + " keywords: " + ", ".join(note.keywords)
             + " tags: " + ", ".join(note.tags)]
        )  # fmt: skip
        self.added.append(content)

    def answer_question(self, question, category, answer):
        self.retriever_llm.llm.get_completion(f"Given the following question, generate several keywords separated by commas.\n\nQuestion: {question}\n\nKeywords:")
        ctx = "memory content: " + " ".join(n.content for n in self.memory_system.memories.values())
        return self.memory_system.llm_controller.llm.get_completion(f"Based on the context: {ctx}, answer the following question. {question}"), "p", ctx


@pytest.fixture
def repro(monkeypatch, tmp_path):
    _Agent.instances = []
    _LLM.answers = {}
    tar = types.ModuleType("test_advanced_robust")
    tar.RobustAdvancedMemAgent = _Agent
    mlr = types.ModuleType("memory_layer_robust")
    mlr.ANALYZE_CONTENT_PROMPT = SEED["note_construction"]
    mlr.EVOLUTION_DECISION_PROMPT = SEED["evolution_decision"]
    mlr.STRENGTHEN_DETAILS_PROMPT = SEED["evolution_strengthen"]
    mlr.UPDATE_NEIGHBORS_PROMPT = SEED["evolution_update_neighbors"]
    mlr.FOCUSED_KEYWORDS_PROMPT = "ORIGINAL FOCUSED {content}"
    mlr.RobustMemoryNote = _FakeRobustMemoryNote
    utils = types.ModuleType("utils")
    utils.calculate_metrics = lambda p, r: {"f1": 1.0 if p.strip() == r.strip() else 0.0}
    parsers = types.ModuleType("llm_text_parsers")
    parsers.parse_plain_text_answer = lambda r: r
    for name, mod in {"test_advanced_robust": tar, "memory_layer_robust": mlr, "utils": utils, "llm_text_parsers": parsers}.items():
        monkeypatch.setitem(sys.modules, name, mod)

    def make(**kw):
        defaults = {"backend": "openai", "model": "m", "cache_dir": tmp_path / "cache", "ensure_nltk": False}
        defaults.update(kw)
        return AMemGEPAAdapter(**defaults)

    return make


def _group(question="Where does Rex like to go?", gold="beach", evidence=("D1:2",)):
    _LLM.answers[question] = gold
    return LoCoMoQuestionGroup(conversation_id="conv-a", turns=TURNS, questions=[_instance(question, gold, evidence=list(evidence))])


# ---------------------------------------------------------------- tests


def test_proxy_mode_requires_seed_and_a_note_construction_stage(repro):
    with pytest.raises(ValueError, match="seed_candidate"):
        repro(proxy_mode="evidence", frozen_components=FROZEN)
    with pytest.raises(ValueError, match="frozen_components"):
        repro(proxy_mode="evidence", seed_candidate=SEED, frozen_components={})
    with pytest.raises(ValueError, match="proxy_mode"):
        repro(proxy_mode="bogus", seed_candidate=SEED, frozen_components=FROZEN)
    repro(proxy_mode="competitors", seed_candidate=SEED, frozen_components=FROZEN)  # ok


def test_evidence_proxy_renotes_only_the_evidence_turn_on_the_seed_build(repro):
    adapter = repro(proxy_mode="evidence", seed_candidate=SEED, frozen_components=FROZEN)
    group = _group()

    seed_batch = adapter.evaluate([group], SEED)  # seed: normal path, one real build
    assert adapter.builds_performed == 1 and adapter.proxy_evaluations == 0
    assert seed_batch.scores == [1.0]

    cand_batch = adapter.evaluate([group], CAND, capture_traces=True)
    assert adapter.builds_performed == 1, "the candidate must not trigger a build"
    assert adapter.proxy_evaluations == 1 and adapter.proxy_renotes == 1
    agent = _Agent.instances[-1]
    assert agent.added == [], "proxy agent is loaded from the seed cache, never replayed"
    notes = list(agent.memory_system.memories.values())
    assert notes[1].keywords == ["cand"] and notes[1].context == "cand ctx"  # evidence turn D1:2 re-noted
    assert notes[0].keywords == ["seed"] and notes[2].keywords == ["seed"]  # others untouched
    r = agent.memory_system.retriever
    assert "keywords: cand" in r.corpus[1] and r.corpus[1].startswith("content:")
    assert np.array_equal(r.embeddings[1], _vec(r.corpus[1])), "row re-embedded from the new document"
    assert np.array_equal(r.embeddings[0], _vec(r.corpus[0])) and "seed" in r.corpus[0]
    assert r.document_ids[r.corpus[1]] == 1
    traj = cand_batch.trajectories[0]
    assert traj.proxy_renoted == 1 and traj.build_cache_hit
    assert [rec.kind for rec in traj.build_records] == ["note_construction"]
    assert cand_batch.scores == [1.0]


def test_competitors_proxy_adds_the_seed_memorys_top_k(repro):
    adapter = repro(proxy_mode="competitors", proxy_competitors_k=2, seed_candidate=SEED, frozen_components=FROZEN)
    group = _group(evidence=("D2:1",))
    adapter.evaluate([group], SEED)
    adapter.evaluate([group], CAND)
    assert adapter.proxy_renotes == 3  # evidence idx 2 + fake top-2 = {0, 1}
    assert [n.keywords for n in _Agent.instances[-1].memory_system.memories.values()] == [["cand"]] * 3


def test_proxy_builds_the_seed_once_if_it_is_not_cached_yet(repro):
    adapter = repro(proxy_mode="evidence", seed_candidate=SEED, frozen_components=FROZEN)
    adapter.evaluate([_group()], CAND)
    assert adapter.builds_performed == 1 and adapter.proxy_evaluations == 1
    adapter.evaluate([_group()], CAND)
    assert adapter.builds_performed == 1 and adapter.proxy_evaluations == 2


def test_cached_seed_build_is_not_mutated_by_proxy_runs(repro):
    adapter = repro(proxy_mode="evidence", seed_candidate=SEED, frozen_components=FROZEN)
    group = _group()
    adapter.evaluate([group], CAND)
    adapter.evaluate([group], SEED)  # seed again: cache hit, must still be seed-noted
    notes = list(_Agent.instances[-1].memory_system.memories.values())
    assert [n.keywords for n in notes] == [["seed"]] * 3


def test_frozen_violation_still_scores_zero_without_proxy(repro):
    adapter = repro(proxy_mode="evidence", seed_candidate=SEED, frozen_components=FROZEN)
    bad = dict(CAND, evolution_decision="CHANGED {context}{content}{keywords}{nearest_neighbors_memories}")
    out = adapter.evaluate([_group()], bad)
    assert out.scores == [0.0] and adapter.proxy_evaluations == 0 and adapter.builds_performed == 0


def test_reflective_dataset_attributes_a_proxy_failure_to_the_candidates_note(repro):
    adapter = repro(proxy_mode="evidence", seed_candidate=SEED, frozen_components=FROZEN)
    group = _group(gold="beach")
    adapter.evaluate([group], SEED)
    _LLM.answers[group.questions[0].question] = "park"  # candidate gets it wrong
    out = adapter.evaluate([group], CAND, capture_traces=True)
    ds = adapter.make_reflective_dataset(CAND, out, ["note_construction"])
    ex = ds["note_construction"][0]
    assert ex["Generated Outputs"] == "KEYWORDS: cand\nCONTEXT: cand ctx\nTAGS: cand"
    assert "Rex loves the beach" in ex["Inputs"]["turn_content (the {content} input)"]


def test_test_eval_uses_the_proxy_for_the_candidate_arm(repro, tmp_path):
    adapter = repro(proxy_mode="evidence", seed_candidate=SEED, frozen_components=FROZEN)
    q = "Where does Rex like to go?"
    _LLM.answers[q] = "beach"
    instances = [_instance(q, "beach", evidence=["D1:2"])]
    report = run_test_eval(adapter, SEED, CAND, instances, tmp_path / "out", bootstrap_resamples=20)
    assert report["adapter_stats"]["builds_performed"] == 1
    assert report["adapter_stats"]["proxy_evaluations"] == 1
    assert report["summary"][CANDIDATE_ARM]["overall"]["f1"] == 1.0

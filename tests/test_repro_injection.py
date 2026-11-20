"""Tests for repro_injection.py (docs/decisions/0014): baseline-file
fidelity, candidate validation, runtime prompt injection, and LLM-call
recording. memory_layer_robust is stubbed (importing the real one pulls
torch via memory_layer), but llm_text_parsers is imported for real -- it's
stdlib-only, and the byte-equality guard is the whole point: a submodule
bump that changes a prompt must fail here loudly."""

import sys
import types

import pytest

from amem_gepa.paper_repro import ensure_repro_repo_importable
from amem_gepa.repro_injection import (
    BASELINE_REPRO_DIR,
    COMPONENT_TO_CONSTANT,
    COMPONENTS,
    BuildTrace,
    LLMCallRecord,
    RecordingLLM,
    candidate_hash,
    injected_prompts,
    load_baseline_candidate,
    validate_candidate,
)


def test_baseline_files_match_the_vendored_constants_byte_for_byte():
    ensure_repro_repo_importable()
    import llm_text_parsers as p

    candidate = load_baseline_candidate()
    for component, constant in COMPONENT_TO_CONSTANT.items():
        assert candidate[component] == getattr(p, constant), (
            f"{BASELINE_REPRO_DIR / (component + '.txt')} drifted from "
            f"llm_text_parsers.{constant} -- regenerate the baseline files "
            "(they are generated, not hand-copied; see docs/decisions/0014)"
        )


def test_baseline_candidate_is_valid():
    assert validate_candidate(load_baseline_candidate()) == []


def test_validate_catches_missing_component():
    candidate = load_baseline_candidate()
    del candidate["evolution_decision"]
    problems = validate_candidate(candidate)
    assert any("evolution_decision" in p and "missing" in p for p in problems)


def test_validate_catches_dropped_placeholder():
    candidate = load_baseline_candidate()
    candidate["note_construction"] = "Analyze this and reply with KEYWORDS/CONTEXT/TAGS."
    problems = validate_candidate(candidate)
    assert any("note_construction" in p and "{content}" in p for p in problems)


def test_validate_catches_invented_placeholder():
    candidate = load_baseline_candidate()
    candidate["note_construction"] += "\nPrior notes: {existing_notes}"
    problems = validate_candidate(candidate)
    assert any("unknown placeholder" in p and "{existing_notes}" in p for p in problems)


def test_validate_catches_positional_and_unbalanced_braces():
    candidate = load_baseline_candidate()
    candidate["evolution_strengthen"] = "Link {} to {content}{keywords}{nearest_neighbors_memories}"
    assert any("positional" in p for p in validate_candidate(candidate))

    candidate = load_baseline_candidate()
    candidate["evolution_strengthen"] = "Unbalanced {content brace {keywords} {nearest_neighbors_memories}"
    assert any("not a valid str.format template" in p for p in validate_candidate(candidate))

    # Literal JSON braces parse as a bogus field, caught as an unknown
    # placeholder -- same crash the real .format() call would hit.
    candidate = load_baseline_candidate()
    candidate["evolution_strengthen"] = (
        "Respond as JSON: {\"tags\": []} using {content} {keywords} {nearest_neighbors_memories}"
    )
    assert any("unknown placeholder" in p for p in validate_candidate(candidate))


def test_candidate_hash_is_stable_and_content_sensitive():
    a = load_baseline_candidate()
    b = dict(reversed(list(a.items())))  # same content, different key order
    assert candidate_hash(a) == candidate_hash(b)

    c = dict(a)
    c["note_construction"] += " "
    assert candidate_hash(c) != candidate_hash(a)


@pytest.fixture
def fake_memory_layer_robust(monkeypatch):
    fake = types.ModuleType("memory_layer_robust")
    for constant in COMPONENT_TO_CONSTANT.values():
        setattr(fake, constant, f"ORIGINAL {constant} {{content}}")
    fake.FOCUSED_KEYWORDS_PROMPT = "ORIGINAL FOCUSED {content}"
    monkeypatch.setitem(sys.modules, "memory_layer_robust", fake)
    return fake


def test_injected_prompts_swaps_and_restores(fake_memory_layer_robust):
    candidate = {component: f"EVOLVED {component}" for component in COMPONENTS}

    with injected_prompts(candidate):
        for component, constant in COMPONENT_TO_CONSTANT.items():
            assert getattr(fake_memory_layer_robust, constant) == f"EVOLVED {component}"
        # The keyword-recovery fallback is frozen, not a component (0014).
        assert fake_memory_layer_robust.FOCUSED_KEYWORDS_PROMPT == "ORIGINAL FOCUSED {content}"

    for constant in COMPONENT_TO_CONSTANT.values():
        assert getattr(fake_memory_layer_robust, constant) == f"ORIGINAL {constant} {{content}}"


def test_injected_prompts_restores_on_exception(fake_memory_layer_robust):
    candidate = {component: f"EVOLVED {component}" for component in COMPONENTS}
    with pytest.raises(RuntimeError), injected_prompts(candidate):
        raise RuntimeError("build blew up")
    for constant in COMPONENT_TO_CONSTANT.values():
        assert getattr(fake_memory_layer_robust, constant).startswith("ORIGINAL")


class _EchoLLM:
    def __init__(self):
        self.calls = []

    def get_completion(self, prompt, temperature=0.7):
        self.calls.append(prompt)
        return f"response to {len(self.calls)}"

    def check_connectivity(self):
        return "ok"


def test_recording_llm_classifies_by_template_prefix():
    candidate = load_baseline_candidate()
    recorder = RecordingLLM(_EchoLLM(), candidate)

    recorder.get_completion(candidate["note_construction"].format(content="hello world"))
    recorder.get_completion(
        candidate["evolution_decision"].format(
            context="c", content="x", keywords="k", nearest_neighbors_memories="n"
        )
    )
    recorder.get_completion("Given the following question, generate several keywords separated by commas.\n\nQuestion: q\n\nKeywords:")
    recorder.get_completion("Based on the context: stuff, answer the following question. q")
    recorder.get_completion("something else entirely")

    assert [r.kind for r in recorder.records] == [
        "note_construction",
        "evolution_decision",
        "query_keywords",
        "qa_answer",
        "other",
    ]
    # Passthrough of non-wrapped attributes still works.
    assert recorder.check_connectivity() == "ok"


def test_build_trace_round_trips_json():
    trace = BuildTrace(
        conversation_id="conv-1",
        candidate_hash="abc123",
        records=[LLMCallRecord(kind="note_construction", prompt="p", response="r", temperature=0.7)],
    )
    restored = BuildTrace.from_json(trace.to_json())
    assert restored.conversation_id == "conv-1"
    assert restored.records[0].kind == "note_construction"
    assert restored.records[0].response == "r"

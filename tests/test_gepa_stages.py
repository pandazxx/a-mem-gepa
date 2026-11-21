"""Tests for sequential single-prompt training (docs/decisions/0019): the
fixed component selector, the frozen-component guard, seed assembly, and an
end-to-end check against the installed gepa package that a run with the
selector never changes the frozen components."""

import pytest
from gepa.core.adapter import EvaluationBatch

from amem_gepa.gepa_stages import (
    COMPONENT_ORDER,
    FixedComponentSelector,
    changed_components,
    frozen_components,
    frozen_violations,
    load_seed_candidate,
    parse_seed_override,
)
from amem_gepa.repro_injection import COMPONENTS

SEED = {name: f"{name} seed {{content}}" for name in COMPONENTS}


def test_component_order_covers_every_component_and_starts_with_note_construction():
    assert set(COMPONENT_ORDER) == set(COMPONENTS)
    assert COMPONENT_ORDER[0] == "note_construction"


def test_fixed_selector_always_returns_its_component_and_ignores_state():
    selector = FixedComponentSelector("note_construction")
    # gepa passes (state, trajectories, subsample_scores, candidate_idx, candidate);
    # none of it may influence the answer.
    for idx in range(3):
        assert selector(object(), [], [0.1], idx, dict(SEED)) == ["note_construction"]


def test_fixed_selector_rejects_unknown_component():
    with pytest.raises(ValueError, match="unknown component"):
        FixedComponentSelector("qa_answer")


def test_frozen_components_is_everything_but_the_target():
    frozen = frozen_components(SEED, "evolution_decision")
    assert set(frozen) == set(COMPONENTS) - {"evolution_decision"}
    assert all(frozen[k] == SEED[k] for k in frozen)


def test_frozen_violations_flag_only_changed_frozen_components():
    frozen = frozen_components(SEED, "note_construction")
    free_change = dict(SEED, note_construction="rewritten {content}")
    assert frozen_violations(free_change, frozen) == []

    bad = dict(free_change, evolution_strengthen="tampered {content}")
    problems = frozen_violations(bad, frozen)
    assert len(problems) == 1
    assert problems[0].startswith("evolution_strengthen:")
    assert "FROZEN" in problems[0] and "note_construction" in problems[0]

    assert frozen_violations(bad, {}) == [], "no frozen components -> nothing to violate"


def test_changed_components_lists_in_canonical_order():
    cand = dict(SEED, evolution_update_neighbors="x {content}", note_construction="y {content}")
    assert changed_components(cand, SEED) == ["note_construction", "evolution_update_neighbors"]
    assert changed_components(SEED, SEED) == []


def test_parse_seed_override():
    assert parse_seed_override("note_construction=prompts/note_construction.v1.txt") == (
        "note_construction",
        __import__("pathlib").Path("prompts/note_construction.v1.txt"),
    )
    with pytest.raises(ValueError):
        parse_seed_override("note_construction")
    with pytest.raises(ValueError):
        parse_seed_override("bogus=x.txt")
    with pytest.raises(ValueError):
        parse_seed_override("note_construction=")


def test_load_seed_candidate_layers_overrides_on_the_seed_dir(tmp_path):
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    for name, text in SEED.items():
        (seed_dir / f"{name}.txt").write_text(text)
    v1 = tmp_path / "note_construction.v1.txt"
    v1.write_text("stage-1 winner {content}")

    assert load_seed_candidate(seed_dir) == SEED
    layered = load_seed_candidate(seed_dir, {"note_construction": v1})
    assert layered["note_construction"] == "stage-1 winner {content}"
    assert {k: v for k, v in layered.items() if k != "note_construction"} == {
        k: v for k, v in SEED.items() if k != "note_construction"
    }


class _ToyAdapter:
    """Minimal GEPAAdapter: a candidate scores by the length of its
    note_construction text, so the proposer below always 'improves' it."""

    propose_new_texts = None

    def __init__(self):
        self.asked_to_update: list[list[str]] = []

    def evaluate(self, batch, candidate, capture_traces=False):
        score = min(len(candidate["note_construction"]) / 1000.0, 0.99)
        return EvaluationBatch(
            outputs=[None for _ in batch],
            scores=[score for _ in batch],
            trajectories=[{} for _ in batch] if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        self.asked_to_update.append(list(components_to_update))
        return {
            c: [{"Inputs": {}, "Generated Outputs": "", "Feedback": "make it longer"}]
            for c in components_to_update
        }


def test_gepa_run_with_fixed_selector_only_ever_touches_the_target_component(tmp_path):
    """End to end against the installed gepa: with FixedComponentSelector
    every reflective proposal asks about note_construction only, and every
    candidate in the pool keeps the other three components byte-identical to
    the seed -- the guarantee docs/decisions/0019 rests on."""
    import gepa

    adapter = _ToyAdapter()
    proposals: list[list[str]] = []

    def proposer(candidate, reflective_dataset, components_to_update):
        proposals.append(list(components_to_update))
        return {c: candidate[c] + " more" for c in components_to_update}

    result = gepa.optimize(
        seed_candidate=dict(SEED),
        trainset=list(range(4)),
        valset=list(range(3)),
        adapter=adapter,
        custom_candidate_proposer=proposer,
        module_selector=FixedComponentSelector("note_construction"),
        reflection_minibatch_size=1,
        max_metric_calls=40,
        run_dir=str(tmp_path / "run"),
        display_progress_bar=False,
        raise_on_exception=True,
        seed=0,
    )

    assert len(result.candidates) > 1, "the toy setup must accept at least one proposal"
    assert proposals and all(p == ["note_construction"] for p in proposals)
    assert all(p == ["note_construction"] for p in adapter.asked_to_update)
    frozen = frozen_components(SEED, "note_construction")
    for cand in result.candidates:
        assert frozen_violations(cand, frozen) == []
    assert changed_components(result.best_candidate, SEED) == ["note_construction"]


def test_gepa_default_round_robin_does_rotate_components(tmp_path):
    """Control for the test above: without the fixed selector gepa's default
    round-robin asks about different components across iterations, which is
    exactly what docs/decisions/0019 moves away from."""
    import gepa

    adapter = _ToyAdapter()
    proposals: list[list[str]] = []

    def proposer(candidate, reflective_dataset, components_to_update):
        proposals.append(list(components_to_update))
        return {c: candidate[c] + " more" for c in components_to_update}

    gepa.optimize(
        seed_candidate=dict(SEED),
        trainset=list(range(4)),
        valset=list(range(3)),
        adapter=adapter,
        custom_candidate_proposer=proposer,
        reflection_minibatch_size=1,
        max_metric_calls=40,
        run_dir=str(tmp_path / "run"),
        display_progress_bar=False,
        raise_on_exception=True,
        seed=0,
    )
    assert len({tuple(p) for p in proposals}) > 1

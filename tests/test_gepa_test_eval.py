"""Tests for the held-out test comparison (docs/decisions/0018), on the same
stubbed repro modules test_gepa_adapter.py uses."""

import json

from amem_gepa.datasets.locomo import LoCoMoQuestionGroup  # noqa: F401 -- keeps fixture import order stable
from amem_gepa.gepa_test_eval import (
    BASELINE_ARM,
    CANDIDATE_ARM,
    QuestionResult,
    format_comparison,
    paired_bootstrap,
    run_test_eval,
    summarize_arm,
)
from tests.test_gepa_adapter import CANDIDATE, TURNS, _FakeAgent, _instance, fake_repro  # noqa: F401

BASELINE = dict(CANDIDATE, note_construction="BASE analyze: {content}")


def _instances():
    _FakeAgent.answers.update({"Where is Rex?": "beach", "What color is the collar?": "Not mentioned in the conversation"})
    return [
        _instance("Where is Rex?", "beach", category=4, conv="conv-a"),
        _instance("What color is the collar?", None, category=5, conv="conv-a", adversarial="red"),
        _instance("Where is Rex?", "beach", category=4, conv="conv-b"),
    ]


def test_run_test_eval_scores_both_arms_and_writes_report(fake_repro, tmp_path):
    adapter = fake_repro()
    report = run_test_eval(adapter, BASELINE, CANDIDATE, _instances(), tmp_path / "out", bootstrap_resamples=200)

    assert set(report["summary"]) == {BASELINE_ARM, CANDIDATE_ARM}
    cand = report["summary"][CANDIDATE_ARM]
    assert cand["overall"]["n"] == 3
    assert cand["single_hop"]["f1"] == 1.0
    # Paper reading: scored against the trap text -> 0; refusal reading -> 1.
    assert cand["adversarial"]["f1"] == 0.0
    assert cand["adversarial"]["refusal_rate"] == 1.0
    # Identical answers in both arms -> zero difference with a zero-width interval.
    bs = report["bootstrap_f1"]["overall"]
    assert bs["n"] == 3 and bs["diff_mean"] == 0.0 and bs["ci_low"] == 0.0 == bs["ci_high"]
    # One build per (arm, conversation).
    assert adapter.builds_performed == 4
    assert (tmp_path / "out" / "test_eval.json").exists()
    table = (tmp_path / "out" / "summary.md").read_text()
    assert "| single_hop | 2 |" in table and "refusal reading" in table


def test_test_eval_resumes_from_the_per_question_progress_log(fake_repro, tmp_path):
    adapter = fake_repro()
    run_test_eval(adapter, BASELINE, CANDIDATE, _instances(), tmp_path / "out", bootstrap_resamples=50)
    agents_before = len(_FakeAgent.instances)
    for arm in (BASELINE_ARM, CANDIDATE_ARM):
        assert len((tmp_path / "out" / f"qa_progress_{arm}.jsonl").read_text().splitlines()) == 3

    _FakeAgent.fail_build = True  # any rebuild would now fail -> proves nothing was rebuilt
    report = run_test_eval(fake_repro(), BASELINE, CANDIDATE, _instances(), tmp_path / "out", bootstrap_resamples=50)
    assert len(_FakeAgent.instances) == agents_before
    assert report["summary"][CANDIDATE_ARM]["overall"]["errors"] == 0


def test_paired_bootstrap_detects_a_consistent_improvement():
    def rs(arm, f1s):
        return [
            QuestionResult(arm, "c", i, 4 if i % 2 else 2, f"q{i}", "p", "r", {"f1": f})
            for i, f in enumerate(f1s)
        ]

    base = rs(BASELINE_ARM, [0.2] * 20)
    cand = rs(CANDIDATE_ARM, [0.5] * 10 + [0.4] * 10)
    out = paired_bootstrap(base, cand, "f1", n_resamples=500, seed=1)
    assert out["overall"]["n"] == 20
    assert abs(out["overall"]["diff_mean"] - 0.25) < 1e-9
    assert out["overall"]["ci_low"] > 0.0, "a uniform gain must not have an interval touching zero"
    assert set(out) >= {"overall", "temporal", "single_hop"}


def test_summarize_and_format_handle_baseline_only():
    base = [QuestionResult(BASELINE_ARM, "c", 0, 1, "q", "p", "r", {"f1": 0.5, "bleu1": 0.25})]
    summary = {BASELINE_ARM: summarize_arm(base)}
    table = format_comparison(summary, {}, None)
    assert "| multi_hop | 1 | 0.5000 | 0.2500 |" in table
    assert json.dumps(summary)  # serializable

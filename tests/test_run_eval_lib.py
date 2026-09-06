import json

import amem_gepa.run_eval_lib as run_eval_lib
from amem_gepa.evaluate import EvalResult, InstanceResult
from amem_gepa.metrics import ScoreSummary


def test_run_and_report_writes_results_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    (tmp_path / "note_construction.txt").write_text("construct {content}")
    (tmp_path / "evolution.txt").write_text("evolve {content} {context} {keywords} {nearest_neighbors_memories} {neighbor_number}")
    (tmp_path / "qa_answer.txt").write_text("{retrieved_memories} {question}")

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        f"""
dataset:
  split_manifest: {tmp_path / "split.json"}
models:
  amem_llm_model: fake-model
  amem_embedding_model: fake-embed
evaluation:
  split: test
  retrieval_k: 5
  qa_prompt: {tmp_path / "qa_answer.txt"}
  bootstrap_resamples: 100
"""
    )

    fake_result = EvalResult(
        instance_results=[
            InstanceResult("conv-x", "Q?", 1, "pred", "gold", None, 1.0),
        ],
        summaries={"aggregate": ScoreSummary(mean=1.0, n=1, ci_low=1.0, ci_high=1.0)},
    )

    monkeypatch.setattr(run_eval_lib, "load_split", lambda split, manifest_path: [object()])
    captured = {}

    def fake_evaluate_candidate(instances, **kwargs):
        captured["kwargs"] = kwargs
        return fake_result

    monkeypatch.setattr(run_eval_lib, "evaluate_candidate", fake_evaluate_candidate)

    summaries = run_eval_lib.run_and_report(
        config_path=str(config_path),
        note_construction_path=tmp_path / "note_construction.txt",
        evolution_path=tmp_path / "evolution.txt",
        run_label="unit-test-run",
    )

    assert summaries["aggregate"].mean == 1.0
    assert captured["kwargs"]["llm_model"] == "fake-model"
    assert captured["kwargs"]["k"] == 5

    out_path = tmp_path / "results" / "unit-test-run" / "test.json"
    assert out_path.exists()
    written = json.loads(out_path.read_text())
    assert written["model"] == "fake-model"
    assert written["summaries"]["aggregate"]["mean"] == 1.0
